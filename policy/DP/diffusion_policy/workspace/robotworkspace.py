if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import time
import threading
import hydra
import torch
import torch.nn as nn
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import copy
import wandb
import tqdm, random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace, _copy_to_cpu
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _cuda_synchronize(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _reduce_train_metrics(metrics: dict, device: torch.device, multi_gpu: bool) -> dict:
    if not multi_gpu:
        return metrics
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return metrics

    reduced = dict(metrics)
    world_size = torch.distributed.get_world_size()

    loss = torch.tensor([float(metrics["train_loss"])], dtype=torch.float32, device=device)
    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
    reduced["train_loss"] = (loss / world_size).item()

    max_keys = [
        "data_time_s",
        "postprocess_time_s",
        "train_time_s",
        "optim_time_s",
        "step_time_s",
        "gpu_mem_gb",
    ]
    values = torch.tensor([float(metrics[key]) for key in max_keys], dtype=torch.float32, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    for key, value in zip(max_keys, values.tolist()):
        reduced[key] = value
    return reduced


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # DDP settings (DDP wrapping happens in run() after device transfer)
        self.rank = cfg.training.rank
        self.world_size = cfg.training.world_size
        self.multi_gpu = cfg.training.multi_gpu

    @property
    def _model(self):
        """Return unwrapped model (handles DDP wrapper in PyTorch 2.0)."""
        if self.multi_gpu and hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        seed = cfg.training.seed

        # resume training (handle DDP module. prefix)
        if cfg.training.resume:
            ckpt_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            # New format: bs{bs}_ep{N}.ckpt + bs{bs}_ep{N}_optimizer.ckpt
            # Old format: last.ckpt (model+ema+optimizer in one file)
            # Sort by epoch NUMBER (not string — "ep30" > "ep5")
            _bs_ep_files = sorted(
                (p for p in ckpt_dir.glob("bs*_ep*.ckpt")
                 if "_optimizer" not in p.name),
                key=lambda p: int(p.stem.rsplit("_ep", 1)[-1]),
            )
            if _bs_ep_files:
                resume_ckpt_path = _bs_ep_files[-1]  # latest epoch
            else:
                resume_ckpt_path = ckpt_dir / "last.ckpt"
            if resume_ckpt_path.is_file():
                if self.rank == 0:
                    print(f"Resuming from checkpoint {resume_ckpt_path}")
                payload = torch.load(resume_ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
                # Strip DDP "module." prefix if present (model not yet wrapped)
                sd = payload.get("state_dicts", {})
                for key in list(sd.keys()):
                    state = sd[key]
                    stripped = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
                    if stripped != state:
                        sd[key] = stripped
                self.load_payload(payload)
                # Load optimizer from separate file (new format) or from
                # the same payload (old format, last.ckpt).
                if "optimizer" not in sd:
                    opt_path = pathlib.Path(str(resume_ckpt_path).replace(".ckpt", "_optimizer.ckpt"))
                    if opt_path.is_file():
                        if self.rank == 0:
                            print(f"Loading optimizer from {opt_path}")
                        opt_payload = torch.load(opt_path.open("rb"), pickle_module=dill, map_location="cpu")
                        self.optimizer.load_state_dict(opt_payload["optimizer"])
                        del opt_payload
                    elif self.rank == 0:
                        print("[DP] WARNING: resume ckpt has no optimizer — "
                              "starting with fresh optimizer state")
                del payload

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader, train_sampler = create_dataloader(
            dataset, **cfg.dataloader,
            multi_gpu=self.multi_gpu, rank=self.rank, world_size=self.world_size,
        )
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = None
        if len(val_dataset) > 0 and self.rank == 0:
            val_dataloader, _ = create_dataloader(
                val_dataset, **cfg.val_dataloader,
                multi_gpu=False, rank=0, world_size=1,
            )
        elif len(val_dataset) == 0 and self.rank == 0:
            print("[DP] Validation disabled; using all episodes for training.")

        self._model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        # total_training_steps covers the ENTIRE schedule (0 → num_epochs).
        # On resume last_epoch skips warmup and positions the scheduler
        # correctly on the cosine curve.
        total_training_steps = (
            (len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every
        )
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=total_training_steps,
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        env_runner = None

        # configure logging (rank 0 only)
        wandb_run = None
        if self.rank == 0:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint (rank 0 only)
        if self.rank == 0:
            topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"),
                                                 **cfg.checkpoint.topk)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # DDP: wrap model after device transfer
        if self.multi_gpu:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[cfg.training.local_rank],
                output_device=cfg.training.local_rank,
                find_unused_parameters=cfg.training.freeze_encoder,
            )

        # save batch for sampling
        train_sampling_batch = None
        disable_train_sampling = bool(cfg.training.get("disable_train_sampling", False))
        log_every_steps = max(1, int(cfg.training.get("log_every_steps", 20)))
        checkpoint_every = max(1, int(cfg.training.checkpoint_every))
        max_keep_checkpoints = max(1, int(cfg.training.get("max_keep_checkpoints", 2)))
        _milestone_raw = cfg.training.get("milestone_epochs", None)
        milestone_epochs = max(1, int(_milestone_raw)) if _milestone_raw is not None else None
        per_gpu_batch_size = int(cfg.dataloader.batch_size)
        configured_global_batch = cfg.training.get("global_batch_size", None)
        effective_batch_size = (
            int(configured_global_batch)
            if configured_global_batch is not None
            else per_gpu_batch_size * self.world_size
        ) * int(cfg.training.gradient_accumulate_every)

        if self.rank == 0:
            print(
                "[DP] batch summary: "
                f"global_batch_size={configured_global_batch or per_gpu_batch_size * self.world_size}, "
                f"per_gpu_batch_size={per_gpu_batch_size}, "
                f"world_size={self.world_size}, "
                f"gradient_accumulate_every={cfg.training.gradient_accumulate_every}, "
                f"effective_batch_size={effective_batch_size}, "
                f"steps_per_epoch={len(train_dataloader)}"
            )

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        json_logger = JsonLogger(log_path) if self.rank == 0 else None
        if json_logger is not None:
            json_logger.__enter__()

        try:
            while self.epoch < cfg.training.num_epochs:
                step_log = dict()

                # set DistributedSampler epoch for shuffling
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)

                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self._model.obs_encoder.eval()
                    self._model.obs_encoder.requires_grad_(False)

                train_losses = list()
                # Only rank 0 shows tqdm (DDP — other ranks stay silent).
                if self.rank == 0:
                    _pbar = tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    )
                else:
                    _pbar = train_dataloader
                data_iter = iter(_pbar)
                batch_idx = 0
                while True:
                    data_start = time.perf_counter()
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        break
                    data_time_s = time.perf_counter() - data_start
                    is_last_batch = batch_idx == (len(train_dataloader) - 1)
                    postprocess_start = time.perf_counter()
                    batch = dataset.postprocess(batch, device)
                    _cuda_synchronize(device)
                    postprocess_time_s = time.perf_counter() - postprocess_start
                    if (
                        self.rank == 0
                        and not disable_train_sampling
                        and train_sampling_batch is None
                    ):
                        train_sampling_batch = batch
                    # compute loss
                    train_start = time.perf_counter()
                    raw_loss = self._model.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    _cuda_synchronize(device)
                    train_time_s = time.perf_counter() - train_start

                    # step optimizer
                    optim_time_s = 0.0
                    if (
                        ((self.global_step + 1) % cfg.training.gradient_accumulate_every == 0)
                        or is_last_batch
                    ):
                        optim_start = time.perf_counter()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                        _cuda_synchronize(device)
                        optim_time_s = time.perf_counter() - optim_start

                    # update ema (use unwrapped model for parameter alignment)
                    if cfg.training.use_ema:
                        ema.step(self._model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    gpu_mem_gb = 0.0
                    if torch.cuda.is_available() and device.type == "cuda":
                        gpu_mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    step_time_s = time.perf_counter() - data_start
                    local_metrics = {
                        "train_loss": raw_loss_cpu,
                        "data_time_s": data_time_s,
                        "postprocess_time_s": postprocess_time_s,
                        "train_time_s": train_time_s,
                        "optim_time_s": optim_time_s,
                        "step_time_s": step_time_s,
                        "gpu_mem_gb": gpu_mem_gb,
                    }
                    # Only synchronise metrics across ranks on logging steps
                    # to avoid 255 all_reduce calls/epoch that create 255
                    # collective stalls when any rank's dataloader hangs.
                    is_log_step = (
                        is_last_batch
                        or (self.global_step % log_every_steps) == 0
                    )
                    if is_log_step and self.multi_gpu:
                        reduced_metrics = _reduce_train_metrics(
                            local_metrics,
                            device=device,
                            multi_gpu=True,
                        )
                    else:
                        reduced_metrics = local_metrics
                    if self.rank == 0:
                        _pbar.set_postfix(loss=local_metrics["train_loss"], refresh=False)
                    train_losses.append(local_metrics["train_loss"])
                    step_log = {
                        "train_loss": reduced_metrics["train_loss"],
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "data_time_s": reduced_metrics["data_time_s"],
                        "postprocess_time_s": reduced_metrics["postprocess_time_s"],
                        "train_time_s": reduced_metrics["train_time_s"],
                        "optim_time_s": reduced_metrics["optim_time_s"],
                        "step_time_s": reduced_metrics["step_time_s"],
                        "gpu_mem_gb": reduced_metrics["gpu_mem_gb"],
                        "effective_batch_size": effective_batch_size,
                        "per_gpu_batch_size": per_gpu_batch_size,
                        "world_size": self.world_size,
                        "steps_per_epoch": len(train_dataloader),
                    }

                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if json_logger is not None and (self.global_step % log_every_steps) == 0:
                            json_logger.log(step_log)
                        self.global_step += 1

                    if (cfg.training.max_train_steps
                            is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                        break
                    batch_idx += 1

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self._model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if val_dataloader is not None and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dataset.postprocess(batch, device)
                                loss = self._model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps
                                        is not None) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch (rank 0 only)
                if (
                    self.rank == 0
                    and not disable_train_sampling
                    and (self.epoch % cfg.training.sample_every) == 0
                ):
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # ========= eval end for this epoch ==========
                policy.train()

                # Advance step/epoch BEFORE save so the checkpoint's internal
                # epoch matches its filename (both use the value representing
                # "after this epoch").  Otherwise the file is ep460 but the
                # pickled epoch is 459, causing every resume to waste 1 epoch.
                self.global_step += 1
                self.epoch += 1

                # end of epoch (rank 0 only for logging & checkpointing)
                if self.rank == 0:
                    checkpoint_time_s = 0.0
                    should_save = (
                        (self.epoch % checkpoint_every == 0)
                        or (self.epoch == cfg.training.num_epochs)
                    )
                    if should_save:
                        checkpoint_start = time.perf_counter()
                        # Wait for any in-flight async save before starting a new one.
                        if (
                            self._saving_thread is not None
                            and self._saving_thread.is_alive()
                        ):
                            self._saving_thread.join()
                        # Split into two files to avoid duplicating model+ema:
                        #   bs{bs}_ep{N}.ckpt           — model + ema  (deploy / upload)
                        #   bs{bs}_ep{N}_optimizer.ckpt — optimizer    (resume only)
                        ckpt_name = f"bs{effective_batch_size}_ep{self.epoch}"
                        ckpt_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
                        model_ckpt_path = ckpt_dir / f"{ckpt_name}.ckpt"
                        self.save_checkpoint(
                            model_ckpt_path,
                            exclude_keys=("optimizer",),
                            use_thread=True,
                        )
                        # Manually save optimizer to a separate file (it has no
                        # state_dict/load_state_dict triple that save_checkpoint expects).
                        opt_path = ckpt_dir / f"{ckpt_name}_optimizer.ckpt"
                        opt_cpu = _copy_to_cpu(self.optimizer.state_dict())
                        threading.Thread(
                            target=lambda: torch.save(
                                {"optimizer": opt_cpu},
                                opt_path.open("wb"),
                                pickle_module=dill,
                            )
                        ).start()
                        checkpoint_time_s = time.perf_counter() - checkpoint_start
                        print(
                            f"Saved checkpoint {ckpt_name} "
                            f"(async, cpu_copy+launch={checkpoint_time_s:.2f}s)"
                        )
                        # --- rotate: keep only latest N checkpoints ---
                        # Milestones (epoch % milestone_epochs == 0) are
                        # exempt from deletion.
                        _all_ckpts = sorted(
                            (p for p in ckpt_dir.glob("bs*_ep*.ckpt")
                             if "_optimizer" not in p.name),
                            key=lambda p: int(p.stem.rsplit("_ep", 1)[-1]),
                        )
                        _milestones = []
                        _regulars = []
                        for _p in _all_ckpts:
                            try:
                                _ep = int(_p.stem.rsplit("_ep", 1)[-1])
                            except ValueError:
                                continue
                            if milestone_epochs and _ep % milestone_epochs == 0:
                                _milestones.append(_p)
                            else:
                                _regulars.append(_p)
                        while len(_regulars) > max_keep_checkpoints:
                            _old = _regulars.pop(0)
                            _old.unlink()
                            _old_opt = _old.with_name(
                                _old.name.replace(".ckpt", "_optimizer.ckpt")
                            )
                            if _old_opt.is_file():
                                _old_opt.unlink()
                        if _milestones:
                            print(
                                f"[DP] {len(_regulars)} rolling + "
                                f"{len(_milestones)} milestone checkpoints kept"
                            )
                        # --------------------------------------------------
                    step_log["checkpoint_time_s"] = checkpoint_time_s
                    json_logger.log(step_log)
                else:
                    should_save = (
                        (self.epoch % checkpoint_every == 0)
                        or (self.epoch == cfg.training.num_epochs)
                    )
                if self.multi_gpu and should_save:
                    torch.distributed.barrier()

        finally:
            if json_logger is not None:
                json_logger.__exit__(None, None, None)


class BatchSampler:

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[:-self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
    multi_gpu: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    def collate(x):
        # Always add batch dimension, even for single samples.
        # A single sample without batch dim breaks downstream shape checks
        # (e.g. multi_image_obs_encoder expects B,C,H,W not C,H,W).
        keys = x[0].keys()
        return {key: np.stack([sample[key] for sample in x], axis=0) for key in keys}

    if multi_gpu:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        return dataloader, sampler

    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)
    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return dataloader, None


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = RobotWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
