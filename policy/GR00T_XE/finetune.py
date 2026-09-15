# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-task finetune, either from a GR00T XE pretrain or from the base model.

Two-stage (default) -- load the cross-embodiment pretrain, finetune on one task::

    python policy/GR00T_XE/finetune.py \\
        --dataset-path /data/73_jigsaw/replay-generalization \\
        --pretrained-ckpt /ckpt/gr00t_xe_pretrain/checkpoint-520000 \\
        --output-dir /ckpt/73/gr00t_xe_ft \\
        --max-steps 5000 --batch-size 64

Single-stage (``--init-from-base``) -- skip the pretrain entirely and train the
one task from GR00T-N1.5-3B, the recipe GR00T n15 uses::

    python policy/GR00T_XE/finetune.py \\
        --dataset-path /data/73_jigsaw/replay-generalization \\
        --init-from-base --base-model-path /models/GR00T-N1.5-3B \\
        --output-dir /ckpt/73/gr00t_xe_full \\
        --max-steps 20000 --batch-size 64 --learning-rate 1e-4
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import tyro
from transformers import TrainingArguments, TrainerCallback

from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.utils.peft import get_lora_model, copy_partial_action_expert_weights


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    """Single-task finetune from GR00T XE pretrained checkpoint."""

    # ---- data -----------------------------------------------------------
    dataset_path: str = ""
    """Single task dataset directory (e.g. .../73_jigsaw/replay-generalization)."""

    data_config: str = "policy.GR00T_XE.xe_config:Dex2BenchXEDataConfig"
    """Cross-embodiment data config (max_action_dim=64, max_state_dim=64)."""

    hdf5_native: bool = True
    hdf5_camera_map: list[str] | None = None
    hdf5_prompt: str | None = None
    hdf5_use_active_dof: bool = True
    hdf5_robot_key: str | None = None
    hdf5_state_dim: int | None = None
    hdf5_action_dim: int | None = None
    hdf5_truncate_at_homing: bool = True

    # ---- model ----------------------------------------------------------
    pretrained_ckpt: str = ""
    """Path to the pretrained GR00T XE checkpoint directory."""

    init_from_base: bool = False
    """Start from ``base_model_path`` instead of a GR00T XE pretrain checkpoint.

    Single-stage training on one task: no cross-embodiment pretrain involved.
    This is the recipe GR00T n15 was trained with, so it isolates the XE data
    pipeline (64-dim unified space, EE arm dims + IK decode) and the inference
    pipeline from the two-stage pretrain/finetune.
    """

    output_dir: str = "/tmp/gr00t_xe_finetune"
    """Directory to save finetuned checkpoints."""

    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Base model path.  Used as the starting point when ``init_from_base``."""

    max_state_dim: int = 64
    max_action_dim: int = 64

    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # ---- training -------------------------------------------------------
    batch_size: int = 64
    max_steps: int = 5000
    num_gpus: int = 1
    save_steps: int = 500
    milestone_steps: int = 0
    seed: int = 42
    resume: bool = False

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False

    dataloader_num_workers: int = 12
    gradient_accumulation_steps: int = 1
    dataloader_prefetch_factor: int = 4

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "tensorboard"

    embodiment_tag: str = "new_embodiment"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchvision_av"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hdf5_camera_map(items: list[str] | None) -> dict[str, str]:
    if items is None:
        items = [
            "stereo_left=cam_stereo_left",
            "stereo_right=cam_stereo_right",
            "right_wrist=cam_wrist_right",
            "left_wrist=cam_wrist_left",
        ]
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        out[k.strip()] = v.strip()
    return out


class MilestoneSaveCallback(TrainerCallback):
    def __init__(self, milestone_steps: int):
        self.milestone_steps = milestone_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.milestone_steps <= 0:
            return control
        if state.global_step > 0 and state.global_step % self.milestone_steps == 0:
            import shutil

            output_dir = Path(args.output_dir)
            last_ckpt = output_dir / f"checkpoint-{state.global_step}"
            milestone = output_dir / f"checkpoint-milestone-{state.global_step}"
            if last_ckpt.is_dir() and not milestone.is_dir():
                shutil.copytree(last_ckpt, milestone)
                print(f"[Milestone] {milestone}")
        return control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: FinetuneConfig) -> None:
    os.environ["GR00T_MAX_STATE_DIM"] = str(config.max_state_dim)
    os.environ["GR00T_MAX_ACTION_DIM"] = str(config.max_action_dim)

    # ---- 1. load dataset ------------------------------------------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    _temp_dir = Path(__file__).resolve().parent  # policy/GR00T_XE/
    if str(_temp_dir) not in sys.path:
        sys.path.insert(0, str(_temp_dir))

    from dataset import build_hdf5_dataset

    camera_map = _parse_hdf5_camera_map(config.hdf5_camera_map)
    train_dataset = build_hdf5_dataset(
        input_dir=config.dataset_path,
        camera_map=camera_map,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=config.embodiment_tag,
        prompt=config.hdf5_prompt,
        use_active_dof=config.hdf5_use_active_dof,
        robot_key=config.hdf5_robot_key,
        state_dim=config.hdf5_state_dim,
        action_dim=config.hdf5_action_dim,
        truncate_at_homing=config.hdf5_truncate_at_homing,
    )
    print(f"[Finetune] Loaded {len(train_dataset)} samples from {config.dataset_path}")

    # ---- 2. load model from pretrained checkpoint -----------------------
    data_action_horizon = len(data_config_cls.action_indices)
    # The dataset applied the frozen artifact to ITS transform in __init__ (and
    # raised if the artifact was missing), so read the dims off that object
    # rather than the unused instance built above.
    last_transform = train_dataset._transforms.transforms[-1]
    from gr00t.model.transforms import GR00TTransform

    assert isinstance(last_transform, GR00TTransform)
    data_max_action_dim = last_transform.max_action_dim

    # ---- starting point --------------------------------------------------
    #
    # Two ways in, and they need DIFFERENT config handling -- getting this wrong
    # is a shape-mismatch crash deep inside from_pretrained:
    #
    #   pretrain ckpt (default): a GR00T XE checkpoint whose saved config.json can
    #       disagree with its own weights (the pretrain was written with
    #       action_dim=32 in the config while the action expert was saved at 64).
    #       Forcing the config to 64 is what makes from_pretrained find a matching
    #       head, so this override has to stay for that path.
    #
    #   base model (--init-from-base): the stock GR00T-N1.5-3B. Its config and its
    #       weights AGREE at action_dim=32, so forcing 64 here would build a 64-dim
    #       head and then try to load 32-dim tensors into it. Load it with its own
    #       config instead and let the mismatch branch below rebuild the head at 64
    #       and carry the 32 trained dims over (copy_partial_action_expert_weights).
    if config.init_from_base:
        if config.pretrained_ckpt:
            raise ValueError(
                "both --init-from-base and --pretrained-ckpt were given; the "
                "starting point must be unambiguous"
            )
        if not config.base_model_path:
            raise ValueError("--init-from-base requires --base-model-path")
        init_dir = Path(config.base_model_path).expanduser()
        if not init_dir.is_dir():
            raise FileNotFoundError(f"Base model not found: {init_dir}")
        print(f"[Finetune] Initialising from the BASE model at {init_dir}")
        print(f"[Finetune]   single-stage: no XE pretrain; the action head will be "
              f"rebuilt at action_dim={data_max_action_dim} and the pretrained "
              f"dims carried over into it")
    else:
        if not config.pretrained_ckpt:
            raise ValueError(
                "no starting point: pass --pretrained-ckpt PATH, or "
                "--init-from-base together with --base-model-path"
            )
        init_dir = Path(config.pretrained_ckpt).expanduser()
        if not init_dir.is_dir():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {init_dir}")
        print(f"[Finetune] Loading pretrained model from {init_dir}")

    from gr00t.model.gr00t_n1 import GR00T_N1_5_Config
    config_override = GR00T_N1_5_Config.from_pretrained(str(init_dir))
    if not config.init_from_base:
        # See the note above: only the XE pretrain needs its config forced to 64.
        config_override.action_dim = data_max_action_dim
        if isinstance(config_override.action_head_cfg, dict):
            config_override.action_head_cfg["action_dim"] = data_max_action_dim
        elif hasattr(config_override.action_head_cfg, "action_dim"):
            config_override.action_head_cfg.action_dim = data_max_action_dim

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=str(init_dir),
        config=config_override,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim

    if action_horizon_mismatch or action_dim_mismatch:
        import copy
        from gr00t.model.action_head.flow_matching_action_head import FlowmatchingActionHead

        new_cfg = copy.deepcopy(model.action_head.config)
        new_cfg.action_horizon = data_action_horizon
        new_cfg.action_dim = data_max_action_dim
        new_head = FlowmatchingActionHead(new_cfg)
        if action_dim_mismatch:
            old_dim = model.action_head.config.action_dim
            merged = copy_partial_action_expert_weights(
                model.action_head.state_dict(), new_head.state_dict(), old_dim, data_max_action_dim
            )
            new_head.load_state_dict(merged, strict=True)
        else:
            new_head.load_state_dict(model.action_head.state_dict(), strict=False)
        model.action_head = new_head
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim
        # ...and so does the config that gets saved next to the weights.
        # ``config.action_head_cfg`` is a plain dict that GR00T_N1_5.__init__ feeds
        # to FlowmatchingActionHeadConfig to build the head; setting
        # config.action_dim does NOT rewrite it.  Leaving it at the old value saves
        # a checkpoint whose config contradicts its own weights -- which is exactly
        # how the XE pretrain ended up unrestorable (deploy built a 32-dim head and
        # then tried to load 64-dim tensors into it).  Only reachable on the
        # rebuild path, so this cannot disturb a checkpoint that already agrees.
        if isinstance(model.config.action_head_cfg, dict):
            model.config.action_head_cfg["action_dim"] = data_max_action_dim
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # ---- 3. training args -----------------------------------------------
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor
        if config.dataloader_num_workers > 1
        else None,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        report_to=config.report_to,
        seed=config.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        ddp_timeout=3600,
        torch_compile_mode=None,
    )

    # ---- 4. run ---------------------------------------------------------
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.trainer.add_callback(MilestoneSaveCallback(milestone_steps=config.milestone_steps))
    experiment.train()


if __name__ == "__main__":
    config = tyro.cli(FinetuneConfig)
    print("\n" + "=" * 60)
    print("GR00T XE SINGLE-TASK FINETUNE")
    print("=" * 60)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, (
        f"Requested {config.num_gpus} GPUs but only {available_gpus} available"
    )
    assert config.num_gpus > 0

    if config.num_gpus == 1:
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *sys.argv[1:],
            ]
            print("Running torchrun:", " ".join(cmd))
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)