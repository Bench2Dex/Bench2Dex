# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-Embodiment GR00T XE pretrain on 26 tasks × 12 embodiments.

Loads all available task datasets and trains a unified policy head (DiT)
while keeping the VLM backbone frozen.  One command::

    python policy/GR00T_XE/pretrain.py \\
        --dataset-path /data/task1 /data/task2 ... \\
        --output-dir /ckpt/gr00t_xe_pretrain \\
        --max-steps 520000 --batch-size 64 --num-gpus 8

Multi-GPU is handled automatically via ``torchrun`` when ``--num-gpus > 1``.
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

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model, copy_partial_action_expert_weights


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_XE_DIR = Path(__file__).resolve().parent  # policy/GR00T_XE/


@dataclass
class PretrainConfig:
    """Cross-embodiment pretrain for GR00T XE."""

    # ---- data -----------------------------------------------------------
    dataset_path: List[str]
    """One directory per task (e.g. .../03_wine_glass_.../replay-generalization)."""

    data_config: str = "policy.GR00T_XE.xe_config:Dex2BenchXEDataConfig"
    """Cross-embodiment data config (max_action_dim=64, max_state_dim=64)."""

    hdf5_native: bool = True
    """Read raw HDF5 files (skips LeRobot conversion)."""

    hdf5_camera_map: list[str] | None = None
    """Camera map entries, e.g. 'stereo_left=cam_stereo_left'."""

    hdf5_prompt: str | None = None
    """Global language prompt, applied to every dataset.  Overrides per-task prompts."""

    hdf5_prompt_from_scene: bool = True
    """Per-task language prompt from ``scenes/<task>.yaml``'s ``description``.

    This is the string the eval feeds the model (``run_policy.py:2184`` passes
    ``task["description"]``) and the one ``finetune.sh`` uses, so pretrain used
    to be the odd one out: with this off (the 2026-09-11 run) every sample of
    all 26 tasks carried the constant ``"perform task"``, which makes the
    language channel carry zero information.  Resolving is all-or-nothing: a
    dataset whose scene has no description aborts the run."""

    scenes_dir: str = "scenes"
    """Scene YAML directory (absolute, or relative to the repo root)."""

    hdf5_use_active_dof: bool = True
    """Apply active-DOF selection from HDF5 metadata."""

    hdf5_robot_key: str | None = None
    """Robot key for active-DOF lookup (auto-detected per dataset when None)."""

    hdf5_state_dim: int | None = None
    """Override state dim (auto-detected when None)."""

    hdf5_action_dim: int | None = None
    """Override action dim (auto-detected when None)."""

    hdf5_truncate_at_homing: bool = True
    """Truncate episodes at homing-start frame."""

    # ---- model ----------------------------------------------------------
    output_dir: str = "/tmp/gr00t_xe_pretrain"
    """Directory to save model checkpoints."""

    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    max_state_dim: int = 64
    """GR00T state padding dimension (64 for cross-embodiment)."""

    max_action_dim: int = 64
    """GR00T action padding dimension (64 for cross-embodiment)."""

    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    # ---- training -------------------------------------------------------
    batch_size: int = 64
    """Per-GPU batch size."""

    max_steps: int = 520000
    """Total training steps.  ~26 × 20000 (single-task steps)."""

    num_gpus: int = 1
    """Number of GPUs.  >1 triggers torchrun."""

    save_steps: int = 2000
    """Save checkpoint every N steps."""

    milestone_steps: int = 0
    """If > 0, keep milestone checkpoints every N steps."""

    seed: int = 42

    resume: bool = False
    """Resume from the latest checkpoint in output_dir."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    lora_rank: int = 0
    """LoRA rank (0 = no LoRA)."""

    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False

    dataloader_num_workers: int = 12
    gradient_accumulation_steps: int = 1
    dataloader_prefetch_factor: int = 4

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "tensorboard"

    embodiment_tag: str = "new_embodiment"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchvision_av"

    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task_name_from_dataset_path(path: str) -> str:
    """``.../08_frypan_stand_pour/replay-generalization`` -> ``08_frypan_stand_pour``."""
    p = Path(str(path).rstrip("/"))
    if p.name in ("replay-generalization", "lerobot", "data"):
        p = p.parent
    return p.name


def _scene_prompt(task_name: str, scenes_dir: Path) -> str:
    """``scenes/<task>.yaml`` -> its ``description``, exactly as the eval reads it.

    ``yaml.safe_load`` folds a wrapped scalar back into one string, so this is
    byte-identical to ``run_policy.py``'s ``task["description"]`` (grepping the
    first physical line of the YAML is *not*: 34/61 wrap).
    """
    import yaml

    yaml_path = scenes_dir / f"{task_name}.yaml"
    if not yaml_path.is_file():
        # The eval resolves scenes/<ID>_*.yaml from the task number; accept the
        # same form here, but only when it is unambiguous.
        matches = sorted(scenes_dir.glob(f"{task_name.split('_')[0]}_*.yaml"))
        if len(matches) == 1:
            yaml_path = matches[0]
        elif not matches:
            raise FileNotFoundError(
                f"no scene YAML for task '{task_name}' under {scenes_dir}. "
                f"Pass --hdf5-prompt '<text>' or --scenes-dir <dir>."
            )
        else:
            raise FileNotFoundError(
                f"ambiguous scene YAML for task '{task_name}' under {scenes_dir}: "
                f"{[m.name for m in matches]}"
            )
    with open(yaml_path) as fh:
        description = (yaml.safe_load(fh) or {}).get("description")
    if description is None or not str(description).strip():
        raise ValueError(
            f"{yaml_path} has no non-empty 'description:' — the eval would feed the "
            f"model an empty instruction. Fix the scene or pass --hdf5-prompt."
        )
    return str(description)


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
    """Copy checkpoint to ``checkpoint-milestone-{step}`` every N steps."""

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

def main(config: PretrainConfig) -> None:
    os.environ["GR00T_MAX_STATE_DIM"] = str(config.max_state_dim)
    os.environ["GR00T_MAX_ACTION_DIM"] = str(config.max_action_dim)

    # ---- 1. load datasets ------------------------------------------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    _temp_dir = Path(__file__).resolve().parent  # policy/GR00T_XE/ — for import
    if str(_temp_dir) not in sys.path:
        sys.path.insert(0, str(_temp_dir))

    from dataset import build_hdf5_dataset

    camera_map = _parse_hdf5_camera_map(config.hdf5_camera_map)

    # ---- language prompt per dataset -------------------------------------
    # Precedence: explicit --hdf5-prompt > scene YAML > (explicitly disabled)
    # the legacy constant "perform task" inside the hdf5 dataset.
    scenes_dir = Path(config.scenes_dir)
    if not scenes_dir.is_absolute():
        scenes_dir = _XE_DIR.parents[1] / scenes_dir  # repo root / scenes

    if not config.hdf5_prompt and not config.hdf5_prompt_from_scene:
        print(
            "[Pretrain] WARNING: --no-hdf5-prompt-from-scene — every sample gets the "
            "constant prompt 'perform task'; the language channel will carry no "
            "information, while eval and finetune will use the real scene description."
        )

    def prompt_for(dataset_path: str) -> str | None:
        if config.hdf5_prompt:
            return config.hdf5_prompt
        if not config.hdf5_prompt_from_scene:
            return None
        return _scene_prompt(_task_name_from_dataset_path(dataset_path), scenes_dir)

    member_datasets: list = []

    if len(config.dataset_path) == 1:
        # Single dataset — use single-dataset path (useful for debugging)
        train_dataset = build_hdf5_dataset(
            input_dir=config.dataset_path[0],
            camera_map=camera_map,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=config.embodiment_tag,
            prompt=prompt_for(config.dataset_path[0]),
            use_active_dof=config.hdf5_use_active_dof,
            robot_key=config.hdf5_robot_key,
            state_dim=config.hdf5_state_dim,
            action_dim=config.hdf5_action_dim,
            truncate_at_homing=config.hdf5_truncate_at_homing,
        )
        print(
            f"[Pretrain] Loaded {len(train_dataset)} samples from {config.dataset_path[0]} "
            f"| prompt={prompt_for(config.dataset_path[0])!r}"
        )
        member_datasets = [train_dataset]
    else:
        # Multi-dataset — build mixture
        single_datasets = []
        for p in sorted(config.dataset_path):
            if not os.path.isdir(p):
                # The launcher discovers these dirs with `find`, so a missing one
                # means a typo or a half-mounted filesystem — not something to
                # train around.
                raise SystemExit(f"[Pretrain] dataset dir does not exist: {p}")
            prompt = prompt_for(p)
            ds = build_hdf5_dataset(
                input_dir=p,
                camera_map=camera_map,
                modality_configs=modality_configs,
                # One transform instance PER dataset.  A transform owns the
                # normalizers, and LeRobotMixtureDataset.update_metadata calls
                # set_transforms_metadata on every member, so a shared object
                # would leave all 26 datasets normalized with whichever robot
                # happened to run last.
                transforms=data_config_cls.transform(),
                embodiment_tag=config.embodiment_tag,
                prompt=prompt,
                use_active_dof=config.hdf5_use_active_dof,
                robot_key=config.hdf5_robot_key,
                state_dim=config.hdf5_state_dim,
                action_dim=config.hdf5_action_dim,
                truncate_at_homing=config.hdf5_truncate_at_homing,
            )
            single_datasets.append(ds)
            print(
                f"[Pretrain] {p.split('/')[-3]}/{p.split('/')[-2]}: {len(ds)} samples "
                f"| prompt={prompt!r}"
            )

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in single_datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=config.seed,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        member_datasets = single_datasets
        print(f"[Pretrain] Mixture: {len(single_datasets)} datasets loaded")

    # ---- 2. load model --------------------------------------------------
    data_action_horizon = len(data_config_cls.action_indices)
    last_transform = transforms.transforms[-1]
    from gr00t.model.transforms import GR00TTransform

    assert isinstance(last_transform, GR00TTransform)
    data_max_action_dim = last_transform.max_action_dim

    # Each member dataset has to own its transform instance (see the mixture
    # loop), and every instance must agree on the model-side dims.  Both are
    # silent-corruption classes: a shared instance means the whole corpus is
    # normalized with one robot's statistics, and a dim mismatch means the
    # loss is computed against the wrong action width.
    if not member_datasets:
        raise RuntimeError("[Pretrain] no member datasets were built")
    _transform_ids = [id(ds._transforms) for ds in member_datasets]
    if len(set(_transform_ids)) != len(_transform_ids):
        raise RuntimeError(
            f"[Pretrain] {len(member_datasets)} datasets share only "
            f"{len(set(_transform_ids))} transform instance(s).  Per-robot "
            f"normalization is impossible with a shared transform: the last "
            f"set_metadata wins and every dataset is normalized with it."
        )
    _action_dims = {ds._transforms.transforms[-1].max_action_dim for ds in member_datasets}
    _action_dims.add(data_max_action_dim)
    if len(_action_dims) != 1:
        raise RuntimeError(
            f"[Pretrain] datasets disagree on max_action_dim: {sorted(_action_dims)}"
        )

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
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
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )

    # ``save_pretrained`` serializes ``model.config``, and the action head is
    # rebuilt from ``config.action_head_cfg`` on load.  ``action_head_cfg``
    # still holds the BASE model's values (32 / horizon 16), so a checkpoint
    # whose weights are 64D ends up with a config that says 32 and cannot be
    # loaded by the deploy path: ``Gr00tPolicy`` builds the head from
    # config.json *before* its own mismatch handling runs, and dies with
    #   RuntimeError: Error(s) in loading state_dict for CategorySpecificLinear:
    #   size mismatch for W: [32, 1024, 64] (ckpt) vs [32, 1024, 32] (model)
    # (hit 2026-09-11 on the 26-task pretrain; the previous 1-task checkpoint
    # had been patched by hand to 64).  Mirror the two fields here.
    if isinstance(model.config.action_head_cfg, dict):
        model.config.action_head_cfg["action_dim"] = data_max_action_dim
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        print(
            f"[Pretrain] action_head_cfg: action_dim={data_max_action_dim} "
            f"action_horizon={data_action_horizon}"
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
    config = tyro.cli(PretrainConfig)

    # ``dataset_path`` is a List[str], so tyro parses it with nargs="+": passing
    # ``--dataset-path A --dataset-path B`` keeps only B, silently shrinking a
    # multi-task pretrain to a single dataset.  The launcher exports the number
    # of paths it passed, so a mismatch fails loudly instead of training on the
    # wrong corpus.
    _expected_paths = os.environ.get("XE_EXPECTED_DATASET_COUNT")
    if _expected_paths is not None and len(config.dataset_path) != int(_expected_paths):
        raise SystemExit(
            f"dataset_path parsed {len(config.dataset_path)} path(s) but the launcher "
            f"passed {_expected_paths}. Repeated --dataset-path flags overwrite each "
            f"other; pass all paths after a single flag. Got: {config.dataset_path}"
        )
    print(f"[Pretrain] dataset_path: {len(config.dataset_path)} path(s)")

    print("\n" + "=" * 60)
    print("GR00T XE CROSS-EMBODIMENT PRETRAIN")
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