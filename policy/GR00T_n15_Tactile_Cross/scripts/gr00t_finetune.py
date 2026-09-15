# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

# Suppress TensorFlow/XLA/albumentations noise BEFORE any import that may
# trigger TF or its dependencies.  TF_CPP_MIN_LOG_LEVEL: 0=all, 1=info, 2=warn, 3=error.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# Ensure the tactile fork of gr00t is imported instead of the editable-installed
# non-tactile one.  This must happen BEFORE the `from gr00t...` imports below:
# when train.sh launches via tmux, the PYTHONPATH it exports does not reach the
# tmux session, so without this the imports would resolve to the non-tactile fork.
_gr00t_src_dir = Path(__file__).resolve().parent.parent / "src"
if str(_gr00t_src_dir) not in sys.path:
    sys.path.insert(0, str(_gr00t_src_dir))

import torch
import tyro
from transformers import TrainingArguments, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories, we assume all datasets have the same data config"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data configuration to use for training.
    Options:
    - Built-in configs: Use predefined config names like 'so100', 'fourier_gr1_arms_only', 'unitree_g1'.
    - External configs: Use 'module:ClassName' format to load custom configs from external files. e.g. 'my_dir.my_configs:RobotConfig'
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    milestone_steps: int = 0
    """If > 0, copy checkpoints to ``checkpoint-milestone-{step}`` every N steps
    (survives HF rotation).  Default 0 = disabled."""

    seed: int = 0
    """Random seed used by torch/transformers dataset sampling and training."""

    max_state_dim: int | None = None
    """Optional GR00T state padding dimension override."""

    max_action_dim: int | None = None
    """Optional GR00T action padding dimension override."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "tensorboard"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"
    """Video backend to use for training. [torchcodec, decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True

    # HDF5-native mode (skip LeRobot conversion)
    hdf5_native: bool = False
    """If True, read dex2bench HDF5 files directly instead of LeRobot format."""

    hdf5_camera_map: list[str] | None = None
    """Camera map entries for HDF5-native mode, e.g. 'stereo_left=cam_stereo_left'."""

    hdf5_prompt: str | None = None
    """Language prompt for HDF5-native mode."""

    hdf5_use_active_dof: bool = False
    """Apply active-DOF selection in HDF5-native mode."""

    hdf5_robot_key: str | None = None
    """Robot key for active-DOF lookup in HDF5-native mode."""

    hdf5_state_dim: int | None = None
    """Override state dim in HDF5-native mode."""

    hdf5_action_dim: int | None = None
    """Override action dim in HDF5-native mode."""

    hdf5_truncate_at_homing: bool = False
    """If True, truncate HDF5 episodes at the homing-start frame (removes post-homing noise)."""

    hdf5_tactile_height: int = 240
    """TacMap height presented to the tactile encoder."""

    hdf5_tactile_width: int = 240
    """TacMap width presented to the tactile encoder."""

    tactile_fusion_stage: Literal["pre_dit", "post_dit"] | None = None
    """Fusion location; default post_dit for a new tactile head, otherwise preserve checkpoint."""

    tactile_grid_size: int | None = None
    """Spatial token grid per site; None preserves the checkpoint setting (legacy: 2)."""

    tactile_dropout_prob: float | None = None
    """Whole-modality dropout; None preserves the checkpoint setting (legacy: 0.3)."""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""


#####################################################################################
# Helper functions
#####################################################################################


from gr00t.utils.peft import copy_partial_action_expert_weights as _copy_partial_action_expert_weights


_DEFAULT_CAMERA_MAP = {
    "stereo_left": "cam_stereo_left",
    "stereo_right": "cam_stereo_right",
    "right_wrist": "cam_wrist_right",
    "left_wrist": "cam_wrist_left",
}


def _parse_hdf5_camera_map(items: list[str] | None) -> dict[str, str]:
    """Parse camera map entries like 'stereo_left=cam_stereo_left' into a dict."""
    camera_map = dict(_DEFAULT_CAMERA_MAP)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Camera map entries must be NAME=DEX2BENCH_CAMERA, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid camera map entry: {item!r}")
        camera_map[key] = value
    return camera_map


def _resolve_tactile_overrides(config: ArgsConfig, *, new_tactile_head: bool) -> dict:
    overrides = {
        key: getattr(config, key)
        for key in ("tactile_fusion_stage", "tactile_grid_size", "tactile_dropout_prob")
        if getattr(config, key) is not None
    }
    # Apply new defaults only when adding tactile to a non-tactile base model.
    if new_tactile_head:
        for key, value in (
            ("tactile_fusion_stage", "post_dit"),
            ("tactile_grid_size", 2),
            ("tactile_dropout_prob", 0.3),
        ):
            overrides.setdefault(key, value)
    return overrides


#####################################################################################
# Milestone checkpoint callback — keeps every-N-step checkpoints that survive rotation.
#####################################################################################


class MilestoneSaveCallback(TrainerCallback):
    """Copy the checkpoint to a ``checkpoint-milestone-{step}`` directory when
    *step* is a multiple of *milestone_steps*.  HF's ``_rotate_checkpoints``
    only touches ``checkpoint-<digits>``, so these milestone copies are safe
    from deletion regardless of ``save_total_limit``.
    """

    def __init__(self, milestone_steps: int = 0):
        self.milestone_steps = milestone_steps

    def on_save(self, args, state, control, **kwargs):
        if self.milestone_steps <= 0:
            return
        import shutil
        step = int(state.global_step)
        if step <= 0 or step % self.milestone_steps != 0:
            return
        src = Path(args.output_dir) / f"checkpoint-{step}"
        dst = Path(args.output_dir) / f"checkpoint-milestone-{step}"
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[gr00t] milestone checkpoint saved: {dst}", flush=True)


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    if config.resume:
        checkpoint = get_last_checkpoint(config.output_dir)
        if checkpoint is None:
            raise ValueError(f"No training checkpoint to resume in {config.output_dir}")
        # Build the saved architecture before Trainer restores optimizer state.
        config.base_model_path = checkpoint
    if config.max_state_dim is not None:
        os.environ["GR00T_MAX_STATE_DIM"] = str(config.max_state_dim)
    if config.max_action_dim is not None:
        os.environ["GR00T_MAX_ACTION_DIM"] = str(config.max_action_dim)

    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if config.hdf5_native:
        # ----- HDF5-native mode: read dex2bench HDF5 files directly ---------
        _gr00t_dir = Path(__file__).resolve().parent.parent  # policy/GR00T_n15_Tactile/
        if str(_gr00t_dir) not in sys.path:
            sys.path.insert(0, str(_gr00t_dir))
        from gr00t_hdf5_dataset import build_hdf5_dataset

        _camera_map = _parse_hdf5_camera_map(config.hdf5_camera_map)
        train_dataset = build_hdf5_dataset(
            input_dir=config.dataset_path[0],
            camera_map=_camera_map,
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=config.embodiment_tag,
            prompt=config.hdf5_prompt,
            use_active_dof=config.hdf5_use_active_dof,
            robot_key=config.hdf5_robot_key,
            state_dim=config.hdf5_state_dim,
            action_dim=config.hdf5_action_dim,
            truncate_at_homing=config.hdf5_truncate_at_homing,
            tactile_height=config.hdf5_tactile_height,
            tactile_width=config.hdf5_tactile_width,
        )
        print(f"[HDF5-native] Loaded {len(train_dataset)} samples from {config.dataset_path[0]}")
    elif len(config.dataset_path) == 1:
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=config.seed,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Assert that the last transform is a GR00TTransform and has max_action_dim
    assert (
        hasattr(transforms, "transforms") and len(transforms.transforms) > 0
    ), "No transforms found"
    last_transform = transforms.transforms[-1]
    from gr00t.model.transforms import GR00TTransform

    assert isinstance(last_transform, GR00TTransform), "Last transform must be GR00TTransform"
    assert hasattr(last_transform, "max_action_dim"), "GR00TTransform must have max_action_dim"
    data_max_action_dim = last_transform.max_action_dim

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )

    # Update action_horizon and max_action_dim to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    action_horizon_mismatch = data_action_horizon != model.action_head.config.action_horizon
    action_dim_mismatch = data_max_action_dim != model.action_head.config.action_dim
    tactile_enabled = config.hdf5_native and hasattr(train_dataset, "tactile_schema")
    tactile_config_mismatch = tactile_enabled and not bool(
        getattr(model.action_head.config, "use_tactile", False)
    )
    tactile_overrides = _resolve_tactile_overrides(
        config, new_tactile_head=tactile_config_mismatch
    )
    if tactile_overrides and not tactile_enabled:
        raise ValueError("Tactile options require an HDF5 dataset with tactile_schema")
    if config.tactile_grid_size is not None and config.tactile_grid_size <= 0:
        raise ValueError("tactile_grid_size must be positive")
    if config.tactile_dropout_prob is not None and not 0.0 <= config.tactile_dropout_prob <= 1.0:
        raise ValueError("tactile_dropout_prob must be in [0,1]")
    if getattr(model.action_head.config, "use_tactile", False):
        for key in ("tactile_fusion_stage", "tactile_grid_size"):
            if key in tactile_overrides and tactile_overrides[key] != getattr(model.action_head.config, key):
                raise ValueError(f"Changing {key} changes the tactile architecture; start from the non-tactile base checkpoint")
    if config.resume:
        for key, value in tactile_overrides.items():
            if value != getattr(model.action_head.config, key):
                raise ValueError(f"Resume must preserve saved {key}; omit the override or start a new run")

    if action_horizon_mismatch or action_dim_mismatch or tactile_config_mismatch:
        # Store old values for logging
        old_action_horizon = model.action_head.config.action_horizon
        old_action_dim = model.action_head.config.action_dim
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {old_action_horizon})"
        )
        if action_dim_mismatch:
            print(f"Updating max_action_dim {data_max_action_dim} (was {old_action_dim})")

        # Update the action head config (need to copy to avoid modifying original)
        import copy

        new_action_head_config = copy.deepcopy(model.action_head.config)
        new_action_head_config.action_horizon = data_action_horizon
        new_action_head_config.action_dim = data_max_action_dim
        for key, value in tactile_overrides.items():
            setattr(new_action_head_config, key, value)
        if tactile_enabled:
            new_action_head_config.use_tactile = True
            new_action_head_config.tactile_num_sites = len(train_dataset.tactile_site_names)
            new_action_head_config.tactile_site_names = list(train_dataset.tactile_site_names)
            new_action_head_config.tactile_height = train_dataset.tactile_output_shape[0]
            new_action_head_config.tactile_width = train_dataset.tactile_output_shape[1]
            new_action_head_config.tactile_native_height = train_dataset.tactile_schema.image_shape[0]
            new_action_head_config.tactile_native_width = train_dataset.tactile_schema.image_shape[1]
            new_action_head_config.tactile_depth_key = train_dataset.tactile_schema.depth_key
            new_action_head_config.tactile_depth_unit = train_dataset.tactile_schema.depth_unit
            new_action_head_config.tactile_d_max_m = train_dataset.tactile_schema.d_max_m
            new_action_head_config.tactile_resolution_step = train_dataset.tactile_schema.resolution_step
            print(
                "Enabling tactile token encoder: "
                f"sites={new_action_head_config.tactile_site_names}, "
                f"shape={train_dataset.tactile_output_shape}"
            )

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        if not action_dim_mismatch:
            print("Copying weights from old action head (compatible dimensions)")
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            print(
                f"Partial weight copy: copying first {old_action_dim} dimensions, initializing last {data_max_action_dim - old_action_dim} dimensions randomly"
            )
            merged_sd = _copy_partial_action_expert_weights(
                model.action_head.state_dict(),
                new_action_head.state_dict(),
                old_action_dim,
                data_max_action_dim,
            )
            new_action_head.load_state_dict(merged_sd, strict=True)

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon
        model.config.action_head_cfg["action_dim"] = data_max_action_dim
        if tactile_enabled:
            for key in (
                "use_tactile",
                "tactile_num_sites",
                "tactile_site_names",
                "tactile_height",
                "tactile_width",
                "tactile_native_height",
                "tactile_native_width",
                "tactile_encoder_dim",
                "tactile_num_heads",
                "tactile_dropout_prob",
                "tactile_fusion_stage",
                "tactile_grid_size",
                "tactile_depth_key",
                "tactile_depth_unit",
                "tactile_d_max_m",
                "tactile_resolution_step",
            ):
                model.config.action_head_cfg[key] = getattr(new_action_head_config, key)

        # Update the main model's action_dim for validation (critical for validate_inputs)
        model.config.action_dim = data_max_action_dim
        model.action_dim = data_max_action_dim

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    if tactile_enabled:
        # Also persist runtime-only overrides when no head recreation was needed.
        for key, value in tactile_overrides.items():
            setattr(model.action_head.config, key, value)
        for key in ("tactile_fusion_stage", "tactile_grid_size", "tactile_dropout_prob"):
            model.config.action_head_cfg[key] = getattr(model.action_head.config, key)
        print("Tactile settings:", {key: model.config.action_head_cfg[key] for key in (
            "tactile_fusion_stage", "tactile_grid_size", "tactile_dropout_prob"
        )})

    # Set the model's compute_dtype to bfloat16
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

    # 2.1 modify training args
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
        dataloader_prefetch_factor=config.dataloader_prefetch_factor if config.dataloader_num_workers > 1 else None,
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
        # evaluation_strategy="no",
        save_total_limit=1,
        report_to=config.report_to,
        seed=config.seed,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        ddp_timeout=3600,   # 1 h: extra headroom for checkpoint I/O on slow fs
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 register milestone checkpoint callback (survives rotation, disabled by default)
    experiment.trainer.add_callback(MilestoneSaveCallback(milestone_steps=config.milestone_steps))

    # 2.4 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
