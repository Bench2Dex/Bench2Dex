"""Pi0.5 in-process inference wrapper for dex2bench remote eval.

Loads a finetuned pi0.5 checkpoint produced by ``finetune.sh`` and exposes the
observation-window API consumed by ``script/policy_sessions.py``.
"""

import dataclasses
import os

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:
    """Wraps an openpi trained policy with the observation-window interface."""

    def __init__(
        self,
        train_config_name: str,
        checkpoint_path: str,
        eval_action_horizon: int,
        *,
        action_dim: int | None = None,
        state_dim: int | None = None,
    ):
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"PI0 checkpoint dir not found: {checkpoint_path}. "
                "Expect the step directory produced by finetune.sh "
                "(e.g. outputs/logs/pi05/pi05_base_dex2bench_lora/<exp>/30000)."
            )

        config = _config.get_config(train_config_name)
        config = self._apply_dimension_overrides(config, action_dim, state_dim)

        assets_dir = os.path.join(checkpoint_path, "assets")
        if not os.path.isdir(assets_dir):
            raise FileNotFoundError(f"PI0 assets dir not found: {assets_dir}")
        asset_entries = [
            name for name in os.listdir(assets_dir)
            if os.path.isdir(os.path.join(assets_dir, name))
        ]
        if not asset_entries:
            raise FileNotFoundError(f"PI0 assets dir is empty: {assets_dir}")
        asset_id = asset_entries[0]

        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_path,
            robotwin_repo_id=asset_id,
        )
        print(
            f"[pi05] Loaded policy from {checkpoint_path} "
            f"(asset_id={asset_id}, action_dim={config.model.action_dim}, "
            f"state_dim={getattr(config.data, 'state_dim', config.model.action_dim)})"
        )

        self.eval_action_horizon = eval_action_horizon
        self.instruction: str | None = None
        self.observation_window: dict | None = None

    @staticmethod
    def _apply_dimension_overrides(config, action_dim: int | None, state_dim: int | None):
        if action_dim is not None:
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, action_dim=action_dim),
            )

        if state_dim is None and action_dim is not None:
            state_dim = action_dim

        if state_dim is not None:
            if state_dim > config.model.action_dim:
                raise ValueError(
                    f"Pi0 state_dim ({state_dim}) cannot exceed action_dim "
                    f"({config.model.action_dim})."
                )
            if hasattr(config.data, "state_dim"):
                config = dataclasses.replace(
                    config,
                    data=dataclasses.replace(config.data, state_dim=state_dim),
                )

        return config

    def set_language(self, instruction: str):
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        self.observation_window = {
            "state": state,
            "images": {
                "cam_stereo_left": img_arr[0],
                "cam_stereo_right": img_arr[1],
                "cam_wrist_left": img_arr[2],
                "cam_wrist_right": img_arr[3],
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_observation_windows(self):
        self.instruction = None
        self.observation_window = None
