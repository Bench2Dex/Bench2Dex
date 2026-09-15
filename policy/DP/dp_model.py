import numpy as np
import torch
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner

class DP:

    def __init__(self, ckpt_file: str, n_obs_steps, n_action_steps, device="cuda:0"):
        self.policy = self.get_policy(ckpt_file, None, device)
        self.runner = DPRunner(n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)

    @staticmethod
    def _filter_arrays(obs):
        """Keep only numpy/torch tensor fields; strip strings added by remote session."""
        if isinstance(obs, dict):
            return {k: v for k, v in obs.items()
                    if isinstance(v, (np.ndarray, torch.Tensor))}
        return obs

    def update_obs(self, observation):
        self.runner.update_obs(self._filter_arrays(observation))

    def reset_obs(self):
        self.runner.reset_obs()

    def reset_model(self):
        self.reset_obs()

    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, self._filter_arrays(observation))
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

    def get_policy(self, checkpoint, output_dir, device):
        # load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace: RobotWorkspace

        # Strip "module." prefix when checkpoint was saved from multi-GPU (DDP/DataParallel)
        # but the target model is a plain single-GPU module.
        for _sd_key, _ckpt_sd in payload.get("state_dicts", {}).items():
            _first_ckpt_key = next(iter(_ckpt_sd), "")
            _target_model = workspace.__dict__.get(_sd_key)
            if _target_model is not None:
                _first_model_key = next(iter(_target_model.state_dict()), "")
                if _first_ckpt_key.startswith("module.") and not _first_model_key.startswith("module."):
                    _ckpt_sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                                for k, v in _ckpt_sd.items()}
            payload["state_dicts"][_sd_key] = _ckpt_sd

        # Skip optimizer when loading inference-only checkpoints
        # (bs*_ep*.ckpt excludes optimizer, saving ~1.6 GB / 50% of file size).
        workspace.load_payload(payload, exclude_keys=("optimizer",), include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.to(device)
        policy.eval()

        return policy
