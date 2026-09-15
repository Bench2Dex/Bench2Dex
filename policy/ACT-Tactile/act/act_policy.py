import torch.nn as nn
import os
import torch
import numpy as np
import pickle
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
except:
    from .detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )

# Mapping from ACT training camera names to observation dict keys used in get_action().
_ACT_CAM_TO_OBS_KEY = {
    "cam_high": "head_cam",
    "cam_right_wrist": "right_cam",
    "cam_left_wrist": "left_cam",
    "cam_chest": "chest_cam",
    "cam_stereo_left": "stereo_left_cam",
    "cam_stereo_right": "stereo_right_cam",
}


class ACTPolicy(nn.Module):

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, RoboTwin_Config)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None, tactile=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.float() / 255
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, actions, is_pad, tactile=tactile
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, tactile=tactile)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT:

    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                "kl_weight": 0.1,  # Default value, can be overridden
                "device": "cuda:0",
            }
        self.policy = ACTPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        # Temporal aggregation settings
        self.temporal_agg = args_override.get("temporal_agg", False)
        self.temporal_agg_k = float(args_override.get("temporal_agg_k", 0.01))
        self.num_queries = args_override["chunk_size"]
        # state_dim: prefer args_override["state_dim"], fall back to RoboTwin_Config, then default 14
        if "state_dim" in args_override:
            self.state_dim = args_override["state_dim"]
        elif RoboTwin_Config is not None:
            self.state_dim = RoboTwin_Config.action_dim
        else:
            self.state_dim = 14  # legacy default
        self.max_timesteps = 3000  # Large enough for deployment

        # Set query frequency based on temporal_agg - matching imitate_episodes.py logic
        self.query_frequency = self.num_queries
        if self.temporal_agg:
            self.query_frequency = 1
            # Initialize with zeros matching imitate_episodes.py format
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")

        # Derive obs camera keys from training camera names (order must match training)
        _cam_names = args_override.get("camera_names", [])
        self.obs_camera_keys = [_ACT_CAM_TO_OBS_KEY[c] for c in _cam_names]

        self.t = 0  # Current timestep

        # Load statistics for normalization
        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            # Load dataset stats for normalization
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                print(f"Warning: Could not find stats file at {stats_path}")
                self.stats = None

            # Load policy weights
            ckpt_name = args_override.get("ckpt_name", "policy_last.ckpt")
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            print("current pwd:", os.getcwd())
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(torch.load(ckpt_path, weights_only=False))
                print(f"Loaded policy weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                print(f"Warning: Could not find policy checkpoint at {ckpt_path}")
        else:
            self.stats = None

    def pre_process(self, qpos):
        """Normalize input joint positions"""
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        """Denormalize model outputs"""
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def get_action(self, obs=None):
        if obs is None:
            return None

        # Convert observations to tensors and normalize qpos - matching imitate_episodes.py
        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Stack images in training camera order (derived from camera_names via _ACT_CAM_TO_OBS_KEY).
        curr_images = []
        for obs_key in self.obs_camera_keys:
            curr_images.append(obs[obs_key])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            # Only query the policy at specified intervals - exactly like imitate_episodes.py
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                # Match temporal aggregation exactly from imitate_episodes.py
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = (self.all_actions)
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                # Use same weighting factor as in imitate_episodes.py
                k = self.temporal_agg_k
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = (torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1))

                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                # Direct action selection, same as imitate_episodes.py
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        # Denormalize action
        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)

        self.t += 1
        return action
