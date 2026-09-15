"""Policy loaders shared by ``run_policy.py`` (in-process / Isaac Sim) and
``policy_server.py`` (RPC server). Single source of truth for the two policies
that have custom checkpoint-loading logic today (ACT, DP).

The other 9 policies in ``policy/<NAME>/`` follow the RoboTwin-style
``deploy_policy.get_model(args)`` convention and are loaded directly by
``policy_server.py`` via ``importlib`` -- they don't need an entry here.

Important: torch and yaml are imported lazily inside each function. This matches
``run_policy.py``'s ordering, where torch can only be imported *after*
Isaac Sim has launched. ``policy_server.py`` doesn't launch Isaac Sim and so is
free to import torch eagerly, but the lazy form works in both processes.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def load_act_policy(args: argparse.Namespace):
    """Load an ACT model from a checkpoint directory."""
    import torch

    act_dir = os.path.join(SCRIPT_DIR, "policy", "ACT")
    if act_dir not in sys.path:
        sys.path.insert(0, act_dir)

    from act_policy import ACT  # type: ignore  # noqa: E402

    ckpt_dir = args.ckpt_dir
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{args.ckpt_name}' not found in '{ckpt_dir}'")

    act_args = {
        "kl_weight": 10,
        "chunk_size": args.chunk_size,
        "state_dim": args.state_dim,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr": 1e-5,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "dropout": 0.1,
        "pre_norm": False,
        "camera_names": [
            "cam_right_wrist",
            "cam_left_wrist",
            "cam_stereo_left",
            "cam_stereo_right",
        ],
        "temporal_agg": args.temporal_agg,
        "temporal_agg_k": args.temporal_agg_k,
        "device": args.device,
        "ckpt_dir": ckpt_dir,
    }

    # DETR's build_ACT_model_and_optimizer calls parser.parse_args() which reads
    # sys.argv. Isaac Sim's args are not valid DETR args, so we temporarily
    # replace sys.argv with the minimal required DETR arguments.
    _saved_argv = sys.argv
    sys.argv = [
        "run_policy.py",
        "--ckpt_dir", ckpt_dir,
        "--policy_class", "ACT",
        "--task_name", "deploy",
        "--seed", "0",
        "--num_epochs", "1",
        "--state_dim", str(args.state_dim),
    ]
    try:
        model = ACT(act_args, None)
    finally:
        sys.argv = _saved_argv

    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.policy.load_state_dict(state_dict)
    model.policy.eval()
    print(f"[policy] Loaded ACT weights from {ckpt_path}", flush=True)
    return model


def load_dp_policy(args: argparse.Namespace):
    """Load a DP model from a checkpoint and training config."""
    dp_dir = os.path.join(SCRIPT_DIR, "policy", "DP")
    if dp_dir not in sys.path:
        sys.path.insert(0, dp_dir)

    from dp_model import DP  # type: ignore  # noqa: E402
    import yaml

    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{args.ckpt_name}' not found in '{args.ckpt_dir}'")

    config_path = args.training_config or os.path.join(
        "policy", "DP", "diffusion_policy", "config", "robot_dp_36_dex2scene.yaml"
    )
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DP training config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = DP(
        ckpt_path,
        n_obs_steps=cfg["n_obs_steps"],
        n_action_steps=cfg["n_action_steps"],
        device=args.device,
    )
    print(f"[policy] Loaded DP weights from {ckpt_path}", flush=True)
    print(f"[policy] Loaded DP training config from {config_path}", flush=True)
    return model


def load_policy(args: argparse.Namespace):
    """Dispatch on ``args.policy_type`` (ACT / DP)."""
    policy_type_upper = args.policy_type.upper()
    if policy_type_upper == "DP":
        return load_dp_policy(args)
    if policy_type_upper == "ACT":
        return load_act_policy(args)
    raise ValueError(f"Unsupported policy type: {args.policy_type}")
