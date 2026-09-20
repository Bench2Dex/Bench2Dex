"""Config-driven client entrypoint for remote policy evaluation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.config_utils import build_config_parser, load_config_with_overrides
from script.policy_rpc import RemotePolicyClient
from utils.seed_policy import EVAL_BASE_SEED_DEFAULT
from utils.eval_devices import append_eval_device_arguments, normalize_eval_device_overrides


def _as_bool(value, default: bool = False) -> bool:
    if value is None or value == "" or value == "null":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


@dataclass
class Dex2SceneLiveEnv:
    """Thin wrapper that delegates live Isaac evaluation to run_policy.py."""

    config: dict

    def build_run_policy_command(self) -> list[str]:
        task_name = self.config["task_name"]
        scene_path = self.config.get("scene_path") or f"scenes/{task_name}.yaml"
        command = [
            sys.executable,
            "run_policy.py",
            "--policy-type",
            "REMOTE",
            "--task",
            str(scene_path),
            "--remote-host",
            str(self.config.get("host") or "127.0.0.1"),
            "--remote-port",
            str(int(self.config.get("port") or 9000)),
            "--enable-rgb",
            "--seed",
            str(int(self.config.get("seed") if self.config.get("seed") is not None else EVAL_BASE_SEED_DEFAULT)),
        ]

        if "use_active_dof" in self.config:
            if _as_bool(self.config.get("use_active_dof"), True):
                command.append("--active-dof")
            else:
                command.append("--no-active-dof")

        robot_key = self.config.get("robot_key")
        if robot_key not in (None, "", "null"):
            command.extend(["--robot-key", str(robot_key)])

        # Resolve ckpt_dir: first-class > fall back to model_path (they are often identical)
        ckpt_dir = self.config.get("ckpt_dir") or self.config.get("model_path")
        if ckpt_dir is not None:
            command.extend(["--ckpt-dir", str(ckpt_dir)])

        # Resolve ckpt_name: first-class > basename of ckpt_dir/model_path
        ckpt_name = self.config.get("ckpt_name") or (
            os.path.basename(str(ckpt_dir).rstrip("/")) if ckpt_dir else None
        )
        if ckpt_name is not None:
            command.extend(["--ckpt-name", str(ckpt_name)])

        episode_steps = self.config.get("episode_steps")
        if episode_steps is not None:
            command.extend(["--episode-steps", str(int(episode_steps))])

        warmup_steps = self.config.get("warmup_steps")
        if warmup_steps is not None:
            command.extend(["--warmup-steps", str(int(warmup_steps))])

        num_episodes = self.config.get("num_episodes")
        if num_episodes is not None:
            command.extend(["--num-episodes", str(int(num_episodes))])

        start_episode = self.config.get("start_episode")
        if start_episode is not None:
            command.extend(["--start-episode", str(int(start_episode))])

        output_dir = self.config.get("output_dir")
        if output_dir is not None:
            command.extend(["--output-dir", str(output_dir)])
        if _as_bool(self.config.get("append_output", False), False):
            command.append("--append-output")

        record_dir = self.config.get("record_dir")
        if record_dir is not None:
            command.extend(["--record-dir", str(record_dir)])
        if _as_bool(self.config.get("record_all", False), False):
            command.append("--record-all")
        if _as_bool(self.config.get("append_record_dir", False), False):
            command.append("--append-record-dir")

        inv_cov_seed = self.config.get("inv_cov_seed")
        if inv_cov_seed is not None:
            command.extend(["--inv-cov-seed", str(int(inv_cov_seed))])

        generalization_profile = self.config.get("generalization_profile")
        if generalization_profile is not None:
            command.extend(["--generalization-profile", str(generalization_profile)])

        generalization_split = self.config.get("generalization_split")
        if generalization_split is not None:
            command.extend(["--generalization-split", str(generalization_split)])

        generalization_config = self.config.get("generalization_config")
        if generalization_config is not None:
            command.extend(["--generalization-config", str(generalization_config)])

        anchor_dir = self.config.get("anchor_dir")
        if anchor_dir is not None:
            command.extend(["--anchor-dir", str(anchor_dir)])

        anchor_hdf5 = self.config.get("anchor_hdf5")
        if anchor_hdf5 is not None and anchor_dir is None:
            command.extend(["--anchor-hdf5", str(anchor_hdf5)])

        if _as_bool(self.config.get("enable_generalization", False), False):
            command.append("--enable-generalization")

        policy_name = self.config.get("policy_name")
        if policy_name is not None:
            command.extend(["--policy-name", str(policy_name)])

        policy_display_name = self.config.get("policy_display_name")
        if policy_display_name is not None:
            command.extend(["--policy-display-name", str(policy_display_name)])

        if _as_bool(self.config.get("headless"), False):
            command.append("--headless")

        append_eval_device_arguments(command, self.config)
        for extra_arg in self.config.get("run_policy_args", []) or []:
            command.append(str(extra_arg))

        return command

    def run(self) -> int:
        command = self.build_run_policy_command()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        return int(completed.returncode)


def main() -> None:
    parser = build_config_parser("Run dex2scene remote evaluation client.")
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Ping the remote socket server before launching run_policy.py.",
    )
    args = parser.parse_args()
    config = load_config_with_overrides(
        args.config,
        normalize_eval_device_overrides(args.overrides),
        extra_updates={
            "host": args.host,
            "port": args.port,
        },
    )

    if args.check_connection:
        client = RemotePolicyClient(
            str(config.get("host") or "127.0.0.1"),
            int(config.get("port") or 9000),
        )
        client.close()

    env = Dex2SceneLiveEnv(config)
    raise SystemExit(env.run())


if __name__ == "__main__":
    main()
