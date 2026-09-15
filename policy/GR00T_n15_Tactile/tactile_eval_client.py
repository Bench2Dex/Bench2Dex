"""Config-driven entrypoint for tactile GR00T remote evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.config_utils import build_config_parser, load_config_with_overrides
from script.policy_rpc import RemotePolicyClient
from utils.seed_policy import EVAL_BASE_SEED_DEFAULT


def _as_bool(value, default: bool = False) -> bool:
    """Parse config booleans using the generic eval client's conventions."""

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


def _is_set(value) -> bool:
    return value not in (None, "", "null")


@dataclass
class TactileEvalClient:
    """Build and execute the tactile-local Isaac evaluation runner."""

    config: dict

    def build_run_policy_command(self) -> list[str]:
        scene_path = self.config.get("scene_path")
        if not _is_set(scene_path):
            scene_path = f"scenes/{self.config['task_name']}.yaml"

        seed = self.config.get("seed")
        if not _is_set(seed):
            seed = EVAL_BASE_SEED_DEFAULT

        command = [
            sys.executable,
            "policy/GR00T_n15_Tactile/run_policy.py",
            "--task",
            str(scene_path),
            "--host",
            str(self.config.get("host") or "127.0.0.1"),
            "--port",
            str(int(self.config.get("port") or 9000)),
            "--enable-rgb",
            "--seed",
            str(int(seed)),
        ]

        if "use_active_dof" in self.config:
            command.append(
                "--active-dof"
                if _as_bool(self.config.get("use_active_dof"), True)
                else "--no-active-dof"
            )

        self._append_value(command, "--robot-key", self.config.get("robot_key"))

        ckpt_dir = self.config.get("ckpt_dir") or self.config.get("model_path")
        self._append_value(command, "--ckpt-dir", ckpt_dir)
        ckpt_name = self.config.get("ckpt_name")
        if not _is_set(ckpt_name) and _is_set(ckpt_dir):
            ckpt_name = os.path.basename(str(ckpt_dir).rstrip("/"))
        self._append_value(command, "--ckpt-name", ckpt_name)

        for config_key, option in (
            ("episode_steps", "--episode-steps"),
            ("warmup_steps", "--warmup-steps"),
            ("num_episodes", "--num-episodes"),
            ("start_episode", "--start-episode"),
        ):
            value = self.config.get(config_key)
            if _is_set(value):
                command.extend([option, str(int(value))])

        for config_key, option in (
            ("output_dir", "--output-dir"),
            ("record_dir", "--record-dir"),
            ("generalization_profile", "--generalization-profile"),
            ("generalization_split", "--generalization-split"),
            ("generalization_config", "--generalization-config"),
        ):
            self._append_value(command, option, self.config.get(config_key))

        anchor_dir = self.config.get("anchor_dir")
        if _is_set(anchor_dir):
            self._append_value(command, "--anchor-dir", anchor_dir)
        else:
            self._append_value(
                command, "--anchor-hdf5", self.config.get("anchor_hdf5")
            )

        for config_key, option in (
            ("policy_name", "--policy-name"),
            ("policy_display_name", "--policy-display-name"),
        ):
            self._append_value(command, option, self.config.get(config_key))

        for config_key, option in (
            ("append_output", "--append-output"),
            ("record_all", "--record-all"),
            ("append_record_dir", "--append-record-dir"),
            ("enable_generalization", "--enable-generalization"),
            ("headless", "--headless"),
        ):
            if _as_bool(self.config.get(config_key), False):
                command.append(option)

        run_policy_args = self.config.get("run_policy_args")
        if run_policy_args is None:
            run_policy_args = []
        elif not isinstance(run_policy_args, (list, tuple)):
            raise ValueError("run_policy_args must be a list or tuple")
        command.extend(str(arg) for arg in run_policy_args)
        return command

    @staticmethod
    def _append_value(command: list[str], option: str, value) -> None:
        if _is_set(value):
            command.extend([option, str(value)])

    def run(self) -> int:
        completed = subprocess.run(
            self.build_run_policy_command(), cwd=REPO_ROOT, check=False
        )
        return int(completed.returncode)


def main() -> None:
    parser = build_config_parser("Run tactile GR00T remote evaluation client.")
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Ping the remote socket server before launching the tactile runner.",
    )
    args = parser.parse_args()
    cli_updates = {
        key: value
        for key, value in (("host", args.host), ("port", args.port))
        if value is not None
    }
    config = load_config_with_overrides(
        args.config,
        args.overrides,
        extra_updates=cli_updates,
    )

    if args.check_connection:
        client = RemotePolicyClient(
            str(config.get("host") or "127.0.0.1"),
            int(config.get("port") or 9000),
        )
        client.close()

    raise SystemExit(TactileEvalClient(config).run())


if __name__ == "__main__":
    main()
