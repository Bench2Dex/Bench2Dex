"""CLI entry point for benchmark evaluation.

Usage:
    python -m tools.run_benchmark --config eval.yaml

Example eval.yaml:
    policy_name: starvla
    mode: dry-run
    scenes:
      - 06_fruit_bowl_loading
    num_episodes: 100
    step_limit: 1000
    output_dir: outputs/benchmarks/my_eval
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class Args:
    config: Path


def main(args: Args) -> None:
    """Run benchmark evaluation from a YAML configuration file."""
    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    from benchmark.harness import EvalConfig, _resolve_task_spec, rollout_scene
    from benchmark.results import write_results

    eval_config = EvalConfig(
        scenes=cfg.get("scenes", []),
        num_episodes=cfg.get("num_episodes", 1),
        step_limit=cfg.get("step_limit", 1000),
        output_dir=cfg.get("output_dir", "eval_result"),
        policy_name=cfg.get("policy_name", "unnamed"),
        allow_legacy_success=bool(cfg.get("allow_legacy_success", cfg.get("mode", "dry-run") == "dry-run")),
        base_seed=int(cfg.get("base_seed", 0)),
    )

    mode = cfg.get("mode", "dry-run")

    if mode == "dry-run":
        _run_dry(eval_config)
    else:
        raise NotImplementedError(
            f"Mode '{mode}' requires a live environment. "
            "Use benchmark.harness.run_benchmark() directly for live evaluation."
        )


def _run_dry(config: EvalConfig) -> None:
    """Dry-run mode: use a trivial env and zero-action policy to test the pipeline."""
    import numpy as np

    from benchmark.harness import _resolve_task_spec, _validate_unique_scene_seed_partitions, rollout_scene
    from benchmark.results import write_results
    from utils.seed_policy import episode_seed

    class _DryRunEnv:
        """Minimal env returning empty states for pipeline testing."""

        def __init__(self, scene_name: str):
            self.scene_name = scene_name

        def reset(self, *, seed: int | None = None) -> None:
            pass

        def get_instruction(self) -> str:
            return self.scene_name

        def get_observation(self) -> dict:
            return {}

        def take_action(self, action: Any) -> None:
            pass

        def get_object_states(self) -> dict:
            return {}

    class _NoOpPolicy:
        """Policy that returns zero actions."""

        def reset(self, instruction: str) -> None:
            pass

        def step(self, example: dict, step: int = 0) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    results = []
    scene_seed_contexts = _validate_unique_scene_seed_partitions(config.scenes, config.base_seed)
    for scene_name in config.scenes:
        task_spec = _resolve_task_spec(scene_name)
        env = _DryRunEnv(scene_name)
        policy = _NoOpPolicy()
        _task_seed_id, scene_base_seed = scene_seed_contexts[scene_name]
        for ep_idx in range(1, config.num_episodes + 1):
            result = rollout_scene(
                task_spec=task_spec,
                env=env,
                policy_adapter=policy,
                episode_index=ep_idx,
                seed=episode_seed(scene_base_seed, ep_idx),
                base_seed=scene_base_seed,
                step_limit=config.step_limit,
                allow_legacy_success=config.allow_legacy_success,
            )
            results.append(result)

    write_results(Path(config.output_dir), config.policy_name, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    parser.add_argument("--config", type=Path, required=True, help="Path to eval YAML config.")
    cli_args = parser.parse_args()
    main(Args(config=cli_args.config))
