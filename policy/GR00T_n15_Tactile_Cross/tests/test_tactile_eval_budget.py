from __future__ import annotations

import ast
import math
import tempfile
import unittest
from pathlib import Path

from policy.GR00T_n15_Tactile_Cross.tactile_eval_budget import (
    resolve_episode_budget,
)


class TactileEvalBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)

    def write_scene(self, contents: str) -> Path:
        scene_path = Path(self._temp_dir.name) / "scene.yaml"
        scene_path.write_text(contents, encoding="utf-8")
        return scene_path

    def test_explicit_800_overrides_scene_budget(self) -> None:
        scene = self.write_scene("expert_time_step: 300\n")

        budget = resolve_episode_budget(scene, explicit_episode_steps=800)

        self.assertEqual(budget.episode_steps, 800)
        self.assertEqual(budget.max_physics_steps, 2400)
        self.assertEqual(budget.source, "explicit")

    def test_uses_expert_time_step_when_budget_is_automatic(self) -> None:
        scene = self.write_scene("expert_time_step: 301\n")

        budget = resolve_episode_budget(scene, explicit_episode_steps=None)

        self.assertEqual(budget.episode_steps, 151)
        self.assertEqual(budget.max_physics_steps, 453)
        self.assertEqual(budget.source, "expert_time_step")

    def test_uses_expert_time_seconds_when_step_count_is_absent(self) -> None:
        scene = self.write_scene("expert_time_s: 10.01\n")

        budget = resolve_episode_budget(scene, explicit_episode_steps=None)

        self.assertEqual(budget.episode_steps, 301)
        self.assertEqual(budget.max_physics_steps, 903)
        self.assertEqual(budget.source, "expert_time_s")

    def test_uses_default_when_scene_has_no_expert_timing(self) -> None:
        scene = self.write_scene("description: no timing metadata\n")

        budget = resolve_episode_budget(scene, explicit_episode_steps=None)

        self.assertEqual(budget.episode_steps, 800)
        self.assertEqual(budget.max_physics_steps, 2400)
        self.assertEqual(budget.source, "default")

    def test_rejects_non_positive_explicit_budget(self) -> None:
        scene = self.write_scene("expert_time_step: 300\n")

        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "explicit_episode_steps must be positive"
                ):
                    resolve_episode_budget(
                        scene,
                        explicit_episode_steps=value,
                    )

    def test_rejects_non_positive_resolver_parameters(self) -> None:
        scene = self.write_scene("")
        invalid_arguments = (
            {"policy_stride": 0},
            {"default_episode_steps": 0},
            {"scale": 0},
            {"physics_hz": 0},
        )

        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    resolve_episode_budget(
                        scene,
                        explicit_episode_steps=None,
                        **arguments,
                    )

    def test_rejects_non_integer_step_parameters_including_bool(self) -> None:
        scene = self.write_scene("")
        invalid_calls = (
            {"explicit_episode_steps": 1.5},
            {"explicit_episode_steps": True},
            {"explicit_episode_steps": None, "policy_stride": 1.5},
            {"explicit_episode_steps": None, "policy_stride": True},
            {"explicit_episode_steps": None, "default_episode_steps": 1.5},
            {"explicit_episode_steps": None, "default_episode_steps": True},
        )

        for arguments in invalid_calls:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    resolve_episode_budget(scene, **arguments)

    def test_rejects_non_finite_or_boolean_scale_and_physics_hz(self) -> None:
        scene = self.write_scene("")
        for name in ("scale", "physics_hz"):
            for value in (math.nan, math.inf, -math.inf, True):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        resolve_episode_budget(
                            scene,
                            explicit_episode_steps=None,
                            **{name: value},
                        )

    def test_accepts_scientific_scene_timing_with_yaml_comment(self) -> None:
        scene = self.write_scene("expert_time_step: 1e3  # physics steps\n")

        budget = resolve_episode_budget(scene, explicit_episode_steps=None)

        self.assertEqual(budget.episode_steps, 500)
        self.assertEqual(budget.max_physics_steps, 1500)
        self.assertEqual(budget.source, "expert_time_step")

    def test_empty_step_timing_is_absent_and_seconds_are_used(self) -> None:
        scene = self.write_scene(
            "expert_time_step:\n"
            "expert_time_s: 10\n"
        )

        budget = resolve_episode_budget(scene, explicit_episode_steps=None)

        self.assertEqual(budget.episode_steps, 300)
        self.assertEqual(budget.max_physics_steps, 900)
        self.assertEqual(budget.source, "expert_time_s")

    def test_comment_only_timing_is_absent_and_falls_back(self) -> None:
        scene = self.write_scene(
            "expert_time_step:  # timing was not recorded\n"
        )

        budget = resolve_episode_budget(scene, explicit_episode_steps=None)

        self.assertEqual(budget.episode_steps, 800)
        self.assertEqual(budget.max_physics_steps, 2400)
        self.assertEqual(budget.source, "default")

    def test_real_scene_with_empty_timing_falls_back(self) -> None:
        scene = (
            Path(__file__).resolve().parents[3]
            / "scenes"
            / "86_short_jigsaw_puzzle.yaml"
        )

        budget = resolve_episode_budget(scene, explicit_episode_steps=None)

        self.assertEqual(budget.episode_steps, 800)
        self.assertEqual(budget.max_physics_steps, 2400)
        self.assertEqual(budget.source, "default")

    def test_hash_without_preceding_whitespace_is_not_a_comment(self) -> None:
        scene = self.write_scene(
            "expert_time_step: 1#not-a-comment\n"
        )

        with self.assertRaisesRegex(
            ValueError,
            "expert_time_step must be a finite positive number",
        ):
            resolve_episode_budget(scene, explicit_episode_steps=None)

    def test_rejects_invalid_or_non_positive_scene_timing(self) -> None:
        invalid_values = ("0", "-1", "nan", "inf", "300oops")
        for key in ("expert_time_step", "expert_time_s"):
            for value in invalid_values:
                with self.subTest(key=key, value=value):
                    scene = self.write_scene(f"{key}: {value}\n")
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{key} must be a finite positive number",
                    ):
                        resolve_episode_budget(
                            scene,
                            explicit_episode_steps=None,
                        )

    def test_runner_structurally_uses_resolved_budget(self) -> None:
        runner_path = Path(__file__).resolve().parents[1] / "run_policy.py"
        runner = runner_path.read_text(encoding="utf-8")
        tree = ast.parse(runner)
        episode_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_episode_impl"
        )

        resolver_calls = [
            node
            for node in ast.walk(episode_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_episode_budget"
        ]
        self.assertEqual(len(resolver_calls), 1)
        resolver_call = resolver_calls[0]
        self.assertEqual(
            ast.dump(resolver_call.args[0]),
            ast.dump(ast.Name(id="scene_path", ctx=ast.Load())),
        )
        explicit_keyword = next(
            keyword
            for keyword in resolver_call.keywords
            if keyword.arg == "explicit_episode_steps"
        )
        self.assertEqual(
            ast.dump(explicit_keyword.value),
            ast.dump(
                ast.Attribute(
                    value=ast.Name(id="args", ctx=ast.Load()),
                    attr="episode_steps",
                    ctx=ast.Load(),
                )
            ),
        )

        scene_assignment = next(
            node
            for node in episode_function.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "scene_path"
                for target in node.targets
            )
        )
        expected_scene_expression = ast.parse(
            "args.task if os.path.isabs(args.task) "
            "else os.path.join(SCRIPT_DIR, args.task)",
            mode="eval",
        ).body
        self.assertEqual(
            ast.dump(scene_assignment.value),
            ast.dump(expected_scene_expression),
        )

        max_steps_assignment = next(
            node
            for node in episode_function.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "max_steps"
                for target in node.targets
            )
        )
        self.assertEqual(
            ast.dump(max_steps_assignment.value),
            ast.dump(
                ast.Attribute(
                    value=ast.Name(id="budget", ctx=ast.Load()),
                    attr="max_physics_steps",
                    ctx=ast.Load(),
                )
            ),
        )

    def test_episode_steps_help_describes_explicit_and_automatic_budgets(self) -> None:
        runner_path = Path(__file__).resolve().parents[1] / "run_policy.py"
        runner = runner_path.read_text(encoding="utf-8")
        tree = ast.parse(runner)
        episode_steps_argument = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(
                isinstance(argument, ast.Constant)
                and argument.value == "--episode-steps"
                for argument in node.args
            )
        )
        help_keyword = next(
            keyword
            for keyword in episode_steps_argument.keywords
            if keyword.arg == "help"
        )

        self.assertEqual(
            help_keyword.value.value,
            "Explicit policy-step override; by default use scene expert timing, "
            "falling back to 800 policy steps.",
        )


if __name__ == "__main__":
    unittest.main()
