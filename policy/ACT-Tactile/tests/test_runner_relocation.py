"""Regression tests for keeping the tactile runner inside ACT-Tactile."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_DIR = REPO_ROOT / "policy" / "ACT-Tactile"
RUNNER_PATH = POLICY_DIR / "tactile_run_policy.py"
WRAPPER_PATH = POLICY_DIR / "eval_direct.sh"


def _bash_major(executable: str | Path) -> int | None:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    match = re.search(r"version\s+(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def _find_bash4() -> str | None:
    candidates = [
        shutil.which("bash"),
        "/usr/bin/bash",
        "/opt/homebrew/bin/bash",
        "/usr/local/bin/bash",
        "/opt/local/bin/bash",
    ]
    for candidate in dict.fromkeys(path for path in candidates if path):
        major = _bash_major(candidate)
        if major is not None and major >= 4:
            return str(candidate)
    return None


BASH4 = _find_bash4()


class RunnerRelocationTests(unittest.TestCase):
    def test_tactile_runner_is_policy_local_only(self) -> None:
        self.assertFalse((REPO_ROOT / "tactile_run_policy.py").exists())
        self.assertTrue(RUNNER_PATH.is_file())

    def test_wrapper_uses_policy_local_runner_at_both_execution_sites(self) -> None:
        script = WRAPPER_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'TACTILE_RUNNER="${SCRIPT_DIR}/tactile_run_policy.py"', script
        )
        self.assertEqual(
            script.count('"${ISAAC_PYTHON}" "${TACTILE_RUNNER}"'), 2
        )
        self.assertNotIn('"${ISAAC_PYTHON}" tactile_run_policy.py', script)

    def test_wrapper_checks_for_bash4_before_array_logic(self) -> None:
        script = WRAPPER_PATH.read_text(encoding="utf-8")

        self.assertIn("BASH_VERSINFO[0]", script)
        guard_index = script.index("BASH_VERSINFO[0]")
        first_array_index = script.index("EXTRA_ARGS=()")
        self.assertLess(guard_index, first_array_index)
        self.assertRegex(script, r"BASH_VERSINFO\[0\]\s*<\s*4")
        self.assertIn("Bash 4+", script)

    @unittest.skipUnless(
        _bash_major("/bin/bash") == 3,
        "requires the macOS system Bash 3 to exercise the real preflight",
    )
    def test_bash3_fails_at_version_preflight_without_environment(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("TACTILE_") and key != "ISAAC_PYTHON"
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                ["/bin/bash", str(WRAPPER_PATH)],
                cwd=temp_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Bash 4+", result.stderr)
        self.assertNotIn("unbound variable", result.stderr)
        self.assertNotIn("mapfile", result.stderr)

    @unittest.skipUnless(
        BASH4 is not None,
        "requires Bash 4+ to execute the wrapper's existing array/mapfile path",
    )
    def test_wrapper_from_external_cwd_forwards_to_absolute_local_runner(self) -> None:
        def arg_value(argv: list[str], option: str) -> str:
            self.assertIn(option, argv)
            return argv[argv.index(option) + 1]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            outside_cwd = temp_root / "outside"
            checkpoint_dir = temp_root / "checkpoint"
            output_root = temp_root / "results"
            invocation_log = temp_root / "isaac_invocations.jsonl"
            fake_isaac_python = temp_root / "fake_isaac_python.py"
            outside_cwd.mkdir()
            checkpoint_dir.mkdir()
            fake_isaac_python.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys

with open(os.environ["FAKE_ISAAC_LOG"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
""",
                encoding="utf-8",
            )
            fake_isaac_python.chmod(0o755)

            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("TACTILE_")
            }
            env.update(
                {
                    "TACTILE_TASK": "06_fruit_bowl_loading",
                    "TACTILE_CKPT_DIR": str(checkpoint_dir),
                    "TACTILE_GEN_PROFILE": "inv_cov",
                    "TACTILE_NUM_EPISODES": "1",
                    "TACTILE_EPISODES_PER_PROCESS": "0",
                    "ISAAC_PYTHON": str(fake_isaac_python),
                    "FAKE_ISAAC_LOG": str(invocation_log),
                }
            )
            result = subprocess.run(
                [str(BASH4), str(WRAPPER_PATH), "--output-root", str(output_root)],
                cwd=outside_cwd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            invocations = [
                json.loads(line)
                for line in invocation_log.read_text(encoding="utf-8").splitlines()
            ]
            runner_argv = invocations[0]
            self.assertEqual(Path(runner_argv[0]), RUNNER_PATH.resolve())
            self.assertEqual(arg_value(runner_argv, "--policy-type"), "TACTILE")
            self.assertEqual(
                arg_value(runner_argv, "--task"),
                "scenes/06_fruit_bowl_loading.yaml",
            )
            self.assertEqual(
                arg_value(runner_argv, "--ckpt-dir"), str(checkpoint_dir)
            )
            self.assertEqual(arg_value(runner_argv, "--num-episodes"), "1")
            self.assertEqual(
                arg_value(runner_argv, "--generalization-profile"), "inv_cov"
            )
            forwarded_output = Path(arg_value(runner_argv, "--output-dir"))
            self.assertTrue(forwarded_output.is_relative_to(output_root))
            self.assertEqual(forwarded_output.name, "inv_cov")
            self.assertIn("--headless", runner_argv)

    def test_readme_documents_bash4_wrapper_requirement(self) -> None:
        readme = (POLICY_DIR / "README.md").read_text(encoding="utf-8")

        self.assertRegex(readme, r"Bash\s+4\+")

    def test_runner_bootstraps_repo_before_repo_local_imports(self) -> None:
        self.assertTrue(RUNNER_PATH.is_file(), f"missing local runner: {RUNNER_PATH}")
        source = RUNNER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        policy_dir_line = None
        repo_root_line = None
        sys_path_line = None
        utils_import_lines = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = {target.id for target in targets if isinstance(target, ast.Name)}
                if "POLICY_DIR" in names:
                    policy_dir_line = node.lineno
                if "REPO_ROOT" in names:
                    repo_root_line = node.lineno
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "insert"
                    and isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "sys"
                    and node.func.value.attr == "path"
                    and "REPO_ROOT" in ast.unparse(node)
                ):
                    sys_path_line = node.lineno
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if any(module == "utils" or module.startswith("utils.") for module in modules):
                    utils_import_lines.append(node.lineno)

        self.assertIsNotNone(policy_dir_line)
        self.assertIsNotNone(repo_root_line)
        self.assertIsNotNone(sys_path_line)
        self.assertTrue(utils_import_lines, "runner must import repository-local utils")
        self.assertLess(policy_dir_line, min(utils_import_lines))
        self.assertLess(repo_root_line, min(utils_import_lines))
        self.assertLess(sys_path_line, min(utils_import_lines))

        self.assertRegex(
            source,
            r"POLICY_DIR\s*=\s*Path\(__file__\)\.resolve\(\)\.parent",
        )
        self.assertRegex(source, r"REPO_ROOT\s*=\s*POLICY_DIR\.parent\.parent")

    def test_runner_dynamically_imports_act_tactile_deployment(self) -> None:
        self.assertTrue(RUNNER_PATH.is_file(), f"missing local runner: {RUNNER_PATH}")
        source = RUNNER_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)

        dynamic_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "policy.ACT-Tactile.deploy_policy"
        ]
        self.assertEqual(len(dynamic_imports), 1)
        self.assertRegex(
            source,
            r"TactilePolicyDeployment\s*=\s*\w+\.TactilePolicyDeployment",
        )
        self.assertNotIn("policy.tactile_policy.deploy_policy", source)

    def test_ordinary_act_wrapper_still_uses_root_runner(self) -> None:
        script = (POLICY_DIR / "act" / "eval_direct.sh").read_text(encoding="utf-8")

        self.assertRegex(
            script,
            re.compile(
                r"python\s+run_policy\.py\s*\\?\s*--policy-type\s+ACT",
                re.MULTILINE,
            ),
        )


if __name__ == "__main__":
    unittest.main()
