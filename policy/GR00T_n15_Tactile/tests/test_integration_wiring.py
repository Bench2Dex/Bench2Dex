from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IntegrationWiringTest(unittest.TestCase):
    def test_training_loader_emits_tactile(self) -> None:
        text = (ROOT / "gr00t_hdf5_dataset.py").read_text(encoding="utf-8")
        self.assertIn('result["tactile"] = read_tactile_frame(', text)
        self.assertIn("validate_tactile_schema", text)

    def test_transform_preserves_tactile(self) -> None:
        text = (ROOT / "gr00t_dex2bench_config.py").read_text(encoding="utf-8")
        self.assertIn("class TactileGR00TTransform", text)
        self.assertIn('transformed["tactile"] = tactile', text)

    def test_action_head_fuses_tactile_token_for_train_and_inference(self) -> None:
        text = (
            ROOT / "src/gr00t/model/action_head/flow_matching_action_head.py"
        ).read_text(encoding="utf-8")
        self.assertIn("TactileTokenEncoder", text)
        self.assertGreaterEqual(
            text.count('self.process_backbone_output(backbone_output, action_input.get("tactile"))'),
            2,
        )
        self.assertIn("torch.cat((backbone_features, tactile_token), dim=1)", text)
        self.assertIn("(attention_mask, tactile_mask), dim=1", text)

    def test_training_serializes_tactile_schema_into_action_head_config(self) -> None:
        text = (ROOT / "scripts/gr00t_finetune.py").read_text(encoding="utf-8")
        for field in (
            "tactile_site_names",
            "tactile_native_height",
            "tactile_native_width",
            "tactile_height",
            "tactile_width",
        ):
            self.assertIn(f'model.config.action_head_cfg[key] = getattr', text)
            self.assertIn(f'"{field}"', text)

    def test_local_runner_uses_tactile_gr00t_adapter(self) -> None:
        runner = (ROOT / "run_policy.py").read_text(encoding="utf-8")
        deployment = (ROOT / "tactile_deployment.py").read_text(encoding="utf-8")
        self.assertIn("GR00TTactileDeployment", runner)
        self.assertIn('"tactile": stack_tactile_mapping(', deployment)
        self.assertIn("returns_action_chunks = True", deployment)

    def test_double_env_uses_tactile_server_and_client(self) -> None:
        launcher = (ROOT / "eval_double_env.sh").read_text(encoding="utf-8")
        discovery = launcher.split(
            "# ── phase 4: auto-discover checkpoint", maxsplit=1
        )[1].split("# ── phase 5: resolve robot key", maxsplit=1)[0]
        checkpoint_helper = (ROOT / "tactile_checkpoint.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('gr00t_n15_tactile*', discovery)
        self.assertIn('tactile_checkpoint_discover', discovery)
        self.assertIn('/*/gr00t_n15_tactile*)', checkpoint_helper)
        self.assertNotIn('/*/gr00t_n15*)', checkpoint_helper)
        self.assertIn('== gr00t_n15_tactile', checkpoint_helper)
        self.assertIn('--policy_name GR00T_n15_Tactile', launcher)
        self.assertIn(
            'python policy/GR00T_n15_Tactile/tactile_eval_client.py', launcher
        )
        self.assertNotIn('python script/eval_policy_client.py', launcher)

    def test_sequential_eval_selects_tactile_policy(self) -> None:
        sequence = (ROOT / "eval_tasks_seq.sh").read_text(encoding="utf-8")
        self.assertIn('POLICY=GR00T_n15_Tactile', sequence)


if __name__ == "__main__":
    unittest.main()
