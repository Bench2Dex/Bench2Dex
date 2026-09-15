from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class TactileTrainingDefaultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolate the pure resolver from heavyweight training imports.
        path = Path(__file__).resolve().parents[1] / "scripts/gr00t_finetune.py"
        tree = ast.parse(path.read_text())
        resolver = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_resolve_tactile_overrides"
        )
        namespace = {"ArgsConfig": SimpleNamespace}
        exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(path), "exec"), namespace)
        cls.resolve = staticmethod(namespace["_resolve_tactile_overrides"])

    def config(self, **overrides):
        values = dict(tactile_fusion_stage=None, tactile_grid_size=None, tactile_dropout_prob=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_new_tactile_head_uses_new_defaults(self):
        self.assertEqual(self.resolve(self.config(), new_tactile_head=True), {
            "tactile_fusion_stage": "post_dit",
            "tactile_grid_size": 2,
            "tactile_dropout_prob": 0.3,
        })

    def test_existing_checkpoint_has_no_implicit_overrides(self):
        self.assertEqual(self.resolve(self.config(), new_tactile_head=False), {})

    def test_explicit_overrides_win_including_zero_dropout(self):
        config = self.config(tactile_fusion_stage="pre_dit", tactile_grid_size=4,
                             tactile_dropout_prob=0.0)
        for new_head in (True, False):
            self.assertEqual(self.resolve(config, new_tactile_head=new_head), vars(config))


if __name__ == "__main__":
    unittest.main()
