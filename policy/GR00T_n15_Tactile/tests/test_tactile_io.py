from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy.GR00T_n15_Tactile.tactile_io import (
    TactileSchema,
    read_tactile_frame,
    read_tactile_schema,
    stack_tactile_mapping,
    validate_tactile_schema,
)


def _write_episode(path: Path, *, site_names=("right_thumb", "left_thumb"), dtype=np.uint8) -> None:
    with h5py.File(path, "w") as file:
        meta = file.require_group("robot/tactile/meta")
        meta.create_dataset("site_names", data=np.asarray(site_names, dtype=h5py.string_dtype("utf-8")))
        tacmap = file.require_group("robot/tactile/tacmap")
        for site_idx, site_name in enumerate(site_names):
            values = np.full((3, 4, 5), 32 + site_idx * 64, dtype=dtype)
            tacmap.create_dataset(site_name, data=values)


class TactileIOTest(unittest.TestCase):
    def test_reads_site_order_and_normalized_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            _write_episode(path)
            with h5py.File(path, "r") as file:
                schema = read_tactile_schema(file, source=str(path))
                frame = read_tactile_frame(file, 1, schema)

        self.assertEqual(schema.site_names, ("right_thumb", "left_thumb"))
        self.assertEqual(schema.image_shape, (4, 5))
        self.assertEqual(frame.shape, (2, 1, 4, 5))
        self.assertEqual(frame.dtype, np.float32)
        np.testing.assert_allclose(frame[0], 32.0 / 255.0)
        np.testing.assert_allclose(frame[1], 96.0 / 255.0)

    def test_rejects_non_uint8_tacmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            _write_episode(path, dtype=np.float32)
            with h5py.File(path, "r") as file:
                with self.assertRaisesRegex(ValueError, "uint8"):
                    read_tactile_schema(file, source=str(path))

    def test_rejects_schema_mismatch(self) -> None:
        expected = TactileSchema(("right_thumb", "left_thumb"), (4, 5))
        actual = TactileSchema(("left_thumb", "right_thumb"), (4, 5))
        with self.assertRaisesRegex(ValueError, "site order"):
            validate_tactile_schema(expected, actual, source="episode_2.hdf5")

    def test_resizes_tactile_maps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            _write_episode(path)
            with h5py.File(path, "r") as file:
                schema = read_tactile_schema(file, source=str(path))
                frame = read_tactile_frame(file, 0, schema, output_shape=(2, 3))
        self.assertEqual(frame.shape, (2, 1, 2, 3))
        np.testing.assert_allclose(frame[0], 32.0 / 255.0)

    def test_stacks_runtime_mapping_in_checkpoint_order(self) -> None:
        schema = TactileSchema(("right_thumb", "left_thumb"), (4, 5))
        mapping = {
            "left_thumb": np.full((4, 5), 96, dtype=np.uint8),
            "right_thumb": np.full((4, 5), 32, dtype=np.uint8),
        }
        frame = stack_tactile_mapping(mapping, schema, output_shape=(2, 3))
        self.assertEqual(frame.shape, (2, 1, 2, 3))
        np.testing.assert_allclose(frame[0], 32.0 / 255.0)
        np.testing.assert_allclose(frame[1], 96.0 / 255.0)


if __name__ == "__main__":
    unittest.main()
