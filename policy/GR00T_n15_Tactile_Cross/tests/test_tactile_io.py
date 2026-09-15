from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from policy.GR00T_n15_Tactile_Cross.tactile_io import (
    TactileSchema,
    read_tactile_frame,
    read_tactile_schema,
    stack_tactile_mapping,
    validate_tactile_schema,
)


def _write_episode(
    path: Path,
    *,
    site_names: tuple[str, ...] = ("right_thumb", "left_thumb"),
    dtype: np.dtype = np.float32,
    d_max_m: float = 0.015,
    resolution_step: int = 1,
) -> None:
    with h5py.File(path, "w") as file:
        meta = file.require_group("robot/tactile/meta")
        meta.create_dataset("site_names", data=np.asarray(site_names, dtype=h5py.string_dtype("utf-8")))
        meta.create_dataset("max_distance_m", data=d_max_m)
        meta.create_dataset("resolution_step", data=resolution_step)
        raw_depth = file.require_group("robot/tactile/distance_along_normal_m")
        for site_idx, site_name in enumerate(site_names):
            values = np.full((3, 4, 5), 0.003 + site_idx * 0.012, dtype=dtype)
            raw_depth.create_dataset(site_name, data=values)


class TactileIOTest(unittest.TestCase):
    def test_reads_ordered_raw_depth_and_normalizes_by_d_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            _write_episode(path)
            with h5py.File(path, "r") as file:
                schema = read_tactile_schema(file, source=str(path))
                frame = read_tactile_frame(file, 1, schema)

        self.assertEqual(schema.site_names, ("right_thumb", "left_thumb"))
        self.assertEqual(schema.image_shape, (4, 5))
        self.assertEqual(schema.depth_key, "distance_along_normal_m")
        self.assertEqual(schema.depth_unit, "m")
        self.assertAlmostEqual(schema.d_max_m, 0.015)
        self.assertEqual(schema.resolution_step, 1)
        self.assertEqual(frame.shape, (2, 1, 4, 5))
        self.assertEqual(frame.dtype, np.float32)
        np.testing.assert_allclose(frame[0], 0.2)
        np.testing.assert_allclose(frame[1], 1.0)

    def test_clips_raw_depth_before_normalizing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episode.hdf5"
            _write_episode(path, site_names=("right_thumb",), d_max_m=0.01)
            with h5py.File(path, "a") as file:
                file["robot/tactile/distance_along_normal_m/right_thumb"][0] = np.array(
                    [[-0.002, 0.005, 0.020, 0.0, 0.010]] * 4,
                    dtype=np.float32,
                )
            with h5py.File(path, "r") as file:
                frame = read_tactile_frame(file, 0, read_tactile_schema(file))
        np.testing.assert_allclose(frame[0, 0, 0], [0.0, 0.5, 1.0, 0.0, 1.0])

    def test_rejects_quantized_or_non_positive_raw_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quantized = Path(tmp) / "quantized.hdf5"
            _write_episode(quantized, dtype=np.uint8)
            with h5py.File(quantized, "r") as file:
                with self.assertRaisesRegex(ValueError, "floating"):
                    read_tactile_schema(file)

            invalid_dmax = Path(tmp) / "invalid_dmax.hdf5"
            _write_episode(invalid_dmax, d_max_m=0.0)
            with h5py.File(invalid_dmax, "r") as file:
                with self.assertRaisesRegex(ValueError, "max_distance_m"):
                    read_tactile_schema(file)

    def test_validates_depth_contract_and_stacks_online_raw_mapping(self) -> None:
        expected = TactileSchema(
            ("right_thumb", "left_thumb"), (4, 5), "distance_along_normal_m", "m", 0.015
        )
        wrong_dmax = TactileSchema(
            ("right_thumb", "left_thumb"), (4, 5), "distance_along_normal_m", "m", 0.010
        )
        with self.assertRaisesRegex(ValueError, "d_max"):
            validate_tactile_schema(expected, wrong_dmax, source="episode_2.hdf5")
        wrong_step = TactileSchema(
            ("right_thumb", "left_thumb"), (4, 5), "distance_along_normal_m", "m", 0.015, 2
        )
        with self.assertRaisesRegex(ValueError, "resolution_step"):
            validate_tactile_schema(expected, wrong_step, source="episode_3.hdf5")

        mapping = {
            "left_thumb": np.full((4, 5), 0.015, dtype=np.float32),
            "right_thumb": np.full((4, 5), 0.003, dtype=np.float32),
        }
        frame = stack_tactile_mapping(mapping, expected, output_shape=(2, 3))
        self.assertEqual(frame.shape, (2, 1, 2, 3))
        np.testing.assert_allclose(frame[0], 0.2)
        np.testing.assert_allclose(frame[1], 1.0)


if __name__ == "__main__":
    unittest.main()
