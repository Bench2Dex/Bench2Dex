import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class LenientCheckpointWeightLoader(WeightLoader):
    """Like ``CheckpointWeightLoader`` but skips parameters whose shape does not
    match the reference (random-initialized) parameters.

    Useful when fine-tuning a pretrained checkpoint with a different
    ``action_dim`` (e.g. dex2bench 36-dim vs. the pretrained 32-dim): the
    state/action projection layers will be kept at their random initialization
    while every other compatible weight is loaded from the checkpoint.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*lora.*",
            skip_shape_mismatch=True,
        )


def _merge_params(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
    skip_shape_mismatch: bool = False,
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.
        skip_shape_mismatch: If True, parameters whose shape does not match the reference are dropped (the
            reference's random initialization is kept).

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    skipped_shape = []
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref = flat_ref[k]
            if skip_shape_mismatch and tuple(v.shape) != tuple(ref.shape):
                skipped_shape.append((k, tuple(v.shape), tuple(ref.shape)))
                continue
            result[k] = v.astype(ref.dtype)

    if skipped_shape:
        logger.warning(
            "Skipped %d weights with shape mismatch (kept random init):\n%s",
            len(skipped_shape),
            "\n".join(f"  {k}: loaded {ls} vs ref {rs}" for k, ls, rs in skipped_shape),
        )

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    # Always keep reference values for keys that were skipped due to shape mismatch.
    for k, _, _ in skipped_shape:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
