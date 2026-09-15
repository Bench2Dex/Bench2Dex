"""Policy-side session adapters for remote evaluation."""

from __future__ import annotations

import copy
import importlib
import inspect
from typing import Any

import numpy as np

from utils.seed_policy import seed_everything


class RemoteEvalNotSupportedError(RuntimeError):
    """Raised when a policy cannot be driven through the generic remote session."""


def load_policy_module(policy_name: str):
    return importlib.import_module(f"policy.{policy_name}.deploy_policy")


def _coerce_seed(seed: int | str | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, str):
        value = seed.strip().lower()
        if value in {"", "none", "null"}:
            return None
        seed = value
    return int(seed)


def _reset_openpi_jax_rng(model: Any | None, seed_int: int) -> None:
    if model is None:
        return
    candidates = [model]
    wrapped_policy = getattr(model, "policy", None)
    if wrapped_policy is not None:
        candidates.append(wrapped_policy)

    try:
        import jax
    except Exception:
        return

    for candidate in candidates:
        if hasattr(candidate, "_rng"):
            candidate._rng = jax.random.key(seed_int)


def _reset_rng(seed: int | str | None, model: Any | None = None) -> None:
    seed_int = _coerce_seed(seed)
    if seed_int is None:
        return
    seed_everything(seed_int)
    _reset_openpi_jax_rng(model, seed_int)


class DefaultRemotePolicySession:
    """Default adapter for stateless get_action(obs) policies."""

    _wrap_instruction_in_list: bool = False

    def __init__(self, policy_module: Any, usr_args: dict[str, Any]) -> None:
        self._module = policy_module
        self._model = policy_module.get_model(usr_args)
        self._encode_obs = getattr(policy_module, "encode_obs", None)
        self._reset_fn = getattr(policy_module, "reset_model", None)
        self._get_action_method = getattr(self._model, "get_action", None)
        if not callable(self._get_action_method):
            raise RemoteEvalNotSupportedError("Model does not expose get_action(obs).")

        signature = inspect.signature(self._get_action_method)
        positional = [
            parameter
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) != 1:
            raise RemoteEvalNotSupportedError(
                "Default remote session requires get_action(obs) with one observation argument.",
            )

    def _prepare_observation(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
    ) -> Any:
        raw = copy.copy(observation)
        if instruction:
            raw.setdefault("language", instruction)
        encoded = self._encode_obs(raw) if callable(self._encode_obs) else raw
        if instruction and isinstance(encoded, dict):
            encoded.setdefault(
                "raw_lang", [instruction] if self._wrap_instruction_in_list else instruction
            )
            encoded.setdefault(
                "instruction", [instruction] if self._wrap_instruction_in_list else instruction
            )
            encoded.setdefault(
                "language", [instruction] if self._wrap_instruction_in_list else instruction
            )
        return encoded

    @staticmethod
    def _normalize_chunk(actions: Any) -> list[np.ndarray]:
        array = np.asarray(actions, dtype=np.float32)
        if array.ndim == 0:
            raise ValueError("Policy returned a scalar action.")
        if array.ndim == 1:
            return [array]
        return [np.asarray(action, dtype=np.float32) for action in array.reshape(-1, array.shape[-1])]

    def reset(self, instruction: str | None = None, *, seed: int | None = None) -> None:
        _reset_rng(seed, self._model)
        if callable(self._reset_fn):
            self._reset_fn(self._model)
        elif hasattr(self._model, "reset_model"):
            self._model.reset_model()

    def get_action_chunk(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
    ) -> list[np.ndarray]:
        prepared = self._prepare_observation(observation, instruction)
        return self._normalize_chunk(self._get_action_method(prepared))

    def update_after_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
    ) -> None:
        updater = getattr(self._model, "update_obs", None)
        if callable(updater):
            prepared = self._prepare_observation(observation, instruction)
            updater(prepared)

    def close(self) -> None:
        return None


class GR00TRemotePolicySession(DefaultRemotePolicySession):
    """GR00T-N15 adapter: wraps instruction fields as single-element lists after encoding."""

    _wrap_instruction_in_list: bool = True


class ObservationWindowRemoteSession:
    """Adapter for policies that keep an internal observation window."""

    def __init__(
        self,
        policy_module: Any,
        usr_args: dict[str, Any],
        *,
        set_instruction_method: str,
        update_method: str,
        get_action_method: str,
        reset_method: str | None = None,
        action_limit_attr: str | None = None,
    ) -> None:
        self._module = policy_module
        self._model = policy_module.get_model(usr_args)
        self._encode_obs = getattr(policy_module, "encode_obs")
        self._set_instruction_method = set_instruction_method
        self._update_method = update_method
        self._get_action_method = get_action_method
        self._reset_method = reset_method
        self._action_limit_attr = action_limit_attr
        self._instruction_applied = False

    def _apply_instruction(self, instruction: str | None) -> None:
        if not instruction or self._instruction_applied:
            return
        setter = getattr(self._model, self._set_instruction_method, None)
        if callable(setter):
            setter(instruction)
        self._instruction_applied = True

    def _update_window(self, observation: dict[str, Any]) -> None:
        encoded = self._encode_obs(observation)
        updater = getattr(self._model, self._update_method)
        if isinstance(encoded, tuple):
            updater(*encoded)
        else:
            updater(encoded)

    def reset(self, instruction: str | None = None, *, seed: int | None = None) -> None:
        _reset_rng(seed, self._model)
        if self._reset_method:
            resetter = getattr(self._model, self._reset_method, None)
            if callable(resetter):
                resetter()
        self._instruction_applied = False

    def get_action_chunk(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
    ) -> list[np.ndarray]:
        self._apply_instruction(instruction)
        self._update_window(observation)
        getter = getattr(self._model, self._get_action_method)
        actions = np.asarray(getter(), dtype=np.float32)
        if self._action_limit_attr:
            limit = int(getattr(self._model, self._action_limit_attr))
            actions = actions[:limit]
        if actions.ndim == 1:
            return [actions]
        return [np.asarray(action, dtype=np.float32) for action in actions.reshape(-1, actions.shape[-1])]

    def update_after_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
    ) -> None:
        self._apply_instruction(instruction)
        self._update_window(observation)

    def close(self) -> None:
        return None


class UnsupportedRemoteSession:
    def __init__(self, message: str) -> None:
        self._message = message

    def reset(self, instruction: str | None = None, *, seed: int | None = None) -> None:
        _reset_rng(seed)
        return None

    def get_action_chunk(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
    ) -> list[np.ndarray]:
        raise RemoteEvalNotSupportedError(self._message)

    def update_after_action(
        self,
        observation: dict[str, Any],
        instruction: str | None = None,
    ) -> None:
        return None

    def close(self) -> None:
        return None


def _load_special_session(policy_name: str, policy_module: Any, usr_args: dict[str, Any]):
    if policy_name == "GR00T_n15":
        return GR00TRemotePolicySession(policy_module, usr_args)
    if policy_name == "DP3":
        return UnsupportedRemoteSession(
            "DP3 remote evaluation is not available yet: the dex2scene live client does not "
            "capture pointcloud observations required by policy/DP3/deploy_policy.py.",
        )
    if policy_name == "GO1":
        return ObservationWindowRemoteSession(
            policy_module,
            usr_args,
            set_instruction_method="set_language",
            update_method="update_observation_window",
            get_action_method="get_action",
            reset_method="reset_observation_window",
        )
    if policy_name in {"pi0", "pi05"}:
        return ObservationWindowRemoteSession(
            policy_module,
            usr_args,
            set_instruction_method="set_language",
            update_method="update_observation_window",
            get_action_method="get_action",
            reset_method="reset_observation_windows",
            action_limit_attr="eval_action_horizon",
        )
    if policy_name == "RDT":
        return ObservationWindowRemoteSession(
            policy_module,
            usr_args,
            set_instruction_method="set_language_instruction",
            update_method="update_observation_window",
            get_action_method="get_action",
            reset_method="reset_observation_windows",
            action_limit_attr="rdt_step",
        )
    return None


def load_policy_session(policy_name: str, usr_args: dict[str, Any]):
    _reset_rng(usr_args.get("seed"))
    policy_module = load_policy_module(policy_name)

    custom_factory = getattr(policy_module, "create_remote_session", None)
    if callable(custom_factory):
        return custom_factory(usr_args)

    special_session = _load_special_session(policy_name, policy_module, usr_args)
    if special_session is not None:
        return special_session

    return DefaultRemotePolicySession(policy_module, usr_args)
