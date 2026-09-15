"""Client-side shim that makes a ``ModelClient`` look like an in-process policy.

``utils.episode_loop.run_one_episode`` accesses the policy via a small duck-typed
surface: ``get_action(obs)``, ``update_obs(obs)``, ``reset_model()``, and the
state-like attributes ``t`` and ``temporal_agg``. ``ClientPolicyHandle`` forwards
the method calls over RPC and tracks ``t`` locally — no extra round-trip per
step. ``temporal_agg`` is set once at construction from the client's CLI flag.
"""

from __future__ import annotations

from typing import Any

from tools.rpc.client import ModelClient


class ClientPolicyHandle:
    def __init__(self, client: ModelClient, *, temporal_agg: bool = False) -> None:
        self._client = client
        self.t = 0
        self.temporal_agg = temporal_agg

    def get_action(self, obs: Any = None) -> Any:
        return self._client.call("get_action", obs)

    def update_obs(self, obs: Any) -> None:
        self._client.call("update_obs", obs)

    def reset_model(self) -> None:
        try:
            self._client.call("reset_model", None)
        except RuntimeError:
            # Not every policy exposes reset_model; best-effort.
            pass
        self.t = 0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ClientPolicyHandle":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
