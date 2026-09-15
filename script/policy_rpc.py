"""Length-prefixed JSON RPC with numpy payload support."""

from __future__ import annotations

import base64
import json
import socket
from typing import Any

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=_NumpyEncoder)


def json_to_numpy(payload: str) -> Any:
    def object_hook(item: dict[str, Any]) -> Any:
        if item.get("__numpy_array__"):
            raw = base64.b64decode(item["data"])
            return np.frombuffer(raw, dtype=np.dtype(item["dtype"])).reshape(item["shape"])
        return item

    return json.loads(payload, object_hook=object_hook)


def send_message(sock: socket.socket, payload: Any) -> None:
    encoded = numpy_to_json(payload).encode("utf-8")
    sock.sendall(len(encoded).to_bytes(4, "big"))
    sock.sendall(encoded)


def recv_message(sock: socket.socket) -> Any | None:
    length_bytes = sock.recv(4)
    if not length_bytes:
        return None  # clean shutdown (FIN), caller decides how to handle
    remaining = int.from_bytes(length_bytes, "big")
    chunks: list[bytes] = []
    while remaining > 0:
        chunk = sock.recv(min(remaining, 4096))
        if not chunk:
            raise ConnectionError("Incomplete payload received (RST or truncated).")
        chunks.append(chunk)
        remaining -= len(chunk)
    return json_to_numpy(b"".join(chunks).decode("utf-8"))


class RemotePolicyClient:
    """Client wrapper used by run_policy remote mode."""

    uses_raw_observation = True
    returns_action_chunks = True

    def __init__(self, host: str, port: int, timeout_s: float = 30.0) -> None:
        self._host = host
        self._port = port
        self._timeout_s = timeout_s
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout_s)
        self._sock.connect((host, port))

    def request(self, command: str, **payload: Any) -> Any:
        send_message(self._sock, {"command": command, **payload})
        response = recv_message(self._sock)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        if isinstance(response, dict) and "result" in response:
            return response["result"]
        return response

    def reset_model(self, seed: int | None = None) -> None:
        self.request("reset", seed=seed)

    def get_action(self, observation: dict[str, Any]) -> Any:
        return self.request(
            "get_action_chunk",
            observation=observation,
            instruction=observation.get("language"),
        )

    def update_obs(self, observation: dict[str, Any]) -> Any:
        return self.request(
            "update_after_action",
            observation=observation,
            instruction=observation.get("language"),
        )

    def close(self) -> None:
        try:
            self.request("close")
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass

