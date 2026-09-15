"""TCP client for the dex2bench policy server (see ``tools/rpc/wire.py``)."""

from __future__ import annotations

import socket
import time
from typing import Any

from tools.rpc.wire import decode_payload, encode_payload, recv_msg, send_msg


class ModelClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6694,
        *,
        connect_retries: int = 600,
        retry_delay_s: float = 1.0,
        socket_timeout_s: float | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._connect(connect_retries, retry_delay_s, socket_timeout_s)

    def _connect(self, retries: int, delay_s: float, timeout_s: float | None) -> None:
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if timeout_s is not None:
                    s.settimeout(timeout_s)
                s.connect((self.host, self.port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock = s
                return
            except (ConnectionRefusedError, OSError) as exc:
                last_exc = exc
                time.sleep(delay_s)
        raise ConnectionError(
            f"ModelClient failed to connect to {self.host}:{self.port} after {retries} attempts"
        ) from last_exc

    def call(self, func_name: str, obs: Any = None) -> Any:
        if self._sock is None:
            raise ConnectionError("ModelClient is closed")
        payload = {"cmd": func_name, "obs": encode_payload(obs) if obs is not None else None}
        send_msg(self._sock, payload)
        resp = recv_msg(self._sock)
        if "error" in resp:
            raise RuntimeError(
                f"policy server error in {func_name}: {resp['error']}\n{resp.get('traceback', '')}"
            )
        return decode_payload(resp.get("res"))

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "ModelClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
