"""Single-threaded TCP server for the dex2bench policy RPC (see wire.py)."""

from __future__ import annotations

import socket
import sys
import traceback
from typing import Any

from tools.rpc.wire import decode_payload, encode_payload, recv_msg, send_msg


def serve(model: Any, host: str = "0.0.0.0", port: int = 6694, *, log: bool = True) -> None:
    """Block forever serving RPC calls dispatched against ``model``.

    The protocol is one-client-at-a-time: when a client disconnects we accept
    the next one. This matches the sim-driver workflow where exactly one Isaac
    Sim process is talking to exactly one policy.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)

    def _log(msg: str) -> None:
        if log:
            print(f"[policy_server] {msg}", flush=True)

    _log(f"listening on {host}:{port}")

    try:
        while True:
            conn, addr = listener.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _log(f"client connected from {addr[0]}:{addr[1]}")
            try:
                _handle_connection(model, conn, log=_log)
            except (ConnectionError, ConnectionResetError, BrokenPipeError) as exc:
                _log(f"client disconnected: {exc}")
            except Exception as exc:  # pragma: no cover -- defensive
                _log(f"connection handler raised: {exc}")
                traceback.print_exc(file=sys.stderr)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        _log("interrupted; shutting down")
    finally:
        listener.close()


def _handle_connection(model: Any, conn: socket.socket, *, log) -> None:
    while True:
        try:
            req = recv_msg(conn)
        except ConnectionError:
            return
        cmd = req.get("cmd")
        obs = req.get("obs")
        try:
            method = getattr(model, cmd)
        except AttributeError:
            send_msg(conn, {"error": f"model has no attribute '{cmd}'", "traceback": ""})
            continue
        try:
            if obs is None:
                result = method() if callable(method) else method
            else:
                result = method(decode_payload(obs))
            send_msg(conn, {"res": encode_payload(result)})
        except Exception as exc:
            send_msg(
                conn,
                {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
            )
            log(f"error in {cmd}: {exc}")
