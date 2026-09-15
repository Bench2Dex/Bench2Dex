"""Socket server for remote policy evaluation."""

from __future__ import annotations

import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.config_utils import build_config_parser, load_config_with_overrides
from script.policy_rpc import recv_message, send_message
from script.policy_sessions import load_policy_session


class PolicyModelServer:
    def __init__(self, session: Any, host: str, port: int) -> None:
        self._session = session
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._running = False
        self._threads: list[threading.Thread] = []

    def serve_forever(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind((self._host, self._port))
        self._sock.listen(5)
        self._sock.settimeout(1.0)
        self._running = True
        print(f"[policy-model-server] listening on {self._host}:{self._port}", flush=True)
        while self._running:
            try:
                client, _addr = self._sock.accept()
            except socket.timeout:
                continue
            worker = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
            worker.start()
            self._threads.append(worker)

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
        try:
            self._session.close()
        except Exception:
            pass

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            while self._running:
                try:
                    payload = recv_message(client)
                except ConnectionError:
                    break  # RST or truncated — client already gone, nothing to send
                except OSError:
                    break  # socket already closed, nothing to send

                if payload is None:
                    break  # clean FIN from client, no error

                try:
                    command = payload.get("command")
                    if command == "reset":
                        result = self._session.reset(
                            payload.get("instruction"),
                            seed=payload.get("seed"),
                        )
                    elif command == "get_action_chunk":
                        result = self._session.get_action_chunk(
                            payload.get("observation", {}),
                            instruction=payload.get("instruction"),
                        )
                    elif command == "update_after_action":
                        result = self._session.update_after_action(
                            payload.get("observation", {}),
                            instruction=payload.get("instruction"),
                        )
                    elif command == "close":
                        try:
                            send_message(client, {"result": "ok"})
                        except OSError:
                            pass
                        break
                    else:
                        raise ValueError(f"Unsupported command: {command}")
                    try:
                        send_message(client, {"result": result})
                    except OSError:
                        break  # client disconnected before we could reply
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                    try:
                        send_message(client, {"error": error})
                    except OSError:
                        pass  # client already gone, cannot inform it
                    break


def main() -> None:
    parser = build_config_parser("Run the dex2scene remote policy model server.")
    args = parser.parse_args()
    config = load_config_with_overrides(
        args.config,
        args.overrides,
        extra_updates={
            "host": args.host,
            "port": args.port,
        },
    )
    policy_name = config["policy_name"]
    host = str(config.get("host") or "127.0.0.1")
    port = int(config.get("port") or 9000)
    session = load_policy_session(policy_name, config)
    server = PolicyModelServer(session, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[policy-model-server] stopping", flush=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
