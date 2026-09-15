"""
Length-prefixed JSON wire format used by the dex2bench policy server / client.

Frame format
------------
    [4 bytes big-endian uint32 length][JSON body]

Request body
    {"cmd": "<method_name>", "obs": <obs_or_null>}

Response body
    {"res": <return_value>}                  # success
    {"error": "<msg>", "traceback": "..."}   # failure

Payload sentinels (recursive in obs / return value)
    {"__np__": true, "dtype": "float32", "shape": [...], "b64": "..."}
    {"__jpg__": true, "shape": [H, W, 3], "b64": "..."}    # opt-in only
"""

from __future__ import annotations

import base64
import json
import socket
import struct
from typing import Any

import numpy as np

_LEN_STRUCT = struct.Struct(">I")


def send_msg(sock: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(_LEN_STRUCT.pack(len(body)) + body)


def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exactly(sock, _LEN_STRUCT.size)
    (length,) = _LEN_STRUCT.unpack(header)
    body = _recv_exactly(sock, length)
    return json.loads(body.decode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"peer closed: wanted {n} bytes, got {n - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_payload(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {
            "__np__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "b64": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii"),
        }
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: encode_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        encoded = [encode_payload(v) for v in obj]
        return encoded if isinstance(obj, list) else encoded
    return obj


def decode_payload(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__np__") is True:
            buf = base64.b64decode(obj["b64"])
            arr = np.frombuffer(buf, dtype=np.dtype(obj["dtype"]))
            return arr.reshape(obj["shape"]) if obj["shape"] else arr.reshape(())
        if obj.get("__jpg__") is True:
            buf = base64.b64decode(obj["b64"])
            try:
                import cv2  # local import: not needed unless JPEG path used
            except ImportError as exc:
                raise RuntimeError("cv2 required to decode __jpg__ payloads") from exc
            arr = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError("cv2.imdecode returned None for __jpg__ payload")
            return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return {k: decode_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_payload(v) for v in obj]
    return obj


def encode_jpeg(rgb_hwc: np.ndarray, quality: int = 90) -> dict:
    """Opt-in helper: encode an HxWx3 uint8 RGB image as a __jpg__ payload."""
    import cv2

    if rgb_hwc.dtype != np.uint8 or rgb_hwc.ndim != 3 or rgb_hwc.shape[-1] != 3:
        raise ValueError(f"encode_jpeg expects HxWx3 uint8 RGB, got {rgb_hwc.shape} {rgb_hwc.dtype}")
    bgr = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for JPEG payload")
    return {
        "__jpg__": True,
        "shape": list(rgb_hwc.shape),
        "b64": base64.b64encode(buf.tobytes()).decode("ascii"),
    }
