"""dex2bench policy server/client RPC (length-prefixed JSON over TCP).

Public API
    ModelClient            -- sim-side client
    ClientPolicyHandle     -- duck-typed shim for utils.episode_loop
    serve                  -- server-side dispatch loop
    send_msg / recv_msg    -- frame helpers
    encode_payload / decode_payload -- numpy-aware serialization
"""

from tools.rpc.client import ModelClient
from tools.rpc.policy_handle import ClientPolicyHandle
from tools.rpc.server import serve
from tools.rpc.wire import (
    decode_payload,
    encode_jpeg,
    encode_payload,
    recv_msg,
    send_msg,
)

__all__ = [
    "ModelClient",
    "ClientPolicyHandle",
    "serve",
    "send_msg",
    "recv_msg",
    "encode_payload",
    "decode_payload",
    "encode_jpeg",
]
