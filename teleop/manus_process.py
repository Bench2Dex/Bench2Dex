"""
manus_process.py — Manus shared-memory readiness check.

The ManusSDK C++ client (SDKClient_Linux.out) is started manually::

    cd teleop/manus_bin && ./SDKClient_Linux.out 1

This module provides a helper to wait until shared memory appears.
"""

from __future__ import annotations

import os
import time

_SHM_PATH = "/dev/shm/manus_hand_data"


def wait_for_shm(timeout: float = 30.0) -> bool:
    """Block until the Manus shared-memory file exists (or *timeout* expires).

    Returns True if SHM is ready, False on timeout.
    """
    if os.path.exists(_SHM_PATH):
        return True
    print(f"[Manus] Waiting for shared memory ({_SHM_PATH})...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if os.path.exists(_SHM_PATH):
            print(f"[Manus] Shared memory ready ({time.monotonic() - t0:.1f}s)")
            return True
        time.sleep(0.5)
    print(f"[Manus] SHM not found within {timeout}s. "
          "Is SDKClient_Linux.out running?")
    return False
