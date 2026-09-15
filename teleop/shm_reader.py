"""
shm_reader.py — Read Manus hand skeleton data from POSIX shared memory.

Wire format must match ShmWriter.hpp exactly:
  ShmBlock (1448 bytes total):
    magic      : uint32  (0x4D414E55 = "MANU")
    version    : uint32  (1)
    seq        : uint64  (frame counter)
    timestamp_us: uint64 (microseconds since epoch)
    hand_count : uint8   (0–2)
    pad        : uint8[7]
    hands[2]   : ShmHandData  ([0]=right, [1]=left)

  ShmHandData (708 bytes):
    valid      : uint8   (1 if data present)
    side       : uint8   (1=left, 2=right)
    pad        : uint8[2]
    node_count : uint32  (usually 25)
    nodes[25]  : ShmNodeData

  ShmNodeData (28 bytes):
    px, py, pz : float32 (position)
    qx, qy, qz, qw : float32 (quaternion)
"""

from __future__ import annotations

import ctypes
import mmap
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Constants (must match ShmWriter.hpp) ────────────────────────────────────

SHM_NAME = "/manus_hand_data"
SHM_MAGIC = 0x4D414E55
SHM_VERSION = 1
MAX_NODES = 25

SIDE_LEFT = 1
SIDE_RIGHT = 2

# Struct sizes (packed)
_NODE_SIZE = 28  # 7 × float32
_HEADER_SIZE = 32  # magic(4) + version(4) + seq(8) + timestamp_us(8) + hand_count(1) + pad(7)
_HAND_DATA_SIZE = 1 + 1 + 2 + 4 + MAX_NODES * _NODE_SIZE  # 708
_SHM_BLOCK_SIZE = _HEADER_SIZE + 2 * _HAND_DATA_SIZE  # 1448


@dataclass
class HandFrame:
    """Parsed data for one hand in one frame."""
    valid: bool = False
    side: int = 0  # 1=left, 2=right
    node_count: int = 0
    positions: np.ndarray = field(default_factory=lambda: np.zeros((MAX_NODES, 3), dtype=np.float32))
    quaternions: np.ndarray = field(default_factory=lambda: np.zeros((MAX_NODES, 4), dtype=np.float32))


@dataclass
class ShmFrame:
    """A complete frame from shared memory."""
    seq: int = 0
    timestamp_us: int = 0
    hand_count: int = 0
    right: HandFrame = field(default_factory=HandFrame)  # slot 0
    left: HandFrame = field(default_factory=HandFrame)    # slot 1


class ShmReader:
    """
    Read Manus hand data from POSIX shared memory.

    Usage:
        reader = ShmReader()
        reader.open()
        frame = reader.read()
        if frame and frame.right.valid:
            positions = frame.right.positions  # (25, 3)
        reader.close()
    """

    def __init__(self, shm_name: str = SHM_NAME):
        self._shm_name = shm_name
        self._fd: Optional[int] = None
        self._mm: Optional[mmap.mmap] = None
        self._last_seq: int = 0

    def open(self) -> bool:
        """Open the shared memory segment. Returns True on success."""
        import os

        shm_path = f"/dev/shm{self._shm_name}"
        try:
            fd = os.open(shm_path, os.O_RDONLY)
            self._mm = mmap.mmap(fd, _SHM_BLOCK_SIZE, access=mmap.ACCESS_READ)
            os.close(fd)
            return True
        except FileNotFoundError:
            print(f"[ShmReader] Shared memory not found: {shm_path}")
            print(f"[ShmReader] ManusSDK may not be running yet.")
            return False
        except Exception as e:
            print(f"[ShmReader] Failed to open shared memory: {e}")
            return False

    def close(self):
        """Close the shared memory mapping."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None

    def is_open(self) -> bool:
        return self._mm is not None

    def read(self) -> Optional[ShmFrame]:
        """
        Read current frame from shared memory.
        Returns ShmFrame or None if not open or invalid.
        """
        if self._mm is None:
            return None

        self._mm.seek(0)
        buf = self._mm.read(_SHM_BLOCK_SIZE)
        if len(buf) < _SHM_BLOCK_SIZE:
            return None

        # Parse header
        magic, version, seq, timestamp_us, hand_count = struct.unpack_from("<IIQQB", buf, 0)
        # 4+4+8+8+1 = 25 bytes, then 7 pad bytes → 32 total header

        if magic != SHM_MAGIC or version != SHM_VERSION:
            return None

        frame = ShmFrame(seq=seq, timestamp_us=timestamp_us, hand_count=hand_count)

        # Parse hands
        for slot in range(2):
            offset = _HEADER_SIZE + slot * _HAND_DATA_SIZE
            hand = self._parse_hand(buf, offset)
            if slot == 0:
                frame.right = hand
            else:
                frame.left = hand

        return frame

    def read_if_new(self) -> Optional[ShmFrame]:
        """Read only if there's a new frame (seq changed)."""
        frame = self.read()
        if frame is None:
            return None
        if frame.seq <= self._last_seq:
            return None
        self._last_seq = frame.seq
        return frame

    @staticmethod
    def _parse_hand(buf: bytes, offset: int) -> HandFrame:
        """Parse one ShmHandData from buffer."""
        hand = HandFrame()

        valid, side = struct.unpack_from("<BB", buf, offset)
        # skip 2 pad bytes
        node_count = struct.unpack_from("<I", buf, offset + 4)[0]

        hand.valid = bool(valid)
        hand.side = side
        hand.node_count = min(node_count, MAX_NODES)

        if hand.valid and hand.node_count > 0:
            nodes_offset = offset + 8  # after valid(1)+side(1)+pad(2)+node_count(4)
            for i in range(hand.node_count):
                n_off = nodes_offset + i * _NODE_SIZE
                vals = struct.unpack_from("<7f", buf, n_off)
                hand.positions[i] = vals[0:3]
                hand.quaternions[i] = vals[3:7]

        return hand

    def get_pos_dict(self, hand: HandFrame) -> dict:
        """
        Convert HandFrame to position dict compatible with manus_nodes_to_mediapipe().
        Returns {node_id: np.array([x,y,z], dtype=float32)}
        """
        if not hand.valid:
            return {}
        return {i: hand.positions[i].copy() for i in range(hand.node_count)}

    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """Wait until valid data appears in shared memory."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            frame = self.read()
            if frame is not None and frame.hand_count > 0:
                return True
            time.sleep(0.01)
        return False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
