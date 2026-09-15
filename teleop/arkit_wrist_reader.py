"""
arkit_wrist_reader.py — 从 iPhone ARKit UDP 流读取手腕 *位置*。
姿态由 Manus 手套提供（更准确），这里只用 ARKit 的位置。

数据来源: ARKitPoseStreamer iOS App (UDP, 37 bytes per packet)
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np


# UDP 端口 (与 ARKitPoseStreamer App 一致)
RIGHT_HAND_PORT = 9999
LEFT_HAND_PORT = 9998

# 坐标轴映射: ARKit (x_a, y_a, z_a) → 仿真世界 (x_s, y_s, z_s)
# AXIS_MAPPING[i] = 仿真第 i 轴从 ARKit 第几轴取值
# AXIS_SIGN[i]    = 仿真第 i 轴的符号 (+1 或 -1)
# 例: AXIS_MAPPING=(0,2,1), AXIS_SIGN=(1,-1,1) → sim_x=+arkit_x, sim_y=-arkit_z, sim_z=+arkit_y
AXIS_MAPPING: Tuple[int, int, int] = (0, 2, 1)
AXIS_SIGN: Tuple[float, float, float] = (1.0, -1.0, 1.0)

# 仿真固定锚点 (世界坐标, 米)
# ARKit 原点 (0,0,0) 始终映射到此仿真世界坐标，与机器人种类无关。
# 桌面高度 0.75m，锚点在桌面中心偏前上方 30cm 处。
SIM_ANCHOR: Tuple[float, float, float] = (0.0, 0.15, 1.05)

MAGIC = b"ARKT"
PACKET_SIZE = 37


class ARKitWristReader:
    """从 iPhone ARKit UDP 流读取手腕位置。

    ARKit 原点 (0,0,0) 始终映射到仿真世界坐标 SIM_ANCHOR。
    sim_pos = SIM_ANCHOR + (mapped_arkit_pos - anchor),  anchor 默认为 (0,0,0)。
    """

    def __init__(self, side: str = "right"):
        self._side = side.lower()
        self._port = RIGHT_HAND_PORT if self._side == "right" else LEFT_HAND_PORT
        self._anchor_arkit = np.zeros(3, dtype=np.float64)
        self._sock: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_pos: Optional[np.ndarray] = None
        self._latest_quat: Optional[np.ndarray] = None  # (qx, qy, qz, qw) ARKit Y-up
        self._last_recv_time = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self._port))
        sock.settimeout(0.01)
        self._sock = sock
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        print(f"[ARKitWrist] Listening {self._side} on port {self._port}")

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def is_receiving(self, stale_timeout: float = 1.0) -> bool:
        return self._last_recv_time > 0 and (time.time() - self._last_recv_time) < stale_timeout

    def reset_anchor(self):
        self._anchor_arkit = np.zeros(3, dtype=np.float64)

    def get_position(self) -> Optional[np.ndarray]:
        """获取仿真世界坐标系中的手腕位置 (3,), 或 None。"""
        with self._lock:
            return self._latest_pos

    def get_quaternion(self) -> Optional[np.ndarray]:
        """获取 ARKit 四元数 (qx, qy, qz, qw)，Y-up 右手系。"""
        with self._lock:
            return self._latest_quat

    def _recv_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(64)
                if len(data) < PACKET_SIZE or data[0:4] != MAGIC:
                    continue
                floats = struct.unpack_from("<7f", data, 9)
                arkit_pos = np.array([floats[0], floats[1], floats[2]], dtype=np.float64)
                arkit_quat = np.array([floats[3], floats[4], floats[5], floats[6]], dtype=np.float64)  # qx,qy,qz,qw
                self._last_recv_time = time.time()

                a, b, c = AXIS_MAPPING
                mapped = np.array([
                    AXIS_SIGN[0] * arkit_pos[a],
                    AXIS_SIGN[1] * arkit_pos[b],
                    AXIS_SIGN[2] * arkit_pos[c],
                ], dtype=np.float64)

                delta = mapped - self._anchor_arkit
                sim_pos = np.array(SIM_ANCHOR, dtype=np.float64) + delta

                with self._lock:
                    self._latest_pos = sim_pos
                    self._latest_quat = arkit_quat.astype(np.float32)
            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    print(f"[ARKitWrist] recv error: {e}")
