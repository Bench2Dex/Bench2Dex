"""Read per-pair contact forces from Isaac Lab ContactSensor instances."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContactSensorReader:
    """Creates and reads ContactSensor instances for specified tool-target pairs.

    Usage::

        reader = ContactSensorReader(
            contact_pairs=[
                {"sensor": "obj_048_hammer_3", "filter_targets": ["obj_061_foam_brick_6"]},
            ],
            object_prim_paths={
                "obj_048_hammer_3": "/World/Objects/obj_048_hammer_3",
                "obj_061_foam_brick_6": "/World/Objects/obj_061_foam_brick_6",
            },
        )
        # After each sim.step():
        reader.update(dt)
        forces = reader.read()
        # forces == {"obj_048_hammer_3": {"obj_061_foam_brick_6": [fx, fy, fz]}}
    """

    def __init__(
        self,
        contact_pairs: list[dict[str, Any]],
        object_prim_paths: dict[str, str] | None = None,
        env_ns: str = "",
    ) -> None:
        self._pair_map: dict[str, dict[str, int]] = {}
        self._sensors: dict[str, Any] = {}
        self._available = False

        if not contact_pairs:
            return

        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg
        except ImportError:
            logger.warning(
                "ContactSensor not available (Isaac Lab not installed). "
                "Contact force data will be unavailable."
            )
            return

        for pair in contact_pairs:
            sensor_id = str(pair["sensor"])
            targets = [str(t) for t in pair["filter_targets"]]

            # Resolve prim paths: prefer direct lookup from object_prim_paths,
            # fall back to env_ns-based construction for backward compat.
            if object_prim_paths and sensor_id in object_prim_paths:
                prim_path = object_prim_paths[sensor_id]
            elif env_ns:
                prim_path = f"{env_ns}/Objects/{sensor_id}"
            else:
                logger.warning(
                    "ContactSensor: no prim path for sensor '%s' "
                    "(provide object_prim_paths or env_ns)", sensor_id,
                )
                continue

            filter_paths: list[str] = []
            skip = False
            for t in targets:
                if object_prim_paths and t in object_prim_paths:
                    filter_paths.append(object_prim_paths[t])
                elif env_ns:
                    filter_paths.append(f"{env_ns}/Objects/{t}")
                else:
                    logger.warning(
                        "ContactSensor: no prim path for target '%s'", t,
                    )
                    skip = True
                    break
            if skip:
                continue

            cfg = ContactSensorCfg(
                prim_path=prim_path,
                update_period=0.0,
                filter_prim_paths_expr=filter_paths,
            )
            try:
                sensor = ContactSensor(cfg=cfg)
                self._sensors[sensor_id] = sensor
                self._pair_map[sensor_id] = {
                    t: i for i, t in enumerate(targets)
                }
            except Exception as exc:
                logger.warning(
                    "Failed to create ContactSensor for %s -> %s: %s",
                    sensor_id,
                    targets,
                    exc,
                )

        self._available = bool(self._sensors)

    @property
    def available(self) -> bool:
        return self._available

    def update(self, dt: float) -> None:
        """Update all sensor buffers. Must be called after each sim.step().

        ContactSensor is a standalone sensor not registered with InteractiveScene,
        so it does NOT auto-update. Without this call, sensor.data will be stale.
        """
        if not self._available:
            return
        for sensor in self._sensors.values():
            try:
                sensor.update(dt, force_recompute=True)
            except Exception:
                pass

    def read(self) -> dict[str, dict[str, list[float]]]:
        """Read per-pair contact forces from all sensors.

        Returns {sensor_id: {target_id: [fx, fy, fz]}}.
        Forces are normal contact forces in world frame (Newtons).
        """
        result: dict[str, dict[str, list[float]]] = {}
        if not self._available:
            return result

        for sensor_id, sensor in self._sensors.items():
            try:
                force_matrix = sensor.data.force_matrix_w  # (N, B, M, 3)
            except Exception:
                result[sensor_id] = {}
                continue

            target_map = self._pair_map[sensor_id]
            forces: dict[str, list[float]] = {}
            for target_id, idx in target_map.items():
                try:
                    if force_matrix.shape[2] > idx:
                        f = force_matrix[0, 0, idx, :].cpu().tolist()
                        forces[target_id] = f
                    else:
                        forces[target_id] = [0.0, 0.0, 0.0]
                except Exception:
                    forces[target_id] = [0.0, 0.0, 0.0]
            result[sensor_id] = forces

        return result

    def close(self) -> None:
        """Release sensor references after an episode or scene rebuild."""
        for sensor_id, sensor in list(self._sensors.items()):
            try:
                destroy = getattr(sensor, "destroy", None)
                if callable(destroy):
                    destroy()
                    continue
                reset = getattr(sensor, "reset", None)
                if callable(reset):
                    reset()
            except Exception as exc:
                logger.warning("ContactSensor cleanup failed for %s: %s", sensor_id, exc)
        self._sensors.clear()
        self._pair_map.clear()
        self._available = False


def install_contact_reader_for_runtime(
    collector: Any | None,
    runtime: dict[str, Any],
    *,
    reader_cls: type | None = None,
    log_fn: Any = print,
) -> bool:
    """Install a fresh ContactSensorReader for the current scene runtime.

    Contact sensors bind to concrete prim paths, so callers must recreate the
    reader after a scene rebuild/resample instead of reusing the old one.
    """
    if collector is None:
        return False

    contact_pairs = runtime.get("contact_pairs", [])
    if not contact_pairs:
        collector.set_contact_reader(None)
        return False

    cls = reader_cls or ContactSensorReader
    try:
        object_prim_paths = runtime.get("object_prim_paths", {})
        contact_reader = cls(contact_pairs, object_prim_paths=object_prim_paths)
        if contact_reader.available:
            collector.set_contact_reader(contact_reader)
            log_fn(f"[contact] ContactSensorReader initialized for {len(contact_pairs)} pair(s)")
            return True
        collector.set_contact_reader(None)
        log_fn("[contact] ContactSensorReader created but no sensors available")
    except Exception as exc:
        collector.set_contact_reader(None)
        log_fn(f"[WARN] Failed to create ContactSensorReader: {exc}")
    return False
