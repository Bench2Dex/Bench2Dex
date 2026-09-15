"""Per-step gravity + Coriolis feed-forward compensation for IsaacLab articulations.

Mirrors RoboTwin's ``entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)``
+ ``entity.set_qf(qf)`` so high-K PD controllers don't have to fight gravity with a
steady-state position error.

Usage (in a robot spawner)::

    robot = Articulation(cfg=robot_cfg)
    compensator = GravityCompensator(robot)

    return {
        "interactive_objects": {"global_robot": robot},
        "pre_step_hooks": [compensator.apply],
        ...
    }

The ``pre_step_hooks`` entries are invoked once per physics step in ``main.py``
just before ``write_articulation_targets`` so that the feed-forward effort buffer
gets flushed to PhysX together with the position targets.

For an ``ImplicitActuatorCfg`` the resulting joint torque seen by PhysX is

    K_p * (q_target - q) + K_d * (qd_target - qd) + applied_effort

so adding ``g(q) + C(q, qd) qd`` to ``applied_effort`` cancels the static droop.
"""

from __future__ import annotations

from typing import Callable


class GravityCompensator:
    """Apply gravity + Coriolis compensation as feed-forward effort each step.

    Args:
        articulation: ``isaaclab.assets.Articulation`` instance. Must already be
            spawned (i.e. ``root_physx_view`` available).
        gravity: include gravity compensation. Default True.
        coriolis: include Coriolis/centrifugal compensation. Default True.
        scale: optional scalar applied to the feed-forward effort. Use < 1.0 to
            slightly under-compensate (helps stability if PhysX inertia is noisy).
            Default 1.0.
    """

    def __init__(
        self,
        articulation,
        *,
        gravity: bool = True,
        coriolis: bool = True,
        scale: float = 1.0,
    ) -> None:
        self._art = articulation
        self._gravity = gravity
        self._coriolis = coriolis
        self._scale = float(scale)

        # Lazily resolved on first apply(): the IsaacLab joint buffer width
        # (excludes the floating-base 6 DOFs if any). At construction time the
        # articulation may not have been initialized yet (sim.reset() hasn't
        # run), so ``articulation.data`` is unavailable.
        self._joint_dim: int | None = None

    # ------------------------------------------------------------------
    # Public entry point used as a pre-step hook.
    # ------------------------------------------------------------------
    def apply(self) -> None:
        view = self._art.root_physx_view
        if view is None:
            # Articulation not yet initialized; skip silently.
            return

        if self._joint_dim is None:
            try:
                self._joint_dim = int(self._art.data.joint_pos.shape[-1])
            except (AttributeError, RuntimeError):
                # Still not ready; try again next step.
                return

        ff = None

        if self._gravity:
            try:
                g = view.get_gravity_compensation_forces()
            except AttributeError:
                # Older Isaac versions only expose the deprecated name.
                g = view.get_generalized_gravity_forces()
            ff = g.clone() if ff is None else ff + g

        if self._coriolis:
            try:
                c = view.get_coriolis_and_centrifugal_compensation_forces()
            except AttributeError:
                c = view.get_coriolis_and_centrifugal_forces()
            ff = c.clone() if ff is None else ff + c

        if ff is None:
            return

        # For floating-base articulations the compensation tensor includes 6
        # extra columns for the root pose; trim to joint DOFs.
        if ff.shape[-1] != self._joint_dim:
            ff = ff[:, : self._joint_dim]

        if self._scale != 1.0:
            ff = ff * self._scale

        # Feed-forward effort -- summed with K*(q_t - q) + D*(qd_t - qd) by the
        # ImplicitActuator inside PhysX.
        self._art.set_joint_effort_target(ff)


def make_hook(
    articulation,
    *,
    gravity: bool = True,
    coriolis: bool = True,
    scale: float = 1.0,
) -> Callable[[], None]:
    """Convenience wrapper that returns the bound ``apply`` method."""
    return GravityCompensator(
        articulation, gravity=gravity, coriolis=coriolis, scale=scale
    ).apply
