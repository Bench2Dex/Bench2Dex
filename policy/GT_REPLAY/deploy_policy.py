# SPDX-License-Identifier: Apache-2.0
"""Ground-truth replay "policy" -- drives Isaac with a recorded demonstration.

Why this exists
---------------
Every GR00T XE rollout so far has been 0/N online (0/29 pretrain, 0/45 ft on
task 08, 0/31 on task 34) while the same network beats GR00T n15 *offline*
(arm error 0.046 vs 0.113).  So the open question is whether the failure lives
in the model or in the code that turns a policy output into motion.

This module takes the model out of the loop and changes nothing else.  It hands
Isaac the joint commands recorded from the successful demonstration of the very
same episode -- and "same episode" is guaranteed by the harness itself:
``--anchor-dir`` applies inference episode N's scene randomization from
``_anchor_idx = (episode_idx - 1) % len(_anchor_samples)`` (run_policy.py), and
the same list is used here to pick the demonstration.

  succeeds -> the sim plus the client plumbing can execute this task, so a
              0/N is the model's fault (or the ee->ik decode's), not the
              harness's.
  fails    -> the execution side itself is broken and no policy could pass.

Conventions that must stay explicit
-----------------------------------
* ``action/commanded`` is stored in Isaac's **articulation** order, not the
  canonical URDF order: ``action/action_names`` interleaves the arms
  (``shoulder_pan, shoulder_lift, L_arm_shoulder_pan, elbow, ...``).  That is
  asserted at load time.  Because of this the launcher runs with the client's
  XE joint-order permutation switched *off* -- see ``zyd/gt_replay_run_eval.sh``.
  Nothing is permuted on the way to the sim; every code path between this
  module and ``robot_art`` is the same one the real eval uses.
* The episode index comes from the seed the client sends with ``reset``
  (``episode_seed = base_seed + episode_index - 1``).  There is no silent
  fallback: if the seed or the base seed is missing this raises.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# episode -> demonstration file, mirroring run_policy.py's --anchor-dir logic
# ---------------------------------------------------------------------------

def _select_anchor_files(anchor_dir: str) -> list[str]:
    """Reproduce run_policy.py's anchor list *exactly*.

    ``--anchor-dir`` loads "main 0-24 + _1 25-49" (25 unique demonstrations and
    25 disjoint backgrounds for each set) and falls back to all main episodes
    for legacy datasets without ``_1`` variants.  Reading the same list in the
    same order is what makes inference episode N and demonstration N the same
    scene; if this ever drifts from run_policy.py the replay is meaningless.
    """
    if not os.path.isdir(anchor_dir):
        raise FileNotFoundError(f"GT replay anchor dir not found: {anchor_dir}")
    all_files = sorted(
        p for p in glob.glob(os.path.join(anchor_dir, "episode_*.hdf5"))
        if "_replay." not in os.path.basename(p)
    )
    main: list[tuple[int, str]] = []
    var: list[tuple[int, str]] = []
    for path in all_files:
        base = os.path.basename(path)
        m = re.match(r"episode_(\d+)(?:_1)?\.hdf5", base)
        if not m:
            continue
        num = int(m.group(1))
        (var if "_1." in base else main).append((num, path))
    main.sort(key=lambda x: x[0])
    var.sort(key=lambda x: x[0])
    selected = [p for num, p in main if num < 25] + [p for num, p in var if num >= 25]
    if not selected:
        selected = [p for num, p in main]
    if not selected:
        raise ValueError(f"--anchor-dir has no episode_*.hdf5 files: {anchor_dir}")
    return selected


# ---------------------------------------------------------------------------
# the policy
# ---------------------------------------------------------------------------

def _robot_manifest(robot_key: str) -> dict[str, Any]:
    """Load this robot's entry from robots/active_dof_maps.yml."""
    import yaml

    path = Path(__file__).resolve().parents[2] / "robots" / "active_dof_maps.yml"
    if not path.is_file():
        raise FileNotFoundError(f"active_dof_maps.yml not found at {path}")
    with open(path) as fh:
        doc = yaml.safe_load(fh)
    # The file nests robot entries under a top-level key; find the one that owns
    # this robot name rather than hardcoding the section name.
    found: dict[str, Any] | None = None
    for _section, entries in doc.items():
        if isinstance(entries, dict) and robot_key in entries:
            found = entries[robot_key]
            break
    if found is None:
        raise KeyError(f"{robot_key} not in {path}; keys: {list(doc.keys())}")
    return found


class GTReplayPolicy:
    """Returns recorded joint commands instead of model output.

    ``get_action`` is stateless from the session's point of view: everything it
    needs (which episode) arrives through :meth:`set_episode`, which the session
    calls from ``reset``.
    """

    # The client queues whatever we return and pops one row per query
    # (script/policy_rpc.py: ``RemotePolicyClient.returns_action_chunks = True``),
    # so returning the whole padded trajectory in one chunk is enough to drive
    # the entire episode without a second query.
    HOLD_ROWS = 256

    def __init__(self, usr_args: dict[str, Any]) -> None:
        self.robot_key = usr_args.get("robot_key")
        if not self.robot_key:
            raise ValueError("GT_REPLAY needs --robot_key")

        anchor_dir = usr_args.get("gt_replay_dir")
        if not anchor_dir:
            raise ValueError(
                "GT_REPLAY needs --gt_replay_dir pointing at the task's "
                "replay-generalization directory (it is the same directory the "
                "eval passes as --anchor-dir)."
            )
        self.anchor_dir = str(Path(str(anchor_dir)).expanduser())
        self.anchor_files = _select_anchor_files(self.anchor_dir)

        self.base_seed = usr_args.get("seed")
        if self.base_seed is None:
            raise ValueError(
                "GT_REPLAY needs --seed (the eval's base seed) to turn the "
                "episode reset seed back into an episode index."
            )
        self.base_seed = int(self.base_seed)

        self.episode_steps = int(usr_args.get("episode_steps") or 0)

        self._episode_idx: int | None = None
        self._anchor_idx: int | None = None
        self._demo_path: str | None = None
        self._served = False

        print(
            f"[GT_REPLAY] robot={self.robot_key} base_seed={self.base_seed} "
            f"anchors={len(self.anchor_files)} from {self.anchor_dir}",
            flush=True,
        )

    # -- episode selection -------------------------------------------------

    def set_episode(self, seed: int | None) -> None:
        if seed is None:
            raise ValueError(
                "GT_REPLAY got a reset with no seed; it cannot tell which "
                "demonstration belongs to this episode. (No silent fallback: a "
                "wrong demo would be reported as a failed replay.)"
            )
        episode_idx = int(seed) - self.base_seed + 1
        if episode_idx < 1:
            raise ValueError(
                f"reset seed {seed} is below base_seed {self.base_seed}: "
                f"computed episode_idx={episode_idx}"
            )
        anchor_idx = (episode_idx - 1) % len(self.anchor_files)
        self._episode_idx = episode_idx
        self._anchor_idx = anchor_idx
        self._demo_path = self.anchor_files[anchor_idx]
        self._served = False
        print(
            f"[GT_REPLAY] episode_idx={episode_idx} seed={seed} -> "
            f"anchor_idx={anchor_idx} demo={os.path.basename(self._demo_path)}",
            flush=True,
        )

    # -- action ------------------------------------------------------------

    def _load_demo(self) -> Any:
        import h5py
        import numpy as np

        path = self._demo_path
        if path is None:
            raise RuntimeError("get_action() called before reset() set the episode")
        with h5py.File(path, "r") as h:
            if "action/commanded" not in h:
                raise KeyError(f"{path} has no action/commanded; keys={list(h.keys())}")
            actions = np.asarray(h["action/commanded"][()], dtype=np.float32)
            names = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in h["action/action_names"][()]
            ]
            joint_names = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in h["robot/joint_names"][()]
            ]
            # Only present in the newer recordings; treated as all-valid when absent.
            action_valid = (
                np.asarray(h["action/action_valid"][()], dtype=bool)
                if "action/action_valid" in h else None
            )

        # The whole point of feeding joint targets straight through is that no
        # permutation is applied, so the stored order must be the sim's own.
        if names != joint_names:
            raise ValueError(
                f"{path}: action/action_names != robot/joint_names, so the stored "
                f"action order is not the articulation order and returning it "
                f"unpermuted would drive the wrong joints."
            )

        # A 52-DOF vector goes straight to the articulation (run_policy.py's
        # expand_to_full only fires for a 30-dim active-DOF action).  That is
        # only sound if the vector is this robot's full DOF in *runtime* order,
        # so check the demonstration against the robot manifest and say out loud
        # which order it turned out to be.  Note the manifest's
        # full_joint_names is the *canonical* order, which is a permutation of
        # the runtime order (canonical: both arms first, then both hands;
        # runtime: arm/hand interleaved) -- so the comparison is by set, and
        # the permutation being non-identity is the expected case.
        manifest = _robot_manifest(self.robot_key)
        full_dof = int(manifest["full_dof"])
        canonical = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in manifest["full_joint_names"]
        ]
        if len(joint_names) != full_dof:
            raise ValueError(
                f"{path}: demonstration has {len(joint_names)} joints but "
                f"{self.robot_key} has full_dof={full_dof}"
            )
        if set(joint_names) != set(canonical):
            missing = sorted(set(canonical) - set(joint_names))
            extra = sorted(set(joint_names) - set(canonical))
            raise ValueError(
                f"{path}: joint names do not match {self.robot_key}. "
                f"missing={missing} extra={extra}"
            )
        order = "canonical" if joint_names == canonical else "runtime (permuted vs canonical)"

        # Frame 0 of every recorded demonstration is an all-NaN command with
        # action_valid[0] == False (the settling frame -- there is no expert
        # action yet).  Feeding it to the sim would poison the articulation with
        # NaN on the very first step and the replay would fail for a reason that
        # has nothing to do with the pipeline being tested, so invalid rows are
        # repaired by holding the nearest valid command.
        valid = np.isfinite(actions).all(axis=1)
        if action_valid is not None:
            valid = valid & action_valid
        if not valid.any():
            raise ValueError(f"{path}: no valid action rows")
        repaired = int((~valid).sum())
        if repaired:
            index = np.where(valid)[0]
            first = int(index[0])
            actions = actions.copy()
            actions[:first] = actions[first]
            last = first
            for t in range(first + 1, len(actions)):
                if valid[t]:
                    last = t
                else:
                    actions[t] = actions[last]
        print(
            f"[GT_REPLAY] {os.path.basename(path)}: {actions.shape} "
            f"(full_dof={full_dof}, order={order}), "
            f"{names[0]}..{names[-1]}; unpermuted, applied directly; "
            f"{repaired} invalid row(s) held at the nearest valid command",
            flush=True,
        )
        return actions

    def get_action(self, obs: Any) -> Any:
        import numpy as np

        if self._served:
            # Should not happen: the first chunk already covers the whole
            # episode.  Holding the last commanded pose is the honest thing to
            # do, and it is worth a line in the log.
            print(
                "[GT_REPLAY] WARNING: queried again after the demo was fully "
                "served; holding the last pose.",
                flush=True,
            )
            return np.tile(self._last_row, (self.HOLD_ROWS, 1))

        actions = self._load_demo()
        self._last_row = actions[-1:].copy()
        self._served = True

        # The recorded tail is a return-to-home segment, and it is truncated in
        # the recording itself: the arm joints are back at their initial values
        # by the last frame while the max deviation over all 52 joints is still
        # ~0.11 rad (hand joints).  Training cuts this tail off with
        # --hdf5-truncate-at-homing, but here the full recording is served: the
        # cut point is not recoverable from the file, and dropping the recorded
        # end state would be an arbitrary edit to the very ground truth this
        # control is supposed to be testing.
        whole = actions
        if self.episode_steps > len(actions):
            pad = np.tile(self._last_row, (self.episode_steps - len(actions), 1))
            whole = np.concatenate([actions, pad], axis=0)
        print(
            f"[GT_REPLAY] serving {whole.shape} ({len(actions)} demo steps"
            + (f" + {len(whole) - len(actions)} hold" if len(whole) > len(actions) else "")
            + "), one row consumed per query",
            flush=True,
        )
        return whole


# ---------------------------------------------------------------------------
# module entry points the harness looks for
# ---------------------------------------------------------------------------

def get_model(usr_args: dict[str, Any]) -> GTReplayPolicy:
    return GTReplayPolicy(usr_args)


def encode_obs(observation: dict[str, Any]) -> dict[str, Any]:
    """Passthrough: the replay never looks at the observation.

    The real XE module reshapes the observation into its 64-dim state here; the
    replay deliberately keeps the observation out of the loop so that a state
    bug cannot be mistaken for an action bug.  Cross-checking the observation
    side is a separate experiment.
    """
    return dict(observation)


def create_remote_session(usr_args: dict[str, Any]):
    """Session that captures the per-episode seed the default session drops.

    ``DefaultRemotePolicySession.reset`` only forwards the seed to jax rng
    helpers; the episode index it implies never reaches the model.  Overriding
    it is the only reason this factory exists.
    """
    from script.policy_sessions import DefaultRemotePolicySession

    class GTReplaySession(DefaultRemotePolicySession):
        def reset(self, instruction: str | None = None, *, seed: int | None = None) -> None:
            self._model.set_episode(seed)
            return super().reset(instruction, seed=seed)

    return GTReplaySession(sys.modules[__name__], usr_args)


# ---------------------------------------------------------------------------
# offline self-check: prove the episode -> demonstration mapping before any GPU
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Show the GT_REPLAY episode mapping.")
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--seed", type=int, default=100000000)
    ap.add_argument("--num-episodes", type=int, default=10)
    args = ap.parse_args()

    files = _select_anchor_files(args.anchor_dir)
    print(f"{len(files)} anchor files in {args.anchor_dir}")
    for i in range(1, args.num_episodes + 1):
        seed = args.seed + i - 1
        idx = (i - 1) % len(files)
        print(f"  episode {i:3d}  seed={seed}  ->  {os.path.basename(files[idx])}")
