from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EpisodeCommand(str, Enum):
    HOME = "home"           # key 1
    START = "start"         # key 2
    STOP = "stop"           # key 3
    HOME_REACHED = "home_reached"  # internal: joints reached home


class EpisodeSessionState(str, Enum):
    READY = "ready"
    HOMING = "homing"
    RECORDING = "recording"


class PendingAfterHome(str, Enum):
    NONE = "none"       # just idle at home
    START = "start"     # start recording
    STOP = "stop"       # stop recording + save


@dataclass(frozen=True)
class EpisodeSessionAction:
    go_home: bool = False
    start_recording: bool = False
    stop_recording: bool = False        # stop + save
    discard_recording: bool = False     # stop + discard (no save)
    suppress_teleop: bool = False
    message: str | None = None


class EpisodeSessionController:
    """Minimal state machine: 3 states + pending_after_home field.

    States: READY → HOMING → READY / RECORDING
            RECORDING → HOMING → READY (save or discard)

    Keys:
      1 (HOME)  → HOMING, pending=NONE.  If was RECORDING, discard episode.
      2 (START) → HOMING, pending=START.
      3 (STOP)  → HOMING, pending=STOP (save on arrival). If not RECORDING, just go home.
    """

    def __init__(self) -> None:
        self._state = EpisodeSessionState.READY
        self._pending = PendingAfterHome.NONE

    @property
    def state(self) -> EpisodeSessionState:
        return self._state

    @property
    def pending_after_home(self) -> PendingAfterHome:
        return self._pending

    def handle(self, command: EpisodeCommand) -> EpisodeSessionAction:
        if command is EpisodeCommand.HOME:
            return self._handle_home()
        if command is EpisodeCommand.START:
            return self._handle_start()
        if command is EpisodeCommand.STOP:
            return self._handle_stop()
        if command is EpisodeCommand.HOME_REACHED:
            return self._handle_home_reached()
        raise ValueError(f"Unknown command: {command!r}")

    # ── key handlers ────────────────────────────────────────────

    def _handle_home(self) -> EpisodeSessionAction:
        if self._state is EpisodeSessionState.HOMING:
            return EpisodeSessionAction(message="already_homing")

        discard = self._state is EpisodeSessionState.RECORDING
        self._state = EpisodeSessionState.HOMING
        self._pending = PendingAfterHome.NONE
        return EpisodeSessionAction(
            go_home=True,
            suppress_teleop=True,
            discard_recording=discard,
            message="discard_recording" if discard else None,
        )

    def _handle_start(self) -> EpisodeSessionAction:
        if self._state is EpisodeSessionState.HOMING:
            return EpisodeSessionAction(message="already_homing")
        if self._state is EpisodeSessionState.RECORDING:
            return EpisodeSessionAction(message="already_recording")

        self._state = EpisodeSessionState.HOMING
        self._pending = PendingAfterHome.START
        return EpisodeSessionAction(go_home=True, suppress_teleop=True)

    def _handle_stop(self) -> EpisodeSessionAction:
        if self._state is EpisodeSessionState.HOMING:
            return EpisodeSessionAction(message="already_homing")

        self._state = EpisodeSessionState.HOMING
        self._pending = PendingAfterHome.STOP
        return EpisodeSessionAction(go_home=True, suppress_teleop=True)

    def _handle_home_reached(self) -> EpisodeSessionAction:
        pending = self._pending
        self._pending = PendingAfterHome.NONE

        if pending is PendingAfterHome.START:
            self._state = EpisodeSessionState.RECORDING
            return EpisodeSessionAction(start_recording=True)
        if pending is PendingAfterHome.STOP:
            self._state = EpisodeSessionState.READY
            return EpisodeSessionAction(stop_recording=True)

        self._state = EpisodeSessionState.READY
        return EpisodeSessionAction()
