"""
ASV Navigation System - Obstacle Handler
Reactive avoidance: intercepts motor commands during obstacle events.
"""

import time
import logging
import threading
from enum import Enum, auto
from typing import Optional, Tuple

import config

log = logging.getLogger(__name__)


class ObstacleSignal(str, Enum):
    NONE           = "NONE"
    OBSTACLE_LEFT  = "OBSTACLE_LEFT"
    OBSTACLE_RIGHT = "OBSTACLE_RIGHT"
    OBSTACLE_FRONT = "OBSTACLE_FRONT"
    CLEAR          = "CLEAR"


class AvoidPhase(Enum):
    INACTIVE = auto()
    TURNING  = auto()   # turning away from obstacle
    FORWARD  = auto()   # moving clear of obstacle
    DONE     = auto()


class ObstacleHandler:
    """
    Intercepts navigator output and applies reactive obstacle avoidance.

    External system calls `receive_signal(signal)` when sensors detect obstacles.
    The handler overrides motor commands until the path is clear, then signals
    the recovery module to rejoin the planned path.

    Thread-safe: signal can be pushed from a separate sensor thread.
    """

    def __init__(self, hardware=None):
        self.hw            = hardware
        self._lock         = threading.Lock()
        self._signal       = ObstacleSignal.NONE
        self._phase        = AvoidPhase.INACTIVE
        self._phase_start  = 0.0
        self._turn_dir     = 0  # +1 right, -1 left
        self.active        = False   # True while avoiding
        self.avoidance_complete = False  # set True when done

    # ── External signal interface ─────────────────────────────────────────────

    def receive_signal(self, signal: str) -> None:
        """Called by sensor system with obstacle signal string."""
        sig = ObstacleSignal(signal.upper())
        with self._lock:
            if sig == ObstacleSignal.CLEAR:
                if self._phase in (AvoidPhase.INACTIVE, AvoidPhase.DONE):
                    return  # no active avoidance to clear
                # Let current avoidance phase finish naturally
                return
            if sig != ObstacleSignal.NONE and not self.active:
                log.warning("Obstacle detected: %s", sig.value)
                self._signal = sig
                self._start_avoidance(sig)

    # ── Motor override ────────────────────────────────────────────────────────

    def process(self, nav_left: float, nav_right: float
                ) -> Tuple[float, float, bool]:
        """
        Called every control loop tick.

        Args:
            nav_left, nav_right: commands from Navigator.

        Returns:
            (left_cmd, right_cmd, overriding)
            overriding=True while avoidance is active.
        """
        with self._lock:
            if not self.active:
                self.avoidance_complete = False
                return nav_left, nav_right, False

            now = time.monotonic()
            elapsed = now - self._phase_start

            if self._phase == AvoidPhase.TURNING:
                left, right = self._turning_commands()
                if elapsed >= config.AVOIDANCE_TURN_TIME:
                    log.info("Avoidance: turn complete → moving forward")
                    self._phase = AvoidPhase.FORWARD
                    self._phase_start = now
                return left, right, True

            elif self._phase == AvoidPhase.FORWARD:
                left  = config.OBSTACLE_SPEED
                right = config.OBSTACLE_SPEED
                if elapsed >= config.AVOIDANCE_FWD_TIME:
                    log.info("Avoidance: forward complete → path recovery")
                    self._phase  = AvoidPhase.DONE
                    self.active  = False
                    self.avoidance_complete = True
                    self._signal = ObstacleSignal.NONE
                return left, right, True

            else:  # DONE / INACTIVE
                self.active = False
                return nav_left, nav_right, False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _start_avoidance(self, sig: ObstacleSignal) -> None:
        self.active       = True
        self.avoidance_complete = False
        self._phase       = AvoidPhase.TURNING
        self._phase_start = time.monotonic()

        if sig == ObstacleSignal.OBSTACLE_LEFT:
            self._turn_dir = +1   # turn right
        elif sig == ObstacleSignal.OBSTACLE_RIGHT:
            self._turn_dir = -1   # turn left
        else:  # FRONT — default turn right
            self._turn_dir = +1

        log.info("Avoidance started: signal=%s turn_dir=%s",
                 sig.value, "RIGHT" if self._turn_dir > 0 else "LEFT")

    def _turning_commands(self) -> Tuple[float, float]:
        spd = config.TURN_SPEED
        if self._turn_dir > 0:   # turn right: left fwd, right back
            return spd, -spd
        else:                    # turn left: right fwd, left back
            return -spd, spd

    def status_line(self) -> str:
        if not self.active:
            return "ObstacleHandler: clear"
        return f"ObstacleHandler: ACTIVE phase={self._phase.name} dir={'R' if self._turn_dir>0 else 'L'}"
