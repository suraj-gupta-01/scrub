"""
ASV Navigation System - Waypoint Navigator
Pure-pursuit + proportional heading control for waypoint following.
"""

import math
import time
import logging
from typing import List, Optional, Tuple
from enum import Enum, auto

from coverage_planner import Waypoint
from utils import haversine, bearing, angle_diff, dist2d, clamp
import config

log = logging.getLogger(__name__)


class NavState(Enum):
    IDLE      = auto()
    NAVIGATING = auto()
    REACHED   = auto()
    COMPLETE  = auto()


class Navigator:
    """
    Follows a list of Waypoints using proportional heading control.

    At each update step call:
        left_cmd, right_cmd = navigator.update(lat, lon, heading_deg)

    The returned values are normalised motor commands [-1.0, 1.0].
    """

    def __init__(self, waypoints: List[Waypoint],
                 start_index: int = 0,
                 hardware=None):
        """
        Args:
            waypoints:   Ordered list from CoveragePlanner.
            start_index: Resume from this waypoint (mission recovery).
            hardware:    Optional hardware interface (has set_motor_speed()).
        """
        self.waypoints    = waypoints
        self.current_idx  = start_index
        self.state        = NavState.IDLE if waypoints else NavState.COMPLETE
        self.hw           = hardware

        # Telemetry
        self.current_lat  = 0.0
        self.current_lon  = 0.0
        self.current_hdg  = 0.0  # degrees
        self.dist_to_wp   = 0.0
        self.desired_hdg  = 0.0
        self.heading_err  = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def active_waypoint(self) -> Optional[Waypoint]:
        if 0 <= self.current_idx < len(self.waypoints):
            return self.waypoints[self.current_idx]
        return None

    @property
    def is_complete(self) -> bool:
        return self.state == NavState.COMPLETE

    @property
    def progress(self) -> float:
        """Mission progress 0.0 → 1.0."""
        if not self.waypoints:
            return 1.0
        return self.current_idx / len(self.waypoints)

    def start(self) -> None:
        if self.state == NavState.IDLE:
            self.state = NavState.NAVIGATING
            log.info("Navigator started. %d waypoints.", len(self.waypoints))

    def update(self, lat: float, lon: float,
               heading_deg: float) -> Tuple[float, float]:
        """
        Compute motor commands for current GPS position and heading.

        Returns:
            (left_speed, right_speed) each normalised to [-1, 1].
        """
        self.current_lat = lat
        self.current_lon = lon
        self.current_hdg = heading_deg

        if self.state != NavState.NAVIGATING:
            return 0.0, 0.0

        wp = self.active_waypoint
        if wp is None:
            self.state = NavState.COMPLETE
            log.info("Mission complete — all waypoints visited.")
            return 0.0, 0.0

        # Distance and desired bearing to active waypoint
        self.dist_to_wp  = haversine(lat, lon, wp.lat, wp.lon)
        self.desired_hdg = bearing(lat, lon, wp.lat, wp.lon)
        self.heading_err = angle_diff(heading_deg, self.desired_hdg)

        # Waypoint reached?
        if self.dist_to_wp < config.WAYPOINT_RADIUS_M:
            log.info("Waypoint %d reached (%.1f m).", wp.index, self.dist_to_wp)
            self.current_idx += 1
            if self.current_idx >= len(self.waypoints):
                self.state = NavState.COMPLETE
                return 0.0, 0.0
            return self.update(lat, lon, heading_deg)  # recurse once

        # ── Proportional heading control ──────────────────────────────────────
        # Normalise error to [-1, 1]
        err_norm   = clamp(self.heading_err / 180.0, -1.0, 1.0)
        turn       = config.HEADING_KP * err_norm

        # Reduce speed on large heading errors (like a skid steer boat)
        fwd        = config.CRUISE_SPEED * (1.0 - 0.5 * abs(err_norm))

        left_cmd   = clamp(fwd - turn * config.MAX_TURN_RATE, -1.0, 1.0)
        right_cmd  = clamp(fwd + turn * config.MAX_TURN_RATE, -1.0, 1.0)

        if self.hw:
            self.hw.set_motor_speed(left_cmd, right_cmd)

        return left_cmd, right_cmd

    def skip_to(self, index: int) -> None:
        """Jump to a specific waypoint index (used by recovery module)."""
        self.current_idx = max(0, min(index, len(self.waypoints) - 1))
        log.info("Navigator skipped to waypoint %d.", self.current_idx)

    def status_line(self) -> str:
        wp = self.active_waypoint
        if wp is None:
            return f"Navigator: {self.state.name}"
        return (f"Nav [{self.state.name}] WP#{self.current_idx}/{len(self.waypoints)} "
                f"dist={self.dist_to_wp:.1f}m "
                f"hdg_err={self.heading_err:+.1f}°")
