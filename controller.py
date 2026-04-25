"""
ASV Navigation System - Mission Controller
Main loop: ties all modules together, handles persistence and fault tolerance.
"""

import json
import time
import signal
import logging
import threading
from pathlib import Path
from typing import Optional

from map_handler import MapHandler
from coverage_planner import CoveragePlanner, Waypoint
from navigator import Navigator, NavState
from obstacle_handler import ObstacleHandler, ObstacleSignal
from recovery import PathRecovery
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Controller")


class MissionController:
    """
    Top-level mission controller.

    Responsibilities:
      • Load map and generate (or reload) coverage waypoints.
      • Run the main control loop at ~10 Hz.
      • Integrate Navigator + ObstacleHandler + PathRecovery.
      • Persist mission state so the boat can resume after a restart.
      • Gracefully stop on Ctrl-C or SIGTERM.
    """

    LOOP_HZ = 10   # control loop frequency

    def __init__(self, boundary_coords,
                 start_lat: float, start_lon: float,
                 hardware=None,
                 sweep_width: float = None,
                 sweep_angle: float = 0.0,
                 resume: bool = True):
        """
        Args:
            boundary_coords: List of (lat, lon) tuples defining the waterbody.
            start_lat/lon:   Ground station / launch position.
            hardware:        HardwareInterface instance.
            sweep_width:     Lane spacing in metres.
            sweep_angle:     Sweep direction angle (0 = E-W lanes).
            resume:          If True, try to resume from saved state.
        """
        self.hw           = hardware
        self.start_lat    = start_lat
        self.start_lon    = start_lon
        self._running     = False

        # ── Map ──────────────────────────────────────────────────────────────
        log.info("Building map from %d boundary vertices.", len(boundary_coords))
        self.map_handler  = MapHandler.from_gps_polygon(boundary_coords)
        log.info("%s", self.map_handler)

        # ── Waypoints ────────────────────────────────────────────────────────
        start_wp_index = 0
        if resume and Path(config.WAYPOINT_FILE).exists():
            log.info("Loading existing waypoints from %s", config.WAYPOINT_FILE)
            self.waypoints = CoveragePlanner.load(config.WAYPOINT_FILE, self.map_handler)
            start_wp_index = self._load_state()
        else:
            log.info("Generating coverage path…")
            planner = CoveragePlanner(
                self.map_handler,
                sweep_width=sweep_width,
                angle_deg=sweep_angle,
            )
            self.waypoints = planner.generate()
            planner.save()
            log.info(planner.summary())

        # ── Sub-systems ──────────────────────────────────────────────────────
        self.navigator       = Navigator(self.waypoints, start_index=start_wp_index,
                                         hardware=self.hw)
        self.obstacle_hdlr   = ObstacleHandler(hardware=self.hw)
        self.path_recovery   = PathRecovery(self.map_handler, self.waypoints)

        # Install SIGTERM handler for graceful shutdown
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT,  self._on_signal)

    # ── Public API ────────────────────────────────────────────────────────────

    def inject_obstacle_signal(self, signal_str: str) -> None:
        """Called by external sensor system."""
        self.obstacle_hdlr.receive_signal(signal_str)

    def run(self) -> None:
        """Block until mission is complete or interrupted."""
        self._running = True
        self.navigator.start()
        dt = 1.0 / self.LOOP_HZ
        log.info("Mission started. Control loop @ %d Hz.", self.LOOP_HZ)

        while self._running and not self.navigator.is_complete:
            loop_start = time.monotonic()

            # 1. Read sensors
            lat, lon = self._get_gps()
            heading  = self._get_heading()

            if lat is None or lon is None:
                log.warning("No GPS fix — waiting…")
                time.sleep(dt)
                continue

            # 2. Navigator computes desired motor commands
            nav_left, nav_right = self.navigator.update(lat, lon, heading)

            # 3. Obstacle handler may override
            left, right, avoiding = self.obstacle_hdlr.process(nav_left, nav_right)

            # 4. After avoidance completes, rejoin planned path
            if self.obstacle_hdlr.avoidance_complete:
                self.obstacle_hdlr.avoidance_complete = False
                resume_idx = self.path_recovery.find_best_waypoint(
                    lat, lon, self.navigator.current_idx
                )
                self.navigator.skip_to(resume_idx)
                self.navigator.state = NavState.NAVIGATING

            # 5. Send to hardware
            if self.hw and not avoiding:
                self.hw.set_motor_speed(left, right)

            # 6. Persist state periodically
            self._save_state()

            # 7. Log status
            if int(loop_start) % 5 == 0:  # every ~5 s
                log.info("%s | %s", self.navigator.status_line(),
                         self.obstacle_hdlr.status_line())

            # Sleep remainder of loop period
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, dt - elapsed))

        # Mission done
        if self.hw:
            self.hw.stop()
        log.info("Mission finished. Progress: %.1f%%",
                 self.navigator.progress * 100)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        state = {
            "current_waypoint_index": self.navigator.current_idx,
            "progress":               self.navigator.progress,
            "waypoint_count":         len(self.waypoints),
        }
        try:
            with open(config.STATE_FILE, "w") as f:
                json.dump(state, f)
        except OSError as e:
            log.error("Could not save state: %s", e)

    def _load_state(self) -> int:
        try:
            with open(config.STATE_FILE) as f:
                state = json.load(f)
            idx = int(state.get("current_waypoint_index", 0))
            log.info("Resuming from waypoint %d (%.1f%% complete).",
                     idx, state.get("progress", 0) * 100)
            return idx
        except (OSError, KeyError, ValueError):
            log.info("No saved state — starting from waypoint 0.")
            return 0

    # ── Sensor helpers ────────────────────────────────────────────────────────

    def _get_gps(self):
        if self.hw:
            return self.hw.get_gps()
        return self.start_lat, self.start_lon

    def _get_heading(self) -> float:
        if self.hw:
            h = self.hw.get_heading()
            return h if h is not None else 0.0
        return 0.0

    # ── Signal handler ────────────────────────────────────────────────────────

    def _on_signal(self, signum, frame):
        log.info("Shutdown signal received — stopping gracefully.")
        self._running = False
        if self.hw:
            self.hw.stop()
