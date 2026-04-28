"""
ASV Navigation System - Mission Controller (v2)
================================================
Integrates all three architectural fixes:

  Fix 1 (Heading)    — HeadingEstimator with IMU priority + EMA filter
  Fix 2 (Sensor I/O) — SensorHub decouples sensor threads from control loop
  Fix 3 (Recovery)   — Forward-only, loop-safe PathRecovery

The control loop is now a pure CONSUMER. It never blocks on I/O.
All sensor data arrives pre-filtered via sensor_hub.snapshot().
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
from sensor_hub import SensorHub, SensorSnapshot
from heading_estimator import HeadingEstimator
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

    Architecture (v2):
      ┌──────────────────────────────────────────────────────────┐
      │                    MissionController                      │
      │                                                          │
      │  SensorHub (producer threads)   Control Loop (consumer) │
      │  ┌──────────┐ ┌──────────┐      snapshot() every tick   │
      │  │ GPS      │ │ Heading  │   →  navigator.update()       │
      │  │ producer │ │ producer │   →  obstacle_handler         │
      │  └──────────┘ └──────────┘   →  path_recovery            │
      │         ↓ RLock ↓                    ↓                   │
      │      SensorSnapshot            hw.set_motor_speed()      │
      └──────────────────────────────────────────────────────────┘
    """

    LOOP_HZ = 10

    def __init__(self,
                 boundary_coords,
                 start_lat:   float,
                 start_lon:   float,
                 hardware     = None,
                 sweep_width: float = None,
                 sweep_angle: float = 0.0,
                 resume:      bool  = True):

        self.hw        = hardware
        self.start_lat = start_lat
        self.start_lon = start_lon
        self._running  = False

        # ── Map ───────────────────────────────────────────────────────────────
        log.info("Building map from %d boundary vertices.", len(boundary_coords))
        self.map_handler = MapHandler.from_gps_polygon(boundary_coords)
        log.info("%s", self.map_handler)

        # ── Waypoints ─────────────────────────────────────────────────────────
        start_wp_index = 0
        if resume and Path(config.WAYPOINT_FILE).exists():
            log.info("Loading existing waypoints from %s", config.WAYPOINT_FILE)
            self.waypoints = CoveragePlanner.load(config.WAYPOINT_FILE,
                                                  self.map_handler)
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

        # ── Fix 1: Heading estimator (IMU-primary, GPS-fallback, EMA-filtered)
        self._heading_est = HeadingEstimator()

        # ── Fix 2: SensorHub (producer-consumer, non-blocking I/O) ───────────
        self.sensor_hub = SensorHub(
            hardware          = self.hw,
            heading_estimator = self._heading_est,
        )

        # ── Sub-systems ───────────────────────────────────────────────────────
        # Navigator: hardware=None because motor commands come from
        # the control loop's single hw.set_motor_speed() call only.
        self.navigator     = Navigator(self.waypoints,
                                       start_index=start_wp_index,
                                       hardware=None)
        self.obstacle_hdlr = ObstacleHandler(hardware=None)

        # Fix 3: improved recovery
        self.path_recovery = PathRecovery(self.map_handler, self.waypoints)

        # Track index at time of avoidance start (for coverage loss logging)
        self._avoidance_start_idx: int = 0

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT,  self._on_signal)

    # ── Public API ────────────────────────────────────────────────────────────

    def inject_obstacle_signal(self, signal_str: str) -> None:
        """Thread-safe. Can be called from any thread (sensor callback, etc.)."""
        self.sensor_hub.inject_obstacle_signal(signal_str)

    def run(self) -> None:
        """Block until mission is complete or interrupted."""
        self._running = True

        # Start producer threads BEFORE the control loop
        self.sensor_hub.start()

        # Wait for first GPS fix (up to 30 s)
        self._wait_for_gps_fix()

        self.navigator.start()
        dt = 1.0 / self.LOOP_HZ
        log.info("Mission started. Control loop @ %d Hz.", self.LOOP_HZ)

        while self._running and not self.navigator.is_complete:
            loop_start = time.monotonic()

            # ── Step 1: Get snapshot (non-blocking, microseconds) ─────────────
            snap = self.sensor_hub.snapshot()

            # ── Step 2: GPS validity gate ─────────────────────────────────────
            if not snap.gps_valid:
                log.warning("GPS not valid (age=%.1f s) — holding position.",
                            snap.gps_age_s)
                time.sleep(dt)
                continue

            lat, lon = snap.lat, snap.lon

            # ── Step 3: Heading (already filtered by HeadingEstimator) ────────
            heading = snap.heading_deg

            # Log heading reliability drop
            if not snap.heading_reliable:
                log.debug("Heading unreliable (src=%s) — using held value %.1f°",
                          snap.heading_source, heading)

            # ── Step 4: Inject obstacle signal from snapshot ──────────────────
            if snap.obstacle_signal != "NONE":
                self.obstacle_hdlr.receive_signal(snap.obstacle_signal)
                # Record where in the mission avoidance started
                self._avoidance_start_idx = self.navigator.current_idx

            # ── Step 5: Navigator computes motor commands ──────────────────────
            nav_left, nav_right = self.navigator.update(lat, lon, heading)

            # ── Step 6: Obstacle handler may override ─────────────────────────
            left, right, avoiding = self.obstacle_hdlr.process(nav_left, nav_right)

            # ── Step 7: Recovery after avoidance (Fix 3) ──────────────────────
            if self.obstacle_hdlr.avoidance_complete:
                self.obstacle_hdlr.avoidance_complete = False
                self.sensor_hub.clear_obstacle_signal()

                prev_idx   = self.navigator.current_idx
                resume_idx = self.path_recovery.find_best_waypoint(
                    lat, lon, self.navigator.current_idx
                )
                self.navigator.skip_to(resume_idx)
                self.navigator.state = NavState.NAVIGATING

                loss_m = self.path_recovery.estimate_coverage_loss(
                    prev_idx, resume_idx
                )
                log.info("Recovery complete: WP %d → %d (%.1f m coverage skipped).",
                         prev_idx, resume_idx, loss_m)

            # ── Step 8: Send motor commands (single authority) ─────────────────
            if self.hw:
                self.hw.set_motor_speed(left, right)

            # ── Step 9: Persist state ─────────────────────────────────────────
            self._save_state()

            # ── Step 10: Log status ───────────────────────────────────────────
            if int(loop_start) % 5 == 0:
                log.info("%s | %s | %s",
                         self.navigator.status_line(),
                         self.obstacle_hdlr.status_line(),
                         self.sensor_hub.status_line())

            # ── Timing: sleep remainder of loop period ────────────────────────
            elapsed = time.monotonic() - loop_start
            overshoot = elapsed - dt
            if overshoot > 0.005:   # warn if >5 ms over budget
                log.debug("Loop overrun by %.1f ms.", overshoot * 1000)
            time.sleep(max(0.0, dt - elapsed))

        # ── Mission complete or interrupted ────────────────────────────────────
        self.sensor_hub.stop()
        if self.hw:
            self.hw.stop()
        log.info("Mission finished. Progress: %.1f%%",
                 self.navigator.progress * 100)

    # ── GPS fix wait ──────────────────────────────────────────────────────────

    def _wait_for_gps_fix(self, timeout_s: float = 30.0) -> None:
        """
        Block until the first valid GPS fix arrives, or timeout.
        Producer threads are already running, so we just poll the snapshot.
        """
        log.info("Waiting for GPS fix (timeout %.0f s)…", timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snap = self.sensor_hub.snapshot()
            if snap.gps_valid:
                log.info("GPS fix acquired: (%.6f, %.6f) fix=%d",
                         snap.lat, snap.lon, snap.gps_fix)
                return
            time.sleep(0.5)
        log.warning("GPS fix not acquired within %.0f s — proceeding anyway.", timeout_s)

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

    # ── Signal handler ────────────────────────────────────────────────────────

    def _on_signal(self, signum, frame):
        log.info("Shutdown signal received — stopping gracefully.")
        self._running = False
        self.sensor_hub.stop()
        if self.hw:
            self.hw.stop()
