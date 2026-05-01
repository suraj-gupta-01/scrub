"""
ASV Navigation System - Mission Controller (v3)
================================================
Integrates all architectural fixes plus trash collection:

  Fix 1 (Heading)    — HeadingEstimator with IMU priority + EMA filter
  Fix 2 (Sensor I/O) — SensorHub decouples sensor threads from control loop
  Fix 3 (Recovery)   — Forward-only, loop-safe PathRecovery
  Fix 4 (Trash)      — Opportunistic trash collection via computer vision

The control loop is now a pure CONSUMER. It never blocks on I/O.
All sensor data arrives pre-filtered via sensor_hub.snapshot().

MOTOR COMMAND PIPELINE (v3)
----------------------------
  Navigator.update()           ← desired path commands
       ↓
  TrashHandler.process()       ← may override if collecting trash
       ↓
  ObstacleHandler.process()    ← may override if avoiding (SAFETY PRIORITY)
       ↓
  hw.set_motor_speed()         ← single authority

  Priority order: Obstacle > Trash > Navigation
  The trash handler is a pass-through when not collecting (zero overhead).
  The obstacle handler can abort any active trash collection.
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
from trash_handler import TrashHandler, TrashConfig, CollectionState
from cv import TrashDetector, MockTrashDetector, create_detector
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

    Architecture (v3):
      ┌──────────────────────────────────────────────────────────┐
      │                    MissionController                     │
      │                                                          │
      │  SensorHub (producer threads)   Control Loop (consumer)  │
      │  ┌──────────┐ ┌──────────┐      snapshot() every tick    │
      │  │ GPS      │ │ Heading  │   →  navigator.update()       │
      │  │ producer │ │ producer │   →  trash_handler.process()   │
      │  └──────────┘ └──────────┘   →  obstacle_handler         │
      │                              →  path_recovery            │
      │  TrashDetector (vision)          ↓                       │
      │  ┌──────────┐               hw.set_motor_speed()         │
      │  │ Camera + │                                            │
      │  │  YOLO    │→ inject_detection()                        │
      │  └──────────┘                                            │
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
                 resume:      bool  = True,
                 enable_trash: bool = None,
                 simulate:    bool  = False,
                 model_path:  str   = None):

        self.hw        = hardware
        self.start_lat = start_lat
        self.start_lon = start_lon
        self._running  = False
        self._simulate = simulate

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

        # ── Add Return-to-Home Waypoint ───────────────────────────────────────
        # Append the ground station as the final mission waypoint.
        gs_x, gs_y = self.map_handler.to_xy(self.start_lat, self.start_lon)
        home_wp = Waypoint(
            index=len(self.waypoints),
            x=gs_x,
            y=gs_y,
            lat=self.start_lat,
            lon=self.start_lon,
            is_turn=True
        )
        self.waypoints.append(home_wp)
        log.info("Added return-to-home waypoint at (%.6f, %.6f). Total waypoints: %d",
                 self.start_lat, self.start_lon, len(self.waypoints))

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

        # ── Fix 4: Trash collection ───────────────────────────────────────────
        self._trash_enabled = (
            enable_trash if enable_trash is not None
            else config.TRASH_DETECTION_ENABLED
        )

        if self._trash_enabled:
            trash_cfg = TrashConfig(
                min_confidence         = config.TRASH_MIN_CONFIDENCE,
                detection_radius_m     = config.TRASH_DETECTION_RADIUS_M,
                min_detection_m        = config.TRASH_MIN_DETECTION_M,
                max_deviation_m        = config.TRASH_MAX_DEVIATION_M,
                max_collection_time_s  = config.TRASH_COLLECTION_TIMEOUT_S,
                max_deviations_per_lane= config.TRASH_MAX_PER_LANE,
                collection_radius_m    = config.TRASH_COLLECTION_RADIUS_M,
                collection_dwell_s     = config.TRASH_DWELL_S,
                collection_approach_speed = config.TRASH_APPROACH_SPEED,
                cooldown_radius_m      = config.TRASH_COOLDOWN_RADIUS_M,
                cooldown_s             = config.TRASH_COOLDOWN_S,
            )
            self.trash_handler = TrashHandler(
                map_handler   = self.map_handler,
                waypoints     = self.waypoints,
                path_recovery = self.path_recovery,
                cfg           = trash_cfg,
            )

            # Vision / detector — initialized but not started until run()
            self._detector = create_detector(
                simulate       = simulate,
                trash_handler  = self.trash_handler,
                get_boat_state = self._get_boat_state,
                model_path     = model_path or config.VISION_MODEL_PATH,
            )
            log.info("Trash collection ENABLED (detector: %s)",
                     type(self._detector).__name__)
        else:
            self.trash_handler = None
            self._detector     = None
            log.info("Trash collection DISABLED.")

        # Track index at time of avoidance start (for coverage loss logging)
        self._avoidance_start_idx: int = 0

        # Shared boat state for the vision detector callback
        self._boat_lat: Optional[float] = None
        self._boat_lon: Optional[float] = None
        self._boat_hdg: float = 0.0
        self._boat_lock = threading.Lock()

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

        # Start vision detector if enabled
        if self._detector is not None:
            try:
                self._detector.start()
            except Exception as exc:
                log.error("Failed to start trash detector: %s — "
                          "trash collection disabled for this mission.", exc)
                self._trash_enabled = False
                self._detector = None

        # Wait for first GPS fix (up to 30 s)
        self._wait_for_gps_fix()

        self.navigator.start()
        dt = 1.0 / self.LOOP_HZ
        log.info("Mission started. Control loop @ %d Hz. Trash=%s",
                 self.LOOP_HZ, "ON" if self._trash_enabled else "OFF")

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

            # Update shared boat state for vision detector
            with self._boat_lock:
                self._boat_lat = lat
                self._boat_lon = lon

            # ── Step 3: Heading (already filtered by HeadingEstimator) ────────
            heading = snap.heading_deg

            with self._boat_lock:
                self._boat_hdg = heading

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

            # ── Step 5.5: Trash handler layer (may override nav commands) ─────
            collecting = False
            if self._trash_enabled and self.trash_handler is not None:
                nav_left, nav_right, collecting = self.trash_handler.process(
                    nav_left, nav_right,
                    lat, lon, heading,
                    obstacle_active=self.obstacle_hdlr.active,
                    current_wp_idx=self.navigator.current_idx,
                )

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

            # ── Step 7.5: Handle trash collection completion ──────────────────
            if (self._trash_enabled and self.trash_handler is not None
                    and self.trash_handler.collection_complete):
                resume_idx = self.trash_handler.return_waypoint_index
                if resume_idx is not None:
                    self.navigator.skip_to(resume_idx)
                    self.navigator.state = NavState.NAVIGATING
                    log.info("Trash collection done: resuming navigation at WP %d.",
                             resume_idx)

            # ── Step 8: Send motor commands (single authority) ─────────────────
            if self.hw:
                self.hw.set_motor_speed(left, right)

            # ── Step 9: Persist state ─────────────────────────────────────────
            self._save_state()

            # ── Step 10: Log status ───────────────────────────────────────────
            if int(loop_start) % 5 == 0:
                parts = [
                    self.navigator.status_line(),
                    self.obstacle_hdlr.status_line(),
                    self.sensor_hub.status_line(),
                ]
                if self._trash_enabled and self.trash_handler is not None:
                    parts.append(self.trash_handler.status_line())
                log.info(" | ".join(parts))

            # ── Timing: sleep remainder of loop period ────────────────────────
            elapsed = time.monotonic() - loop_start
            overshoot = elapsed - dt
            if overshoot > 0.005:   # warn if >5 ms over budget
                log.debug("Loop overrun by %.1f ms.", overshoot * 1000)
            time.sleep(max(0.0, dt - elapsed))

        # ── Mission complete or interrupted ────────────────────────────────────
        self.sensor_hub.stop()
        if self._detector is not None:
            self._detector.stop()
        if self.hw:
            self.hw.stop()

        # Log trash collection statistics
        if self._trash_enabled and self.trash_handler is not None:
            stats = self.trash_handler.stats()
            log.info("Trash collection stats: %s", stats)

        log.info("Mission finished. Progress: %.1f%%",
                 self.navigator.progress * 100)

    # ── Boat state callback for vision detector ───────────────────────────────

    def _get_boat_state(self):
        """
        Callable for the TrashDetector: returns (lat, lon, heading_deg).
        Thread-safe — called from the vision thread.
        """
        with self._boat_lock:
            return self._boat_lat, self._boat_lon, self._boat_hdg

    # ── GPS fix wait ──────────────────────────────────────────────────────────

    def _wait_for_gps_fix(self, timeout_s: float = 900.0) -> None:
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
        # Include trash stats if available
        if self._trash_enabled and self.trash_handler is not None:
            state["trash_stats"] = self.trash_handler.stats()

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
        if self._detector is not None:
            self._detector.stop()
        if self.hw:
            self.hw.stop()
