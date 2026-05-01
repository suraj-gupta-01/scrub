"""
ASV Navigation System - Trash Collection Handler
=================================================
Opportunistic trash collection layer that sits between the Navigator
and the ObstacleHandler in the motor-command pipeline.

DESIGN PHILOSOPHY
-----------------
This module was designed after careful analysis of three alternative
approaches:

  Option A — Reduce sweep width (e.g. 1.5 m instead of 3 m)
  ─────────────────────────────────────────────────────────
  Pros:  Guaranteed coverage. Simple. No new failure modes.
  Cons:  2× path length, 2× time, 2× energy. Diminishing returns for
         sparse trash. Misses floating/moving trash anyway.

  Option B — Opportunistic deviation (THIS MODULE)
  ─────────────────────────────────────────────────
  Pros:  Efficient on sparse trash. Coverage remains broad. Collects
         trash missed by fixed sweeps (e.e. trash between lanes).
  Cons:  Adds deviation overhead. Risk of loops. Vision confidence
         dependency. New failure modes require careful design.

  Option C — Hybrid (reduced sweep + opportunistic)
  ─────────────────────────────────────────────────
  Pros:  Best coverage completeness.
  Cons:  Complex. Both failure modes present. Only justified for very
         dense, high-value trash scenarios.

VERDICT: Option B (this module) wins for the stated problem context
(sparse trash, large area, existing lawnmower coverage). The key
insight is that opportunistic collection COMPLEMENTS fixed coverage —
it does not replace it. The fixed sweep handles systematic coverage;
this module handles the in-between misses.

CONDITIONS UNDER WHICH OPPORTUNISTIC COLLECTION FAILS
------------------------------------------------------
1. Dense trash fields: constant deviations destroy coverage pattern.
   Mitigation: MAX_DEVIATIONS_PER_LANE limit + cooldown.

2. Low GPS accuracy: post-deviation path return may be inaccurate.
   Mitigation: PathRecovery (already existing) handles this.

3. False-positive detections: wasted diversions, coverage loss.
   Mitigation: confidence threshold + minimum detection duration.

4. Trash near boundary: deviation exits polygon.
   Mitigation: polygon boundary check before committing to deviation.

5. Obstacle during collection: position becomes uncertain.
   Mitigation: abort collection, use PathRecovery to rejoin path.

6. Same trash re-detected: infinite loop toward already-collected trash.
   Mitigation: cooldown registry with GPS-position-based deduplication.

MOTOR COMMAND PIPELINE
----------------------
(Each stage may override the previous. Obstacle always wins.)

  Navigator.update()          ← desired path commands
       ↓
  TrashHandler.process()      ← may override if collecting
       ↓
  ObstacleHandler.process()   ← may override if avoiding (SAFETY PRIORITY)
       ↓
  hw.set_motor_speed()        ← single authority

STATE MACHINE
-------------

  ┌───────────────────────────────────────────────────────────────┐
  │                    NORMAL_NAVIGATION                          │
  │         (pass-through: navigator commands unchanged)          │
  └─────────────────────────┬─────────────────────────────────────┘
                            │ trash detected within range
                            │ AND confidence ≥ threshold
                            │ AND not already collecting
                            │ AND obstacle handler not active
                            │ AND not in cooldown
                            │ AND trash inside polygon
                            ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                    DEVIATING_TO_TRASH                         │
  │  Steer toward trash GPS position using bearing/heading        │
  │  control (same P-controller as Navigator)                     │
  │  Timeout: MAX_COLLECTION_TIME_S                               │
  │  Abort if: obstacle detected, out of polygon, timeout         │
  └─────────────────────────┬─────────────────────────────────────┘
                            │ within COLLECTION_RADIUS_M
                            ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                    COLLECTING                                  │
  │  Stop motors for COLLECTION_DWELL_S seconds                   │
  │  (physical collector mechanism activates here)                │
  │  Log collection event                                         │
  └─────────────────────────┬─────────────────────────────────────┘
                            │ dwell complete
                            ▼
  ┌───────────────────────────────────────────────────────────────┐
  │                    RETURNING_TO_PATH                          │
  │  Call PathRecovery.find_best_waypoint()                       │
  │  Navigate back to nearest valid path point                    │
  │  When reached: notify controller to resume Navigator          │
  └─────────────────────────┬─────────────────────────────────────┘
                            │ back on path
                            ▼
                   NORMAL_NAVIGATION

  ABORT path (from any collecting state → NORMAL_NAVIGATION):
    • Obstacle detected while deviating/collecting
    • Deviation timeout exceeded
    • Trash exits polygon boundary
    • Max deviations per lane exceeded
"""

import math
import time
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict

from map_handler import MapHandler
from coverage_planner import Waypoint
from recovery import PathRecovery
from utils import haversine, bearing, angle_diff, clamp, dist2d
import config

log = logging.getLogger("TrashHandler")


# ═════════════════════════════════════════════════════════════════════════════
#  Configuration  (all tunable, separate from config.py to keep scope clear)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrashConfig:
    """
    All tuning parameters for the trash collection behaviour.
    Defaults are conservative — err on the side of not deviating.
    Adjust based on your vision system's accuracy and your
    collector's physical capabilities.
    """

    # ── Detection gates ───────────────────────────────────────────────────────
    # Minimum confidence from vision system to consider acting.
    # Range [0.0, 1.0]. Below this: always ignore.
    min_confidence: float = 0.65

    # Only divert if trash is within this radius of the boat (metres).
    # Beyond this, the deviation cost outweighs the benefit.
    # Typical: 1 × sweep_width (3 m default) — trash just outside the lane.
    detection_radius_m: float = 4.0

    # Minimum distance: ignore trash that is too close (already being collected
    # by hull/collector, or noise in vision system at very short range).
    min_detection_m: float = 0.5

    # ── Path deviation limits ─────────────────────────────────────────────────
    # Maximum distance the boat will travel from the collection trigger point.
    # If trash is not reached within this distance, abort.
    max_deviation_m: float = 8.0

    # Maximum time (seconds) allowed for a full collection attempt
    # (deviation + dwell + return starts). If exceeded: abort.
    max_collection_time_s: float = 30.0

    # Maximum number of trash collections per coverage lane segment.
    # Prevents dense trash fields from trapping the boat in a collection loop.
    max_deviations_per_lane: int = 3

    # ── Collection mechanics ──────────────────────────────────────────────────
    # Distance from trash GPS position at which we consider it "reached"
    # and activate the collector.
    collection_radius_m: float = 1.5

    # How long to hold position for physical collection (seconds).
    # Set based on your collector mechanism speed.
    collection_dwell_s: float = 3.0

    # Speed while navigating toward trash (normalised 0-1).
    # Slower than cruise: more precise approach, collector works better.
    collection_approach_speed: float = 0.4

    # ── Anti-loop / deduplication ─────────────────────────────────────────────
    # After attempting collection of a trash item (success or fail),
    # ignore any trash detected within this radius for this many seconds.
    cooldown_radius_m: float = 3.0
    cooldown_s: float = 60.0

    # ── Safety gates ──────────────────────────────────────────────────────────
    # Abort if the deviation point is outside the polygon boundary.
    require_inside_polygon: bool = True

    # Abort collection if obstacle handler becomes active.
    abort_on_obstacle: bool = True


# ═════════════════════════════════════════════════════════════════════════════
#  Data types
# ═════════════════════════════════════════════════════════════════════════════

class CollectionState(Enum):
    NORMAL_NAVIGATION  = auto()   # pass-through: not collecting
    DEVIATING_TO_TRASH = auto()   # steering toward detected trash
    COLLECTING         = auto()   # at trash position, dwell active
    RETURNING_TO_PATH  = auto()   # heading back to coverage path


@dataclass
class TrashDetection:
    """
    A single trash detection event from the vision system.

    The vision system should produce detections in GPS coordinates.
    If your camera gives pixel/range/bearing outputs, a converter
    (see TrashDetection.from_bearing_range()) builds the GPS position.
    """
    lat:        float           # GPS latitude of detected trash
    lon:        float           # GPS longitude of detected trash
    confidence: float           # Detection confidence [0.0, 1.0]
    timestamp:  float = field(default_factory=time.monotonic)
    label:      str   = "trash" # Optional: trash type label from classifier

    @classmethod
    def from_bearing_range(cls,
                            boat_lat: float, boat_lon: float,
                            bearing_deg: float,
                            range_m: float,
                            confidence: float,
                            label: str = "trash") -> "TrashDetection":
        """
        Construct a TrashDetection from a bearing + range measurement
        (e.g. from a radar, lidar, or bearing-only camera with depth estimate).

        Args:
            boat_lat, boat_lon: Current boat GPS position.
            bearing_deg:        True compass bearing to trash (degrees).
            range_m:            Distance to trash (metres).
            confidence:         Detection confidence [0.0, 1.0].
        """
        EARTH_R = 6_371_000.0
        rad     = math.radians(bearing_deg)
        dlat    = math.degrees(range_m * math.cos(rad) / EARTH_R)
        dlon    = math.degrees(
            range_m * math.sin(rad) /
            (EARTH_R * math.cos(math.radians(boat_lat)))
        )
        return cls(
            lat        = boat_lat + dlat,
            lon        = boat_lon + dlon,
            confidence = confidence,
            label      = label,
        )


@dataclass
class CollectionRecord:
    """Persistent record of a collection attempt (success or abort)."""
    lat:       float
    lon:       float
    timestamp: float
    success:   bool
    label:     str = "trash"
    reason:    str = ""    # "collected", "timeout", "obstacle", "out_of_polygon"


# ═════════════════════════════════════════════════════════════════════════════
#  Main TrashHandler class
# ═════════════════════════════════════════════════════════════════════════════

class TrashHandler:
    """
    Opportunistic trash collection behaviour layer.

    Sits between Navigator and ObstacleHandler in the motor pipeline.
    When not collecting, it is a perfect pass-through (zero overhead).
    When collecting, it overrides motor commands to steer toward trash,
    dwell for collection, then uses PathRecovery to rejoin the path.

    Thread-safe: vision detections can be injected from any thread
    (camera callback, ROS subscriber, etc.) via inject_detection().

    Integration (controller.py):
        # In __init__:
        self.trash_handler = TrashHandler(map_handler, waypoints, path_recovery, cfg=TrashConfig())

        # In run() loop, after navigator.update(), before obstacle_handler.process():
        nav_l, nav_r = self.navigator.update(lat, lon, heading)
        nav_l, nav_r, collecting = self.trash_handler.process(
            nav_l, nav_r, lat, lon, heading,
            obstacle_active=self.obstacle_hdlr.active,
            current_wp_idx=self.navigator.current_idx,
        )
        left, right, avoiding = self.obstacle_hdlr.process(nav_l, nav_r)
        # ... handle recovery, send motors as before
    """

    def __init__(self,
                 map_handler:   MapHandler,
                 waypoints:     List[Waypoint],
                 path_recovery: PathRecovery,
                 cfg:           TrashConfig = None):

        self.mh       = map_handler
        self.waypoints = waypoints
        self.recovery  = path_recovery
        self.cfg       = cfg or TrashConfig()

        # ── Internal state (guarded by lock for thread safety) ────────────────
        self._lock  = threading.Lock()
        self._state = CollectionState.NORMAL_NAVIGATION

        # Queued detections from vision system (producer thread)
        self._pending_detections: List[TrashDetection] = []

        # Active collection target
        self._target:        Optional[TrashDetection] = None
        self._deviation_origin_lat: float = 0.0
        self._deviation_origin_lon: float = 0.0
        self._deviation_start_time: float = 0.0
        self._wp_idx_at_deviation:  int   = 0   # for PathRecovery
        self._dwell_start_time:     float = 0.0
        self._deviations_this_lane: int   = 0
        self._last_lane_idx:        int   = -1

        # Return path target (set when leaving COLLECTING → RETURNING_TO_PATH)
        self._return_wp_idx: int = 0

        # Cooldown registry: list of (lat, lon, expiry_monotonic)
        self._cooldowns: List[Tuple[float, float, float]] = []

        # Persistent history for logging and analysis
        self.collection_log: List[CollectionRecord] = []

        # Public flags (readable by controller without lock for status display)
        self.is_collecting:   bool = False
        self.collection_complete: bool = False  # True for one tick when done

        log.info("TrashHandler initialised. Config: radius=%.1fm conf≥%.2f timeout=%.0fs",
                 self.cfg.detection_radius_m, self.cfg.min_confidence,
                 self.cfg.max_collection_time_s)

    # ── Public API: vision system interface ───────────────────────────────────

    def inject_detection(self, detection: TrashDetection) -> None:
        """
        Thread-safe. Call from vision system thread (camera callback, etc.)
        whenever a trash item is detected.

        The handler will decide on the next process() tick whether to act.
        Multiple detections queue up; only the best candidate is acted upon.

        Args:
            detection: TrashDetection with lat/lon/confidence/label.
        """
        with self._lock:
            self._pending_detections.append(detection)
        log.debug("Detection queued: (%.6f, %.6f) conf=%.2f label=%s",
                  detection.lat, detection.lon, detection.confidence, detection.label)

    def inject_detections_batch(self, detections: List[TrashDetection]) -> None:
        """Batch version of inject_detection for vision frames with multiple targets."""
        with self._lock:
            self._pending_detections.extend(detections)

    # ── Public API: motor command pipeline ────────────────────────────────────

    def process(self,
                nav_left:       float,
                nav_right:      float,
                lat:            float,
                lon:            float,
                heading_deg:    float,
                obstacle_active: bool,
                current_wp_idx: int) -> Tuple[float, float, bool]:
        """
        Called every control loop tick (10 Hz).

        Args:
            nav_left, nav_right: Motor commands from Navigator.
            lat, lon:            Current boat GPS position.
            heading_deg:         Current heading (degrees).
            obstacle_active:     True if ObstacleHandler is currently avoiding.
            current_wp_idx:      Navigator's current waypoint index.

        Returns:
            (left_cmd, right_cmd, is_collecting)
            is_collecting=True while in any collection sub-state.
            When is_collecting=False, returned commands equal nav inputs (pass-through).
        """
        with self._lock:
            self.collection_complete = False  # reset one-tick flag

            # ── Safety gate: obstacle always wins ─────────────────────────────
            if obstacle_active:
                if self._state != CollectionState.NORMAL_NAVIGATION:
                    log.warning("TrashHandler: obstacle detected — aborting collection.")
                    self._abort_collection("obstacle_preemption", lat, lon)
                # Pass through navigator commands — obstacle handler will override
                return nav_left, nav_right, False

            # ── Per-lane deviation counter reset ─────────────────────────────
            self._update_lane_counter(current_wp_idx)

            # ── Dispatch to current state ─────────────────────────────────────
            state = self._state

            if state == CollectionState.NORMAL_NAVIGATION:
                return self._tick_normal(nav_left, nav_right, lat, lon,
                                         heading_deg, current_wp_idx)

            elif state == CollectionState.DEVIATING_TO_TRASH:
                return self._tick_deviating(nav_left, nav_right, lat, lon,
                                             heading_deg, current_wp_idx)

            elif state == CollectionState.COLLECTING:
                return self._tick_collecting(nav_left, nav_right, lat, lon)

            elif state == CollectionState.RETURNING_TO_PATH:
                return self._tick_returning(nav_left, nav_right, lat, lon,
                                             heading_deg)

            # Should never reach here
            return nav_left, nav_right, False

    # ── Public API: controller integration ───────────────────────────────────

    @property
    def return_waypoint_index(self) -> Optional[int]:
        """
        When collection_complete is True (one tick only), the controller
        should call navigator.skip_to(trash_handler.return_waypoint_index)
        and set navigator.state = NavState.NAVIGATING.
        """
        return self._return_wp_idx

    def status_line(self) -> str:
        with self._lock:
            state = self._state
            target = self._target
        if state == CollectionState.NORMAL_NAVIGATION:
            return "TrashHandler: monitoring"
        if target:
            dist = haversine(
                self._deviation_origin_lat, self._deviation_origin_lon,
                target.lat, target.lon
            )
            return (f"TrashHandler: {state.name} | "
                    f"target=({target.lat:.5f},{target.lon:.5f}) "
                    f"dist={dist:.1f}m conf={target.confidence:.2f}")
        return f"TrashHandler: {state.name}"

    def stats(self) -> Dict:
        """Return collection statistics for mission summary."""
        success = [r for r in self.collection_log if r.success]
        failed  = [r for r in self.collection_log if not r.success]
        return {
            "total_attempts":    len(self.collection_log),
            "successful":        len(success),
            "failed":            len(failed),
            "abort_reasons":     [r.reason for r in failed],
        }

    # ── State machine: tick functions ─────────────────────────────────────────

    def _tick_normal(self, nav_left, nav_right, lat, lon,
                     heading_deg, current_wp_idx):
        """
        NORMAL_NAVIGATION state.
        Check pending detections and decide whether to divert.
        If not diverting: pure pass-through.
        """
        # Expire stale cooldowns
        self._prune_cooldowns()

        # Pick best detection from queue
        best = self._pick_best_detection(lat, lon)
        if best is None:
            self.is_collecting = False
            return nav_left, nav_right, False

        # ── Decision gates ────────────────────────────────────────────────────

        # Gate 1: confidence
        if best.confidence < self.cfg.min_confidence:
            log.debug("Ignoring detection: confidence %.2f < %.2f",
                      best.confidence, self.cfg.min_confidence)
            return nav_left, nav_right, False

        # Gate 2: distance check
        dist = haversine(lat, lon, best.lat, best.lon)
        if dist < self.cfg.min_detection_m:
            log.debug("Ignoring detection: too close (%.1f m)", dist)
            return nav_left, nav_right, False
        if dist > self.cfg.detection_radius_m:
            log.debug("Ignoring detection: out of range (%.1f m > %.1f m)",
                      dist, self.cfg.detection_radius_m)
            return nav_left, nav_right, False

        # Gate 3: in cooldown?
        if self._is_in_cooldown(best.lat, best.lon):
            log.debug("Ignoring detection: in cooldown zone")
            return nav_left, nav_right, False

        # Gate 4: max deviations per lane
        if self._deviations_this_lane >= self.cfg.max_deviations_per_lane:
            log.warning("TrashHandler: max deviations per lane (%d) reached — ignoring.",
                        self.cfg.max_deviations_per_lane)
            return nav_left, nav_right, False

        # Gate 5: polygon boundary check
        if self.cfg.require_inside_polygon:
            tx, ty = self.mh.to_xy(best.lat, best.lon)
            if not self.mh.is_inside_xy(tx, ty):
                log.info("Ignoring detection: trash is outside polygon boundary.")
                return nav_left, nav_right, False

        # Gate 6: deviation distance feasibility
        if dist > self.cfg.max_deviation_m:
            log.info("Ignoring detection: required deviation %.1f m > max %.1f m",
                     dist, self.cfg.max_deviation_m)
            return nav_left, nav_right, False

        # ── All gates passed: commit to collection ────────────────────────────
        self._target                = best
        self._deviation_origin_lat  = lat
        self._deviation_origin_lon  = lon
        self._deviation_start_time  = time.monotonic()
        self._wp_idx_at_deviation   = current_wp_idx
        self._deviations_this_lane += 1
        self._state                 = CollectionState.DEVIATING_TO_TRASH
        self.is_collecting          = True

        log.info(
            "TrashHandler: DIVERTING → (%.6f, %.6f) dist=%.1fm conf=%.2f label=%s",
            best.lat, best.lon, dist, best.confidence, best.label
        )
        # Immediately compute and return approach commands for this tick
        return self._compute_approach_commands(lat, lon, heading_deg, best)

    def _tick_deviating(self, nav_left, nav_right, lat, lon,
                         heading_deg, current_wp_idx):
        """
        DEVIATING_TO_TRASH state.
        Steer toward trash target. Check for timeout, abort conditions,
        and arrival.
        """
        target = self._target
        if target is None:
            self._state = CollectionState.NORMAL_NAVIGATION
            return nav_left, nav_right, False

        now     = time.monotonic()
        elapsed = now - self._deviation_start_time

        # ── Timeout abort ─────────────────────────────────────────────────────
        if elapsed > self.cfg.max_collection_time_s:
            log.warning("TrashHandler: collection timeout (%.0f s) — aborting.",
                        elapsed)
            self._abort_collection("timeout", lat, lon)
            return nav_left, nav_right, False

        # ── Distance from origin abort ────────────────────────────────────────
        dist_from_origin = haversine(
            self._deviation_origin_lat, self._deviation_origin_lon, lat, lon
        )
        if dist_from_origin > self.cfg.max_deviation_m:
            log.warning("TrashHandler: max deviation distance exceeded (%.1f m) — aborting.",
                        dist_from_origin)
            self._abort_collection("max_deviation_exceeded", lat, lon)
            return nav_left, nav_right, False

        # ── Check arrival at trash ────────────────────────────────────────────
        dist_to_trash = haversine(lat, lon, target.lat, target.lon)
        if dist_to_trash <= self.cfg.collection_radius_m:
            log.info("TrashHandler: ARRIVED at trash (%.2f m). Starting collection dwell.",
                     dist_to_trash)
            self._state          = CollectionState.COLLECTING
            self._dwell_start_time = time.monotonic()
            # Stop motors during collection
            return 0.0, 0.0, True

        # ── Continue approach ──────────────────────────────────────────────────
        return self._compute_approach_commands(lat, lon, heading_deg, target)

    def _tick_collecting(self, nav_left, nav_right, lat, lon):
        """
        COLLECTING state.
        Motors stopped. Wait for dwell period (physical collector operating).
        After dwell: log success, register cooldown, transition to RETURNING.
        """
        elapsed = time.monotonic() - self._dwell_start_time

        if elapsed < self.cfg.collection_dwell_s:
            # Still dwelling — motors off
            pct = elapsed / self.cfg.collection_dwell_s * 100
            log.debug("TrashHandler: collecting... %.0f%%", pct)
            return 0.0, 0.0, True

        # ── Dwell complete: collection successful ─────────────────────────────
        target = self._target
        log.info("TrashHandler: COLLECTION COMPLETE — %s at (%.6f, %.6f).",
                 target.label if target else "trash",
                 target.lat if target else lat,
                 target.lon if target else lon)

        # Register cooldown at collection location
        self._register_cooldown(
            target.lat if target else lat,
            target.lon if target else lon
        )

        # Log to history
        self.collection_log.append(CollectionRecord(
            lat=target.lat if target else lat,
            lon=target.lon if target else lon,
            timestamp=time.monotonic(),
            success=True,
            label=target.label if target else "trash",
            reason="collected",
        ))

        # Compute best return waypoint using PathRecovery
        self._return_wp_idx = self.recovery.find_best_waypoint(
            lat, lon, self._wp_idx_at_deviation
        )
        log.info("TrashHandler: returning to path at WP %d.", self._return_wp_idx)

        self._state        = CollectionState.RETURNING_TO_PATH
        self._target       = None
        self.is_collecting = True

        return 0.0, 0.0, True   # hold position; next tick _tick_returning computes return commands

    def _tick_returning(self, nav_left, nav_right, lat, lon, heading_deg):
        """
        RETURNING_TO_PATH state.
        Navigate toward the recovery waypoint using same P-controller as Navigator.
        When within WAYPOINT_RADIUS_M: signal controller to resume Navigator.
        """
        wp_idx = self._return_wp_idx
        if wp_idx >= len(self.waypoints):
            wp_idx = len(self.waypoints) - 1
        wp = self.waypoints[wp_idx]

        dist_to_wp = haversine(lat, lon, wp.lat, wp.lon)

        if dist_to_wp <= config.WAYPOINT_RADIUS_M:
            # Arrived at return point
            log.info("TrashHandler: returned to path at WP %d (%.1f m). "
                     "Resuming normal navigation.", wp_idx, dist_to_wp)
            self._state               = CollectionState.NORMAL_NAVIGATION
            self.is_collecting        = False
            self.collection_complete  = True   # one-tick signal to controller
            return nav_left, nav_right, False

        # Steer toward recovery waypoint (same P-controller as Navigator)
        desired_hdg = bearing(lat, lon, wp.lat, wp.lon)
        err         = angle_diff(heading_deg, desired_hdg)
        err_norm    = clamp(err / 180.0, -1.0, 1.0)
        turn        = config.HEADING_KP * err_norm
        fwd         = self.cfg.collection_approach_speed * (1.0 - 0.4 * abs(err_norm))

        left  = clamp(fwd - turn * config.MAX_TURN_RATE, -1.0, 1.0)
        right = clamp(fwd + turn * config.MAX_TURN_RATE, -1.0, 1.0)

        log.debug("TrashHandler: returning WP=%d dist=%.1fm hdg_err=%+.1f°",
                  wp_idx, dist_to_wp, err)
        return left, right, True

    # ── Detection selection ───────────────────────────────────────────────────

    def _pick_best_detection(self,
                              lat: float,
                              lon: float) -> Optional[TrashDetection]:
        """
        Select the best actionable detection from the pending queue.

        Strategy: among detections within detection_radius_m, pick the
        NEAREST one above confidence threshold. Nearest = least deviation cost.

        Clears the queue after selection (old detections are stale).
        """
        if not self._pending_detections:
            return None

        # Snapshot and clear queue atomically
        detections = list(self._pending_detections)
        self._pending_detections.clear()

        # Filter by confidence
        qualified = [d for d in detections if d.confidence >= self.cfg.min_confidence]
        if not qualified:
            return None

        # Sort by distance (nearest first)
        qualified.sort(key=lambda d: haversine(lat, lon, d.lat, d.lon))
        return qualified[0]

    # ── Cooldown management ───────────────────────────────────────────────────

    def _register_cooldown(self, lat: float, lon: float) -> None:
        """Mark a GPS position as cooled-down for cooldown_s seconds."""
        expiry = time.monotonic() + self.cfg.cooldown_s
        self._cooldowns.append((lat, lon, expiry))
        log.debug("Cooldown registered at (%.6f, %.6f) for %.0f s.",
                  lat, lon, self.cfg.cooldown_s)

    def _is_in_cooldown(self, lat: float, lon: float) -> bool:
        """Return True if (lat, lon) is within cooldown_radius_m of any active cooldown."""
        now = time.monotonic()
        for clat, clon, expiry in self._cooldowns:
            if expiry < now:
                continue
            if haversine(lat, lon, clat, clon) <= self.cfg.cooldown_radius_m:
                return True
        return False

    def _prune_cooldowns(self) -> None:
        """Remove expired cooldown entries."""
        now = time.monotonic()
        before = len(self._cooldowns)
        self._cooldowns = [(la, lo, ex) for la, lo, ex in self._cooldowns if ex > now]
        removed = before - len(self._cooldowns)
        if removed > 0:
            log.debug("Pruned %d expired cooldowns.", removed)

    # ── Per-lane deviation counter ────────────────────────────────────────────

    def _update_lane_counter(self, current_wp_idx: int) -> None:
        """
        Reset deviation counter when the navigator moves to a new lane.
        A "lane" in the lawnmower pattern is every pair of waypoints.
        Lane index = current_wp_idx // 2.
        """
        lane_idx = current_wp_idx // 2
        if lane_idx != self._last_lane_idx:
            if self._last_lane_idx != -1:
                log.debug("TrashHandler: new lane %d (was %d). Deviations reset.",
                          lane_idx, self._last_lane_idx)
            self._last_lane_idx        = lane_idx
            self._deviations_this_lane = 0

    # ── Abort logic ───────────────────────────────────────────────────────────

    def _abort_collection(self, reason: str, lat: float, lon: float) -> None:
        """
        Abort any active collection attempt and return to NORMAL_NAVIGATION.
        Registers a cooldown so the same trash doesn't re-trigger immediately.
        Also logs the failure.
        """
        target = self._target

        log.warning("TrashHandler: ABORTING collection. Reason: %s", reason)

        # Register cooldown at trash position (if known) to prevent re-trigger
        if target:
            self._register_cooldown(target.lat, target.lon)

        # Also register cooldown at current position (to prevent immediate re-detection)
        self._register_cooldown(lat, lon)

        # Log failure
        self.collection_log.append(CollectionRecord(
            lat=target.lat if target else lat,
            lon=target.lon if target else lon,
            timestamp=time.monotonic(),
            success=False,
            label=target.label if target else "trash",
            reason=reason,
        ))

        # Use PathRecovery to find where to rejoin the path
        self._return_wp_idx = self.recovery.find_best_waypoint(
            lat, lon, self._wp_idx_at_deviation
        )

        # Signal controller to resume Navigator at recovery waypoint
        self.collection_complete = True   # one-tick signal
        self._state              = CollectionState.NORMAL_NAVIGATION
        self._target             = None
        self.is_collecting       = False

        log.info("TrashHandler: aborted. Will resume at WP %d.", self._return_wp_idx)

    # ── Motor command computation ─────────────────────────────────────────────

    def _compute_approach_commands(self,
                                    lat:         float,
                                    lon:         float,
                                    heading_deg: float,
                                    target:      TrashDetection
                                    ) -> Tuple[float, float, bool]:
        """
        P-controller steering toward target GPS position.
        Same algorithm as Navigator.update() for consistency.
        Slower approach speed than cruise for precision.
        """
        desired_hdg = bearing(lat, lon, target.lat, target.lon)
        err         = angle_diff(heading_deg, desired_hdg)
        err_norm    = clamp(err / 180.0, -1.0, 1.0)
        turn        = config.HEADING_KP * err_norm
        fwd         = self.cfg.collection_approach_speed * (1.0 - 0.5 * abs(err_norm))

        left  = clamp(fwd - turn * config.MAX_TURN_RATE, -1.0, 1.0)
        right = clamp(fwd + turn * config.MAX_TURN_RATE, -1.0, 1.0)

        dist  = haversine(lat, lon, target.lat, target.lon)
        log.debug("Approach: dist=%.1fm bearing=%.1f° hdg_err=%+.1f°",
                  dist, desired_hdg, err)

        return left, right, True