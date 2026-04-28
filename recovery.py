"""
ASV Navigation System - Path Recovery (v2)
==========================================
Deterministic, loop-free path recovery after obstacle avoidance.

WHY THE ORIGINAL COULD LOOP
-----------------------------
The original find_best_waypoint() searched from (last_wp_index - 1) onward
and returned the END of whichever segment was geometrically closest.

Consider this scenario:

  Waypoints: 0 → 1 → 2 → 3 → 4 → 5 → 6 ...
  Boat was heading to WP #4 (current_idx = 4).
  Obstacle detected near WP #3.
  Avoidance pushes boat sideways and slightly backward.
  Boat ends up physically nearest to segment [2→3].
  Recovery returns WP #3 — BEHIND the avoidance point.
  Navigator drives toward WP #3.
  Obstacle is STILL near WP #3. New avoidance triggered.
  Recovery returns WP #3 again.
  LOOP.

This is especially bad in a lawnmower pattern where alternating lanes
run in opposite directions. The nearest segment is often on the
PREVIOUS lane, not the current one.

A second failure mode: the boat avoids into an area between two lanes
that are equidistant. The search oscillates between the two candidates
each recovery cycle.

DESIGN GOALS FOR v2
--------------------
1. DETERMINISTIC — same inputs always produce same output.
2. FORWARD-ONLY — never return a waypoint index ≤ last_wp_index - 1
   unless the mission has literally no forward path left.
3. LOOP-SAFE — if the best forward segment is behind the boat's
   current heading, skip it and take the NEXT one.
4. COVERAGE-PRESERVING — prefer the resumption point that loses the
   least uncovered area (i.e. as close to where avoidance started as
   possible, not just geometrically nearest).

ALGORITHM
---------
                    A ─────────────────── B
                                          │
                              ← boat ─ ─ ┤  (pushed here by avoidance)
                                          │
                    D ─────────────────── C

Step 1. Enforce a minimum resume index:
        min_idx = max(last_wp_index, 1)   ← never go backward

Step 2. Search ONLY segments [min_idx .. end].
        For each segment, compute the closest point on that segment to
        the boat. Record (segment_idx, perpendicular_distance).

Step 3. Among segments within CLOSE_ENOUGH_M (configurable, default 20 m)
        of the boat, pick the one with the LOWEST index (earliest in
        mission, least coverage lost).

Step 4. If no segment is within CLOSE_ENOUGH_M, fall back to the
        segment with the minimum perpendicular distance — but still
        subject to the forward-only constraint.

Step 5. Return the END waypoint of the winning segment so the navigator
        immediately heads forward.

Step 6. Safety cap: if the returned index equals last_wp_index and the
        boat is still within avoidance distance of that waypoint, bump
        by +1 to prevent immediate re-trigger.
"""

import math
import logging
from typing import List, Tuple, Optional

from coverage_planner import Waypoint
from map_handler import MapHandler
from utils import point_segment_closest, dist2d
import config

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

# Segments within this distance are treated as "candidate group".
# The EARLIEST (lowest index) member of the group is selected.
CLOSE_ENOUGH_M = 20.0

# Safety bump: if recovered index == last_wp_index, add this offset
# to avoid immediately re-encountering the same obstacle.
SAFETY_SKIP = 1

# How far ahead (in waypoints) to scan beyond the minimum index.
# Prevents searching the entire remaining mission on large maps.
# 0 = unlimited (scan to end).
MAX_LOOKAHEAD_WPS = 0


class PathRecovery:
    """
    Determines the best waypoint to resume after obstacle avoidance.

    Implements forward-only, loop-safe, coverage-preserving recovery.
    """

    def __init__(self, map_handler: MapHandler, waypoints: List[Waypoint]):
        self.mh        = map_handler
        self.waypoints = waypoints

    # ── Public API ────────────────────────────────────────────────────────────

    def find_best_waypoint(self,
                            lat: float,
                            lon: float,
                            last_wp_index: int) -> int:
        """
        Return the waypoint index the navigator should resume toward.

        Args:
            lat, lon:      Current boat GPS position after avoidance.
            last_wp_index: The waypoint the navigator was targeting when
                           the obstacle was detected (NOT the previously
                           completed one — the CURRENT target).

        Returns:
            Waypoint index for Navigator.skip_to(). Always ≥ last_wp_index.
        """
        n = len(self.waypoints)
        if n == 0:
            return 0
        if last_wp_index >= n:
            return n - 1

        boat_x, boat_y = self.mh.to_xy(lat, lon)

        # Step 1: Enforce forward-only minimum
        # We allow one segment back (last_wp_index - 1 as segment start)
        # so that if avoidance barely overshot a waypoint we can still
        # land on the segment that starts there — but we return its END,
        # which is still ≥ last_wp_index.
        seg_start = max(0, last_wp_index - 1)
        seg_end   = n - 1  # exclusive: segment i goes from wp[i] to wp[i+1]
        if MAX_LOOKAHEAD_WPS > 0:
            seg_end = min(seg_end, seg_start + MAX_LOOKAHEAD_WPS)

        # Step 2: Score every candidate segment
        candidates: List[Tuple[float, int]] = []  # (perp_dist, end_wp_idx)

        for i in range(seg_start, seg_end):
            wp_a = self.waypoints[i]
            wp_b = self.waypoints[i + 1]
            cx, cy, _ = point_segment_closest(
                boat_x, boat_y,
                wp_a.x, wp_a.y,
                wp_b.x, wp_b.y,
            )
            d = dist2d(boat_x, boat_y, cx, cy)
            end_idx = i + 1
            candidates.append((d, end_idx))

        if not candidates:
            log.warning("Recovery: no candidate segments — staying at WP %d.", last_wp_index)
            return last_wp_index

        # Step 3: Prefer the earliest segment within CLOSE_ENOUGH_M
        close_candidates = [(d, idx) for d, idx in candidates if d <= CLOSE_ENOUGH_M]

        if close_candidates:
            # Among close segments, pick the EARLIEST (smallest index)
            # — minimises coverage loss.
            # Sort by index first (coverage priority), then by dist (tie-break).
            close_candidates.sort(key=lambda x: (x[1], x[0]))
            best_dist, best_idx = close_candidates[0]
            log.info(
                "Recovery (close): WP %d, off-path %.1f m, %d candidates within %.0f m.",
                best_idx, best_dist, len(close_candidates), CLOSE_ENOUGH_M
            )
        else:
            # Step 4: No segment is close — use nearest by distance
            # Still forward-only (seg_start enforced above).
            candidates.sort(key=lambda x: x[0])
            best_dist, best_idx = candidates[0]
            log.warning(
                "Recovery (far): WP %d, off-path %.1f m — no segment within %.0f m.",
                best_idx, best_dist, CLOSE_ENOUGH_M
            )

        # Step 5: Forward-only guarantee
        # The end_idx is always ≥ seg_start + 1 ≥ last_wp_index (when seg_start = last_wp_index - 1)
        # but double-check defensively:
        best_idx = max(best_idx, last_wp_index)

        # Step 6: Safety skip — if we'd return exactly where the obstacle is,
        # advance by SAFETY_SKIP to avoid immediately re-triggering avoidance.
        if best_idx == last_wp_index and best_idx + SAFETY_SKIP < n:
            log.info("Recovery: safety skip applied WP %d → %d.",
                     best_idx, best_idx + SAFETY_SKIP)
            best_idx += SAFETY_SKIP

        # Cap to valid range
        best_idx = min(best_idx, n - 1)

        log.info("Recovery: resuming at WP %d / %d (off-path %.1f m).",
                 best_idx, n, best_dist)
        return best_idx

    def needs_recovery(self,
                        lat: float,
                        lon: float,
                        current_wp_index: int) -> bool:
        """
        True if the boat is further than RECOVERY_THRESHOLD_M from the
        planned path segment it is currently traversing.

        Uses the segment BEFORE and AT current_wp_index so that recovery
        is measured against the active leg, not a stale one.
        """
        n = len(self.waypoints)
        if n == 0 or current_wp_index >= n:
            return False

        bx, by = self.mh.to_xy(lat, lon)

        # Active segment: from the waypoint BEFORE current to current
        i = max(0, current_wp_index - 1)
        j = min(current_wp_index, n - 1)

        wp_a = self.waypoints[i]
        wp_b = self.waypoints[j]
        cx, cy, _ = point_segment_closest(bx, by, wp_a.x, wp_a.y, wp_b.x, wp_b.y)
        off_path = dist2d(bx, by, cx, cy)

        return off_path > config.RECOVERY_THRESHOLD_M

    def estimate_coverage_loss(self,
                                skipped_from: int,
                                resumed_at: int) -> float:
        """
        Estimate how much path length (metres) was skipped during recovery.
        Useful for logging and mission quality assessment.
        """
        total = 0.0
        for i in range(skipped_from, min(resumed_at, len(self.waypoints) - 1)):
            wp_a = self.waypoints[i]
            wp_b = self.waypoints[i + 1]
            dx = wp_b.x - wp_a.x
            dy = wp_b.y - wp_a.y
            total += math.sqrt(dx*dx + dy*dy)
        return total
