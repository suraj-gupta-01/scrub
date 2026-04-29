"""
ASV Navigation System - Path Recovery
After obstacle avoidance, finds the nearest point on the planned path
and resumes waypoint navigation smoothly.
"""

import math
import logging
from typing import List, Tuple, Optional

from coverage_planner import Waypoint
from map_handler import MapHandler
from utils import point_segment_closest, dist2d, haversine
import config

log = logging.getLogger(__name__)


class PathRecovery:
    """
    Determines the best waypoint to resume after an obstacle avoidance manoeuvre.

    Strategy:
      1. Project the boat's current XY position onto every path segment
         (between consecutive waypoints).
      2. Find the segment whose closest point is nearest to the boat.
      3. Return the *end* waypoint of that segment so the navigator
         resumes forward along the original path.
      4. Prefer segments ahead of the last known waypoint index to avoid
         going backwards.
    """

    def __init__(self, map_handler: MapHandler, waypoints: List[Waypoint]):
        self.mh        = map_handler
        self.waypoints = waypoints

    def find_best_waypoint(self, lat: float, lon: float,
                            last_wp_index: int) -> int:
        """
        Return the index of the waypoint to resume navigation toward.

        Args:
            lat, lon:      Current boat GPS position.
            last_wp_index: Index of the waypoint the navigator was heading to.

        Returns:
            Waypoint index to pass to Navigator.skip_to().
        """
        if not self.waypoints:
            return 0

        boat_x, boat_y = self.mh.to_xy(lat, lon)

        best_idx  = last_wp_index
        best_dist = float("inf")

        # Search from last_wp_index onward (don't reverse mission)
        search_start = max(0, last_wp_index - 1)

        for i in range(search_start, len(self.waypoints) - 1):
            wp_a = self.waypoints[i]
            wp_b = self.waypoints[i + 1]

            cx, cy, _ = point_segment_closest(
                boat_x, boat_y,
                wp_a.x, wp_a.y,
                wp_b.x, wp_b.y,
            )
            d = dist2d(boat_x, boat_y, cx, cy)

            if d < best_dist:
                best_dist = d
                best_idx  = i + 1  # head toward the far end of the segment

        if best_dist > config.RECOVERY_THRESHOLD_M * 5:
            log.warning("Recovery: very far off-path (%.1f m) — jumping to nearest waypoint.", best_dist)
            # Fall back to nearest waypoint (not segment)
            best_idx = self._nearest_waypoint_ahead(boat_x, boat_y, search_start)

        log.info("Recovery: resume at waypoint %d (off-path by %.1f m).",
                 best_idx, best_dist)
        return best_idx

    def _nearest_waypoint_ahead(self, bx: float, by: float,
                                  start: int) -> int:
        best_idx  = start
        best_dist = float("inf")
        for i in range(start, len(self.waypoints)):
            wp = self.waypoints[i]
            d  = dist2d(bx, by, wp.x, wp.y)
            if d < best_dist:
                best_dist = d
                best_idx  = i
        return best_idx

    def needs_recovery(self, lat: float, lon: float,
                        current_wp_index: int) -> bool:
        """
        True if the boat is further than RECOVERY_THRESHOLD_M from the
        planned path (measured between waypoints on either side of current).
        """
        if not self.waypoints or current_wp_index >= len(self.waypoints):
            return False

        bx, by = self.mh.to_xy(lat, lon)
        i = max(0, current_wp_index - 1)
        j = min(current_wp_index, len(self.waypoints) - 1)

        wp_a = self.waypoints[i]
        wp_b = self.waypoints[j]
        cx, cy, _ = point_segment_closest(bx, by, wp_a.x, wp_a.y, wp_b.x, wp_b.y)
        off_path_dist = dist2d(bx, by, cx, cy)

        return off_path_dist > config.RECOVERY_THRESHOLD_M
