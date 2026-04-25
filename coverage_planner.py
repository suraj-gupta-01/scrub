"""
ASV Navigation System - Coverage Planner
Generates a boustrophedon (lawnmower) sweep path inside the waterbody polygon.
"""

import json
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

from map_handler import MapHandler, XYPoint
from utils import xy_to_gps
import config


@dataclass
class Waypoint:
    """A single mission waypoint in both coordinate frames."""
    index:  int
    x:      float   # local metres
    y:      float
    lat:    float   # GPS
    lon:    float
    is_turn: bool = False   # True for lane-transition points

    def as_gps(self) -> Tuple[float, float]:
        return self.lat, self.lon

    def as_xy(self) -> Tuple[float, float]:
        return self.x, self.y


class CoveragePlanner:
    """
    Generates a lawnmower coverage path inside the polygon managed by map_handler.

    Algorithm:
      1. Optionally inset the polygon by POLYGON_BUFFER_M to keep the boat
         away from shores.
      2. Scan horizontal (or rotated) lines separated by SWEEP_WIDTH_M.
      3. For each scan line, clip against the polygon using the Sutherland-
         Hodgman algorithm to find entry/exit points.
      4. Alternate the direction of travel each lane (boustrophedon).
      5. Convert XY waypoints back to GPS.
    """

    def __init__(self, map_handler: MapHandler,
                 sweep_width: float = None,
                 buffer:      float = None,
                 angle_deg:   float = 0.0):
        """
        Args:
            map_handler: Loaded boundary map.
            sweep_width: Lane spacing in metres (default: config.SWEEP_WIDTH_M).
            buffer:      Inset from boundary in metres.
            angle_deg:   Sweep direction angle in degrees (0 = horizontal lanes).
        """
        self.mh          = map_handler
        self.sweep_width = sweep_width or config.SWEEP_WIDTH_M
        self.buffer      = buffer      if buffer is not None else config.POLYGON_BUFFER_M
        self.angle_rad   = math.radians(angle_deg)
        self.waypoints:  List[Waypoint] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> List[Waypoint]:
        """Compute and return the coverage waypoints."""
        # Get working polygon (inset from boundary)
        poly = self.mh.inset_polygon_xy(self.buffer)
        if len(poly) < 4:
            raise ValueError("Polygon too small after inset — reduce POLYGON_BUFFER_M.")

        pts = [(p.x, p.y) for p in poly]

        # Rotate polygon into sweep frame
        cos_a, sin_a = math.cos(-self.angle_rad), math.sin(-self.angle_rad)
        rotated = [self._rotate(x, y, cos_a, sin_a) for x, y in pts]

        y_vals = [p[1] for p in rotated]
        y_min, y_max = min(y_vals), max(y_vals)

        # Generate sweep lines
        raw_waypoints: List[Tuple[float, float]] = []
        forward = True
        y = y_min + self.sweep_width / 2.0

        while y <= y_max:
            # Clip horizontal line y=const against rotated polygon
            xs = self._intersect_horizontal(rotated, y)
            if len(xs) >= 2:
                x_left, x_right = min(xs), max(xs)
                if forward:
                    raw_waypoints.append((x_left,  y))
                    raw_waypoints.append((x_right, y))
                else:
                    raw_waypoints.append((x_right, y))
                    raw_waypoints.append((x_left,  y))
                forward = not forward
            y += self.sweep_width

        # Rotate back to world frame and convert to GPS
        cos_b, sin_b = math.cos(self.angle_rad), math.sin(self.angle_rad)
        self.waypoints = []
        for i, (rx, ry) in enumerate(raw_waypoints):
            wx, wy = self._rotate(rx, ry, cos_b, sin_b)
            lat, lon = self.mh.to_gps(wx, wy)
            # Mark turn waypoints (even-indexed are row-start, odd are row-end)
            is_turn = (i % 2 == 1) and (i < len(raw_waypoints) - 1)
            self.waypoints.append(Waypoint(
                index=i, x=wx, y=wy, lat=lat, lon=lon, is_turn=is_turn
            ))

        return self.waypoints

    def save(self, path: str = None) -> None:
        """Persist waypoints to JSON."""
        path = path or config.WAYPOINT_FILE
        data = [
            {"index": w.index, "lat": w.lat, "lon": w.lon,
             "x": w.x, "y": w.y, "is_turn": w.is_turn}
            for w in self.waypoints
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CoveragePlanner] Saved {len(data)} waypoints → {path}")

    @staticmethod
    def load(path: str, map_handler: MapHandler) -> List[Waypoint]:
        """Load previously saved waypoints from JSON."""
        with open(path) as f:
            data = json.load(f)
        waypoints = []
        for d in data:
            waypoints.append(Waypoint(
                index=d["index"], x=d["x"], y=d["y"],
                lat=d["lat"], lon=d["lon"], is_turn=d.get("is_turn", False)
            ))
        return waypoints

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rotate(x: float, y: float, cos_a: float, sin_a: float) -> Tuple[float, float]:
        return x * cos_a - y * sin_a, x * sin_a + y * cos_a

    @staticmethod
    def _intersect_horizontal(poly: List[Tuple[float, float]], y: float) -> List[float]:
        """
        Find all x-coordinates where the horizontal line at height y
        intersects the polygon edges.
        """
        xs = []
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 <= y < y2) or (y2 <= y < y1):
                # Linear interpolation
                t = (y - y1) / (y2 - y1)
                xs.append(x1 + t * (x2 - x1))
        return xs

    def summary(self) -> str:
        n = len(self.waypoints)
        if n == 0:
            return "No waypoints generated yet."
        total_dist = 0.0
        for i in range(1, n):
            dx = self.waypoints[i].x - self.waypoints[i-1].x
            dy = self.waypoints[i].y - self.waypoints[i-1].y
            total_dist += math.sqrt(dx*dx + dy*dy)
        lanes = n // 2
        return (f"CoveragePlanner: {n} waypoints, ~{lanes} lanes, "
                f"~{total_dist:.0f} m total path length")
