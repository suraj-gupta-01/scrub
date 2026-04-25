"""
ASV Navigation System - Map Handler
GPS ↔ local XY conversion, polygon storage and query.
"""

import math
import json
from typing import List, Tuple, Optional
from dataclasses import dataclass, field, asdict

from utils import gps_to_xy, xy_to_gps


@dataclass
class GPSPoint:
    lat: float
    lon: float

    def as_tuple(self) -> Tuple[float, float]:
        return self.lat, self.lon


@dataclass
class XYPoint:
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class MapHandler:
    """
    Manages the waterbody boundary polygon in both GPS and local XY frames.

    Usage:
        mh = MapHandler.from_gps_polygon(gps_coords)
        x, y = mh.to_xy(lat, lon)
        lat, lon = mh.to_gps(x, y)
        inside = mh.is_inside_xy(x, y)
    """

    # GPS polygon as provided by user / GIS tool
    gps_polygon: List[GPSPoint] = field(default_factory=list)

    # Local XY polygon (metres, relative to origin)
    xy_polygon:  List[XYPoint]  = field(default_factory=list)

    # Origin of the local coordinate frame (centroid of the polygon)
    origin_lat: float = 0.0
    origin_lon: float = 0.0

    # Bounding box in XY space
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_gps_polygon(cls, coords: List[Tuple[float, float]]) -> "MapHandler":
        """
        Build a MapHandler from a list of (lat, lon) tuples.
        The polygon is automatically closed if needed.
        """
        if not coords or len(coords) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")

        gps_pts = [GPSPoint(lat, lon) for lat, lon in coords]

        # Ensure polygon is closed
        if gps_pts[0].as_tuple() != gps_pts[-1].as_tuple():
            gps_pts.append(gps_pts[0])

        # Compute centroid as origin
        n = len(gps_pts) - 1  # exclude duplicate closing point
        origin_lat = sum(p.lat for p in gps_pts[:n]) / n
        origin_lon = sum(p.lon for p in gps_pts[:n]) / n

        # Project to XY
        xy_pts = [
            XYPoint(*gps_to_xy(p.lat, p.lon, origin_lat, origin_lon))
            for p in gps_pts
        ]

        # Bounding box
        xs = [p.x for p in xy_pts]
        ys = [p.y for p in xy_pts]

        return cls(
            gps_polygon=gps_pts,
            xy_polygon=xy_pts,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            x_min=min(xs), x_max=max(xs),
            y_min=min(ys), y_max=max(ys),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "MapHandler":
        with open(path) as f:
            data = json.load(f)
        coords = [(p["lat"], p["lon"]) for p in data["boundary"]]
        return cls.from_gps_polygon(coords)

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        return gps_to_xy(lat, lon, self.origin_lat, self.origin_lon)

    def to_gps(self, x: float, y: float) -> Tuple[float, float]:
        return xy_to_gps(x, y, self.origin_lat, self.origin_lon)

    # ── Polygon tests ─────────────────────────────────────────────────────────

    def is_inside_xy(self, x: float, y: float) -> bool:
        """Ray-casting algorithm — O(n)."""
        poly = self.xy_polygon
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i].x, poly[i].y
            xj, yj = poly[j].x, poly[j].y
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def is_inside_gps(self, lat: float, lon: float) -> bool:
        x, y = self.to_xy(lat, lon)
        return self.is_inside_xy(x, y)

    def inset_polygon_xy(self, buffer_m: float) -> List[XYPoint]:
        """
        Return an inset (shrunk) version of the XY polygon.
        Uses simple vertex-moving toward centroid — good for convex shapes.
        For production, replace with Shapely's buffer(-buffer_m).
        """
        # Centroid (excluding duplicate closing point)
        pts = self.xy_polygon[:-1]
        cx = sum(p.x for p in pts) / len(pts)
        cy = sum(p.y for p in pts) / len(pts)

        inset = []
        for p in pts:
            dx, dy = p.x - cx, p.y - cy
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < 1e-9:
                inset.append(XYPoint(p.x, p.y))
            else:
                ratio = max(0.0, (dist - buffer_m) / dist)
                inset.append(XYPoint(cx + dx * ratio, cy + dy * ratio))

        # Re-close
        inset.append(inset[0])
        return inset

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save_json(self, path: str) -> None:
        data = {
            "origin": {"lat": self.origin_lat, "lon": self.origin_lon},
            "boundary": [{"lat": p.lat, "lon": p.lon} for p in self.gps_polygon],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def __repr__(self) -> str:
        n = len(self.gps_polygon) - 1
        return (f"MapHandler(vertices={n}, "
                f"bbox=[{self.x_min:.1f},{self.x_max:.1f}]×"
                f"[{self.y_min:.1f},{self.y_max:.1f}] m)")
