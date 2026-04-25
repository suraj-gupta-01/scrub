"""
ASV Navigation System - Math Utilities
Haversine distance, bearing, coordinate helpers.
"""

import math
from typing import Tuple

# Earth radius in metres
EARTH_R = 6_371_000.0


# ── Haversine & bearing ───────────────────────────────────────────────────────

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in metres between two GPS points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlam       = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return initial bearing in degrees [0, 360) from point 1 → point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam       = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a: float, b: float) -> float:
    """Signed difference b - a in degrees, result in (-180, 180]."""
    return ((b - a + 180) % 360) - 180


# ── Equirectangular projection ────────────────────────────────────────────────

def gps_to_xy(lat: float, lon: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """
    Convert (lat, lon) → (x, y) metres relative to origin.
    Equirectangular approximation — accurate for small areas (< 50 km).
    """
    x = math.radians(lon - origin_lon) * EARTH_R * math.cos(math.radians(origin_lat))
    y = math.radians(lat - origin_lat) * EARTH_R
    return x, y


def xy_to_gps(x: float, y: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    """Inverse of gps_to_xy — (x, y) metres → (lat, lon)."""
    lat = origin_lat + math.degrees(y / EARTH_R)
    lon = origin_lon + math.degrees(x / (EARTH_R * math.cos(math.radians(origin_lat))))
    return lat, lon


# ── Geometry helpers ──────────────────────────────────────────────────────────

def point_segment_closest(px: float, py: float,
                           ax: float, ay: float,
                           bx: float, by: float) -> Tuple[float, float, float]:
    """
    Return (cx, cy, t) where (cx, cy) is the closest point on segment AB
    to point P, and t ∈ [0, 1] is the parameter along AB.
    """
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-12:
        return ax, ay, 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
    return ax + t * dx, ay + t * dy, t


def dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_angle(deg: float) -> float:
    """Wrap angle to (-180, 180]."""
    return ((deg + 180) % 360) - 180
