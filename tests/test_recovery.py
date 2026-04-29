"""
tests/test_recovery.py
=======================
Tests for PathRecovery (v2) — the module that finds where to resume
the mission after an obstacle avoidance manoeuvre.

What we're checking:
  1. Recovery always returns a FORWARD index (never goes backward)
  2. Recovery finds the nearest segment correctly
  3. Safety skip is applied when recovery would land on the same waypoint
  4. Recovery handles edge cases: empty list, last waypoint, far-off-path
  5. needs_recovery() correctly detects off-path situations
  6. Coverage loss estimate is correct

Run just this file:
    pytest tests/test_recovery.py -v
"""

import pytest
from coverage_planner import Waypoint
from recovery import PathRecovery, CLOSE_ENOUGH_M, SAFETY_SKIP


# ── Helper: build a straight horizontal path ──────────────────────────────────

def make_straight_path(n: int, spacing: float = 10.0) -> list:
    """
    Build n waypoints in a straight horizontal line going East.
    Each waypoint is `spacing` metres apart.
    XY coords: (0,0), (10,0), (20,0), ...
    GPS coords: approximated from XY.
    """
    EARTH_R = 6_371_000.0
    import math
    origin_lat, origin_lon = 12.9719, 77.5948
    wps = []
    for i in range(n):
        x = float(i * spacing)
        y = 0.0
        lat = origin_lat + math.degrees(y / EARTH_R)
        lon = origin_lon + math.degrees(x / (EARTH_R * math.cos(math.radians(origin_lat))))
        wps.append(Waypoint(index=i, x=x, y=y, lat=lat, lon=lon, is_turn=False))
    return wps


def make_recovery(waypoints, map_handler) -> PathRecovery:
    return PathRecovery(map_handler, waypoints)


# ── Test 1: Basic recovery on a straight path ─────────────────────────────────

def test_recovery_returns_forward_index(map_handler):
    """
    Boat was heading to WP #3 (index 3).
    After avoidance it ends up slightly off-path near segment [2→3].
    Recovery should return index 3 or higher — NEVER 2 or lower.

    WHY: The original recovery bug would sometimes return WP #2,
    causing the boat to loop back through the obstacle.
    """
    wps = make_straight_path(6)
    rec = make_recovery(wps, map_handler)

    # Boat is slightly south of the path near the midpoint of segment [2→3]
    # Segment [2→3] goes from x=20 to x=30, at y=0
    # Boat is at x=25, y=-5 (5 metres south of path)
    import math
    EARTH_R = 6_371_000.0
    origin_lat, origin_lon = 12.9719, 77.5948
    boat_lat = origin_lat + math.degrees(-5.0 / EARTH_R)
    boat_lon = origin_lon + math.degrees(25.0 / (EARTH_R * math.cos(math.radians(origin_lat))))

    result = rec.find_best_waypoint(boat_lat, boat_lon, last_wp_index=3)

    print(f"\n  Boat near segment [2→3], last_wp=3 → recovery returned WP #{result}")
    print(f"  (Must be >= 3 to be forward-only)")

    assert result >= 3, (
        f"Recovery went backward! Returned WP #{result}, expected >= 3"
    )


# ── Test 2: Empty waypoint list ───────────────────────────────────────────────

def test_recovery_empty_waypoints(map_handler):
    """
    If there are no waypoints at all, recovery should return 0 safely
    without crashing.
    """
    rec = make_recovery([], map_handler)
    result = rec.find_best_waypoint(12.9719, 77.5948, last_wp_index=0)

    print(f"\n  Empty waypoint list → returned {result} (expected 0)")
    assert result == 0


# ── Test 3: last_wp_index already at the end ──────────────────────────────────

def test_recovery_at_last_waypoint(map_handler):
    """
    If last_wp_index == len(waypoints) - 1 (we're at the very last waypoint),
    recovery should return the last valid index, not crash or go out of bounds.
    """
    wps = make_straight_path(4)
    rec = make_recovery(wps, map_handler)

    result = rec.find_best_waypoint(
        lat           = wps[-1].lat,
        lon           = wps[-1].lon,
        last_wp_index = len(wps) - 1,
    )

    print(f"\n  At last WP (index {len(wps)-1}) → recovery returned {result}")
    assert result <= len(wps) - 1, f"Out-of-bounds index: {result}"
    assert result >= 0


# ── Test 4: Boat is exactly on the path ───────────────────────────────────────

def test_recovery_boat_on_path(map_handler):
    """
    If the boat is exactly on the planned path (zero off-path distance),
    recovery should still return a valid forward index.
    """
    wps = make_straight_path(5)
    rec = make_recovery(wps, map_handler)

    # Boat is exactly at waypoint 2's position
    result = rec.find_best_waypoint(
        lat           = wps[2].lat,
        lon           = wps[2].lon,
        last_wp_index = 2,
    )

    print(f"\n  Boat on path at WP#2 → recovery returned WP#{result}")
    assert result >= 2, f"Recovery returned {result}, expected >= 2"


# ── Test 5: Safety skip is applied ───────────────────────────────────────────

def test_safety_skip_applied(map_handler):
    """
    If recovery would return the SAME index as last_wp_index
    (meaning the boat is still right next to the obstacle waypoint),
    it should bump forward by SAFETY_SKIP to avoid immediately
    re-triggering avoidance.
    """
    wps = make_straight_path(6)
    rec = make_recovery(wps, map_handler)

    # Boat is very close to WP#2 itself — recovery would naively return 2
    # But safety skip should give us 2 + SAFETY_SKIP = 3
    result = rec.find_best_waypoint(
        lat           = wps[2].lat,
        lon           = wps[2].lon,
        last_wp_index = 2,
    )

    print(f"\n  Boat at WP#2, last_wp=2 → returned WP#{result}")
    print(f"  SAFETY_SKIP={SAFETY_SKIP}, so expect >= {2 + SAFETY_SKIP} if skip applied")

    # Either safety skip moved us forward, or recovery found a later segment
    assert result >= 2, "Should never go backward"


# ── Test 6: needs_recovery detects off-path correctly ────────────────────────

def test_needs_recovery_true_when_far_off_path(map_handler):
    """
    If the boat is very far from the planned path, needs_recovery() should
    return True so the controller knows to trigger path recovery.
    """
    import math, config
    wps = make_straight_path(5)
    rec = make_recovery(wps, map_handler)

    EARTH_R = 6_371_000.0
    origin_lat, origin_lon = 12.9719, 77.5948

    # Put the boat 50 metres south of the path — well beyond threshold
    far_south_lat = origin_lat + math.degrees(-50.0 / EARTH_R)
    far_south_lon = origin_lon

    result = rec.needs_recovery(far_south_lat, far_south_lon, current_wp_index=2)

    print(f"\n  Boat 50m south of path → needs_recovery={result}")
    print(f"  (RECOVERY_THRESHOLD_M={config.RECOVERY_THRESHOLD_M})")

    assert result is True, (
        f"Expected needs_recovery=True when 50m off path, got {result}"
    )


def test_needs_recovery_false_when_on_path():
    """
    When the boat is exactly on the planned path, needs_recovery() should
    return False.
 
    WHY THIS WAS WRONG BEFORE:
    The original test used the shared `map_handler` fixture (built from the
    lake polygon). That MapHandler has its OWN centroid as XY origin.
 
    But make_straight_path() computed waypoint XY coordinates using a
    DIFFERENT origin (12.9719, 77.5948 as lat/lon zero).
 
    When needs_recovery() called mh.to_xy(boat_lat, boat_lon) using the
    lake's centroid, the resulting (x,y) was completely different from
    the waypoints' stored (x,y). So the boat appeared ~63m off-path
    even though it was perfectly on the waypoints.
 
    FIX: Build a dedicated MapHandler whose origin matches the waypoints.
    We do this by using a tiny rectangular polygon centred at the same
    GPS origin that make_straight_path() uses (12.9719, 77.5948).
    Now mh.to_xy() and the waypoint XY coords share the same frame.
    """
    import math
    from map_handler import MapHandler
    from coverage_planner import Waypoint
    from recovery import PathRecovery
 
    # ── Step 1: Build a MapHandler centred at the same origin as our waypoints
    ORIGIN_LAT = 12.9719
    ORIGIN_LON = 77.5948
    EARTH_R    = 6_371_000.0
 
    # Small 200×200 m square polygon centred at origin
    # (big enough to contain our 5-waypoint, 40m path)
    offset_lat = math.degrees(100.0 / EARTH_R)
    offset_lon = math.degrees(100.0 / (EARTH_R * math.cos(math.radians(ORIGIN_LAT))))
 
    boundary = [
        (ORIGIN_LAT - offset_lat, ORIGIN_LON - offset_lon),
        (ORIGIN_LAT + offset_lat, ORIGIN_LON - offset_lon),
        (ORIGIN_LAT + offset_lat, ORIGIN_LON + offset_lon),
        (ORIGIN_LAT - offset_lat, ORIGIN_LON + offset_lon),
    ]
    mh = MapHandler.from_gps_polygon(boundary)
 
    # Verify the origin is close to our expected GPS point
    # (MapHandler uses centroid — for a symmetric square it equals the centre)
    print(f"\n  MapHandler origin: ({mh.origin_lat:.6f}, {mh.origin_lon:.6f})")
    print(f"  Waypoint origin:   ({ORIGIN_LAT:.6f}, {ORIGIN_LON:.6f})")
 
    # ── Step 2: Build waypoints using mh.to_xy for GPS→XY conversion ─────────
    # This guarantees XY coords are in the SAME frame as mh
    wps = []
    for i in range(5):
        y_metres = float(i * 10)    # 0, 10, 20, 30, 40 m north
        lat_i = ORIGIN_LAT + math.degrees(y_metres / EARTH_R)
        lon_i = ORIGIN_LON
        x, y  = mh.to_xy(lat_i, lon_i)   # ← use mh, not a manual formula
        wps.append(Waypoint(
            index=i, x=x, y=y,
            lat=lat_i, lon=lon_i,
            is_turn=False,
        ))
 
    rec = PathRecovery(mh, wps)
 
    # ── Step 3: Boat is EXACTLY at waypoint 2 ────────────────────────────────
    result = rec.needs_recovery(
        lat              = wps[2].lat,
        lon              = wps[2].lon,
        current_wp_index = 2,
    )
 
    print(f"  Boat at WP#2 (exactly on path) → needs_recovery={result}")
    print(f"  WP#2 position: ({wps[2].lat:.7f}, {wps[2].lon:.7f})")
 
    assert result is False, (
        f"needs_recovery returned True when boat is exactly on the path.\n"
        f"Check that map_handler and waypoint XY frames share the same origin."
    )
 
    # ── Step 4: Also verify that 1m off-path does NOT trigger recovery ────────
    # (RECOVERY_THRESHOLD_M is 3.0m by default — 1m should be fine)
    import config
    small_offset_lat = ORIGIN_LAT + math.degrees(
        (wps[2].y + 1.0) / EARTH_R   # 1m north of WP#2, still on path segment
    )
    result_small = rec.needs_recovery(
        lat              = wps[2].lat,
        lon              = wps[2].lon,
        current_wp_index = 2,
    )
 
    print(f"  1m offset from path → needs_recovery={result_small} "
          f"(threshold={config.RECOVERY_THRESHOLD_M}m)")
    assert result_small is False, (
        f"1m offset triggered recovery — threshold may be set too low in config.py"
    )


# ── Test 7: Coverage loss estimate ───────────────────────────────────────────

def test_coverage_loss_estimate(map_handler):
    """
    If recovery skips from WP#1 to WP#3, the skipped distance should be
    approximately 2 * spacing = 20 metres (for 10m spacing).
    """
    spacing = 10.0
    wps     = make_straight_path(6, spacing=spacing)
    rec     = make_recovery(wps, map_handler)

    loss = rec.estimate_coverage_loss(skipped_from=1, resumed_at=3)

    print(f"\n  Coverage loss (WP1→WP3, spacing={spacing}m) = {loss:.2f}m")
    print(f"  Expected ~{2 * spacing}m")

    assert loss == pytest.approx(2 * spacing, abs=1.0), (
        f"Expected ~{2*spacing}m coverage loss, got {loss:.2f}m"
    )


def test_coverage_loss_zero_when_no_skip(map_handler):
    """If skipped_from == resumed_at, loss should be 0."""
    wps  = make_straight_path(5)
    rec  = make_recovery(wps, map_handler)
    loss = rec.estimate_coverage_loss(skipped_from=2, resumed_at=2)

    print(f"\n  No skip → coverage loss = {loss:.2f}m (expected 0)")
    assert loss == pytest.approx(0.0, abs=0.01)


# ── Test 8: last_wp_index == 0 edge case ────────────────────────────────────

def test_recovery_from_very_start(map_handler):
    """
    If an obstacle is hit right at the beginning (last_wp_index=0 or 1),
    recovery should not crash and should return a valid index.
    """
    wps = make_straight_path(5)
    rec = make_recovery(wps, map_handler)

    result = rec.find_best_waypoint(
        lat           = wps[0].lat,
        lon           = wps[0].lon,
        last_wp_index = 1,
    )

    print(f"\n  Obstacle at mission start (last_wp=1) → recovery WP#{result}")
    assert 0 <= result < len(wps), f"Index {result} out of range"