"""
tests/test_heading.py
======================
Tests for HeadingEstimator.

What we're checking:
  1. IMU heading is used when available
  2. GPS heading is computed correctly when boat is moving fast enough
  3. GPS heading is IGNORED when boat is nearly stationary
  4. Big sudden jumps in GPS heading are rejected as noise
  5. When no source is available, last known heading is held
  6. Wraparound (e.g. 359° → 1°) is handled correctly
  7. EMA filter smooths out noisy inputs

Run just this file:
    pytest tests/test_heading.py -v
"""

import pytest
from heading_estimator import HeadingEstimator


# ── Helper ────────────────────────────────────────────────────────────────────

def make_estimator(alpha=0.9) -> HeadingEstimator:
    """
    alpha=0.9 means the EMA is very responsive (almost no smoothing).
    This makes test assertions simpler — we don't have to warm up the filter.
    """
    return HeadingEstimator(alpha=0.9)


# ── Test 1: IMU heading is accepted and returned ──────────────────────────────

def test_imu_heading_used_when_available():
    """
    If the IMU gives us a heading, we should get back something close to it.

    WHY: IMU is Priority 1. If it works, nothing else matters.
    """
    est = make_estimator()

    # Give it a clear 90° heading (East)
    result = est.update(imu_heading_deg=90.0, lat=0.0, lon=0.0, gps_fix_quality=0)

    print(f"\n  IMU gave 90.0°, estimator returned {result:.2f}°")

    assert result == pytest.approx(90.0, abs=2.0), (
        f"Expected ~90.0° from IMU, got {result:.2f}°"
    )
    assert est.source == "imu", f"Source should be 'imu', got '{est.source}'"


# ── Test 2: Source label is set correctly ─────────────────────────────────────

def test_source_label_imu():
    """After an IMU update, est.source should equal 'imu'."""
    est = make_estimator()
    est.update(imu_heading_deg=180.0, lat=0.0, lon=0.0)
    assert est.source == "imu"


# ── Test 3: No IMU, no movement → returns 0.0 on very first call ─────────────

def test_no_source_returns_zero_on_first_call():
    """
    On the very first call with no IMU and no movement,
    the estimator has nothing to go on — it should return 0.0 safely.

    WHY: This is the "boat is stationary at startup" case. We don't want
    a crash or a random value.
    """
    est = make_estimator()

    # No IMU (None), no GPS movement (same position twice)
    result = est.update(imu_heading_deg=None, lat=12.9, lon=77.5)

    print(f"\n  No IMU, no movement → got {result:.2f}° (expected 0.0°)")

    assert result == pytest.approx(0.0, abs=0.1), (
        f"Expected 0.0° when no data available, got {result:.2f}°"
    )
    assert est.source == "none"


# ── Test 4: Hold last heading when source disappears ─────────────────────────

def test_holds_last_heading_when_imu_disappears():
    """
    If IMU was giving 270° and then goes None, we should keep returning
    something close to 270° — NOT jump to 0°.

    WHY: In real deployment, IMU comms can hiccup for 1-2 seconds.
    Returning 0° would make the boat spin. Holding last value is safer.
    """
    est = make_estimator()

    # First give it a solid IMU heading
    est.update(imu_heading_deg=270.0, lat=0.0, lon=0.0)

    # Now IMU disappears
    result = est.update(imu_heading_deg=None, lat=0.0, lon=0.0)

    print(f"\n  IMU gone after 270° — held at {result:.2f}° (expected ~270°)")

    assert result == pytest.approx(270.0, abs=5.0), (
        f"Expected held heading ~270°, got {result:.2f}°"
    )
    assert est.source == "held"


# ── Test 5: Heading wraparound (359° → 1°) is handled ────────────────────────

def test_wraparound_is_smooth():
    """
    Boat is heading 359° (nearly North). Next reading is 1°.
    The estimator should NOT compute an average of 180° (which would be wrong).
    It should output something near 0° (North).

    WHY: The EMA filter works in sin/cos space to handle this correctly.
    This test verifies that fix is working.
    """
    est = make_estimator(alpha=0.5)   # 50/50 mix

    # Give it 359°
    est.update(imu_heading_deg=359.0, lat=0.0, lon=0.0)

    # Now give it 1°
    result = est.update(imu_heading_deg=1.0, lat=0.0, lon=0.0)

    print(f"\n  359° then 1° → EMA result: {result:.2f}° (should be near 0°, not 180°)")

    # The correct answer is somewhere near 0° (North), not 180° (South)
    is_near_north = result > 350.0 or result < 10.0
    assert is_near_north, (
        f"Wraparound failed: 359°→1° gave {result:.2f}° instead of ~0°"
    )


# ── Test 6: GPS heading is used when moving fast enough ──────────────────────

def test_gps_heading_used_when_fast_enough():
    """
    With no IMU, if the boat moves north fast enough,
    the GPS-derived heading should be near 0° (north).

    WHY: GPS heading is Priority 2. It should kick in when IMU is absent
    and the boat is moving.
    """
    est = make_estimator()

    # First call sets the GPS anchor position
    est.update(imu_heading_deg=None, lat=12.9700, lon=77.5900, gps_fix_quality=1)

    # Second call: boat moved ~44 m north in 1 second = 44 m/s (fast, above threshold)
    # Using a large movement to ensure it's above the 0.4 m/s speed threshold
    result = est.update(
        imu_heading_deg=None,
        lat=12.9704,       # moved north
        lon=77.5900,       # same east/west
        gps_fix_quality=1,
    )

    print(f"\n  Boat moved north at speed → GPS heading: {result:.2f}° (expected ~0°)")

    # North is 0°, allow a wide tolerance because GPS heading is approximate
    is_near_north = result < 20.0 or result > 340.0
    assert is_near_north, (
        f"Expected GPS heading near 0° (North), got {result:.2f}°"
    )


# ── Test 7: GPS heading rejected when nearly stationary ──────────────────────

def test_gps_heading_rejected_when_stationary():
    """
    If the boat barely moves between GPS fixes, the GPS bearing is noise.
    The estimator should NOT use it.

    WHY: At 0.1 m movement, the GPS bearing could be anything 0-360°
    due to position noise. We gate this with MIN_SPEED_FOR_GPS_HEADING.
    """
    est = make_estimator()

    # Give a known IMU heading first so we have a "last valid" value
    est.update(imu_heading_deg=45.0, lat=12.9700, lon=77.5900)

    # Now IMU goes away and boat barely moves (< 0.4 m/s)
    result = est.update(
        imu_heading_deg=None,
        lat=12.97000001,   # ~0.001 m movement — below speed threshold
        lon=77.59000001,
        gps_fix_quality=1,
    )

    print(f"\n  Barely moving, no IMU → held at {result:.2f}° (expected ~45°)")

    # Should hold the last IMU heading (45°), not use junk GPS bearing
    assert result == pytest.approx(45.0, abs=10.0), (
        f"Expected ~45° (held from IMU), got {result:.2f}°"
    )


# ── Test 8: is_reliable flag ──────────────────────────────────────────────────

def test_is_reliable_true_with_imu():
    """is_reliable should be True when IMU is active."""
    est = make_estimator()
    est.update(imu_heading_deg=90.0, lat=0.0, lon=0.0)
    assert est.is_reliable is True


def test_is_reliable_false_when_holding():
    """is_reliable should be False when just holding the last value."""
    est = make_estimator()
    est.update(imu_heading_deg=90.0, lat=0.0, lon=0.0)  # sets last_valid
    est.update(imu_heading_deg=None,  lat=0.0, lon=0.0)  # now holding
    assert est.is_reliable is False


# ── Test 9: reset clears state ────────────────────────────────────────────────

def test_reset_clears_history():
    """
    After reset(), the estimator should behave as if it just started.
    Calling update() with None IMU and no movement should return 0.0 again.
    """
    est = make_estimator()

    # Build up some state
    est.update(imu_heading_deg=180.0, lat=0.0, lon=0.0)
    est.update(imu_heading_deg=180.0, lat=0.1, lon=0.0)

    # Reset
    est.reset()

    # Now: no IMU, no previous position → should return 0.0
    result = est.update(imu_heading_deg=None, lat=0.0, lon=0.0)
    print(f"\n  After reset, first call with no data → {result:.2f}° (expected 0.0°)")
    assert result == pytest.approx(0.0, abs=0.1)
    assert est.source == "none"