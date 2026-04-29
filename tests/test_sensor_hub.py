"""
tests/test_sensor_hub.py
=========================
Tests for SensorHub — the thread-safe producer-consumer bridge.

What we're checking:
  1. snapshot() returns valid data after producers update it
  2. GPS is marked stale after the timeout expires
  3. Obstacle signal is stored and retrieved correctly
  4. Obstacle signal auto-clears after OBS_STALE_S seconds
  5. Injecting an obstacle signal from outside is thread-safe
  6. SensorHub can be started and stopped without hanging

IMPORTANT: These tests use time.sleep() sparingly because producer
threads run at 10-20 Hz. Tests wait just long enough for one cycle,
then check the result.

Run just this file:
    pytest tests/test_sensor_hub.py -v --timeout=10
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from sensor_hub import SensorHub, SensorSnapshot, GPS_STALE_S, OBS_STALE_S
from heading_estimator import HeadingEstimator


# ── Helper: build a SensorHub wired to a MockHardware ────────────────────────

def make_hub(init_lat=12.9719, init_lon=77.5948):
    """
    Create a SensorHub backed by a real MockHardware.
    Does NOT call hub.start() — tests that need threads call it explicitly.
    """
    from hardware import MockHardware
    hw  = MockHardware(init_lat=init_lat, init_lon=init_lon)
    est = HeadingEstimator(alpha=0.9)
    hub = SensorHub(hardware=hw, heading_estimator=est)
    return hub, hw


# ── Test 1: snapshot() returns a SensorSnapshot object ───────────────────────

def test_snapshot_returns_correct_type():
    """
    snapshot() should always return a SensorSnapshot dataclass,
    even before any producer thread has run.

    WHY: The control loop calls snapshot() every tick. It must never
    crash or return None.
    """
    hub, _ = make_hub()
    snap   = hub.snapshot()

    print(f"\n  snapshot() type: {type(snap).__name__}")
    assert isinstance(snap, SensorSnapshot), (
        f"Expected SensorSnapshot, got {type(snap).__name__}"
    )


# ── Test 2: Initial snapshot has no valid GPS ─────────────────────────────────

def test_initial_snapshot_has_no_gps():
    """
    Before any producer thread runs, GPS should be invalid (no fix yet).
    lat and lon should be None.

    WHY: We should never navigate with uninitialized GPS data.
    """
    hub, _ = make_hub()
    snap   = hub.snapshot()

    print(f"\n  Initial GPS valid={snap.gps_valid}, lat={snap.lat}, lon={snap.lon}")

    assert snap.gps_valid is False
    assert snap.lat is None
    assert snap.lon is None


# ── Test 3: GPS data appears after producer thread runs ──────────────────────

@pytest.mark.timeout(5)
def test_gps_available_after_producer_runs():
    """
    After start() and a brief wait, the GPS producer thread should have
    polled MockHardware and the snapshot should show a valid GPS fix.

    WHY: Core feature — the producer-consumer pattern must actually work.
    """
    hub, hw = make_hub(init_lat=12.9719, init_lon=77.5948)
    hub.start()

    # Wait up to 2 seconds for the GPS producer to fire at least once (10 Hz)
    deadline = time.monotonic() + 2.0
    snap     = hub.snapshot()
    while not snap.gps_valid and time.monotonic() < deadline:
        time.sleep(0.15)
        snap = hub.snapshot()

    hub.stop()

    print(f"\n  After producer thread: GPS valid={snap.gps_valid}")
    print(f"  lat={snap.lat:.6f}, lon={snap.lon:.6f}")

    assert snap.gps_valid is True, "GPS should be valid after producer runs"
    assert snap.lat  == pytest.approx(12.9719, abs=0.0001)
    assert snap.lon  == pytest.approx(77.5948, abs=0.0001)


# ── Test 4: Heading appears after producer runs ───────────────────────────────

@pytest.mark.timeout(5)
def test_heading_available_after_producer_runs():
    """
    The heading producer runs at 20 Hz. After start() + a short wait,
    heading_valid should be True and heading_deg should be a number.
    """
    hub, hw = make_hub()
    hub.start()

    time.sleep(0.3)   # give the 20 Hz producer 6 cycles
    snap = hub.snapshot()
    hub.stop()

    print(f"\n  Heading valid={snap.heading_valid}, deg={snap.heading_deg:.1f}°")
    print(f"  Source: {snap.heading_source}")

    assert snap.heading_valid is True, "Heading should be valid after producer runs"
    assert isinstance(snap.heading_deg, float)
    assert 0.0 <= snap.heading_deg < 360.0


# ── Test 5: Obstacle signal injection ────────────────────────────────────────

def test_inject_obstacle_signal():
    """
    inject_obstacle_signal() stores the signal so the next snapshot()
    reflects it immediately — no thread needed, this is a direct write.

    WHY: The sensor bridge or controller may inject signals from any thread.
    We need to verify this works.
    """
    hub, _ = make_hub()

    hub.inject_obstacle_signal("OBSTACLE_FRONT")
    snap = hub.snapshot()

    print(f"\n  Injected OBSTACLE_FRONT → snapshot has: '{snap.obstacle_signal}'")

    assert snap.obstacle_signal == "OBSTACLE_FRONT"


def test_inject_signal_is_uppercased():
    """Signal strings are uppercased on injection regardless of input case."""
    hub, _ = make_hub()
    hub.inject_obstacle_signal("obstacle_left")
    snap = hub.snapshot()

    print(f"\n  Injected 'obstacle_left' → stored as '{snap.obstacle_signal}'")
    assert snap.obstacle_signal == "OBSTACLE_LEFT"


# ── Test 6: Obstacle signal clears ───────────────────────────────────────────

def test_clear_obstacle_signal():
    """
    After clear_obstacle_signal(), the next snapshot should show "NONE".
    """
    hub, _ = make_hub()
    hub.inject_obstacle_signal("OBSTACLE_RIGHT")
    hub.clear_obstacle_signal()
    snap = hub.snapshot()

    print(f"\n  After clear → obstacle_signal='{snap.obstacle_signal}'")
    assert snap.obstacle_signal == "NONE"


# ── Test 7: Stale GPS is reported correctly ───────────────────────────────────

def test_stale_gps_flag():
    """
    Manually write a GPS timestamp far in the past so it appears stale.
    snapshot() should then report gps_valid=False.

    WHY: We don't want to wait GPS_STALE_S (3 seconds) in a real test.
    We simulate the passage of time by backdating the timestamp.
    """
    hub, _ = make_hub()

    # Manually inject a GPS fix but with an old timestamp
    with hub._lock:
        hub._lat     = 12.9719
        hub._lon     = 77.5948
        hub._gps_fix = 1
        hub._gps_ts  = time.monotonic() - (GPS_STALE_S + 1.0)  # expired

    snap = hub.snapshot()

    print(f"\n  GPS timestamp expired → gps_valid={snap.gps_valid} (expected False)")
    print(f"  GPS age = {snap.gps_age_s:.1f}s (stale threshold = {GPS_STALE_S}s)")

    assert snap.gps_valid is False
    assert snap.gps_age_s > GPS_STALE_S


# ── Test 8: Auto-clear of stale obstacle signal ───────────────────────────────

def test_stale_obstacle_signal_auto_clears():
    """
    If an obstacle signal is older than OBS_STALE_S, snapshot() should
    automatically clear it to "NONE".

    WHY: If the sensor bridge crashes mid-mission, we don't want the boat
    to keep thinking there's an obstacle forever.
    """
    hub, _ = make_hub()

    hub.inject_obstacle_signal("OBSTACLE_FRONT")

    # Backdate the obstacle timestamp so it appears stale
    with hub._lock:
        hub._obstacle_ts = time.monotonic() - (OBS_STALE_S + 1.0)

    snap = hub.snapshot()

    print(f"\n  Stale obstacle signal → auto-cleared to '{snap.obstacle_signal}'")
    print(f"  Obstacle age = {snap.obstacle_age_s:.1f}s (threshold = {OBS_STALE_S}s)")

    assert snap.obstacle_signal == "NONE", (
        f"Expected 'NONE' after auto-clear, got '{snap.obstacle_signal}'"
    )


# ── Test 9: snapshot() timestamp is recent ───────────────────────────────────

def test_snapshot_timestamp_is_fresh():
    """
    The timestamp on the snapshot should be within 0.1 seconds of now.
    This confirms the snapshot is freshly created, not cached.
    """
    hub, _ = make_hub()
    before = time.monotonic()
    snap   = hub.snapshot()
    after  = time.monotonic()

    print(f"\n  Snapshot timestamp age: {after - snap.timestamp:.4f}s")
    assert before <= snap.timestamp <= after + 0.01


# ── Test 10: start() and stop() don't hang ───────────────────────────────────

@pytest.mark.timeout(5)
def test_start_stop_does_not_hang():
    """
    Calling start() then stop() should complete cleanly without
    the test hanging. This checks thread lifecycle is correct.

    WHY: daemon threads should die when stop() sets self._running=False.
    If they don't, the test framework hangs and pytest-timeout kills it
    after 5 seconds — which itself is a test failure signal.
    """
    hub, _ = make_hub()
    hub.start()
    time.sleep(0.2)
    hub.stop()

    print("\n  start() then stop() completed without hanging ✓")
    assert hub._running is False


# ── Test 11: status_line() returns a string ───────────────────────────────────

def test_status_line_returns_string():
    """
    status_line() is called in logs. It must never crash and must
    return a non-empty string.
    """
    hub, _ = make_hub()
    line   = hub.status_line()

    print(f"\n  status_line(): '{line}'")
    assert isinstance(line, str)
    assert len(line) > 0
    assert "GPS" in line
    assert "HDG" in line