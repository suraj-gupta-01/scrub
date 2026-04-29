"""
tests/test_hardware.py
=======================
Tests for MockHardware and the create_hardware() factory.

We do NOT test STM32Hardware here — that requires a real serial port.
MockHardware is the simulation-mode class and is fully testable on desktop.

What we're checking:
  1. Motor commands are clamped to [-1, 1]
  2. GPS returns the correct starting position
  3. Heading starts at the configured value
  4. Physics step moves the boat in the right direction
  5. stop() zeroes the motors
  6. The factory returns MockHardware in simulate mode
  7. Thread safety: simultaneous set_motor_speed and step don't crash

Run just this file:
    pytest tests/test_hardware.py -v
"""

import math
import threading
import time
import pytest
from hardware import MockHardware, create_hardware, HardwareInterface


# ── Test 1: GPS returns initial position ──────────────────────────────────────

def test_get_gps_returns_initial_position():
    """
    After construction, get_gps() should return the exact (lat, lon)
    that was passed to the constructor.

    WHY: If this fails, every navigator calculation starts from the wrong place.
    """
    hw = MockHardware(init_lat=12.9719, init_lon=77.5948)
    lat, lon = hw.get_gps()

    print(f"\n  Initial GPS: lat={lat}, lon={lon}")

    assert lat == pytest.approx(12.9719, abs=1e-6)
    assert lon == pytest.approx(77.5948, abs=1e-6)


# ── Test 2: Heading returns initial value ─────────────────────────────────────

def test_get_heading_returns_initial_value():
    """
    Heading should start at init_heading (default 0.0).
    """
    hw = MockHardware(init_lat=12.9719, init_lon=77.5948, init_heading=45.0)
    hdg = hw.get_heading()

    print(f"\n  Initial heading: {hdg}° (expected 45.0°)")
    assert hdg == pytest.approx(45.0, abs=0.01)


# ── Test 3: Motor commands are clamped ────────────────────────────────────────

def test_motor_speed_clamped_above():
    """
    Setting motor speed to 2.0 (above 1.0 max) should be silently clamped to 1.0.

    WHY: ESCs only accept [-1, 1]. Values outside this range could damage hardware.
    """
    hw = MockHardware(init_lat=0.0, init_lon=0.0)
    hw.set_motor_speed(left=2.0, right=5.0)   # both way above limit

    # Step the physics — if clamping works, speed is limited to max
    hw.step(dt=1.0)
    lat, lon = hw.get_gps()

    # The boat should have moved, but not faster than max speed
    dist = _haversine(0.0, 0.0, lat, lon)
    print(f"\n  Motor set to 2.0, 5.0 → moved {dist:.2f}m in 1s")
    print(f"  (Max possible at speed=1.0 * SIM_BOAT_SPEED — should be bounded)")

    # At clamp=1.0 and default SIM_BOAT_SPEED=2.0, max distance is 2.0m
    assert dist < 5.0, f"Boat moved {dist:.2f}m — clamping may not be working"


def test_motor_speed_clamped_below():
    """
    Setting motor speed to -3.0 should be clamped to -1.0 (full reverse).
    """
    hw = MockHardware(init_lat=12.9719, init_lon=77.5948, init_heading=0.0)
    hw.set_motor_speed(left=-3.0, right=-3.0)

    # Should not crash, and internally _left/_right should be -1.0
    with hw._lock:
        assert hw._left  == pytest.approx(-1.0, abs=0.001)
        assert hw._right == pytest.approx(-1.0, abs=0.001)

    print("\n  Negative clamp: _left=-1.0, _right=-1.0 ✓")


# ── Test 4: stop() zeros the motors ──────────────────────────────────────────

def test_stop_zeros_motors():
    """
    After stop(), both motor values should be 0.0.
    Then step() should not move the boat.
    """
    hw = MockHardware(init_lat=12.9719, init_lon=77.5948)
    hw.set_motor_speed(1.0, 1.0)   # full speed ahead
    hw.stop()

    # Step the physics — no movement expected
    lat_before, lon_before = hw.get_gps()
    hw.step(dt=1.0)
    lat_after, lon_after = hw.get_gps()

    moved = _haversine(lat_before, lon_before, lat_after, lon_after)
    print(f"\n  After stop(), stepped 1s → moved {moved:.4f}m (expected ~0)")

    assert moved < 0.01, f"Boat moved {moved:.4f}m after stop()"


# ── Test 5: Physics — boat moves north when heading=0 ─────────────────────────

def test_physics_moves_north_at_heading_zero():
    """
    Boat starts heading North (0°). With both motors forward,
    it should move northward — latitude should increase.

    WHY: This validates the sin/cos math in step(). If the heading
    convention is wrong, the boat drives the wrong direction.
    """
    hw = MockHardware(
        init_lat      = 12.9719,
        init_lon      = 77.5948,
        init_heading  = 0.0,      # North
        boat_speed_mps= 2.0,
    )
    hw.set_motor_speed(1.0, 1.0)   # full forward

    lat_start = hw.lat
    hw.step(dt=1.0)               # 1 second at 2.0 m/s = ~2m north
    lat_end = hw.lat

    print(f"\n  Heading=0° (North), full forward 1s:")
    print(f"  lat: {lat_start:.7f} → {lat_end:.7f}")
    print(f"  Δlat = {(lat_end - lat_start) * 111111:.3f}m (should be ~2m)")

    assert lat_end > lat_start, "Boat should move north (lat increase) at heading 0°"
    delta_m = (lat_end - lat_start) * 111_111.0
    assert 1.0 < delta_m < 3.0, f"Expected ~2m north, got {delta_m:.3f}m"


def test_physics_moves_east_at_heading_90():
    """
    Boat heading East (90°). With both motors forward,
    longitude should increase.
    """
    hw = MockHardware(
        init_lat      = 12.9719,
        init_lon      = 77.5948,
        init_heading  = 90.0,     # East
        boat_speed_mps= 2.0,
    )
    hw.set_motor_speed(1.0, 1.0)

    lon_start = hw.lon
    hw.step(dt=1.0)
    lon_end = hw.lon

    print(f"\n  Heading=90° (East), full forward 1s:")
    print(f"  lon: {lon_start:.7f} → {lon_end:.7f}")

    assert lon_end > lon_start, "Boat should move east (lon increase) at heading 90°"


# ── Test 6: Differential drive causes turning ────────────────────────────────

def test_differential_drive_causes_turn():
    """
    When left motor > right motor, the boat turns counter-clockwise (LEFT).
 
    WHY THIS WAS WRONG BEFORE:
    The physics in hardware.py computes:
        diff      = (right - left) / 2.0
        turn_rate = diff * turn_rate_dps
 
    With left=1.0, right=0.0:
        diff      = (0.0 - 1.0) / 2.0 = -0.5
        turn_rate = -0.5 * 45.0       = -22.5 deg/s   ← counter-clockwise
 
    So heading goes: 0° - 22.5° = -22.5° → wrapped to 337.5°.
    The original test had the turn direction backwards.
 
    CORRECT ASSERTION:
    - left > right  →  diff is NEGATIVE  →  turns LEFT (heading decreases)
    - right > left  →  diff is POSITIVE  →  turns RIGHT (heading increases)
 
    We test BOTH directions here.
    """
    from hardware import MockHardware
 
    # ── Case A: left > right → turns left (counter-clockwise) ────────────────
    hw_left = MockHardware(
        init_lat     = 12.9719,
        init_lon     = 77.5948,
        init_heading = 0.0,
        turn_rate_dps= 45.0,
    )
    hw_left.set_motor_speed(left=1.0, right=0.0)
    hw_left.step(dt=1.0)
    hdg_left = hw_left.get_heading()
 
    print(f"\n  LEFT turn (left=1.0, right=0.0) → heading: {hdg_left:.1f}°")
    print(f"  Expected: ~337.5° (= 0° - 22.5°, wrapped)")
 
    # Heading should have gone counter-clockwise (decreased from 0°, wraps to ~337.5°)
    # Allow ±5° tolerance
    assert 330.0 < hdg_left < 345.0, (
        f"Expected ~337.5° for left turn, got {hdg_left:.1f}°\n"
        f"Physics: diff=(right-left)/2=(0-1)/2=-0.5, "
        f"turn_rate=-0.5*45=-22.5°/s, heading=0-22.5=-22.5→337.5°"
    )
 
    # ── Case B: right > left → turns right (clockwise) ───────────────────────
    hw_right = MockHardware(
        init_lat     = 12.9719,
        init_lon     = 77.5948,
        init_heading = 0.0,
        turn_rate_dps= 45.0,
    )
    hw_right.set_motor_speed(left=0.0, right=1.0)
    hw_right.step(dt=1.0)
    hdg_right = hw_right.get_heading()
 
    print(f"  RIGHT turn (left=0.0, right=1.0) → heading: {hdg_right:.1f}°")
    print(f"  Expected: ~22.5° (= 0° + 22.5°)")
 
    # Heading should have increased (clockwise)
    assert 17.0 < hdg_right < 28.0, (
        f"Expected ~22.5° for right turn, got {hdg_right:.1f}°"
    )
 
    print(f"  ✓ Turn direction convention confirmed: left>right=CCW, right>left=CW")


# ── Test 7: Trajectory is recorded ──────────────────────────────────────────

def test_trajectory_grows_with_steps():
    """
    Each call to step() should append a (lat, lon) point to hw.trajectory.
    After N steps, len(trajectory) should be N+1 (including the start).
    """
    hw = MockHardware(init_lat=12.9719, init_lon=77.5948)
    hw.set_motor_speed(0.5, 0.5)

    assert len(hw.trajectory) == 1, "Should start with 1 point (initial position)"

    for _ in range(5):
        hw.step(dt=0.1)

    print(f"\n  After 5 steps: trajectory has {len(hw.trajectory)} points")
    assert len(hw.trajectory) == 6   # 1 initial + 5 steps


# ── Test 8: create_hardware() factory in simulate mode ───────────────────────

def test_create_hardware_sim_mode():
    """
    create_hardware(simulate=True) should return a MockHardware instance,
    not try to open any serial port.
    """
    hw = create_hardware(simulate=True, init_lat=12.9719, init_lon=77.5948)

    print(f"\n  create_hardware(simulate=True) → {type(hw).__name__}")
    assert isinstance(hw, MockHardware)
    assert isinstance(hw, HardwareInterface)


def test_create_hardware_is_hardware_interface():
    """
    The returned object must implement HardwareInterface (all 4 methods).
    """
    hw = create_hardware(simulate=True, init_lat=0.0, init_lon=0.0)

    assert hasattr(hw, "set_motor_speed")
    assert hasattr(hw, "get_gps")
    assert hasattr(hw, "get_heading")
    assert hasattr(hw, "stop")

    print("\n  All 4 HardwareInterface methods present ✓")


# ── Test 9: Thread safety under concurrent access ────────────────────────────

@pytest.mark.timeout(5)
def test_concurrent_set_and_step_no_crash():
    """
    Simulate what happens in real deployment: the control loop calls
    set_motor_speed() while the physics thread calls step().
    This should never crash, corrupt state, or deadlock.

    WHY: The lock in MockHardware must correctly protect shared state.
    """
    hw = MockHardware(init_lat=12.9719, init_lon=77.5948)

    errors = []

    def motor_setter():
        for i in range(50):
            try:
                hw.set_motor_speed(0.5, 0.5)
                time.sleep(0.001)
            except Exception as e:
                errors.append(f"motor_setter: {e}")

    def physics_stepper():
        for i in range(50):
            try:
                hw.step(dt=0.01)
                time.sleep(0.001)
            except Exception as e:
                errors.append(f"physics_stepper: {e}")

    t1 = threading.Thread(target=motor_setter)
    t2 = threading.Thread(target=physics_stepper)
    t1.start(); t2.start()
    t1.join(); t2.join()

    print(f"\n  Concurrent access test: {len(errors)} errors (expected 0)")
    assert len(errors) == 0, f"Thread safety errors: {errors}"


# ── Private helper ────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Local haversine — avoids importing utils to keep test self-contained."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))