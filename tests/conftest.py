"""
tests/conftest.py
==================
Shared setup that pytest loads automatically before any test.

You never import this file yourself — pytest finds it automatically.

What this does:
  1. Turns on logging so you can see log messages during tests
  2. Provides reusable "fixtures" — little helper objects that
     your tests can ask for by name in their function arguments.

What is a fixture?
  A fixture is a function that creates something your test needs
  (like a fake boat, or a heading estimator) and hands it over.
  pytest handles creating and cleaning up fixtures automatically.
"""

import sys
import os
import logging
import pytest

# ── Make sure the project root is on the Python path ─────────────────────────
# This lets tests import hardware.py, sensor_hub.py, etc. directly
# without needing to install the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Turn on logging during tests so you can see what's happening ──────────────
def pytest_configure(config):
    """Called once by pytest before any tests run."""
    logging.basicConfig(
        level   = logging.DEBUG,
        format  = "  %(name)-20s [%(levelname)-8s] %(message)s",
        stream  = sys.stdout,
    )


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_hw():
    """
    A simulated boat starting at a known GPS position.
    Tests that need a hardware object ask for 'mock_hw' as an argument.

    Example usage in a test:
        def test_something(mock_hw):
            mock_hw.set_motor_speed(0.5, 0.5)
            lat, lon = mock_hw.get_gps()
    """
    from hardware import MockHardware
    hw = MockHardware(
        init_lat      = 12.971920,
        init_lon      = 77.594800,
        init_heading  = 0.0,
        boat_speed_mps= 2.0,
        turn_rate_dps = 45.0,
    )
    return hw


@pytest.fixture
def heading_estimator():
    """
    A fresh HeadingEstimator with default settings.

    Example usage:
        def test_imu_heading(heading_estimator):
            result = heading_estimator.update(imu_heading_deg=90.0, lat=0, lon=0)
            assert result == pytest.approx(90.0, abs=1.0)
    """
    from heading_estimator import HeadingEstimator
    return HeadingEstimator(alpha=0.4)


@pytest.fixture
def simple_waypoints():
    """
    A small straight-line list of 5 waypoints for testing recovery.
    Returns a list of Waypoint objects spaced 10 metres apart going north.
    """
    from coverage_planner import Waypoint
    wps = []
    for i in range(5):
        y = float(i * 10)          # 0, 10, 20, 30, 40 metres north
        lat = 12.971920 + (y / 6_371_000.0) * (180.0 / 3.14159)
        wps.append(Waypoint(
            index   = i,
            x       = 0.0,
            y       = y,
            lat     = lat,
            lon     = 77.594800,
            is_turn = False,
        ))
    return wps


@pytest.fixture
def map_handler():
    """A MapHandler built from a small test polygon."""
    from map_handler import MapHandler
    # Small rectangular lake boundary (~120 x 80 m)
    boundary = [
        (12.97192, 77.59480),
        (12.97230, 77.59510),
        (12.97265, 77.59525),
        (12.97290, 77.59510),
        (12.97295, 77.59470),
        (12.97270, 77.59440),
        (12.97230, 77.59430),
        (12.97200, 77.59445),
    ]
    return MapHandler.from_gps_polygon(boundary)