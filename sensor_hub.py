"""
ASV Navigation System - Sensor Hub
====================================
Thread-safe producer-consumer bridge between hardware (sensor threads)
and the mission control loop (consumer).

WHY THIS EXISTS
---------------
PROBLEM: The original controller.py runs a tight 10 Hz loop using
time.sleep(). It calls hw.get_gps() and hw.get_heading() inline.

In the STM32Hardware backend these are fast non-blocking reads from a
shared dict (good). But consider what happens when:

  a) Serial readline() inside the reader thread stalls (USB hiccup,
     buffer overflow, STM32 reset). The reader thread hangs. The main
     loop still calls get_gps() but gets stale data from 3+ seconds ago.
     The stale-check returns None. The main loop waits. The boat coasts.

  b) A future sensor is added (e.g. YOLO object detection on a camera
     frame). YOLO inference takes 80-200 ms on RPi5. If called inside
     the main loop, the 10 Hz timing collapses to ~5 Hz. Motor commands
     become irregular. The heading controller becomes unstable.

  c) Two threads (heartbeat + main loop) call _send() simultaneously
     on the same serial port. The current code has no send-side lock,
     so partial writes corrupt the STM32's command parser.

SOLUTION: Producer-Consumer with a SensorHub
---------------------------------------------
Each sensor runs in its own daemon thread at its natural update rate.
The SensorHub is the ONLY shared object. It holds the most recent
snapshot of all sensor data behind a single RLock.

The main control loop calls sensor_hub.snapshot() — one fast lock
acquire — and gets a frozen copy of all sensor data for that tick.
No blocking on serial. No stale-data surprises. Adding a new sensor
means adding one producer thread; the control loop is untouched.

                ┌─────────────────────────────────────────────────────┐
                │                  SensorHub                          │
                │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
                │  │  GPS     │  │ Heading  │  │ Obstacle signal  │   │
                │  │ producer │  │ producer │  │   producer       │   │
                │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘   │
                │       │             │                 │             │
                │       └─────────────┴─────────────────┘             │
                │                     │  RLock                        │
                │             shared SensorState                      │
                └─────────────────────┬───────────────────────────────┘
                                      │ snapshot() — non-blocking
                              ┌───────▼───────┐
                              │  Control Loop │   10 Hz, deterministic
                              │  (controller) │
                              └───────────────┘

THREADING MODEL
---------------
• One RLock guards all writes AND the snapshot read.
  RLock (re-entrant) is used instead of Lock so that a producer thread
  can call update methods multiple times without deadlocking itself.

• Producers never block the consumer for more than microseconds
  (just the time to copy a few floats into the state dict).

• The control loop never waits on I/O. It calls snapshot() and moves on.

• HeadingEstimator (Problem 1 fix) is called inside the heading producer
  thread at the IMU update rate — NOT inside the 10 Hz control loop.
  This means the EMA filter runs at full sensor rate (e.g. 50 Hz IMU)
  and the control loop just reads the already-filtered result.
"""

import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

from heading_estimator import HeadingEstimator

log = logging.getLogger(__name__)

# ── How long before a sensor value is considered stale ───────────────────────
GPS_STALE_S     = 3.0
HEADING_STALE_S = 3.0
OBS_STALE_S     = 10.0   # obstacle signal auto-clears after this long


@dataclass
class SensorSnapshot:
    """
    Immutable snapshot of all sensor data for one control loop tick.
    The control loop works exclusively with this — never touches raw hardware.
    """
    # GPS
    lat:            Optional[float] = None
    lon:            Optional[float] = None
    gps_fix:        int             = 0      # 0=none,1=GPS,2=DGPS
    gps_age_s:      float           = 999.0  # seconds since last valid fix

    # Heading (already filtered by HeadingEstimator)
    heading_deg:    float           = 0.0
    heading_source: str             = "none" # "imu","gps","held","none"
    heading_reliable: bool          = False

    # Obstacle signal (from sensor thread or external injection)
    obstacle_signal: str            = "NONE"
    obstacle_age_s:  float          = 999.0

    # System health
    gps_valid:      bool            = False
    heading_valid:  bool            = False
    timestamp:      float           = field(default_factory=time.monotonic)


class SensorHub:
    """
    Central, thread-safe sensor aggregation point.

    Instantiate once. Pass to producers (hardware reader threads) and
    to the control loop (via snapshot()).

    The hardware layer (STM32Hardware / MockHardware) is NOT changed.
    The SensorHub wraps the hardware and polls it in producer threads.
    """

    def __init__(self, hardware, heading_estimator: HeadingEstimator):
        """
        Args:
            hardware:          Any HardwareInterface instance.
            heading_estimator: HeadingEstimator instance (Problem 1 fix).
        """
        self._hw   = hardware
        self._hest = heading_estimator
        self._lock = threading.RLock()

        # Mutable internal state — only modified by producers
        self._lat:             Optional[float] = None
        self._lon:             Optional[float] = None
        self._gps_fix:         int             = 0
        self._gps_ts:          float           = 0.0

        self._heading_deg:     float           = 0.0
        self._heading_source:  str             = "none"
        self._heading_reliable: bool           = False
        self._heading_ts:      float           = 0.0

        self._obstacle_signal: str             = "NONE"
        self._obstacle_ts:     float           = 0.0

        # Producer threads
        self._running = True
        self._threads: list = []

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all producer threads. Call once before the control loop."""
        self._threads = [
            threading.Thread(target=self._gps_producer,
                             name="SensorHub-GPS",     daemon=True),
            threading.Thread(target=self._heading_producer,
                             name="SensorHub-HDG",     daemon=True),
        ]
        for t in self._threads:
            t.start()
        log.info("SensorHub started (%d producer threads).", len(self._threads))

    def stop(self) -> None:
        """Signal all producer threads to exit."""
        self._running = False
        log.info("SensorHub stopped.")

    # ── Consumer API (called by control loop) ─────────────────────────────────

    def snapshot(self) -> SensorSnapshot:
        """
        Return a frozen copy of all sensor data.
        Fast — only acquires lock for a dict copy. Never blocks on I/O.
        """
        now = time.monotonic()
        with self._lock:
            gps_age = now - self._gps_ts if self._gps_ts > 0 else 999.0
            hdg_age = now - self._heading_ts if self._heading_ts > 0 else 999.0
            obs_age = now - self._obstacle_ts if self._obstacle_ts > 0 else 999.0

            gps_valid = (
                self._lat is not None
                and self._gps_fix > 0
                and gps_age < GPS_STALE_S
            )
            hdg_valid = hdg_age < HEADING_STALE_S

            # Auto-clear obstacle signal if stale
            obs_sig = self._obstacle_signal
            if obs_age > OBS_STALE_S and obs_sig != "NONE":
                self._obstacle_signal = "NONE"
                obs_sig = "NONE"
                log.debug("Obstacle signal auto-cleared (stale).")

            return SensorSnapshot(
                lat             = self._lat if gps_valid else None,
                lon             = self._lon if gps_valid else None,
                gps_fix         = self._gps_fix,
                gps_age_s       = gps_age,
                heading_deg     = self._heading_deg,
                heading_source  = self._heading_source,
                heading_reliable= self._heading_reliable,
                heading_valid   = hdg_valid,
                obstacle_signal = obs_sig,
                obstacle_age_s  = obs_age,
                gps_valid       = gps_valid,
                timestamp       = now,
            )

    # ── Producer API (called by sensor threads or external code) ──────────────

    def inject_obstacle_signal(self, signal: str) -> None:
        """
        Thread-safe injection of obstacle signal from any thread
        (sensor callback, ROS node, test harness, etc.).
        """
        with self._lock:
            self._obstacle_signal = signal.upper()
            self._obstacle_ts     = time.monotonic()
        log.info("SensorHub: obstacle signal injected → %s", signal.upper())

    def clear_obstacle_signal(self) -> None:
        with self._lock:
            self._obstacle_signal = "NONE"

    # ── Producer threads ──────────────────────────────────────────────────────

    def _gps_producer(self) -> None:
        """
        Polls hardware GPS at ~10 Hz and updates shared state.
        Runs independently from the control loop — never blocks it.
        """
        log.debug("GPS producer thread started.")
        interval = 0.1   # 10 Hz poll rate

        while self._running:
            t0 = time.monotonic()
            try:
                lat, lon = self._hw.get_gps()
                # hardware.get_gps() returns (None, None) on no fix
                fix = 1 if (lat is not None and lon is not None) else 0

                with self._lock:
                    if fix:
                        self._lat     = lat
                        self._lon     = lon
                        self._gps_ts  = time.monotonic()
                    self._gps_fix = fix

            except Exception as exc:
                log.warning("GPS producer error: %s", exc)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _heading_producer(self) -> None:
        """
        Polls IMU heading and GPS position at ~20 Hz, runs HeadingEstimator,
        and updates shared heading state.

        This runs at HIGHER than the control loop rate so the EMA filter
        gets more data points and produces a smoother output.
        """
        log.debug("Heading producer thread started.")
        interval = 0.05   # 20 Hz

        while self._running:
            t0 = time.monotonic()
            try:
                imu_hdg = self._hw.get_heading()

                # Read current GPS snapshot safely
                with self._lock:
                    lat = self._lat
                    lon = self._lon
                    fix = self._gps_fix

                if lat is None or lon is None:
                    # No GPS yet — still update heading if IMU is available
                    lat, lon, fix = 0.0, 0.0, 0

                hdg = self._hest.update(
                    imu_heading_deg = imu_hdg,
                    lat             = lat,
                    lon             = lon,
                    gps_fix_quality = fix,
                )

                with self._lock:
                    self._heading_deg      = hdg
                    self._heading_source   = self._hest.source
                    self._heading_reliable = self._hest.is_reliable
                    self._heading_ts       = time.monotonic()

            except Exception as exc:
                log.warning("Heading producer error: %s", exc)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, interval - elapsed))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status_line(self) -> str:
        snap = self.snapshot()
        return (
            f"SensorHub | GPS={'OK' if snap.gps_valid else 'STALE'} "
            f"({snap.gps_age_s:.1f}s) | "
            f"HDG={snap.heading_deg:.1f}°[{snap.heading_source}] | "
            f"OBS={snap.obstacle_signal}"
        )
