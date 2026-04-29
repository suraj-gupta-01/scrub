"""
ASV Navigation System - Hardware Abstraction Layer
Swap MockHardware for RealHardware to deploy on the boat.
"""

import time
import math
import threading
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional

log = logging.getLogger(__name__)


# ── Abstract interface ────────────────────────────────────────────────────────

class HardwareInterface(ABC):
    """All hardware adapters implement this interface."""

    @abstractmethod
    def set_motor_speed(self, left: float, right: float) -> None:
        """Set thruster speeds. Values normalised to [-1, 1]."""

    @abstractmethod
    def get_gps(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (lat, lon) or (None, None) if fix unavailable."""

    @abstractmethod
    def get_heading(self) -> Optional[float]:
        """Return current heading in degrees [0, 360) or None."""

    @abstractmethod
    def stop(self) -> None:
        """Emergency stop all motors."""


# ── Mock (simulation) hardware ────────────────────────────────────────────────

class MockHardware(HardwareInterface):
    """
    Simulated hardware for desktop testing.
    Internal physics: differential drive boat model.
    Update by calling step(dt) from the simulation loop.
    """

    def __init__(self, init_lat: float, init_lon: float,
                 init_heading: float = 0.0,
                 boat_speed_mps: float = None,
                 turn_rate_dps: float = None):
        import config as cfg
        self.lat     = init_lat
        self.lon     = init_lon
        self.heading = init_heading   # degrees
        self._left   = 0.0
        self._right  = 0.0
        self._speed  = boat_speed_mps or cfg.SIM_BOAT_SPEED
        self._trn    = turn_rate_dps  or cfg.SIM_TURN_RATE
        self._lock   = threading.Lock()
        self.trajectory: list = [(init_lat, init_lon)]

    # Hardware interface implementation

    def set_motor_speed(self, left: float, right: float) -> None:
        with self._lock:
            self._left  = max(-1.0, min(1.0, left))
            self._right = max(-1.0, min(1.0, right))

    def get_gps(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            return self.lat, self.lon

    def get_heading(self) -> Optional[float]:
        with self._lock:
            return self.heading

    def stop(self) -> None:
        with self._lock:
            self._left = self._right = 0.0

    # Simulation physics

    def step(self, dt: float) -> None:
        """Advance simulated physics by dt seconds."""
        with self._lock:
            left, right = self._left, self._right

        # Average speed → forward movement
        fwd  = (left + right) / 2.0 * self._speed
        # Differential → turn
        diff = (right - left) / 2.0
        turn_rate = diff * self._trn  # degrees/sec

        with self._lock:
            self.heading = (self.heading + turn_rate * dt) % 360.0
            hdg_rad      = math.radians(self.heading)

            dx = fwd * dt * math.sin(hdg_rad)   # east
            dy = fwd * dt * math.cos(hdg_rad)   # north

            EARTH_R = 6_371_000.0
            self.lat += math.degrees(dy / EARTH_R)
            self.lon += math.degrees(dx / (EARTH_R * math.cos(math.radians(self.lat))))
            self.trajectory.append((self.lat, self.lon))


# ── Real Raspberry Pi hardware ────────────────────────────────────────────────

class RealHardware(HardwareInterface):
    """
    Real hardware adapter for Raspberry Pi 5.

    Requires:
      - pigpio daemon (sudo pigpiod) for PWM motor control
      - pyserial + pynmea2 for GPS UART
      - Optional: BNO055 IMU for heading (else computed from GPS track)

    Pin mapping (edit to match your ESC wiring):
        LEFT_MOTOR_PIN  = 12   # PWM capable GPIO
        RIGHT_MOTOR_PIN = 13
    """

    LEFT_MOTOR_PIN  = 12
    RIGHT_MOTOR_PIN = 13
    PWM_FREQ        = 50       # Hz — standard RC servo/ESC
    PWM_MID         = 1500     # µs neutral
    PWM_RANGE       = 400      # µs each side

    def __init__(self, gps_port: str = None, gps_baud: int = None):
        import config as cfg
        self._gps_port = gps_port or cfg.GPS_PORT
        self._gps_baud = gps_baud or cfg.GPS_BAUD
        self._lat:     Optional[float] = None
        self._lon:     Optional[float] = None
        self._heading: Optional[float] = None
        self._lock = threading.Lock()
        self._running = True

        # Import hardware libraries
        try:
            import pigpio
            self._pi = pigpio.pi()
            if not self._pi.connected:
                raise RuntimeError("pigpio daemon not running — run: sudo pigpiod")
            self._pi.set_servo_pulsewidth(self.LEFT_MOTOR_PIN,  self.PWM_MID)
            self._pi.set_servo_pulsewidth(self.RIGHT_MOTOR_PIN, self.PWM_MID)
        except ImportError:
            raise ImportError("Install pigpio: pip install pigpio")

        # Start GPS reader thread
        self._gps_thread = threading.Thread(target=self._gps_reader, daemon=True)
        self._gps_thread.start()
        log.info("RealHardware initialised. GPS on %s @ %d baud.",
                 self._gps_port, self._gps_baud)

    # ── Motor control ─────────────────────────────────────────────────────────

    def set_motor_speed(self, left: float, right: float) -> None:
        lw = self.PWM_MID + int(left  * self.PWM_RANGE)
        rw = self.PWM_MID + int(right * self.PWM_RANGE)
        lw = max(1000, min(2000, lw))
        rw = max(1000, min(2000, rw))
        self._pi.set_servo_pulsewidth(self.LEFT_MOTOR_PIN,  lw)
        self._pi.set_servo_pulsewidth(self.RIGHT_MOTOR_PIN, rw)

    def stop(self) -> None:
        self.set_motor_speed(0.0, 0.0)
        self._pi.set_servo_pulsewidth(self.LEFT_MOTOR_PIN,  self.PWM_MID)
        self._pi.set_servo_pulsewidth(self.RIGHT_MOTOR_PIN, self.PWM_MID)

    # ── GPS ───────────────────────────────────────────────────────────────────

    def get_gps(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            return self._lat, self._lon

    def get_heading(self) -> Optional[float]:
        with self._lock:
            return self._heading

    def _gps_reader(self) -> None:
        """Background thread — reads NMEA sentences from UART."""
        try:
            import serial
            import pynmea2
        except ImportError:
            raise ImportError("Install: pip install pyserial pynmea2")

        with serial.Serial(self._gps_port, self._gps_baud, timeout=1) as ser:
            prev_lat = prev_lon = None
            while self._running:
                try:
                    line = ser.readline().decode("ascii", errors="ignore").strip()
                    if not line.startswith("$"):
                        continue
                    msg = pynmea2.parse(line)

                    if isinstance(msg, pynmea2.GGA) and msg.gps_qual > 0:
                        with self._lock:
                            self._lat = msg.latitude
                            self._lon = msg.longitude
                            # Derive heading from consecutive GPS fixes
                            if prev_lat is not None:
                                from utils import bearing as _bearing
                                hdg = _bearing(prev_lat, prev_lon,
                                               self._lat, self._lon)
                                self._heading = hdg
                            prev_lat, prev_lon = self._lat, self._lon

                except Exception as exc:
                    log.debug("GPS parse error: %s", exc)

    def __del__(self):
        self._running = False
        if hasattr(self, "_pi"):
            self.stop()
            self._pi.stop()
