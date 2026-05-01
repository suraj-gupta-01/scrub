"""
ASV Navigation System - Hardware Abstraction Layer (v2)
=======================================================
Architecture change: Raspberry Pi 5 no longer talks to GPS or motors directly.
All real-time hardware (GPS, motor ESCs) is owned by an STM32 microcontroller.
The RPi communicates with the STM32 over a single UART serial link.

                ┌──────────────────────────────────────┐
                │         Raspberry Pi 5               │
                │   (mission logic / navigation)       │
                │          hardware.py                 │
                └────────────┬─────────┬───────────────┘
                             │  UART   │
                             │ 115200  │
                ┌────────────▼─────────▼────────────────┐
                │           STM32 MCU                   │
                │   - GPS UART (NMEA parsing)           │
                │   - PWM ESC left/right                │
                │   - Sensor watchdog                   │
                │   - Emergency stop if RPi silent      │
                └───────────────────────────────────────┘

PROTOCOL
--------
RPi  → STM32  (commands):
    M <left> <right>\n       Motor set    e.g. "M 0.60 0.45\n"
    S\n                      Emergency stop
    P\n                      Ping (heartbeat)

STM32 → RPi   (telemetry, 10 Hz):
    G <lat> <lon> <fix>\n    GPS fix      e.g. "G 12.9719 77.5948 1\n"
    H <heading>\n            Heading      e.g. "H 270.5\n"
    A <status>\n             ACK/status   e.g. "A OK\n" or "A ERR\n"
    T <pong>\n               Ping reply   e.g. "T PONG\n"

All tokens are ASCII. Lines end with '\n'. Values use '.' as decimal sep.
Thread-safe: a background reader thread continuously drains the serial
buffer and updates shared state protected by a lock.
"""

import math
import time
import threading
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional

log = logging.getLogger(__name__)

# ── Shared constants ──────────────────────────────────────────────────────────

BAUD_RATE        = 115_200   # match your STM32 UART config
READ_TIMEOUT_S   = 0.05      # serial read timeout (non-blocking feel)
HEARTBEAT_HZ     = 2.0       # how often RPi pings STM32
GPS_STALE_S      = 3.0       # seconds before GPS considered stale
HEADING_STALE_S  = 3.0       # seconds before heading considered stale
MAX_SEND_RETRIES = 3         # retries for motor commands on ACK timeout


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface  (unchanged — rest of codebase depends on this contract)
# ─────────────────────────────────────────────────────────────────────────────

class HardwareInterface(ABC):
    """
    Every hardware backend must implement exactly these four methods.
    The navigator, controller, and obstacle handler never touch hardware
    directly — they call only these methods.
    """

    @abstractmethod
    def set_motor_speed(self, left: float, right: float) -> None:
        """Send normalised thruster commands [-1, 1] to hardware."""

    @abstractmethod
    def get_gps(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (lat, lon) or (None, None) when no valid fix."""

    @abstractmethod
    def get_heading(self) -> Optional[float]:
        """Return heading [0, 360) degrees or None when unavailable."""

    @abstractmethod
    def stop(self) -> None:
        """Immediate stop — call on shutdown or emergency."""


# ─────────────────────────────────────────────────────────────────────────────
# STM32HardWare  — the real deployment class
# ─────────────────────────────────────────────────────────────────────────────

class STM32Hardware(HardwareInterface):
    """
    Communicates with the STM32 over a single UART serial port.

    Responsibilities of THIS class (RPi side):
      • Send motor commands as ASCII over serial.
      • Parse incoming telemetry lines from the STM32.
      • Run a background heartbeat so STM32 can watchdog the RPi.
      • Expose get_gps() / get_heading() from latest parsed data.

    Responsibilities of the STM32 firmware (NOT this file):
      • Parse NMEA from the GPS module.
      • Drive PWM on both ESC channels.
      • Emergency-stop motors if no 'M' or 'P' received within ~2 s.
      • Send telemetry lines at ~10 Hz.
    """

    def __init__(self, port: str = "/dev/ttyAMA0", baud: int = BAUD_RATE):
        """
        Args:
            port:  Serial port where STM32 TX/RX is wired.
                   RPi 5 default UART: /dev/ttyAMA0
                   Check: ls /dev/tty* | grep -E 'AMA|USB'
            baud:  Must match STM32 UART peripheral configuration.
        """
        try:
            import serial
        except ImportError:
            raise ImportError("Install pyserial:  pip install pyserial")

        self._port    = port
        self._baud    = baud
        self._ser: Optional["serial.Serial"] = None

        # ── Shared state (protected by lock) ──────────────────────────────
        self._lock        = threading.Lock()
        self._lat:        Optional[float] = None
        self._lon:        Optional[float] = None
        self._gps_fix:    bool            = False
        self._heading:    Optional[float] = None
        self._gps_ts:     float           = 0.0   # monotonic timestamp
        self._hdg_ts:     float           = 0.0
        self._last_ack:   str             = ""

        # ── Background threads ────────────────────────────────────────────
        self._running     = True
        self._reader_thread    = threading.Thread(
            target=self._reader_loop, name="STM32-reader", daemon=True)
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="STM32-hb", daemon=True)

        self._open_serial(serial)
        self._reader_thread.start()
        self._heartbeat_thread.start()
        log.info("STM32Hardware ready on %s @ %d baud.", port, baud)

    # ── HardwareInterface implementation ──────────────────────────────────

    def set_motor_speed(self, left: float, right: float) -> None:
        """
        Send  'M <left> <right>\\n'  to the STM32.
        Values clamped to [-1.0, 1.0]. Blocks only for the serial write,
        not for an ACK (fire-and-forget keeps the control loop smooth).
        """
        left  = _clamp(left,  -1.0, 1.0)
        right = _clamp(right, -1.0, 1.0)
        cmd   = f"M {left:.4f} {right:.4f}\n"
        self._send(cmd)

    def get_gps(self) -> Tuple[Optional[float], Optional[float]]:
        """Return latest GPS fix, or (None, None) if stale / no fix."""
        with self._lock:
            age = time.monotonic() - self._gps_ts
            if self._gps_fix and age < GPS_STALE_S:
                return self._lat, self._lon
        log.debug("GPS stale or no fix (age=%.1f s).", age)
        return None, None

    def get_heading(self) -> Optional[float]:
        """Return latest heading, or None if stale."""
        with self._lock:
            age = time.monotonic() - self._hdg_ts
            if self._heading is not None and age < HEADING_STALE_S:
                return self._heading
        return None

    def stop(self) -> None:
        """Send emergency stop, then close serial."""
        self._running = False
        try:
            self._send("S\n")
            log.info("STM32Hardware: stop command sent.")
        except Exception:
            pass
        finally:
            if self._ser and self._ser.is_open:
                self._ser.close()

    # ── Serial open ───────────────────────────────────────────────────────

    def _open_serial(self, serial_module) -> None:
        try:
            self._ser = serial_module.Serial(
                port     = self._port,
                baudrate = self._baud,
                timeout  = READ_TIMEOUT_S,
                bytesize = serial_module.EIGHTBITS,
                parity   = serial_module.PARITY_NONE,
                stopbits = serial_module.STOPBITS_ONE,
            )
            log.info("Serial port %s opened.", self._port)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot open {self._port}: {exc}\n"
                "Check wiring, port name, and that STM32 is powered."
            ) from exc

    # ── Background: serial reader ─────────────────────────────────────────

    def _reader_loop(self) -> None:
        """
        Continuously reads lines from the STM32 and updates shared state.
        Runs in its own daemon thread — never blocks the control loop.
        """
        log.debug("STM32 reader thread started.")
        while self._running:
            try:
                raw = self._ser.readline()        # blocks up to READ_TIMEOUT_S
                if not raw:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()
                if line:
                    self._parse_line(line)
            except Exception as exc:
                if self._running:
                    log.warning("Serial read error: %s", exc)
                    time.sleep(0.1)

    def _parse_line(self, line: str) -> None:
        """
        Dispatch an incoming telemetry line from the STM32.

        Message formats:
            G <lat> <lon> <fix>   →  GPS update
            H <degrees>           →  Heading update
            A <status>            →  Command acknowledgement
            T PONG                →  Heartbeat reply (no action needed)
        """
        parts = line.split()
        if not parts:
            return

        msg_type = parts[0].upper()

        try:
            if msg_type == "G" and len(parts) >= 4:
                # GPS:  G 12.971900 77.594700 1
                lat = float(parts[1])
                lon = float(parts[2])
                fix = int(parts[3])            # 0=no fix, 1=GPS, 2=DGPS
                with self._lock:
                    self._lat     = lat
                    self._lon     = lon
                    self._gps_fix = fix > 0
                    self._gps_ts  = time.monotonic()
                log.debug("GPS: %.6f, %.6f fix=%d", lat, lon, fix)

            elif msg_type == "H" and len(parts) >= 2:
                # Heading:  H 270.5
                hdg = float(parts[1]) % 360.0
                with self._lock:
                    self._heading = hdg
                    self._hdg_ts  = time.monotonic()
                log.debug("Heading: %.1f°", hdg)

            elif msg_type == "A" and len(parts) >= 2:
                # ACK:  A OK  or  A ERR
                status = parts[1].upper()
                with self._lock:
                    self._last_ack = status
                if status == "ERR":
                    log.warning("STM32 reported ERR on last command.")

            elif msg_type == "T":
                # Heartbeat pong — nothing to do
                log.debug("STM32 pong received.")

            else:
                log.debug("Unknown STM32 message: %s", line)

        except (ValueError, IndexError) as exc:
            log.debug("Parse error on '%s': %s", line, exc)

    # ── Background: heartbeat ─────────────────────────────────────────────

    def _heartbeat_loop(self) -> None:
        """
        Sends a 'P\\n' ping to the STM32 at HEARTBEAT_HZ frequency.
        If the STM32 firmware has a watchdog, it uses this to confirm
        the RPi is alive. If pings stop, the STM32 should stop motors.
        """
        interval = 1.0 / HEARTBEAT_HZ
        log.debug("Heartbeat thread started at %.1f Hz.", HEARTBEAT_HZ)
        while self._running:
            try:
                self._send("P\n")
            except Exception as exc:
                log.warning("Heartbeat send failed: %s", exc)
            time.sleep(interval)

    # ── Serial send ───────────────────────────────────────────────────────

    def _send(self, cmd: str) -> None:
        """Write an ASCII command to the STM32. Not re-entrant — caller manages."""
        if self._ser and self._ser.is_open:
            self._ser.write(cmd.encode("ascii"))
            log.debug("→ STM32: %r", cmd.strip())
        else:
            log.error("Serial not open — cannot send: %r", cmd.strip())


# ─────────────────────────────────────────────────────────────────────────────
# MockHardware  — desktop / CI simulation (no STM32, no GPIO required)
# ─────────────────────────────────────────────────────────────────────────────

class MockHardware(HardwareInterface):
    """
    Pure-Python simulated boat for desktop testing and CI.
    Implements the same HardwareInterface as STM32Hardware so the
    controller, navigator, and obstacle handler are completely unaware
    of whether real hardware is present.

    Physics model: differential-drive boat (same as original v1).
    Call step(dt) from the simulation loop to advance the physics.
    """

    def __init__(self, init_lat: float, init_lon: float,
                 init_heading: float = 0.0,
                 boat_speed_mps: float = None,
                 turn_rate_dps:  float = None):
        import config as cfg
        self.lat      = init_lat
        self.lon      = init_lon
        self.heading  = init_heading   # degrees

        self._left    = 0.0
        self._right   = 0.0
        self._speed   = boat_speed_mps or cfg.SIM_BOAT_SPEED
        self._trn     = turn_rate_dps  or cfg.SIM_TURN_RATE
        self._lock    = threading.Lock()
        self.trajectory: list = [(init_lat, init_lon)]

        log.info("MockHardware initialised at (%.6f, %.6f).", init_lat, init_lon)

    # HardwareInterface implementation

    def set_motor_speed(self, left: float, right: float) -> None:
        with self._lock:
            self._left  = _clamp(left,  -1.0, 1.0)
            self._right = _clamp(right, -1.0, 1.0)

    def get_gps(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            return self.lat, self.lon

    def get_heading(self) -> Optional[float]:
        with self._lock:
            return self.heading

    def stop(self) -> None:
        with self._lock:
            self._left = self._right = 0.0
        log.info("MockHardware: motors stopped.")

    # Simulation-only method

    def step(self, dt: float) -> None:
        """Advance the simulated physics by dt seconds."""
        with self._lock:
            left, right = self._left, self._right

        fwd       = (left + right) / 2.0 * self._speed
        turn_rate = ((right - left) / 2.0) * self._trn   # degrees/s

        with self._lock:
            self.heading  = (self.heading + turn_rate * dt) % 360.0
            hdg_rad       = math.radians(self.heading)
            dx            = fwd * dt * math.sin(hdg_rad)   # east  (m)
            dy            = fwd * dt * math.cos(hdg_rad)   # north (m)
            EARTH_R       = 6_371_000.0
            self.lat     += math.degrees(dy / EARTH_R)
            self.lon     += math.degrees(
                dx / (EARTH_R * math.cos(math.radians(self.lat)))
            )
            self.trajectory.append((self.lat, self.lon))


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper  — use this in main.py
# ─────────────────────────────────────────────────────────────────────────────

def create_hardware(simulate: bool = False,
                    init_lat: float = 0.0,
                    init_lon: float = 0.0,
                    stm32_port: str = "/dev/ttyAMA0") -> HardwareInterface:
    """
    Factory that returns the correct hardware backend.

    Args:
        simulate:   If True, return MockHardware (no serial needed).
        init_lat:   Starting latitude  (MockHardware only).
        init_lon:   Starting longitude (MockHardware only).
        stm32_port: Serial port for real STM32 connection.

    Usage:
        hw = create_hardware(simulate=False, stm32_port="/dev/ttyAMA0")
        hw = create_hardware(simulate=True,  init_lat=12.97, init_lon=77.59)
    """
    if simulate:
        log.info("Hardware mode: SIMULATION (MockHardware)")
        return MockHardware(init_lat, init_lon)
    else:
        log.info("Hardware mode: REAL (STM32 over %s)", stm32_port)
        return STM32Hardware(port=stm32_port)


# ─────────────────────────────────────────────────────────────────────────────
# Private utility
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
