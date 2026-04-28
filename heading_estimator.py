"""
ASV Navigation System - Heading Estimator
==========================================
Provides a single, robust heading value to the Navigator by fusing
multiple sources in a priority-ranked, filtered pipeline.

WHY THIS EXISTS
---------------
The original system derived heading by computing the compass bearing
between two consecutive GPS fixes (done inside RealHardware._gps_reader).

This fails in three common real-world scenarios:

  1. STATIONARY / SLOW SPEED
     GPS position noise (~1-3 m CEP for typical modules) creates phantom
     movement vectors. At 0.5 m/s the boat moves only 5 cm per 100 ms
     update — well inside noise floor. The computed bearing can jump
     anywhere in [0°, 360°] between consecutive fixes.

  2. GPS MULTIPATH / MOMENTARY OUTAGE
     Near buildings, trees, or when a satellite drops, a single bad fix
     causes a 180° heading spike. With KP=1.2 the navigator immediately
     commands full reverse-turn, which can put the boat into the shore.

  3. HEADING USED BEFORE MOVEMENT STARTS
     At mission start the boat is stationary. The first bearing() call
     returns 0.0 (or last remembered value). The navigator then turns
     toward 0° instead of the actual first waypoint direction — wasting
     time and often crossing its own path.

SOLUTION ARCHITECTURE
---------------------
Priority 1  IMU heading from STM32 (BNO055 / ICM-42688-P)
              • Absolute, drift-corrected, ~50 Hz update
              • Valid at any speed including zero
              • Use directly through complementary filter

Priority 2  GPS-track heading
              • Computed from consecutive GPS fixes
              • Only accepted when:
                  - Speed ≥ MIN_SPEED_FOR_GPS_HEADING (0.4 m/s)
                  - Fix quality is good (fix_type > 0)
                  - Angular jump ≤ MAX_HEADING_JUMP_DEG (45°)
              • Fused into EMA filter

Priority 3  Hold last valid heading
              • If no source is fresh / valid, freeze the last known heading
              • This is far safer than returning 0.0 or random noise

EMA FILTER
----------
Exponential Moving Average applied to the heading signal.
Alpha ∈ (0, 1]:  high alpha = more responsive, low alpha = smoother.
Operates in sin/cos space to correctly handle 359°→1° wraparound.
"""

import math
import time
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
# Minimum GPS ground speed (m/s) before GPS-derived heading is trusted.
# Below this the boat is considered "nearly stationary" and GPS bearing is
# too noisy to use.
MIN_SPEED_FOR_GPS_HEADING = 0.4   # m/s  — raise if your GPS is noisier

# Maximum allowed single-step heading change (degrees).
# Anything larger is treated as a noise spike and discarded.
MAX_HEADING_JUMP_DEG      = 45.0

# EMA alpha for heading smoothing.
# 0.3 = smooth (slower response), 0.7 = snappy (less filtering).
# Tune based on your IMU noise characteristics.
HEADING_EMA_ALPHA         = 0.4

# After this many seconds without any valid heading source,
# log a warning and keep holding last known value.
HEADING_STALE_WARN_S      = 2.0

# Minimum distance (m) between two GPS points before computing bearing.
# Prevents division artifacts on identical consecutive fixes.
MIN_GPS_DIST_FOR_BEARING  = 0.15  # metres


class HeadingEstimator:
    """
    Fuses IMU heading (primary) and GPS-track heading (fallback) into a
    single, filtered, reliable heading value for the Navigator.

    Usage:
        estimator = HeadingEstimator()

        # Call every control loop tick (10 Hz):
        heading = estimator.update(
            imu_heading_deg = hw.get_heading(),   # from STM32/IMU
            lat=lat, lon=lon,                     # current GPS fix
            gps_fix_quality=1,                    # 0=no fix, 1=GPS, 2=DGPS
        )

        # Pass `heading` directly to navigator.update()
    """

    def __init__(self, alpha: float = HEADING_EMA_ALPHA):
        self._alpha = alpha

        # EMA state (stored as sin/cos to avoid wraparound errors)
        self._ema_sin: Optional[float] = None
        self._ema_cos: Optional[float] = None

        # GPS tracking for ground-speed and bearing computation
        self._prev_lat:  Optional[float] = None
        self._prev_lon:  Optional[float] = None
        self._prev_time: float = 0.0

        # Last valid heading for hold-last strategy
        self._last_valid_hdg:  Optional[float] = None
        self._last_valid_time: float = 0.0

        # Diagnostics
        self.source: str = "none"   # "imu", "gps", "held"
        self.speed_mps: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self,
               imu_heading_deg: Optional[float],
               lat: float,
               lon: float,
               gps_fix_quality: int = 1) -> float:
        """
        Compute best heading estimate for this control tick.

        Args:
            imu_heading_deg:  Raw heading from IMU (degrees 0-360), or None.
            lat, lon:         Current GPS position.
            gps_fix_quality:  0=no fix, 1=GPS, 2=DGPS.

        Returns:
            Smoothed heading in degrees [0, 360).
            Falls back to last valid heading if no source is fresh.
            Falls back to 0.0 only on very first tick with no data.
        """
        now = time.monotonic()
        candidate: Optional[float] = None

        # ── Source 1: IMU (highest trust) ─────────────────────────────────
        if imu_heading_deg is not None:
            candidate   = imu_heading_deg % 360.0
            self.source = "imu"

        # ── Source 2: GPS track heading (fallback) ─────────────────────────
        gps_heading = self._compute_gps_heading(lat, lon, now, gps_fix_quality)

        if candidate is None and gps_heading is not None:
            candidate   = gps_heading
            self.source = "gps"

        # ── Filter candidate through EMA ───────────────────────────────────
        if candidate is not None:
            # Validate against max jump to reject noise spikes
            if self._last_valid_hdg is not None:
                jump = abs(_angle_diff(self._last_valid_hdg, candidate))
                if jump > MAX_HEADING_JUMP_DEG and self.source == "gps":
                    # GPS spike — discard candidate, hold last
                    log.debug("Heading spike rejected: %.1f° jump (src=%s)",
                              jump, self.source)
                    candidate = None

        if candidate is not None:
            filtered = self._ema_update(candidate)
            self._last_valid_hdg  = filtered
            self._last_valid_time = now
            return filtered

        # ── Source 3: Hold last valid ──────────────────────────────────────
        if self._last_valid_hdg is not None:
            age = now - self._last_valid_time
            if age > HEADING_STALE_WARN_S:
                log.warning("Heading stale for %.1f s — holding %.1f°",
                            age, self._last_valid_hdg)
            self.source = "held"
            return self._last_valid_hdg

        # ── No data at all (first tick, no IMU, no GPS movement) ──────────
        log.warning("No heading source available — returning 0.0°")
        self.source = "none"
        return 0.0

    @property
    def is_reliable(self) -> bool:
        """
        True if current heading comes from IMU or recent GPS track.
        False if holding stale data. Use this to gate aggressive manoeuvres.
        """
        return self.source in ("imu", "gps")

    def reset(self) -> None:
        """Clear EMA state — call after a large position jump or resume."""
        self._ema_sin = None
        self._ema_cos = None
        self._prev_lat = None
        self._prev_lon = None
        self._last_valid_hdg = None
        log.info("HeadingEstimator reset.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_gps_heading(self, lat: float, lon: float,
                              now: float, fix_quality: int) -> Optional[float]:
        """
        Derive heading from consecutive GPS fixes.
        Returns None if conditions for trustworthy GPS bearing are not met.
        """
        # No previous fix to compare against
        if self._prev_lat is None:
            self._prev_lat  = lat
            self._prev_lon  = lon
            self._prev_time = now
            return None

        # No GPS fix — don't even compute
        if fix_quality == 0:
            return None

        dt   = now - self._prev_time
        if dt < 1e-6:
            return None

        dist = _haversine(self._prev_lat, self._prev_lon, lat, lon)

        # Too close — GPS noise dominates
        if dist < MIN_GPS_DIST_FOR_BEARING:
            # Still update prev so we don't accumulate a stale anchor
            self._prev_lat  = lat
            self._prev_lon  = lon
            self._prev_time = now
            return None

        self.speed_mps = dist / dt

        # Speed gate — below threshold, GPS bearing is meaningless
        if self.speed_mps < MIN_SPEED_FOR_GPS_HEADING:
            self._prev_lat  = lat
            self._prev_lon  = lon
            self._prev_time = now
            return None

        hdg = _bearing(self._prev_lat, self._prev_lon, lat, lon)
        self._prev_lat  = lat
        self._prev_lon  = lon
        self._prev_time = now
        return hdg

    def _ema_update(self, heading_deg: float) -> float:
        """
        Apply EMA in sin/cos space so 359°→1° wraps correctly.
        Returns filtered heading in [0, 360).
        """
        rad = math.radians(heading_deg)
        s, c = math.sin(rad), math.cos(rad)

        if self._ema_sin is None:
            # Cold start — seed the filter
            self._ema_sin = s
            self._ema_cos = c
        else:
            self._ema_sin = self._alpha * s + (1 - self._alpha) * self._ema_sin
            self._ema_cos = self._alpha * c + (1 - self._alpha) * self._ema_cos

        return (math.degrees(math.atan2(self._ema_sin, self._ema_cos)) + 360) % 360

    def status_line(self) -> str:
        hdg = self._last_valid_hdg
        return (f"Heading[{self.source.upper()}] "
                f"{hdg:.1f}°  spd={self.speed_mps:.2f}m/s  "
                f"reliable={'Y' if self.is_reliable else 'N'}")


# ── Pure math helpers (no external dependencies) ─────────────────────────────

_EARTH_R = 6_371_000.0

def _haversine(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * _EARTH_R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _bearing(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def _angle_diff(a: float, b: float) -> float:
    """Signed difference b - a in (-180, 180]."""
    return ((b - a + 180) % 360) - 180
