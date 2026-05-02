"""
sensor_bridge.py — RPi-side HTTP bridge between the navigation system
and the SCRUB Streamlit dashboard.

WHY THIS EXISTS
---------------
The navigation code (controller.py / sensor_hub.py) runs as a separate
process focused on motor control at 10 Hz.  The Streamlit dashboard runs
as another process and needs live sensor + GPS data to display.

Rather than making the two processes share memory (fragile) or polling a
file (laggy), this tiny Flask server:
  • Runs in its own thread inside the navigation process, OR
  • Runs as a standalone process that reads the shared state file.

This file supports BOTH modes:

  Mode A — embedded (recommended):
      Call SensorBridge(sensor_hub).start() from main.py or controller.py.
      The bridge reads directly from the live SensorHub snapshot.
      Zero file I/O — always current.

  Mode B — standalone (fallback):
      python3 sensor_bridge.py
      Reads from STATE_FILE (mission_state.json) and a new SENSOR_FILE
      written by the navigation process.  Useful when the dashboard runs
      on a different machine on the same LAN.

DASHBOARD INTEGRATION
---------------------
The existing app.py already calls:
    resp = requests.get(f"http://{clean_ip}:5000/sensor_data", timeout=2)

This server answers that request with ALL 8 sensor values PLUS the real
GPS coordinates from the navigation system.  The dashboard therefore shows
the boat's actual position on the map and real sensor readings.

ENDPOINT
--------
GET /sensor_data
    Returns JSON:
    {
        "pH":         7.12,
        "Turbidity":  4.5,
        "TDS":        285.0,
        "WaterTemp":  26.1,
        "AmbiTemp":   31.2,
        "Humidity":   65.0,
        "GasCO":      3.8,
        "GasCH4":     420.0,
        "latitude":   12.918316,
        "longitude":  77.490639,
        "gps_fix":    1,
        "heading":    270.5,
        "sensor_age": 0.8,
        "gps_age":    0.1,
        "timestamp":  "2025-05-01T14:32:01.123"
    }

    On error / no data yet:
    {"error": "no sensor data", "gps_valid": false}

RUNNING
-------
    # Embedded (inside navigation process) — call from main.py:
    from sensor_bridge import SensorBridge
    bridge = SensorBridge(sensor_hub=mc.sensor_hub, port=5000)
    bridge.start()   # starts a daemon thread — non-blocking

    # Standalone (separate terminal on RPi):
    python3 sensor_bridge.py --port 5000

DEPENDENCIES
------------
    pip install flask

    Flask is chosen over FastAPI because it has zero extra deps and
    the single endpoint does not benefit from async I/O.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import argparse
from datetime import datetime
from typing import Optional

log = logging.getLogger("SensorBridge")


# ── Flask import (optional — bridge degrades gracefully if not installed) ─────
try:
    from flask import Flask, jsonify
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False
    log.warning(
        "Flask not installed — SensorBridge will not start. "
        "Install with:  pip install flask"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Embedded bridge  (Mode A — reads live from SensorHub)
# ═════════════════════════════════════════════════════════════════════════════

class SensorBridge:
    """
    Tiny Flask HTTP server that exposes the SensorHub snapshot to the
    Streamlit dashboard over a local HTTP endpoint.

    Runs in a daemon thread so it never blocks the navigation loop.

    Usage:
        bridge = SensorBridge(sensor_hub=mc.sensor_hub, port=5000)
        bridge.start()
    """

    def __init__(self, sensor_hub, port: int = 5000):
        """
        Args:
            sensor_hub: A live SensorHub instance (must have .snapshot()).
            port:       TCP port the server listens on (default 5000).
        """
        self._hub  = sensor_hub
        self._port = port
        self._app  = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        if not _FLASK_AVAILABLE:
            log.error("Cannot start SensorBridge — Flask is not installed.")
            return

        self._app = _build_flask_app(self._hub)
        self._thread = threading.Thread(
            target=self._run_server,
            name="SensorBridge",
            daemon=True,   # exits automatically when main process exits
        )
        self._thread.start()
        log.info("SensorBridge started on port %d (daemon thread).", self._port)

    def _run_server(self) -> None:
        import logging as _logging
        # Suppress Flask's default access log to keep the terminal clean
        _logging.getLogger("werkzeug").setLevel(_logging.WARNING)
        self._app.run(host="0.0.0.0", port=self._port, use_reloader=False)


def _build_flask_app(sensor_hub) -> "Flask":
    """Build and return the Flask app with the /sensor_data route."""
    app = Flask("SensorBridge")

    @app.route("/sensor_data")
    def sensor_data():
        snap = sensor_hub.snapshot()

        if not snap.sensor_valid and not snap.gps_valid:
            return jsonify({"error": "no sensor data", "gps_valid": False}), 503

        payload = {
            # ── Water-quality / gas sensors ───────────────────────────────────
            # Keys match what app.py already reads (live_data.get(...))
            "pH":         snap.ph        if snap.ph        is not None else None,
            "Turbidity":  snap.turbidity if snap.turbidity is not None else None,
            "TDS":        snap.tds       if snap.tds       is not None else None,
            "WaterTemp":  snap.watertemp if snap.watertemp is not None else None,
            "AmbiTemp":   snap.ambitemp  if snap.ambitemp  is not None else None,
            "Humidity":   snap.humidity  if snap.humidity  is not None else None,
            "GasCO":      snap.gasCO     if snap.gasCO     is not None else None,
            "GasCH4":     snap.gasCH4    if snap.gasCH4    is not None else None,
            # ── GPS (real boat position) ───────────────────────────────────────
            # Passed separately so the dashboard can plot the actual track.
            "latitude":   snap.lat,
            "longitude":  snap.lon,
            "gps_fix":    snap.gps_fix,
            "gps_valid":  snap.gps_valid,
            "gps_age":    round(snap.gps_age_s, 2),
            # ── Navigation state ───────────────────────────────────────────────
            "heading":    round(snap.heading_deg, 1),
            "heading_src": snap.heading_source,
            # ── Freshness indicators ───────────────────────────────────────────
            "sensor_age": round(snap.sensor_age_s, 2),
            "sensor_valid": snap.sensor_valid,
            "timestamp":  datetime.now().isoformat(timespec="milliseconds"),
        }
        return jsonify(payload)

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "time": time.monotonic()})

    return app


# ═════════════════════════════════════════════════════════════════════════════
# Standalone bridge  (Mode B — reads from state/sensor JSON files)
# ═════════════════════════════════════════════════════════════════════════════

SENSOR_FILE = "live_sensors.json"   # written by navigation process (see below)


class _FileSensorHub:
    """
    Minimal SensorHub-compatible shim that reads from a JSON file on disk.
    Used when the bridge runs as a standalone process (Mode B).
    """

    def snapshot(self):
        from dataclasses import dataclass

        @dataclass
        class _Snap:
            ph: object = None; turbidity: object = None; tds: object = None
            watertemp: object = None; ambitemp: object = None
            humidity: object = None; gasCO: object = None; gasCH4: object = None
            lat: object = None; lon: object = None; gps_fix: int = 0
            gps_valid: bool = False; gps_age_s: float = 999.0
            heading_deg: float = 0.0; heading_source: str = "none"
            sensor_valid: bool = False; sensor_age_s: float = 999.0

        snap = _Snap()
        try:
            with open(SENSOR_FILE) as f:
                data = json.load(f)
            age = time.monotonic() - data.get("_monotonic", 0)
            snap.ph        = data.get("ph")
            snap.turbidity = data.get("turbidity")
            snap.tds       = data.get("tds")
            snap.watertemp = data.get("watertemp")
            snap.ambitemp  = data.get("ambitemp")
            snap.humidity  = data.get("humidity")
            snap.gasCO     = data.get("gasCO")
            snap.gasCH4    = data.get("gasCH4")
            snap.lat       = data.get("lat")
            snap.lon       = data.get("lon")
            snap.gps_fix   = data.get("gps_fix", 0)
            snap.gps_valid = snap.lat is not None and snap.gps_fix > 0 and age < 5
            snap.heading_deg    = data.get("heading", 0.0)
            snap.heading_source = data.get("heading_src", "none")
            snap.sensor_valid   = snap.ph is not None and age < 15
            snap.sensor_age_s   = age
            snap.gps_age_s      = age
        except Exception as exc:
            log.debug("Could not read %s: %s", SENSOR_FILE, exc)
        return snap


# ── Utility: write live_sensors.json from controller (call ~1 Hz) ─────────────

def write_sensor_file(snap, path: str = SENSOR_FILE) -> None:
    """
    Write the current SensorHub snapshot to a JSON file so a standalone
    sensor_bridge.py can serve it to the dashboard.

    Call this from the controller's status-log block (~1 Hz is fine):
        from sensor_bridge import write_sensor_file
        write_sensor_file(snap)

    The file is written atomically (temp file + rename) to prevent the
    dashboard from reading a half-written file.
    """
    import os, tempfile
    data = {
        "_monotonic": time.monotonic(),
        "ph":        snap.ph,
        "turbidity": snap.turbidity,
        "tds":       snap.tds,
        "watertemp": snap.watertemp,
        "ambitemp":  snap.ambitemp,
        "humidity":  snap.humidity,
        "gasCO":     snap.gasCO,
        "gasCH4":    snap.gasCH4,
        "lat":       snap.lat,
        "lon":       snap.lon,
        "gps_fix":   snap.gps_fix,
        "heading":   snap.heading_deg,
        "heading_src": snap.heading_source,
    }
    try:
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)   # atomic on POSIX
    except OSError as exc:
        log.warning("write_sensor_file failed: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# CLI entry point  (Mode B standalone)
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="SCRUB sensor bridge — serves live data to the dashboard"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="HTTP port (default 5000, must match app.py)"
    )
    parser.add_argument(
        "--file", type=str, default=SENSOR_FILE,
        help=f"Sensor JSON file written by the navigation process (default: {SENSOR_FILE})"
    )
    args = parser.parse_args()

    if not _FLASK_AVAILABLE:
        print("ERROR: Flask is required.  pip install flask")
        raise SystemExit(1)

    hub = _FileSensorHub()
    hub.sensor_file = args.file   # allow override

    app = _build_flask_app(hub)

    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.WARNING)

    log.info("Standalone SensorBridge running on http://0.0.0.0:%d/sensor_data", args.port)
    log.info("Reading sensor data from: %s", args.file)
    app.run(host="0.0.0.0", port=args.port, use_reloader=False)


if __name__ == "__main__":
    main()