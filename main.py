"""
ASV Navigation System - Main Entry Point
=========================================
Reads lake boundary and mission config from  mission_boundary.json
(same format as sample_lake.json).

Usage:
    python3 main.py                                  # real STM32 hardware
    python3 main.py --sim                            # desktop simulation
    python3 main.py --file my_lake.json              # custom boundary file
    python3 main.py --sim --sweep 4 --angle 30      # sim with custom sweep
    python3 main.py --no-resume                      # restart mission fresh
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from controller import MissionController
from hardware import create_hardware

"""
# Replace default logging config with our custom setup from log_setup.py.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
"""
log = logging.getLogger("Main")

from log_setup import setup_logging
setup_logging()

# ── Default file name ─────────────────────────────────────────────────────────
DEFAULT_BOUNDARY_FILE = "mission_boundary.json"
DEFAULT_STM32_PORT    = "/dev/ttyAMA0"


# ── JSON loader ───────────────────────────────────────────────────────────────

def load_mission_file(path: str) -> dict:
    """
    Load and validate mission_boundary.json.

    Expected format (same as sample_lake.json):
    {
        "ground_station": { "lat": ..., "lon": ... },
        "boundary": [
            { "lat": ..., "lon": ... },
            ...
        ],
        "config": {                        <- optional section
            "sweep_width_m": 3.0,
            "sweep_angle_deg": 0
        }
    }
    """
    fpath = Path(path)
    if not fpath.exists():
        log.error("Boundary file not found: %s", fpath.resolve())
        log.error("Create it from sample_lake.json and rename to mission_boundary.json")
        sys.exit(1)

    with open(fpath) as f:
        data = json.load(f)

    # Validate required keys
    if "boundary" not in data:
        log.error("JSON is missing the 'boundary' key.")
        sys.exit(1)
    if len(data["boundary"]) < 3:
        log.error("'boundary' must have at least 3 points.")
        sys.exit(1)
    if "ground_station" not in data:
        log.error("JSON is missing 'ground_station' (lat/lon of launch point).")
        sys.exit(1)

    log.info("Loaded mission file: %s  (%d boundary points)",
             fpath.name, len(data["boundary"]))
    return data


def parse_boundary(data: dict):
    """Extract boundary as list of (lat, lon) tuples."""
    return [(p["lat"], p["lon"]) for p in data["boundary"]]


def parse_ground_station(data: dict):
    """Extract ground station as (lat, lon)."""
    gs = data["ground_station"]
    return gs["lat"], gs["lon"]


def parse_config(data: dict, args: argparse.Namespace):
    """
    Merge sweep settings: CLI args take priority over JSON config section,
    which takes priority over built-in defaults.
    """
    json_cfg = data.get("config", {})

    # sweep_width: CLI > JSON > default 3.0
    if args.sweep is not None:
        sweep_width = args.sweep
        log.info("Sweep width from CLI: %.1f m", sweep_width)
    elif "sweep_width_m" in json_cfg:
        sweep_width = float(json_cfg["sweep_width_m"])
        log.info("Sweep width from JSON: %.1f m", sweep_width)
    else:
        sweep_width = 3.0
        log.info("Sweep width default: %.1f m", sweep_width)

    # sweep_angle: CLI > JSON > default 0 degrees
    if args.angle is not None:
        sweep_angle = args.angle
        log.info("Sweep angle from CLI: %.1f deg", sweep_angle)
    elif "sweep_angle_deg" in json_cfg:
        sweep_angle = float(json_cfg["sweep_angle_deg"])
        log.info("Sweep angle from JSON: %.1f deg", sweep_angle)
    else:
        sweep_angle = 0.0

    return sweep_width, sweep_angle


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ASV Mission Runner — reads boundary from mission_boundary.json"
    )
    parser.add_argument(
        "--file", type=str, default=DEFAULT_BOUNDARY_FILE,
        help=f"Path to boundary JSON file (default: {DEFAULT_BOUNDARY_FILE})"
    )
    parser.add_argument(
        "--sim", action="store_true",
        help="Use MockHardware simulation (no STM32 or GPS required)"
    )
    parser.add_argument(
        "--port", type=str, default=DEFAULT_STM32_PORT,
        help=f"STM32 serial port (default: {DEFAULT_STM32_PORT})"
    )
    parser.add_argument(
        "--sweep", type=float, default=None,
        help="Override lane spacing in metres (overrides JSON config)"
    )
    parser.add_argument(
        "--angle", type=float, default=None,
        help="Override sweep angle in degrees (overrides JSON config)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore saved mission state and restart from waypoint 0"
    )
    args = parser.parse_args()

    # ── Load mission boundary ─────────────────────────────────────────────
    data           = load_mission_file(args.file)
    boundary       = parse_boundary(data)
    gs_lat, gs_lon = parse_ground_station(data)
    sweep_width, sweep_angle = parse_config(data, args)

    log.info("Ground station : (%.6f, %.6f)", gs_lat, gs_lon)
    log.info("Boundary points: %d", len(boundary))
    log.info("Sweep width    : %.1f m", sweep_width)
    log.info("Sweep angle    : %.1f deg", sweep_angle)

    # ── Select hardware backend ───────────────────────────────────────────
    hw = create_hardware(
        simulate   = args.sim,
        init_lat   = gs_lat,
        init_lon   = gs_lon,
        stm32_port = args.port,
    )

    # ── Launch mission ────────────────────────────────────────────────────
    mc = MissionController(
        boundary_coords = boundary,
        start_lat       = gs_lat,
        start_lon       = gs_lon,
        hardware        = hw,
        sweep_width     = sweep_width,
        sweep_angle     = sweep_angle,
        resume          = not args.no_resume,
    )

    log.info("Mission starting. Press Ctrl-C to stop gracefully.")
    mc.run()


if __name__ == "__main__":
    main()
