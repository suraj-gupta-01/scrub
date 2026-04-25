"""
ASV Navigation System - Simulation & Visualisation
Run a full mission simulation and show an animated matplotlib plot.

Usage:
    python simulate.py                     # default sample lake
    python simulate.py --obstacles         # inject random obstacles
    python simulate.py --angle 45          # sweep at 45°
    python simulate.py --sweep 5           # 5 m lane spacing
"""

import argparse
import math
import time
import random
import threading
import logging

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from map_handler import MapHandler
from coverage_planner import CoveragePlanner
from navigator import Navigator, NavState
from obstacle_handler import ObstacleHandler, ObstacleSignal
from recovery import PathRecovery
from hardware import MockHardware
import config

log = logging.getLogger("Simulate")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] — %(message)s",
                    datefmt="%H:%M:%S")


# ── Sample waterbody boundary ─────────────────────────────────────────────────
# Represents a small lake (~120 × 80 m). Replace with your actual GPS polygon.
SAMPLE_LAKE = [
    (12.97192, 77.59480),
    (12.97230, 77.59510),
    (12.97265, 77.59525),
    (12.97290, 77.59510),
    (12.97295, 77.59470),
    (12.97270, 77.59440),
    (12.97230, 77.59430),
    (12.97200, 77.59445),
    (12.97192, 77.59480),
]

# Ground station — just outside the lake
GROUND_STATION = (12.97180, 77.59475)


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_simulation(sweep_width: float, sweep_angle: float,
                   inject_obstacles: bool, speed: float) -> None:

    # ── Build map ─────────────────────────────────────────────────────────────
    mh = MapHandler.from_gps_polygon(SAMPLE_LAKE)
    log.info("Map: %s", mh)

    # ── Generate coverage path ────────────────────────────────────────────────
    planner = CoveragePlanner(mh, sweep_width=sweep_width, angle_deg=sweep_angle)
    waypoints = planner.generate()
    planner.save()
    log.info(planner.summary())

    if not waypoints:
        log.error("No waypoints generated — check polygon / sweep settings.")
        return

    # ── Hardware mock ─────────────────────────────────────────────────────────
    hw = MockHardware(*GROUND_STATION, boat_speed_mps=speed)

    # ── Sub-systems ───────────────────────────────────────────────────────────
    nav       = Navigator(waypoints, hardware=hw)
    obs_hdlr  = ObstacleHandler(hardware=hw)
    recovery  = PathRecovery(mh, waypoints)

    nav.start()

    # ── Matplotlib setup ──────────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.set_title("ASV Autonomous Navigation Simulation", color="#e0e0e0",
                 fontsize=13, pad=10)
    ax.set_xlabel("X (metres)", color="#888888")
    ax.set_ylabel("Y (metres)", color="#888888")

    # XY coordinates for plotting
    def to_xy_arr(pts):
        xs = [mh.to_xy(p[0], p[1])[0] for p in pts]
        ys = [mh.to_xy(p[0], p[1])[1] for p in pts]
        return xs, ys

    # Polygon boundary
    poly_xs, poly_ys = to_xy_arr([(p.lat, p.lon) for p in mh.gps_polygon])
    ax.fill(poly_xs, poly_ys, alpha=0.15, color="#1e88e5")
    ax.plot(poly_xs, poly_ys, color="#1e88e5", lw=1.5, label="Lake boundary")

    # Coverage waypoints
    wp_xs = [w.x for w in waypoints]
    wp_ys = [w.y for w in waypoints]
    ax.plot(wp_xs, wp_ys, "--", color="#555555", lw=0.8, zorder=1)
    ax.scatter(wp_xs, wp_ys, s=20, color="#aaaaaa", zorder=2, label="Waypoints")

    # Dynamic elements
    traj_line, = ax.plot([], [], "-", color="#00e5ff", lw=1.2,
                         alpha=0.8, label="Boat trajectory", zorder=3)
    boat_dot,  = ax.plot([], [], "o", color="#ff6d00", ms=8, zorder=5)
    wp_marker, = ax.plot([], [], "*", color="#ffeb3b", ms=12, zorder=4,
                         label="Active waypoint")

    legend = ax.legend(loc="upper right", facecolor="#1a1a2e",
                       edgecolor="#333333", labelcolor="#cccccc", fontsize=8)
    progress_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                            color="#90caf9", fontsize=9, va="top")
    status_text   = ax.text(0.02, 0.93, "", transform=ax.transAxes,
                            color="#a5d6a7", fontsize=8, va="top")

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.pause(0.01)

    # ── Obstacle injection thread ─────────────────────────────────────────────
    if inject_obstacles:
        def obstacle_injector():
            time.sleep(5)  # let navigation start first
            signals = ["OBSTACLE_FRONT", "OBSTACLE_LEFT", "OBSTACLE_RIGHT"]
            while not nav.is_complete:
                time.sleep(random.uniform(15, 30))
                if not nav.is_complete:
                    sig = random.choice(signals)
                    log.info(">>> Injecting simulated obstacle: %s", sig)
                    obs_hdlr.receive_signal(sig)

        threading.Thread(target=obstacle_injector, daemon=True).start()

    # ── Main simulation loop ──────────────────────────────────────────────────
    DT = config.SIM_DT

    while not nav.is_complete:
        # Physics
        hw.step(DT)

        lat, lon     = hw.get_gps()
        heading      = hw.get_heading()
        nav_l, nav_r = nav.update(lat, lon, heading)

        # Obstacle override
        left, right, avoiding = obs_hdlr.process(nav_l, nav_r)
        hw.set_motor_speed(left, right)

        # Recovery after avoidance
        if obs_hdlr.avoidance_complete:
            obs_hdlr.avoidance_complete = False
            resume = recovery.find_best_waypoint(lat, lon, nav.current_idx)
            nav.skip_to(resume)
            nav.state = NavState.NAVIGATING

        # Update plot
        traj_xs = [mh.to_xy(p[0], p[1])[0] for p in hw.trajectory[-500:]]
        traj_ys = [mh.to_xy(p[0], p[1])[1] for p in hw.trajectory[-500:]]
        traj_line.set_data(traj_xs, traj_ys)

        bx, by = mh.to_xy(lat, lon)
        boat_dot.set_data([bx], [by])

        wp = nav.active_waypoint
        if wp:
            wp_marker.set_data([wp.x], [wp.y])

        progress_text.set_text(
            f"Progress: {nav.progress*100:.1f}%  "
            f"WP {nav.current_idx}/{len(waypoints)}"
        )
        status_text.set_text(
            f"{'⚠ AVOIDING' if avoiding else '◉ NAVIGATING'}  "
            f"dist={nav.dist_to_wp:.1f}m  "
            f"hdg_err={nav.heading_err:+.1f}°"
        )

        # Auto-scale
        if len(traj_xs) > 1:
            all_xs = wp_xs + traj_xs + [bx]
            all_ys = wp_ys + traj_ys + [by]
            pad = 10
            ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
            ax.set_ylim(min(all_ys) - pad, max(all_ys) + pad)

        plt.pause(DT / 2)

    # Mission complete
    hw.stop()
    log.info("=== SIMULATION COMPLETE ===")
    log.info("Total trajectory points: %d", len(hw.trajectory))
    progress_text.set_text("✓ MISSION COMPLETE")
    progress_text.set_color("#69f0ae")
    plt.ioff()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASV Navigation Simulation")
    parser.add_argument("--sweep",     type=float, default=config.SWEEP_WIDTH_M,
                        help=f"Lane spacing in metres (default {config.SWEEP_WIDTH_M})")
    parser.add_argument("--angle",     type=float, default=0.0,
                        help="Sweep angle in degrees (default 0 = E-W lanes)")
    parser.add_argument("--obstacles", action="store_true",
                        help="Inject random obstacle signals during simulation")
    parser.add_argument("--speed",     type=float, default=config.SIM_BOAT_SPEED,
                        help=f"Boat speed m/s (default {config.SIM_BOAT_SPEED})")
    args = parser.parse_args()

    run_simulation(
        sweep_width=args.sweep,
        sweep_angle=args.angle,
        inject_obstacles=args.obstacles,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
