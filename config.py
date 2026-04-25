"""
ASV Navigation System - Configuration
All tunable parameters in one place.
"""

# ── Sweep / Coverage ──────────────────────────────────────────────────────────
SWEEP_WIDTH_M        = 3.0     # metres between parallel lanes
POLYGON_BUFFER_M     = 1.5     # inset from boundary before sweeping

# ── Waypoint Following ────────────────────────────────────────────────────────
WAYPOINT_RADIUS_M    = 2.0     # distance to consider a waypoint reached
LOOKAHEAD_M          = 4.0     # pure-pursuit lookahead distance
HEADING_KP           = 1.2     # proportional gain for heading control
MAX_TURN_RATE        = 1.0     # normalised [-1, 1] max differential

# ── Motor / Speed ─────────────────────────────────────────────────────────────
CRUISE_SPEED         = 0.6     # normalised 0-1
OBSTACLE_SPEED       = 0.4     # speed during avoidance manoeuvre
TURN_SPEED           = 0.35    # base speed while turning

# ── Obstacle Avoidance ────────────────────────────────────────────────────────
AVOIDANCE_TURN_TIME  = 2.0     # seconds to turn away from obstacle
AVOIDANCE_FWD_TIME   = 1.5     # seconds to move forward after turn
OBSTACLE_CLEAR_DIST  = 5.0     # metres: obstacle considered clear

# ── Recovery ─────────────────────────────────────────────────────────────────
RECOVERY_THRESHOLD_M = 3.0     # how far off-path before re-routing
RECOVERY_SPEED       = 0.5

# ── Simulation ────────────────────────────────────────────────────────────────
SIM_DT               = 0.1     # seconds per simulation step
SIM_BOAT_SPEED       = 2.0     # m/s simulated speed
SIM_TURN_RATE        = 45.0    # degrees/s simulated turn rate

# ── Persistence ───────────────────────────────────────────────────────────────
WAYPOINT_FILE        = "mission_waypoints.json"
STATE_FILE           = "mission_state.json"

# ── GPS (real hardware) ───────────────────────────────────────────────────────
GPS_PORT             = "/dev/ttyAMA0"
GPS_BAUD             = 9600
GPS_UPDATE_HZ        = 1.0
