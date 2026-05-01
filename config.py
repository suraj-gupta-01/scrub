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

# ── Trash Collection ──────────────────────────────────────────────────────────
TRASH_DETECTION_ENABLED   = True    # master switch for trash collection
TRASH_MIN_CONFIDENCE      = 0.65    # minimum YOLO confidence to act
TRASH_DETECTION_RADIUS_M  = 4.0     # only divert if trash within this range
TRASH_MIN_DETECTION_M     = 0.5     # ignore trash closer than this
TRASH_MAX_DEVIATION_M     = 8.0     # max distance boat will travel from path
TRASH_COLLECTION_TIMEOUT_S = 30.0   # max time for entire collection attempt
TRASH_MAX_PER_LANE        = 3       # max collections per coverage lane
TRASH_COLLECTION_RADIUS_M = 1.5     # distance to consider trash "reached"
TRASH_DWELL_S             = 3.0     # seconds to hold for physical collection
TRASH_APPROACH_SPEED      = 0.4     # normalised 0-1, slower for precision
TRASH_COOLDOWN_RADIUS_M   = 3.0     # ignore detections near recent collections
TRASH_COOLDOWN_S          = 60.0    # cooldown duration (seconds)

# ── Vision / Camera ───────────────────────────────────────────────────────────
VISION_MODEL_PATH    = "best.pt"     # YOLO model: .pt (PyTorch) or .ncnn dir
VISION_CAMERA_ID     = 0             # camera device index (0 = default)
VISION_FRAME_WIDTH   = 640           # capture resolution width
VISION_FRAME_HEIGHT  = 480           # capture resolution height
VISION_INFERENCE_HZ  = 5.0           # max inference rate (Hz)
VISION_CONF_THRESH   = 0.5           # YOLO confidence threshold
VISION_IOU_THRESH    = 0.45          # YOLO NMS IoU threshold
VISION_CAMERA_FOV_H  = 62.2          # horizontal FOV degrees (PiCam v2)
VISION_CAMERA_FOV_V  = 48.8          # vertical FOV degrees
VISION_CAMERA_HEIGHT_M = 0.5         # camera mount height above waterline
VISION_CAMERA_TILT_DEG = 30.0        # downward tilt angle from horizontal

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
