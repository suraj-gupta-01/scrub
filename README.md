# ASV Autonomous Navigation System
## Raspberry Pi 5 Deployment Guide

---

### Architecture Overview

```
controller.py          ← Main mission loop (10 Hz)
  ├── map_handler.py   ← GPS ↔ XY, polygon management
  ├── coverage_planner.py  ← Boustrophedon path generation
  ├── navigator.py     ← Pure-pursuit waypoint following
  ├── obstacle_handler.py  ← Reactive avoidance (thread-safe)
  ├── converter.py     ← Convert different coordinate input format to expected format
  ├── recovery.py      ← Path rejoin after avoidance
  ├── hardware.py      ← MockHardware / RealHardware
  ├── utils.py         ← Haversine, bearing, geometry
  └── config.py        ← All tunable parameters
```

---

### Data Flow

```
GPS fix → Navigator.update() → [left_cmd, right_cmd]
                                      ↓
              ObstacleHandler.process() → [override or pass-through]
                                      ↓
                          hw.set_motor_speed(L, R)

Obstacle signal → ObstacleHandler.receive_signal()  (any thread)
                       ↓ (when done)
              PathRecovery.find_best_waypoint()
                       ↓
              Navigator.skip_to(resume_idx)
```

---

### Quickstart (Desktop Simulation)

```bash
# 1. Install dependencies
pip install numpy matplotlib

# 2. Run simulation (default lake boundary)
python simulate.py

# 3. Simulate with obstacle injection
python simulate.py --obstacles

# 4. Custom sweep angle and lane spacing
python simulate.py --sweep 5 --angle 30 --obstacles
```

---

### Raspberry Pi 5 Deployment

#### 1. System Requirements

```bash
sudo apt update && sudo apt install -y python3-pip python3-numpy pigpio
pip3 install pyserial pynmea2 matplotlib numpy
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

#### 2. GPS Wiring

| GPS Module Pin | RPi 5 GPIO |
|----------------|------------|
| VCC            | 3.3 V (Pin 1) |
| GND            | GND (Pin 6)   |
| TX             | GPIO 15 / UART RX (Pin 10) |
| RX             | GPIO 14 / UART TX (Pin 8)  |

Enable UART in `/boot/config.txt`:
```
enable_uart=1
dtoverlay=disable-bt
```

#### 3. Motor ESC Wiring (PWM via pigpio)

| Signal       | GPIO Pin | Notes              |
|--------------|----------|--------------------|
| Left ESC PWM | GPIO 12  | 50 Hz servo PWM    |
| Right ESC PWM| GPIO 13  | 50 Hz servo PWM    |
| GND          | Any GND  | Common ground req. |

Edit `hardware.py` → `RealHardware.LEFT_MOTOR_PIN / RIGHT_MOTOR_PIN` to match
your actual wiring.

#### 4. Run the Mission

```python
# main.py — edit boundary and run
from controller import MissionController
from hardware import RealHardware

LAKE_BOUNDARY = [
    (12.97192, 77.59480),
    (12.97230, 77.59510),
    # ... your GPS polygon from GIS / OSM
]
GROUND_STATION_GPS = (12.97180, 77.59475)

hw = RealHardware(gps_port="/dev/ttyAMA0", gps_baud=9600)

mc = MissionController(
    boundary_coords=LAKE_BOUNDARY,
    start_lat=GROUND_STATION_GPS[0],
    start_lon=GROUND_STATION_GPS[1],
    hardware=hw,
    sweep_width=3.0,   # metres
    resume=True,       # auto-resume if restarted
)
mc.run()
```

```bash
python3 main.py
```

#### 5. Run as a systemd Service (auto-start on boot)

```ini
# /etc/systemd/system/asv-mission.service
[Unit]
Description=ASV Autonomous Mission
After=network.target pigpiod.service

[Service]
ExecStart=/usr/bin/python3 /home/pi/asv_navigation/main.py
WorkingDirectory=/home/pi/asv_navigation
Restart=on-failure
RestartSec=10
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable asv-mission
sudo systemctl start asv-mission
sudo journalctl -u asv-mission -f   # live logs
```

---

### Integrating Your Obstacle Sensor

```python
# From your sensor thread / ROS node / serial listener:
mc.inject_obstacle_signal("OBSTACLE_FRONT")   # or LEFT / RIGHT / CLEAR
```

The controller is thread-safe. Call `inject_obstacle_signal` from any thread.

---

### Tuning Parameters (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SWEEP_WIDTH_M` | 3.0 | Metres between coverage lanes |
| `POLYGON_BUFFER_M` | 1.5 | Safety margin from shore |
| `WAYPOINT_RADIUS_M` | 2.0 | Distance to consider WP reached |
| `HEADING_KP` | 1.2 | Heading proportional gain |
| `CRUISE_SPEED` | 0.6 | Normalised thruster speed |
| `AVOIDANCE_TURN_TIME` | 2.0 | Seconds to turn during avoidance |
| `AVOIDANCE_FWD_TIME` | 1.5 | Seconds forward after avoidance turn |
| `RECOVERY_THRESHOLD_M` | 3.0 | Off-path distance triggering recovery |

---

### Output Files

| File | Description |
|------|-------------|
| `mission_waypoints.json` | Generated coverage waypoints (GPS + XY) |
| `mission_state.json` | Live mission progress (auto-saved) |
| `sample_lake.json` | Example polygon input format |

---

### Extracting Your Lake Boundary from OpenStreetMap

1. Open **JOSM** or **QGIS** and download OSM data for your area.
2. Select the water body polygon.
3. Export vertex coordinates as CSV (lat, lon).
4. Format as Python list of tuples and pass to `MissionController`.

Or use **Overpass API**:
```
[out:json];
way["natural"="water"](around:500, YOUR_LAT, YOUR_LON);
(._;>;);
out body;
```
Then parse the returned node coordinates.

---

### Mission State Recovery

If the RPi loses power mid-mission:
- `mission_state.json` stores the last waypoint index.
- On restart with `resume=True`, the mission continues from where it stopped.
- The saved `mission_waypoints.json` is reloaded (no re-planning needed).

---

### File Structure

```
asv_navigation/
├── config.py               # All parameters
├── utils.py                # Math helpers
├── map_handler.py          # Polygon + coordinate conversion
├── coverage_planner.py     # Lawnmower path generation
├── navigator.py            # Waypoint following
├── obstacle_handler.py     # Reactive avoidance
├── recovery.py             # Path rejoin logic
├── hardware.py             # Mock + Real hardware drivers
├── controller.py           # Main mission loop
├── simulate.py             # Desktop simulation + visualisation
├── sample_lake.json        # Example boundary input
├── mission_waypoints.json  # Generated waypoints (auto-created)
└── mission_state.json      # Mission progress  (auto-created)
```
