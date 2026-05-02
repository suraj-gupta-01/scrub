"""
ASV Navigation System — Computer Vision / Trash Detection
==========================================================
Detects trash using a YOLO model (Ultralytics) and converts bounding-box
detections into GPS coordinates that the TrashHandler can act on.

SUPPORTED MODEL FORMATS
-----------------------
  .pt    — Standard PyTorch / Ultralytics checkpoint.
           Requires: pip install ultralytics torch
  .ncnn  — NCNN export folder (exported via yolo export format=ncnn).
           Requires: pip install ultralytics  (NCNN backend is built-in)

Both formats use the exact same Ultralytics `YOLO(model_path)` API —
the library auto-detects the format from the file extension / directory
structure.  So passing "best.pt" or "best_ncnn_model/" both work.

CAMERA-TO-GPS GEOMETRY
----------------------
We estimate trash distance from the camera using a pinhole model:

    ┌─────────────────────┐  ← image top (horizon)
    │                     │
    │        ┌──┐         │  ← bounding box centre (u, v)
    │        │  │         │
    │        └──┘         │
    │                     │
    └─────────────────────┘  ← image bottom (closest to boat)

  1.  The camera is mounted at height `h` above the waterline and
      tilted downward by angle `tilt` from horizontal.
  2.  A pixel at row `v` (0 = top) corresponds to a vertical angle
      offset from the camera's optical axis.
  3.  The depression angle below horizontal for that pixel is:
          angle_below_horiz = tilt + (v - cy) * fov_v / height
  4.  Ground distance from boat:
          d = h / tan(angle_below_horiz)           (flat water)
  5.  Horizontal angle from camera centre:
          theta = (u - cx) * fov_h / width          (degrees)
  6.  Bearing to trash:
          trash_bearing = boat_heading + theta
  7.  Convert (bearing, distance) → GPS via simple projection.

This is approximate (ignores lens distortion, wave tilt, etc.) but
sufficient for the 1-4 m detection range we operate in.

THREADING
---------
The TrashDetector runs its own daemon thread (like SensorHub producers).
Each frame:
  1. Capture from camera
  2. Run YOLO inference
  3. Convert detections to TrashDetection objects
  4. Inject into TrashHandler via inject_detection() / inject_detections_batch()

The control loop is never blocked by inference.

MOCK MODE
---------
MockTrashDetector is a test double that generates synthetic detections
at random positions near the boat.  Used by simulate.py and unit tests.
"""

import math
import os
import time
import threading
import logging
from typing import Optional, List, Tuple

import config

log = logging.getLogger("Vision")


# ═════════════════════════════════════════════════════════════════════════════
#  Camera-to-GPS geometry helpers
# ═════════════════════════════════════════════════════════════════════════════

EARTH_R = 6_371_000.0


def _pixel_to_range_bearing(
    u: float, v: float,
    img_w: int, img_h: int,
    fov_h_deg: float, fov_v_deg: float,
    cam_height_m: float,
    cam_tilt_deg: float,
) -> Optional[Tuple[float, float]]:
    """
    Convert a bounding-box centre pixel (u, v) to (range_m, bearing_offset_deg).

    Returns None if the geometry implies the target is at/above the horizon
    (i.e. infinite distance — unreachable).

    Args:
        u, v:           Pixel centre of the detection.
        img_w, img_h:   Frame dimensions.
        fov_h_deg:      Horizontal field of view (degrees).
        fov_v_deg:      Vertical field of view (degrees).
        cam_height_m:   Camera height above waterline (metres).
        cam_tilt_deg:   Downward tilt from horizontal (degrees, positive = down).

    Returns:
        (range_m, bearing_offset_deg) or None.
        bearing_offset_deg is relative to camera optical axis (+ = right).
    """
    cx, cy = img_w / 2.0, img_h / 2.0

    # Vertical angle of the pixel relative to optical axis
    # Positive v_offset means pixel is BELOW centre → closer to boat
    v_offset_deg = (v - cy) / img_h * fov_v_deg

    # Depression angle below horizontal for this pixel
    depression_deg = cam_tilt_deg + v_offset_deg

    if depression_deg <= 0.5:
        # At or above horizon — infinite / unreachable
        return None

    depression_rad = math.radians(depression_deg)
    range_m = cam_height_m / math.tan(depression_rad)

    # Sanity cap — beyond 15 m the geometry is too unreliable
    if range_m > 15.0 or range_m < 0.1:
        return None

    # Horizontal bearing offset from camera centre
    bearing_offset_deg = (u - cx) / img_w * fov_h_deg

    return range_m, bearing_offset_deg


def _project_to_gps(
    boat_lat: float, boat_lon: float,
    boat_heading_deg: float,
    range_m: float,
    bearing_offset_deg: float,
) -> Tuple[float, float]:
    """
    Project a (range, bearing_offset) detection into GPS coordinates.

    Args:
        boat_lat, boat_lon:  Current boat position.
        boat_heading_deg:    Current heading (degrees, true north).
        range_m:             Distance to target.
        bearing_offset_deg:  Horizontal offset from boat heading (+ = right).

    Returns:
        (lat, lon) of the projected trash position.
    """
    abs_bearing_deg = (boat_heading_deg + bearing_offset_deg) % 360.0
    abs_bearing_rad = math.radians(abs_bearing_deg)

    dlat = math.degrees(range_m * math.cos(abs_bearing_rad) / EARTH_R)
    dlon = math.degrees(
        range_m * math.sin(abs_bearing_rad)
        / (EARTH_R * math.cos(math.radians(boat_lat)))
    )

    return boat_lat + dlat, boat_lon + dlon


# ═════════════════════════════════════════════════════════════════════════════
#  TrashDetector — real camera + YOLO inference
# ═════════════════════════════════════════════════════════════════════════════

class TrashDetector:
    """
    Runs YOLO inference on camera frames in a background thread.

    Supports both .pt (PyTorch) and .ncnn (NCNN export) model formats.
    The Ultralytics YOLO() constructor auto-detects the format.

    Integration:
        from cv import TrashDetector
        from trash_handler import TrashHandler

        detector = TrashDetector(
            model_path="best.pt",      # or "best_ncnn_model/"
            trash_handler=trash_handler,
            get_boat_state=lambda: (lat, lon, heading),
        )
        detector.start()   # spawns camera + inference thread
        ...
        detector.stop()    # on shutdown
    """

    def __init__(
        self,
        model_path: str = None,
        trash_handler=None,
        get_boat_state=None,
        camera_id: int = None,
        frame_w: int = None,
        frame_h: int = None,
        inference_hz: float = None,
        conf_thresh: float = None,
        iou_thresh: float = None,
        fov_h_deg: float = None,
        fov_v_deg: float = None,
        cam_height_m: float = None,
        cam_tilt_deg: float = None,
    ):
        """
        Args:
            model_path:     Path to YOLO model (.pt file or .ncnn directory).
            trash_handler:  TrashHandler instance — detections injected here.
            get_boat_state: Callable returning (lat, lon, heading_deg).
                            Called each frame to get current boat position.
            camera_id:      OpenCV camera index.
            frame_w/h:      Capture resolution.
            inference_hz:   Max inference rate.
            conf_thresh:    YOLO confidence threshold.
            iou_thresh:     YOLO NMS IoU threshold.
            fov_h/v_deg:    Camera field of view.
            cam_height_m:   Camera mount height above waterline.
            cam_tilt_deg:   Camera tilt angle (degrees, positive = down).
        """
        self._model_path   = model_path   or config.VISION_MODEL_PATH
        self._trash_handler = trash_handler
        self._get_state    = get_boat_state

        self._camera_id    = camera_id    if camera_id    is not None else config.VISION_CAMERA_ID
        self._frame_w      = frame_w      if frame_w      is not None else config.VISION_FRAME_WIDTH
        self._frame_h      = frame_h      if frame_h      is not None else config.VISION_FRAME_HEIGHT
        self._inference_hz = inference_hz  if inference_hz is not None else config.VISION_INFERENCE_HZ
        self._conf_thresh  = conf_thresh   if conf_thresh  is not None else config.VISION_CONF_THRESH
        self._iou_thresh   = iou_thresh    if iou_thresh   is not None else config.VISION_IOU_THRESH
        self._fov_h        = fov_h_deg     if fov_h_deg    is not None else config.VISION_CAMERA_FOV_H
        self._fov_v        = fov_v_deg     if fov_v_deg    is not None else config.VISION_CAMERA_FOV_V
        self._cam_h        = cam_height_m  if cam_height_m is not None else config.VISION_CAMERA_HEIGHT_M
        self._cam_tilt     = cam_tilt_deg  if cam_tilt_deg is not None else config.VISION_CAMERA_TILT_DEG

        self._model  = None
        self._cap    = None
        self._running = False
        self._thread  = None

        # Stats
        self.frames_processed: int = 0
        self.detections_total: int = 0
        self.last_inference_ms: float = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Load model, open camera, start inference thread."""
        if self._running:
            return

        # ── Load YOLO model ───────────────────────────────────────────────
        log.info("Loading YOLO model from: %s", self._model_path)
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics not installed.  Run:  pip install ultralytics\n"
                "For NCNN support it is included automatically.\n"
                "For PyTorch .pt models, also install torch."
            )

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"YOLO model not found at: {self._model_path}\n"
                "Set VISION_MODEL_PATH in config.py or pass model_path= argument."
            )

        self._model = YOLO(self._model_path)

        # Detect format for logging
        is_ncnn = os.path.isdir(self._model_path)
        fmt = "NCNN" if is_ncnn else "PyTorch"
        log.info("Model loaded successfully (format: %s)", fmt)

        # ── Open camera ───────────────────────────────────────────────────
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV not installed.  Run:  pip install opencv-python")

        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self._camera_id}. "
                "Check device connection and permissions."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._frame_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_h)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera opened: requested %dx%d, actual %dx%d",
                 self._frame_w, self._frame_h, actual_w, actual_h)
        self._frame_w = actual_w
        self._frame_h = actual_h

        # ── Start inference thread ────────────────────────────────────────
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="TrashDetector",
            daemon=True,
        )
        self._thread.start()
        log.info("TrashDetector started @ %.1f Hz", self._inference_hz)

    def stop(self) -> None:
        """Signal thread to stop and release camera."""
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        log.info("TrashDetector stopped. Processed %d frames, %d detections.",
                 self.frames_processed, self.detections_total)

    # ── Inference thread ──────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """
        Background thread: capture → infer → convert → inject.
        Runs at self._inference_hz or slower (if inference takes longer).
        """
        interval = 1.0 / self._inference_hz

        while self._running:
            t0 = time.monotonic()

            try:
                detections = self._process_frame()
                if detections and self._trash_handler is not None:
                    self._trash_handler.inject_detections_batch(detections)
                    self.detections_total += len(detections)
            except Exception as exc:
                log.warning("Vision inference error: %s", exc)

            self.frames_processed += 1
            elapsed = time.monotonic() - t0
            self.last_inference_ms = elapsed * 1000

            time.sleep(max(0.0, interval - elapsed))

    def _process_frame(self) -> List:
        """
        Capture one frame, run YOLO, convert detections to TrashDetection objects.

        Returns:
            List of TrashDetection objects (may be empty).
        """
        from trash_handler import TrashDetection

        if self._cap is None or not self._cap.isOpened():
            return []

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return []

        # ── Get current boat state ────────────────────────────────────────
        if self._get_state is None:
            return []

        boat_lat, boat_lon, boat_heading = self._get_state()
        if boat_lat is None or boat_lon is None:
            return []   # no GPS fix yet

        # ── YOLO inference ────────────────────────────────────────────────
        results = self._model.predict(
            source=frame,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            verbose=False,
            stream=False,
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        # ── Convert detections ────────────────────────────────────────────
        detections = []
        for box in result.boxes:
            conf = float(box.conf[0])

            # Bounding box centre (xyxy → centre)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Class label
            cls_id = int(box.cls[0])
            label = result.names.get(cls_id, "trash") if result.names else "trash"

            # Convert pixel to range/bearing
            rb = _pixel_to_range_bearing(
                u=cx, v=cy,
                img_w=self._frame_w, img_h=self._frame_h,
                fov_h_deg=self._fov_h, fov_v_deg=self._fov_v,
                cam_height_m=self._cam_h,
                cam_tilt_deg=self._cam_tilt,
            )

            if rb is None:
                log.debug("Detection at pixel (%.0f, %.0f) maps to horizon — skipping.", cx, cy)
                continue

            range_m, bearing_offset_deg = rb

            # Project to GPS
            trash_lat, trash_lon = _project_to_gps(
                boat_lat, boat_lon, boat_heading,
                range_m, bearing_offset_deg,
            )

            detection = TrashDetection(
                lat=trash_lat,
                lon=trash_lon,
                confidence=conf,
                label=label,
            )

            detections.append(detection)
            log.debug(
                "Detection: label=%s conf=%.2f range=%.1fm bearing=%+.1f° → (%.6f, %.6f)",
                label, conf, range_m, bearing_offset_deg, trash_lat, trash_lon,
            )

        if detections:
            log.info("Frame: %d detection(s) converted to GPS.", len(detections))

        return detections

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status_line(self) -> str:
        return (
            f"Vision: {'RUN' if self._running else 'OFF'} | "
            f"frames={self.frames_processed} det={self.detections_total} "
            f"inf={self.last_inference_ms:.0f}ms"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  MockTrashDetector — for simulation and testing without camera/model
# ═════════════════════════════════════════════════════════════════════════════

class MockTrashDetector:
    """
    Test double that generates synthetic trash detections at fixed positions.

    Usage:
        mock = MockTrashDetector(
            trash_positions=[(12.972, 77.595), (12.973, 77.594)],
            trash_handler=handler,
            get_boat_state=lambda: (lat, lon, heading),
            detection_radius_m=5.0,
        )
        mock.start()
    """

    def __init__(
        self,
        trash_positions: List[Tuple[float, float]] = None,
        trash_handler=None,
        get_boat_state=None,
        detection_radius_m: float = 5.0,
        confidence: float = 0.85,
        update_hz: float = 2.0,
    ):
        self._positions      = trash_positions or []
        self._trash_handler  = trash_handler
        self._get_state      = get_boat_state
        self._detect_radius  = detection_radius_m
        self._confidence     = confidence
        self._update_hz      = update_hz
        self._running        = False
        self._thread         = None

        # Track which positions have been "collected" (removed from detection)
        self._collected = set()

        self.frames_processed = 0
        self.detections_total = 0
        self.last_inference_ms = 0.0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._mock_loop,
            name="MockTrashDetector",
            daemon=True,
        )
        self._thread.start()
        log.info("MockTrashDetector started with %d trash items.", len(self._positions))

    def stop(self) -> None:
        self._running = False
        log.info("MockTrashDetector stopped.")

    def _mock_loop(self) -> None:
        from trash_handler import TrashDetection
        from utils import haversine

        interval = 1.0 / self._update_hz

        while self._running:
            t0 = time.monotonic()

            if self._get_state and self._trash_handler:
                boat_lat, boat_lon, heading = self._get_state()

                if boat_lat is not None and boat_lon is not None:
                    detections = []
                    for i, (tlat, tlon) in enumerate(self._positions):
                        if i in self._collected:
                            continue
                        dist = haversine(boat_lat, boat_lon, tlat, tlon)
                        if dist <= self._detect_radius:
                            det = TrashDetection(
                                lat=tlat,
                                lon=tlon,
                                confidence=self._confidence,
                                label=f"trash_{i}",
                            )
                            detections.append(det)
                            log.debug("Mock detection: item %d at %.1f m", i, dist)

                    if detections:
                        self._trash_handler.inject_detections_batch(detections)
                        self.detections_total += len(detections)

            self.frames_processed += 1
            elapsed = time.monotonic() - t0
            self.last_inference_ms = elapsed * 1000
            time.sleep(max(0.0, interval - elapsed))

    def mark_collected(self, index: int) -> None:
        """Mark a trash item as collected (won't be detected again)."""
        self._collected.add(index)

    def status_line(self) -> str:
        remaining = len(self._positions) - len(self._collected)
        return (
            f"MockVision: {'RUN' if self._running else 'OFF'} | "
            f"remaining={remaining}/{len(self._positions)} "
            f"det={self.detections_total}"
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Factory helper
# ═════════════════════════════════════════════════════════════════════════════

def create_detector(
    simulate: bool = False,
    trash_handler=None,
    get_boat_state=None,
    model_path: str = None,
    trash_positions: List[Tuple[float, float]] = None,
):
    """
    Factory that returns the correct detector backend.

    Args:
        simulate:         If True, return MockTrashDetector.
        trash_handler:    TrashHandler instance.
        get_boat_state:   Callable → (lat, lon, heading_deg).
        model_path:       Path to YOLO model (.pt or .ncnn dir).
        trash_positions:  Mock trash positions (simulate mode only).

    Returns:
        TrashDetector or MockTrashDetector instance (not yet started).
    """
    if simulate:
        log.info("Vision mode: MOCK (no camera, synthetic detections)")
        return MockTrashDetector(
            trash_positions=trash_positions or [],
            trash_handler=trash_handler,
            get_boat_state=get_boat_state,
        )
    else:
        log.info("Vision mode: REAL (camera + YOLO)")
        return TrashDetector(
            model_path=model_path,
            trash_handler=trash_handler,
            get_boat_state=get_boat_state,
        )
