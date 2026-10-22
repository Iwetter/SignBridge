"""OpenCV camera + MediaPipe Tasks Hand Landmarker, imported only when used.

Official implementation reference:
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python
Model bundle: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from src.contracts import FrameObservation, HandObservation
from .features import validated_landmarks

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


class CameraError(RuntimeError):
    """Actionable camera/dependency/model failure for the UI to surface."""


@dataclass(frozen=True)
class CapturedFrame:
    """One canonical, unmirrored BGR frame; controller stamps its generation."""
    frame_id: int
    timestamp_ms: int
    bgr: np.ndarray
    capture_ms: float
    generation: int = 0


class CameraCapture:
    """Capture only. The creating worker must also call read() and close()."""
    def __init__(self, camera_index: int | None = None, *, backend: str | None = None,
                 width: int | None = None, height: int | None = None, input_mirrored: bool = False,
                 capture_factory=None, cv2_module=None, diagnostic=None):
        from src.camera_settings import validate_selection
        validate_selection(camera_index, backend)
        self._diagnostic = diagnostic or (lambda **fields: None)
        self._description = f'Camera index {camera_index}, backend {backend}'
        self._capture = None
        self._last_timestamp = -1
        self._frame_id = 0
        self._closed = False
        self.input_mirrored = bool(input_mirrored)
        started = time.perf_counter()
        try:
            if cv2_module is None:
                import cv2 as cv2_module
            self._cv2 = cv2_module
            self._capture = (capture_factory or cv2_module.VideoCapture)(camera_index, getattr(cv2_module,'CAP_'+backend))
            opened = bool(self._capture.isOpened())
            actual_backend = None
            if opened:
                try: actual_backend = self._capture.getBackendName()
                except Exception: pass
            self._diagnostic(event='opened', cap_is_opened=opened, actual_backend=actual_backend)
            if not opened:
                raise CameraError(f'{self._description}: cap.isOpened() returned False; first frame was not attempted.')
            if width is not None:
                self._capture.set(cv2_module.CAP_PROP_FRAME_WIDTH, int(width))
            if height is not None:
                self._capture.set(cv2_module.CAP_PROP_FRAME_HEIGHT, int(height))
            self._capture.set(cv2_module.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            self.close()
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"Camera could not start: {exc}") from exc
        self.startup_ms = (time.perf_counter() - started) * 1000

    def read(self) -> CapturedFrame:
        if self._closed:
            raise CameraError("CameraCapture is closed")
        started = time.perf_counter()
        first = self._frame_id == 0
        try:
            ok, frame = self._capture.read()
        except Exception as exc:
            self._diagnostic(event='first_frame' if first else 'read_error',
                             first_frame_success=False if first else None, error=str(exc))
            raise CameraError(f'{self._description}: {"first frame" if first else "frame read"} raised {exc}') from exc
        capture_ms = (time.perf_counter() - started) * 1000
        if not ok or frame is None or frame.size == 0:
            message = f'{self._description}: {"first frame" if first else "frame read"} failed (read success={bool(ok)}, frame={"None" if frame is None else "empty"}).'
            self._diagnostic(event='first_frame' if first else 'read_error',
                             first_frame_success=False if first else None, error=message)
            raise CameraError(message)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            message = f'{self._description}: unexpected frame shape {frame.shape}, dtype {frame.dtype}.'
            self._diagnostic(event='first_frame' if first else 'read_error',
                             first_frame_success=False if first else None, error=message)
            raise CameraError(message)
        if first:
            self._diagnostic(event='first_frame', first_frame_success=True,
                             frame_dimensions=[int(frame.shape[1]),int(frame.shape[0])])
        timestamp = max(time.monotonic_ns() // 1_000_000, self._last_timestamp + 1)
        self._last_timestamp = timestamp
        self._frame_id += 1
        # A driver mirror is corrected once here. Qt independently mirrors only
        # presentation; inference and stored coordinates see these same pixels.
        frame = np.ascontiguousarray(frame[:, ::-1]) if self.input_mirrored else np.ascontiguousarray(frame)
        frame.flags.writeable = False
        return CapturedFrame(self._frame_id, timestamp, frame, capture_ms)

    def close(self) -> None:
        self._closed = True
        if self._capture is not None:
            capture, self._capture = self._capture, None
            capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class HandTracker:
    """Resident MediaPipe Tasks inference, independent of camera lifetime.

    detect() receives unmirrored BGR pixels. Unlike legacy Solutions Hands,
    Tasks already reverses the model's category mapping (0=Right, 1=Left), so
    its category is anatomical on unmirrored pixels: do NOT swap it again.
    Source: MediaPipe v0.10.35 hand_landmarks_detector_graph.cc, lines 116-129:
    https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarks_detector_graph.cc#L116-L129
    Legacy mapping (0=Left, 1=Right):
    https://github.com/google-ai-edge/mediapipe/blob/v0.10.35/mediapipe/modules/hand_landmark/handedness.txt
    Never use screen position to infer physical hand. Driver mirrors must be
    corrected by CameraCapture/SourceController before calling detect().
    """
    normalization_note = "Tasks category retained; unmirrored ML coordinates; selfie preview only"

    def __init__(self, model_path="models/hand_landmarker.task", *, confidence: float = 0.5,
                 landmarker_factory=None, image_factory=None):
        self._landmarker = None
        self._last_timestamp = -1
        self._closed = False
        self.last_latency_ms = None
        started = time.perf_counter()
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        # Both injections are required to avoid importing optional camera/ML
        # runtimes in unit tests and to keep the exact RGB input observable.
        if landmarker_factory is not None:
            if image_factory is None:
                raise ValueError("image_factory is required with landmarker_factory")
            self._landmarker = landmarker_factory()
            self._image_factory = image_factory
            self.startup_ms = (time.perf_counter() - started) * 1000
            return
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise CameraError(
                f"Hand Landmarker model missing: {model_path}. Run the project setup "
                "to obtain Google's hand_landmarker.task bundle; this is separate "
                "from the ASL model learned during calibration."
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except (ImportError, OSError) as exc:
            raise CameraError(
                "Hand tracking needs a working mediapipe installation. "
                "Run the project setup with the supported Python environment."
            ) from exc
        self._image_factory = lambda rgb: mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        try:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=confidence,
                min_hand_presence_confidence=confidence,
                min_tracking_confidence=confidence,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            self.close()
            if isinstance(exc, CameraError):
                raise
            raise CameraError(f"Hand tracker could not start: {exc}") from exc
        self.startup_ms = (time.perf_counter() - started) * 1000

    def detect(self, bgr: np.ndarray, timestamp_ms: int) -> FrameObservation:
        if self._closed:
            raise CameraError("HandTracker is closed")
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8 or bgr.size == 0:
            raise CameraError("HandTracker needs a nonempty uint8 BGR image")
        started = time.perf_counter()
        timestamp = max(int(timestamp_ms), self._last_timestamp + 1)
        self._last_timestamp = timestamp
        mp_image = self._image_factory(np.ascontiguousarray(bgr[:, :, ::-1]))
        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp)
        except Exception as exc:
            raise CameraError(f"Hand landmark inference failed: {exc}") from exc
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000
        hands = []
        for points, categories in zip(result.hand_landmarks, result.handedness):
            if not categories:
                continue
            category = max(categories, key=lambda item: item.score)
            if category.category_name not in ("Right", "Left"):
                continue
            coordinates = np.asarray([(p.x, p.y, p.z) for p in points], dtype=np.float32)
            try:
                validated_landmarks(coordinates)
            except ValueError:
                continue
            score = float(category.score)
            if not np.isfinite(score) or not 0 <= score <= 1:
                continue
            hands.append(HandObservation(coordinates, category.category_name, score,
                                         raw_handedness=category.category_name))
        height, width = bgr.shape[:2]
        return FrameObservation(timestamp, hands, width, height)

    def close(self) -> None:
        self._closed = True
        if self._landmarker is not None:
            landmarker, self._landmarker = self._landmarker, None
            landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class CameraTracker:
    """Synchronous compatibility adapter for the existing collection CLI."""
    def __init__(self, camera_index: int = 0, model_path="models/hand_landmarker.task",
                 *, width: int = 640, height: int = 480, confidence: float = 0.5,
                 input_mirrored: bool = False, capture_factory=None, tracker_factory=None, backend='ANY'):
        self._capture = None
        self._tracker = None
        self._closed = False
        try:
            self._tracker = (tracker_factory or HandTracker)(model_path, confidence=confidence)
            self._capture = (capture_factory or CameraCapture)(camera_index, width=width, height=height,
                                                              input_mirrored=input_mirrored, backend=backend)
        except Exception:
            self.close()
            raise

    def read(self) -> tuple[np.ndarray, FrameObservation]:
        if self._closed:
            raise CameraError("CameraTracker is closed")
        frame = self._capture.read()
        return frame.bgr, self._tracker.detect(frame.bgr, frame.timestamp_ms)

    def close(self) -> None:
        self._closed = True
        try:
            if self._capture is not None:
                capture, self._capture = self._capture, None
                capture.close()
        finally:
            if self._tracker is not None:
                tracker, self._tracker = self._tracker, None
                tracker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def draw_landmarks(frame: np.ndarray, observation: FrameObservation, mirror: bool = False) -> np.ndarray:
    """Return an annotated copy; mirror pixels and landmark positions together."""
    import cv2

    display = cv2.flip(frame, 1) if mirror else frame.copy()
    height, width = display.shape[:2]
    for hand in observation.hands:
        try:
            points = validated_landmarks(hand.landmarks)
        except (TypeError, ValueError):
            continue
        pixels = []
        for x, y, _ in points:
            px = int(round(float(x) * (width - 1)))
            if mirror:
                px = width - 1 - px
            py = int(round(float(y) * (height - 1)))
            pixels.append((int(np.clip(px, -width, width * 2)), int(np.clip(py, -height, height * 2))))
        color = (80, 230, 140) if hand.handedness == "Right" else (240, 175, 85)
        for start, end in HAND_CONNECTIONS:
            cv2.line(display, pixels[start], pixels[end], color, 2, cv2.LINE_AA)
        for point in pixels:
            cv2.circle(display, point, 3, (250, 250, 250), -1, cv2.LINE_AA)
        wrist_x, wrist_y = pixels[0]
        cv2.putText(display, hand.handedness, (max(0, wrist_x - 20), max(18, wrist_y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return display
