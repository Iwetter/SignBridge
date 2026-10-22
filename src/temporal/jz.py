"""Conservative geometric J/Z trajectories, not a trained ASL probability model.

Input coordinates are physical, unmirrored camera landmarks. Handedness is
anatomical; preview mirroring must never change these coordinates. Geometry is
measured in palm lengths and resampled by distance, not by frame number.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from src.contracts import FrameObservation, Prediction
from src.vision.features import primary_hand


def _shape(points: np.ndarray) -> str | None:
    palm = float(np.linalg.norm(points[[5, 9, 13, 17]] - points[0], axis=1).mean())
    # A tucked thumb near the curled middle/ring tips excludes L and Y without
    # overriding any static prediction. A pointing Z keeps its index upright.
    tucked = min(np.linalg.norm(points[4] - points[i]) for i in (12, 16)) < .65 * palm
    if not tucked:
        return None
    extended = []
    for base in (5, 9, 13, 17):
        joints = points[base:base + 4]
        length = float(np.linalg.norm(np.diff(joints, axis=0), axis=1).sum())
        if length < 1e-7:
            return None
        extended.append(float(np.linalg.norm(joints[-1] - joints[0])) / length)
    # Chain straightness survives J's wrist turn, unlike image-y comparisons.
    if extended[3] >= .78 and max(extended[:3]) <= .70:
        return 'J'
    direction = points[8] - points[5]
    if extended[0] >= .85 and max(extended[1:]) <= .88 and direction[1] < -.25 * np.linalg.norm(direction):
        return 'Z'
    return None


def _resample(points: np.ndarray, count: int = 48) -> np.ndarray:
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distances) > 1e-8]
    distances, points = distances[keep], points[keep]
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0)
    target = np.linspace(0., distances[-1], count)
    return np.column_stack([np.interp(target, distances, points[:, axis]) for axis in (0, 1)])


class JZRecognizer:
    """Emit a completed J/Z once, after settling; ordinary holds emit nothing.

    ``pending`` briefly reserves a potential starting pose. ``motion_active``
    requires a correctly directed trajectory prefix with its required shape.
    Returning from the endpoint only rearms; it cannot repeat a completed sign.
    """
    supported_labels = ('J', 'Z')

    def __init__(self, *, pending_ms: int = 650, settle_ms: int = 140,
                 min_duration_ms: int = 220, max_duration_ms: int = 4000,
                 max_gap_ms: int = 250, cooldown_ms: int = 350):
        for name, value in locals().copy().items():
            if name not in ('self',) and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f'{name} must be a positive integer')
        if pending_ms > 650 or min_duration_ms >= max_duration_ms:
            raise ValueError('pending must be <=650ms and min duration less than max duration')
        self.pending_ms, self.settle_ms = pending_ms, settle_ms
        self.min_duration_ms, self.max_duration_ms = min_duration_ms, max_duration_ms
        self.max_gap_ms, self.cooldown_ms = max_gap_ms, cooldown_ms
        self.reset()

    def reset(self) -> None:
        self._last_time = None
        self._hand = None
        self._cooldown_until = 0
        self._held_endpoint = None
        self._held_scale = None
        self._clear('Waiting for J/Z starting handshape')

    def _clear(self, reason: str) -> None:
        self._history = deque(maxlen=256)
        self._raw = deque(maxlen=256)
        self._label = None
        self._shape_since = None
        self._last_move = None
        self._scale = None
        self.motion_active = self.pending = False
        self._state, self._reason, self._progress = 'idle', reason, 0.

    def snapshot(self) -> dict:
        return {'mode': 'MOTION' if self.motion_active else 'STATIC',
                'state': self._state, 'label': self._label, 'progress': self._progress,
                'reason': self._reason, 'pending': self.pending,
                'trajectory': [list(point) for point in self._raw],
                'score_kind': 'geometric match; not a calibrated probability',
                'thresholds': {'pending_ms': self.pending_ms, 'settle_ms': self.settle_ms,
                               'min_duration_ms': self.min_duration_ms,
                               'max_duration_ms': self.max_duration_ms,
                               'max_gap_ms': self.max_gap_ms, 'cooldown_ms': self.cooldown_ms,
                               'minimum_palm_extent': {'J': .55, 'Z': .75}}}

    def update(self, observation: FrameObservation,
               static_prediction: Prediction | None = None) -> Prediction | None:
        # Static predictions are optional evidence only; omitted static J/Z
        # classes must not prevent the independently checked motion handshape.
        now = observation.timestamp_ms
        if isinstance(now, bool) or not isinstance(now, (int, np.integer)) or now < 0:
            raise ValueError('timestamp_ms must be a nonnegative integer')
        if self._last_time is not None and now < self._last_time:
            raise ValueError('JZRecognizer timestamps moved backwards; reset the stream')
        if now == self._last_time:
            return None
        if self._last_time is not None and now - self._last_time > self.max_gap_ms:
            self._clear('Tracking gap: incomplete trajectory discarded')
        self._last_time = now
        hand = primary_hand(observation)
        if hand is None:
            self._clear('No hand: incomplete trajectory discarded')
            self._held_endpoint = None
            self._hand = None
            return None
        if hand.handedness != self._hand:
            self._clear('Hand changed: incomplete trajectory discarded')
            self._held_endpoint = None
            self._hand = hand.handedness
        raw = np.asarray(hand.landmarks, dtype=float)
        if raw.shape != (21, 3) or not np.isfinite(raw).all():
            self._clear('Invalid hand landmarks')
            return None
        points = raw.copy()
        points[:, (0, 2)] *= observation.width / max(1, observation.height)
        scale = float(np.mean(np.linalg.norm(points[[5, 9, 13, 17]] - points[0], axis=1)))
        if scale < .015:
            self._clear('Hand too small for reliable motion geometry')
            return None
        label = _shape(points)
        if label is None:
            self._clear('Required pinky/index handshape not maintained')
            self._held_endpoint = None
            return None
        tip = 20 if label == 'J' else 8
        position = points[tip, :2].copy()
        position[0] *= -1 if hand.handedness == 'Right' else 1
        if self._held_endpoint is not None:
            self.pending = self.motion_active = False
            self._state, self._reason = 'cooldown', 'Completed; move away to begin a new stroke'
            if now < self._cooldown_until or np.linalg.norm(position - self._held_endpoint) < .40 * self._held_scale:
                return None
            self._held_endpoint = None
            self._clear('Rearmed for a new trajectory')
        if label != self._label:
            self._clear('Starting handshape; waiting for directed movement')
            self._label, self._shape_since, self._scale = label, now, scale
        if not .55 <= scale / self._scale <= 1.8:
            self._clear('Large depth change: trajectory discarded')
            return None
        self.pending = now - self._shape_since < self.pending_ms
        if not self._history:
            self._history.append((now, position))
            self._raw.append(tuple(raw[tip, :2]))
            self._last_move = now
            self._state = 'pending'
            return None
        movement = np.linalg.norm(position - self._history[-1][1]) / self._scale
        if movement >= .035:
            self._history.append((now, position))
            self._raw.append(tuple(raw[tip, :2]))
            self._last_move = now
        path = (np.asarray([item[1] for item in self._history]) - self._history[0][1]) / self._scale
        elapsed = now - self._history[0][0]
        if elapsed > self.max_duration_ms:
            self._clear('Trajectory timed out')
            return None
        # Retain a recent stationary origin so holding I/D first does not make
        # the later moving stroke artificially old.
        if len(path) == 1:
            self._history[0] = (now, position)
            self.motion_active = False
            self._state = 'pending' if self.pending else 'idle'
            return None
        if not self.motion_active and now - self._last_move >= self.settle_ms:
            # A settled return establishes the next stroke's origin. Do not
            # include the tail of that return in the next Z/J trajectory.
            self._history.clear()
            self._raw.clear()
            self._history.append((now, position))
            self._raw.append(tuple(raw[tip, :2]))
            self._last_move = now
            return None
        x, y = path[:, 0], path[:, 1]
        prefix = ((y.max() >= .15 and y[-1] > -.10 and x.max() < .30)
                  if label == 'J' else (x.max() >= .35 and y.min() > -.20))
        if not prefix:
            self.motion_active = False
            wrong_direction = (x.max() > .30 or y[-1] < -.30) if label == 'J' else (x[-1] < -.30 or y[-1] < -.30 or (y.max() > .50 and x.max() < .35))
            if wrong_direction:
                self._history.clear()
                self._raw.clear()
            return None
        self.motion_active = True
        self._state, self._reason = 'tracking', f'Tracking {label}; complete the stroke and briefly settle'
        self._progress = min(.90, float(np.linalg.norm(np.ptp(path, axis=0))) / 2.)
        score = self._match(label, path)
        if score is None:
            if now - self._last_move >= 500:
                # A single ordinary swipe is not a motion letter. Let the
                # stationary handshape resume the usual static dwell gate.
                self._history.clear()
                self._raw.clear()
                self.motion_active = self.pending = False
                self._shape_since = now - self.pending_ms
                self._state, self._reason = 'idle', 'Incomplete trajectory rejected'
            return None
        self._progress = .95
        moving_duration = self._history[-1][0] - self._history[0][0]
        if moving_duration < self.min_duration_ms or now - self._last_move < self.settle_ms:
            return None
        self._held_endpoint, self._held_scale = position.copy(), self._scale
        self._cooldown_until = now + self.cooldown_ms
        self.motion_active = self.pending = False
        self._state, self._reason, self._progress = 'completed', f'Completed {label} trajectory', 1.
        return Prediction(label, score, int(now), 'motion-geometric', scores={label: score})

    @staticmethod
    def _match(label: str, path: np.ndarray) -> float | None:
        x, y = path[:, 0], path[:, 1]
        if label == 'Z':
            if len(path) < 7:
                return None
            first = int(np.argmax(x))
            # Last horizontal may extend beyond the first, so detect the first
            # significant right-to-left reversal, not the global x maximum.
            for i in range(1, len(path) - 2):
                if x[i] >= .40 and x[i] - x[i + 1:].min() >= .40:
                    first = i
                    while first + 1 < len(path) and x[first + 1] > x[first]:
                        first += 1
                    break
            if first >= len(path) - 3:
                return None
            second = first + 1 + int(np.argmin(x[first + 1:]))
            a, b, c = path[first], path[second], path[-1]
            if (a[0] < .40 or abs(a[1]) > .35 * a[0] or
                    a[0] - b[0] < .40 or b[1] - a[1] < .35 or
                    c[0] - b[0] < .40 or abs(c[1] - b[1]) > .35 * (c[0] - b[0]) or
                    np.ptp(x) < .75 or np.ptp(y) < .35):
                return None
            template = np.array([[0., 0.], a, b, c])
        else:
            bottom = int(np.argmax(y))
            if bottom < 2 or bottom >= len(path) - 1:
                return None
            if (y[bottom] < .55 or x[-1] > -.25 or
                    y[bottom] - y[-1] < .13 or x.max() > .25 or
                    abs(x[np.argmax(y >= y[bottom] * .5)]) > .32):
                return None
            # J's lower curve has moderate natural radius variation. Compare
            # to several hook profiles; never accept a mere down/side swipe.
            height, width = y[bottom], -x[-1]
            templates = [np.array([[0., 0.], [0., .60], [-.08, .84],
                                   [-.35, 1.], [-.72, .95], [-1., end]]) * [width, height]
                         for end in (.55, .70, .85)]
            residual = min(float(np.sqrt(np.mean((_resample(path) - _resample(t)) ** 2)))
                           for t in templates)
            normalized_error = residual / max(height, width)
            return None if normalized_error > .16 else float(np.clip(1. - normalized_error, .80, .99))
        residual = float(np.sqrt(np.mean((_resample(path) - _resample(template)) ** 2)))
        normalized_error = residual / max(np.ptp(x), np.ptp(y))
        return None if normalized_error > .11 else float(np.clip(1. - normalized_error, .80, .99))
