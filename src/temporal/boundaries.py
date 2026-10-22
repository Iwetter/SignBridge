"""Observed motion boundaries; never modify the static handshape feature contract.

Inputs are original, unmirrored camera landmarks. A signer's right is decreasing
camera x for either anatomical hand. Preview mirroring belongs to rendering.
"""
from __future__ import annotations

import math
from collections import deque
from numbers import Integral

import numpy as np

from src.contracts import FrameObservation, Prediction, STATIC_LABELS
from src.temporal.decoder import _timestamp
from src.vision.features import normalize_landmarks, primary_hand, validated_landmarks


def _duration(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class RepeatDetector:
    """A stable static letter, a deliberate rightward shift, then a settled hand.

    ``threshold`` is displacement in palm radii, measured independently at wrist
    and palm center. Gaps, confidence drops, shape changes and hand switches
    discard partial movements. This supplements the decoder's existing release
    mechanism; it does not unlock or commit anything by itself.
    """

    max_gap_ms = 250
    settle_ms = 120
    max_motion_ms = 2000
    shape_tolerance = .18
    settle_tolerance = .10

    def __init__(self, threshold: float = .65, cooldown_ms: int = 700,
                 confidence_threshold: float = .8):
        if not math.isfinite(threshold) or threshold <= .15:
            raise ValueError("repeat movement threshold must be finite and > 0.15 palm radii")
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.threshold = float(threshold)
        self.cooldown_ms = _duration("cooldown_ms", cooldown_ms)
        self.confidence_threshold = float(confidence_threshold)
        self.reset()

    def reset(self) -> None:
        self._last_timestamp = None
        self._last_commit = None
        self._last_confident = None
        self._discard("Waiting for a committed static letter")

    def _discard(self, reason: str) -> None:
        self._key = None
        self._baseline = None
        self._shape = None
        self._scale = None
        self._movement_since = None
        self._history = deque()
        self._state = "WAITING"
        self._progress = 0.
        self._reason = reason

    def snapshot(self) -> dict:
        return {"state": self._state, "progress": self._progress,
                "reason": self._reason, "threshold": self.threshold,
                "cooldown_ms": self.cooldown_ms,
                "last_repeat_ms": self._last_commit}

    def _settled(self, now: int, positions: np.ndarray, scale: float) -> bool:
        self._history.append((now, positions.copy()))
        # Keep the last observation at/before the window boundary.
        while len(self._history) > 1 and self._history[1][0] <= now - self.settle_ms:
            self._history.popleft()
        if now - self._history[0][0] < self.settle_ms:
            return False
        points = np.stack([sample for _, sample in self._history])
        return bool(np.max(np.linalg.norm(points - positions, axis=2)) / scale <= self.settle_tolerance)

    def prime(self, observation, prediction, locked_label):
        """The static dwell already proved this pose stable at commitment."""
        self.update(observation, prediction, locked_label)
        hand = primary_hand(observation)
        if hand is not None and self._key is not None and self._history and self._baseline is None:
            xy = hand.landmarks[:, :2] * [observation.width, observation.height]
            self._baseline = self._history[-1][1].copy()
            self._shape = normalize_landmarks(hand.landmarks, hand.handedness)
            self._scale = float(np.linalg.norm(xy[[5, 9, 13, 17]] - xy[0], axis=1).max())
            self._state, self._reason = 'ARMED', 'Move to your right, then hold briefly to repeat'

    def update(self, observation: FrameObservation, prediction: Prediction | None,
               locked_label: str | None) -> bool:
        now = _timestamp(observation.timestamp_ms)
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("RepeatDetector timestamps moved backwards; reset before a new stream")
        if self._last_timestamp is not None and now - self._last_timestamp > self.max_gap_ms:
            self._discard("Observation gap: movement discarded")
        self._last_timestamp = now
        valid_prediction = (prediction is not None and locked_label in STATIC_LABELS
                            and prediction.label == locked_label
                            and math.isfinite(prediction.confidence)
                            and self.confidence_threshold <= prediction.confidence <= 1
                            and 0 <= now - prediction.timestamp_ms <= self.max_gap_ms)
        hand = primary_hand(observation)
        if not valid_prediction or hand is None:
            # Short classifier flicker during movement may retain the origin, but
            # cannot commit. Fresh confident evidence is required at the endpoint.
            confident_other = (prediction is not None and prediction.label != locked_label
                               and prediction.confidence >= self.confidence_threshold)
            if (hand is not None and not confident_other and self._last_confident is not None
                    and now-self._last_confident <= 160 and self._key is not None
                    and hand.handedness == self._key[1]):
                self._state, self._reason = 'TRACKING', 'Brief confidence dip; waiting for confident endpoint'
                return False
            self._discard("Waiting for the same confident committed handshape")
            return False
        self._last_confident = now
        if observation.width <= 0 or observation.height <= 0:
            self._discard("Invalid camera dimensions")
            return False
        points = validated_landmarks(hand.landmarks).astype(np.float64)
        # Pixels correct for non-square images; z is irrelevant to displacement.
        xy = points[:, :2] * [observation.width, observation.height]
        scale = float(np.linalg.norm(xy[[5, 9, 13, 17]] - xy[0], axis=1).max())
        if not math.isfinite(scale) or scale < 2:
            self._discard("Hand too small for reliable movement")
            return False
        positions = np.stack((xy[0], xy[[0, 5, 9, 13, 17]].mean(axis=0)))
        shape = normalize_landmarks(points, hand.handedness)
        key = (locked_label, hand.handedness, observation.width, observation.height)
        if key != self._key:
            self._discard("Establishing a stable hand position")
            self._key = key
        settled = self._settled(now, positions, scale)
        if self._baseline is None:
            if settled:
                self._baseline, self._shape, self._scale = positions.copy(), shape, scale
                self._state, self._reason = "ARMED", "Move to your right, then hold briefly to repeat"
            return False
        if (float(np.sqrt(np.mean((shape - self._shape) ** 2))) > self.shape_tolerance
                or not .65 <= scale / self._scale <= 1.5):
            self._discard("Handshape or camera distance changed: movement discarded")
            return False
        displacement = (self._baseline - positions) / self._scale
        rightward = float(displacement[:, 0].min())
        vertical = float(np.abs(displacement[:, 1]).max())
        self._progress = max(0., min(1., rightward / self.threshold))
        if rightward > .15 and self._movement_since is None:
            self._movement_since = now
        if self._movement_since is not None and now - self._movement_since > self.max_motion_ms:
            self._discard("Movement timed out; hold steady to rearm")
            return False
        if vertical > max(.5, self.threshold * .65):
            self._discard("Movement was not primarily lateral")
            return False
        cooling = self._last_commit is not None and now - self._last_commit < self.cooldown_ms
        if cooling:
            self._state, self._reason = "COOLDOWN", "Repeat accepted; cooldown"
            # Motion during cooldown cannot become a delayed extra commitment.
            self._baseline = positions.copy()
            self._movement_since = None
            self._history.clear()
            self._progress = 0.
            return False
        if rightward >= self.threshold:
            self._state, self._reason = "SETTLING", "Rightward shift detected; hold briefly"
            if settled and self._movement_since is not None and now - self._movement_since >= self.settle_ms:
                self._last_commit = now
                self._baseline = positions.copy()
                self._movement_since = None
                self._history.clear()
                self._state, self._reason = "COOLDOWN", "One intentional repeat accepted"
                return True
        else:
            self._state, self._reason = "ARMED", "Move to your right, then hold briefly to repeat"
            # A settled leftward return repositions the baseline but never fires.
            if rightward < -.15 and settled:
                self._baseline = positions.copy()
                self._movement_since = None
                self._progress = 0.
        return False


class AutoSpaceDetector:
    """Optional continuously observed hand withdrawal after text exists.

    No-hand time before signing, a stalled frame stream, and short detection
    losses do not create a space. The existing text buffer also enforces its
    no-leading/no-duplicate-space rules.
    """

    max_gap_ms = 250

    def __init__(self, enabled: bool = False, pause_ms: int = 1200):
        self.enabled = bool(enabled)
        self.pause_ms = _duration("pause_ms", pause_ms, 500)
        self.reset()

    def reset(self) -> None:
        self._last_timestamp = None
        self._withdrawal_since = None
        self._armed = False

    def snapshot(self):
        elapsed = (self._last_timestamp-self._withdrawal_since if self._withdrawal_since is not None else 0)
        return dict(enabled=self.enabled, armed=self._armed, elapsed_ms=elapsed,
                    remaining_ms=max(0,self.pause_ms-elapsed),
                    state='counting' if self._withdrawal_since is not None else ('armed' if self._armed else 'idle'))

    def update(self, timestamp_ms: int, hand_present: bool, has_text: bool) -> bool:
        now = _timestamp(timestamp_ms)
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("AutoSpaceDetector timestamps moved backwards; reset before a new stream")
        if self._last_timestamp is not None and now - self._last_timestamp > self.max_gap_ms:
            self._withdrawal_since = None
        self._last_timestamp = now
        if not self.enabled or not has_text:
            self._withdrawal_since = None
            self._armed = False
            return False
        if hand_present:
            self._armed = True
            self._withdrawal_since = None
            return False
        if not self._armed:
            return False
        if self._withdrawal_since is None:
            self._withdrawal_since = now
        if now - self._withdrawal_since >= self.pause_ms:
            self._armed = False
            self._withdrawal_since = None
            return True
        return False

