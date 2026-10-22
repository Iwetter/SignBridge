"""Pure NumPy landmark preparation shared by collection and recognition."""
from __future__ import annotations

import numpy as np

from src.contracts import FEATURE_DIM, SEQUENCE_DIM, FrameObservation, HandObservation


def validated_landmarks(landmarks) -> np.ndarray:
    """Return finite float32 coordinates, raising instead of poisoning a dataset."""
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (21, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Landmarks contain non-finite coordinates")
    return points


def normalize_landmarks(landmarks, handedness: str = "Right") -> np.ndarray:
    """Wrist-center, scale by MCP radius and canonicalize anatomical Left.

    Orientation is deliberately preserved. Coordinates are xyz-interleaved.
    This exactly implements the v1 feature contract, including the scale floor.
    """
    if handedness not in ("Left", "Right"):
        raise ValueError("Handedness must be Left or Right")
    points = validated_landmarks(landmarks).astype(np.float64)
    centered = points - points[0]
    radius = max(float(np.linalg.norm(centered[[5, 9, 13, 17]], axis=1).max()), 1e-6)
    centered /= radius
    if handedness == "Left":
        centered[:, 0] *= -1
    return np.clip(centered, -4, 4).astype(np.float32).reshape(FEATURE_DIM)


def _valid_hands(observation: FrameObservation) -> list[HandObservation]:
    hands = []
    for hand in observation.hands:
        try:
            validated_landmarks(hand.landmarks)
            if hand.handedness not in ("Right", "Left"):
                continue
            if not np.isfinite(float(hand.score)) or not 0 <= float(hand.score) <= 1:
                continue
            hands.append(hand)
        except (TypeError, ValueError, OverflowError):
            continue
    return hands


def primary_hand(observation: FrameObservation, preferred: str = "Right") -> HandObservation | None:
    """Prefer the configured anatomical hand; fall back only to one valid hand."""
    if preferred not in ("Right", "Left"):
        raise ValueError("Preferred hand must be Left or Right")
    hands = _valid_hands(observation)
    matches = [hand for hand in hands if hand.handedness == preferred]
    if matches:
        return max(matches, key=lambda hand: hand.score)
    return hands[0] if len(hands) == 1 else None


def primary_features(observation: FrameObservation, preferred: str = "Right") -> np.ndarray | None:
    hand = primary_hand(observation, preferred)
    return None if hand is None else normalize_landmarks(hand.landmarks, hand.handedness)


def sequence_features(observation: FrameObservation) -> np.ndarray:
    """Right63, Left63, right-wrist3, left-wrist3, right/left presence."""
    result = np.zeros(SEQUENCE_DIM, dtype=np.float32)
    hands = _valid_hands(observation)
    for index, side in enumerate(("Right", "Left")):
        matches = [hand for hand in hands if hand.handedness == side]
        if not matches:
            continue
        hand = max(matches, key=lambda item: item.score)
        result[index * 63:(index + 1) * 63] = normalize_landmarks(hand.landmarks, side)
        result[126 + index * 3:129 + index * 3] = validated_landmarks(hand.landmarks)[0]
        result[132 + index] = 1
    return result
