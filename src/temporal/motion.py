"""Session-calibrated motion/sign matching; never an untrained ASL classifier.

Templates are dataset-v1 NPZ clips. Optional scalar JSON ``metadata`` may contain
``reject: true`` for a negative example. UNKNOWN/REST/BACKGROUND/NONE and any
noncanonical label are reject-only templates, never emitted as canonical signs.
Every recognition requires a close clip match and separation from other labels.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from src.contracts import ALPHABET, CLASS_TO_ID, FrameObservation, Prediction, SEQUENCE_DIM
from .decoder import _timestamp


@dataclass(frozen=True)
class _Template:
    label: str
    sequence: np.ndarray
    duration_ms: int
    reject: bool
    requires_motion: bool
    extent: float
    path: str
    session_id: str


def _resample(sequence: np.ndarray, timestamps: np.ndarray, count: int = 32) -> np.ndarray:
    """Resample actual capture times, retaining binary hand-presence channels."""
    target = np.linspace(float(timestamps[0]), float(timestamps[-1]), count)
    sampled = np.column_stack([np.interp(target, timestamps, sequence[:, i])
                               for i in range(SEQUENCE_DIM)]).astype(np.float32)
    nearest = np.searchsorted(timestamps, target).clip(0, len(timestamps) - 1)
    before = (nearest - 1).clip(0)
    nearest = np.where(np.abs(timestamps[before] - target) <
                       np.abs(timestamps[nearest] - target), before, nearest)
    sampled[:, 132:134] = sequence[nearest, 132:134]
    return sampled


def _relative_wrists(sequence: np.ndarray) -> np.ndarray:
    result = sequence.copy()
    for slot, start in ((0, 126), (1, 129)):
        present = result[:, 132 + slot] > 0.5
        if present.any():
            result[present, start:start + 3] -= result[np.flatnonzero(present)[0], start:start + 3]
    return result


def _distance(a: np.ndarray, b: np.ndarray, relative: bool = False) -> float:
    """Weighted pose/wrist/presence distance; wrists keep vocabulary framing."""
    if relative:
        a, b = _relative_wrists(a), _relative_wrists(b)
    presence = float(np.mean(np.abs(a[:, 132:134] - b[:, 132:134])))
    pose_errors, wrist_errors = [], []
    for slot, start, wrist in ((0, 0, 126), (1, 63, 129)):
        common = (a[:, 132 + slot] > 0.5) & (b[:, 132 + slot] > 0.5)
        if common.any():
            pose_errors.append(float(np.mean((a[common, start:start + 63] -
                                               b[common, start:start + 63]) ** 2)))
            wrist_errors.append(float(np.mean((a[common, wrist:wrist + 3] -
                                                b[common, wrist:wrist + 3]) ** 2)))
    pose = np.sqrt(np.mean(pose_errors)) if pose_errors else 0.0
    wrist = np.sqrt(np.mean(wrist_errors)) if wrist_errors else 0.0
    return float(0.55 * pose + 2.0 * wrist + 0.8 * presence)


def _motion_extent(sequence: np.ndarray) -> float:
    """Zero for frozen frames, including stationary hand-presence changes.

    A score of 1 represents ~0.025 image-width wrist motion or ~0.16
    normalized-landmark RMS pose motion. Only continuously present samples of
    each hand contribute; inserting/removing an absent slot is not a trajectory.
    """
    extent = 0.0
    for slot, start, wrist in ((0, 0, 126), (1, 63, 129)):
        present = sequence[:, 132 + slot] > 0.5
        if np.count_nonzero(present) < 3:
            continue
        poses = sequence[present, start:start + 63]
        wrists = sequence[present, wrist:wrist + 2]
        pose_extent = float(np.sqrt(np.mean(np.ptp(poses, axis=0) ** 2))) / 0.16
        wrist_extent = float(np.linalg.norm(np.ptp(wrists, axis=0))) / 0.025
        extent = max(extent, pose_extent, wrist_extent)
    return extent


class MotionRecognizer:
    """Match J/Z and calibrated vocabulary clips against a bounded history.

    ``motion_active`` is true while recent hand motion is underway. A completed
    match returns a current-timestamp Prediction for confirmation_ms (default
    500), allowing TemporalDecoder's 350ms dwell. During confirmation,
    motion_active is false: the caller must prioritize this prediction over
    static recognition. No-hand observations cancel confirmation immediately.
    No templates means empty coverage and no detections or static suppression.

    Threshold and margin are distances, not validated accuracy. Collect multiple
    examples and reject clips under consistent framing, then evaluate held-out
    sessions. ``diagnostics`` exposes every skipped file and calibration status.
    """

    def __init__(self, data_dir: str | Path = "data/samples", *, threshold: float = 0.20,
                 margin: float = 0.025, min_duration_ms: int = 300,
                 cooldown_ms: int = 800, confirmation_ms: int = 500,
                 max_gap_ms: int = 250, settle_ms: int = 150):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive and finite")
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("margin must be nonnegative and finite")
        for name, value in (("min_duration_ms", min_duration_ms), ("cooldown_ms", cooldown_ms),
                            ("confirmation_ms", confirmation_ms), ("max_gap_ms", max_gap_ms),
                            ("settle_ms", settle_ms)):
            if _timestamp(value) < 1:
                raise ValueError(f"{name} must be positive")
        self.threshold, self.margin = float(threshold), float(margin)
        self.min_duration_ms, self.cooldown_ms = min_duration_ms, cooldown_ms
        self.confirmation_ms, self.max_gap_ms, self.settle_ms = confirmation_ms, max_gap_ms, settle_ms
        self.diagnostics: list[str] = []
        self._templates = self._load(Path(data_dir))
        self.supported_labels = tuple(sorted({t.label for t in self._templates if not t.reject},
                                             key=CLASS_TO_ID.__getitem__))
        self.calibration_info = {
            "method": "session-calibrated timestamp-resampled nearest template",
            "supported_labels": list(self.supported_labels),
            "template_count": len(self._templates),
            "reject_count": sum(t.reject for t in self._templates),
            "sessions": sorted({t.session_id for t in self._templates}),
            "threshold": self.threshold, "margin": self.margin,
            "accuracy": None,
        }
        self._history_ms = max([int(t.duration_ms * 1.6) + 400 for t in self._templates] + [1200])
        self.reset()

    def _load(self, directory: Path) -> list[_Template]:
        templates = []
        if not directory.exists():
            self.diagnostics.append(f"No calibration data directory: {directory}")
            return templates
        for path in sorted(directory.rglob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    required = {"sequence", "timestamps_ms", "label", "session_id", "schema_version"}
                    if not required.issubset(data.files):
                        raise ValueError("missing dataset-v1 keys")
                    if data["label"].shape != () or data["label"].dtype.kind != "U":
                        raise ValueError("label must be scalar Unicode")
                    label = str(data["label"].item()).strip().upper()
                    if label in ALPHABET and label not in ("J", "Z"):
                        continue  # Static capture sessions are not clip templates.
                    if data["schema_version"].shape != () or int(data["schema_version"].item()) != 1:
                        raise ValueError("unsupported dataset schema")
                    if data["session_id"].shape != () or data["session_id"].dtype.kind != "U":
                        raise ValueError("session_id must be scalar Unicode")
                    session_id = str(data["session_id"].item())
                    seq = np.asarray(data["sequence"], dtype=np.float32)
                    stamps = np.asarray(data["timestamps_ms"])
                    if seq.ndim != 2 or seq.shape[1] != SEQUENCE_DIM or len(seq) < 6:
                        raise ValueError("sequence must contain at least 6 frames of shape (T,134)")
                    if not np.isfinite(seq).all() or not np.isin(seq[:, 132:134], [0, 1]).all():
                        raise ValueError("sequence contains nonfinite values or invalid presence flags")
                    if stamps.shape != (len(seq),) or stamps.dtype.kind not in "iu":
                        raise ValueError("timestamps_ms must be integer (T,)")
                    stamps = stamps.astype(np.int64)
                    if stamps[0] < 0 or np.any(np.diff(stamps) <= 0):
                        raise ValueError("clip timestamps must strictly increase")
                    if np.max(np.diff(stamps)) > self.max_gap_ms:
                        raise ValueError("clip contains excessive dropped-frame gaps")
                    duration = int(stamps[-1] - stamps[0])
                    if not self.min_duration_ms <= duration <= 6000:
                        raise ValueError("clip duration must be between min_duration_ms and 6000ms")
                    if np.mean(np.any(seq[:, 132:134] > 0.5, axis=1)) < 0.8:
                        raise ValueError("clip has too many no-hand frames")
                    metadata = {}
                    if "metadata" in data.files:
                        if data["metadata"].shape != () or data["metadata"].dtype.kind != "U":
                            raise ValueError("metadata must be scalar JSON Unicode")
                        metadata = json.loads(str(data["metadata"].item()))
                        if not isinstance(metadata, dict):
                            raise ValueError("metadata must be a JSON object")
                    reject = bool(metadata.get("reject", False)) or label not in CLASS_TO_ID
                    extent = _motion_extent(seq)
                    if label in ("J", "Z") and extent < 1.0 and not reject:
                        raise ValueError("J/Z calibration must contain a real trajectory")
                    templates.append(_Template(label, _resample(seq, stamps), duration, reject,
                                               label in ("J", "Z") or extent >= 1.0,
                                               extent, str(path), session_id))
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                self.diagnostics.append(f"Skipped {path.name}: {exc}")
        if not any(not t.reject for t in templates):
            self.diagnostics.append("No usable J/Z or vocabulary clips; record calibration performances")
        return templates

    def reset(self) -> None:
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=512)
        self.last_timestamp_ms: int | None = None
        self.motion_active = False
        self._last_motion_ms: int | None = None
        self._confirmed: tuple[str, float, int] | None = None
        self._cooldown_until = 0
        self._held_label: str | None = None
        self._held_feature: np.ndarray | None = None
        self._held_change_since: int | None = None
        self._release_since: int | None = None
        self.last_match: dict | None = None

    def update(self, observation: FrameObservation) -> Prediction | None:
        started = time.perf_counter()
        now = _timestamp(observation.timestamp_ms)
        if self.last_timestamp_ms is not None and now < self.last_timestamp_ms:
            raise ValueError("MotionRecognizer timestamps moved backwards; reset before a new stream")
        if self.last_timestamp_ms == now:
            return None  # A duplicate cannot add movement or extend a confirmation.
        if self.last_timestamp_ms is not None and now - self.last_timestamp_ms > self.max_gap_ms:
            self._history.clear()
            self._confirmed = None
            self._last_motion_ms = None
            self._release_since = self._held_change_since = None
        self.last_timestamp_ms = now
        self.motion_active = False
        if not self.supported_labels:
            return None
        from src.vision import sequence_features
        feature = np.asarray(sequence_features(observation), dtype=np.float32)
        if feature.shape != (SEQUENCE_DIM,) or not np.isfinite(feature).all():
            raise ValueError("sequence_features must return finite shape (134,)")
        present = bool(np.any(feature[132:134] > 0.5))
        if not present:
            self._confirmed = None
            self._history.clear()
            self._last_motion_ms = None
            self._held_change_since = None
            if self._release_since is None:
                self._release_since = now
            if now - self._release_since >= 200:
                self._held_label = self._held_feature = None
            return None
        self._release_since = None
        if self._confirmed is not None:
            label, confidence, until = self._confirmed
            if now <= until:
                # A confirmation survives minor pose changes but not a new sign.
                if self._held_feature is not None and _distance(feature[None], self._held_feature[None]) > 0.35:
                    self._confirmed = None
                else:
                    return Prediction(label, confidence, now, "temporal-calibrated",
                                      (time.perf_counter() - started) * 1000)
            else:
                self._confirmed = None
        if self._held_feature is not None:
            if _distance(feature[None], self._held_feature[None]) > 0.18:
                if self._held_change_since is None:
                    self._held_change_since = now
                elif now - self._held_change_since >= 200:
                    self._held_label = self._held_feature = None
                    self._held_change_since = None
            else:
                self._held_change_since = None
        self._history.append((now, feature.copy()))
        while self._history and now - self._history[0][0] > self._history_ms:
            self._history.popleft()
        times = np.asarray([x[0] for x in self._history], dtype=np.int64)
        seq = np.asarray([x[1] for x in self._history], dtype=np.float32)
        recent = seq[times >= now - 200]
        if len(recent) >= 3 and _motion_extent(recent) >= 0.65:
            self._last_motion_ms = now
        self.motion_active = self._last_motion_ms is not None and now - self._last_motion_ms < self.settle_ms
        if now < self._cooldown_until or len(seq) < 6 or now - times[0] < self.min_duration_ms:
            return None
        by_label: dict[str, tuple[float, _Template]] = {}
        best_reject = float("inf")
        # End-aligned windows permit moderately different signing speeds. Each
        # window uses real timestamps; no extrapolation across missing frames.
        for template in self._templates:
            if not template.requires_motion and template.label == self._held_label and not template.reject:
                continue
            best_distance = float("inf")
            for speed in (0.75, 1.0, 1.3):
                duration = max(self.min_duration_ms, int(template.duration_ms * speed))
                first = int(np.searchsorted(times, now - duration, side="left"))
                clip, clip_times = seq[first:], times[first:]
                if len(clip) < 6 or clip_times[-1] - clip_times[0] < duration * 0.88:
                    continue
                if template.requires_motion and _motion_extent(clip) < max(1.0, template.extent * 0.35):
                    continue
                sampled = _resample(clip, clip_times)
                distance = _distance(sampled, template.sequence, relative=template.label in ("J", "Z"))
                best_distance = min(best_distance, distance)
            if template.reject:
                best_reject = min(best_reject, best_distance)
            elif best_distance < by_label.get(template.label, (float("inf"), template))[0]:
                by_label[template.label] = (best_distance, template)
        ranked = sorted(by_label.items(), key=lambda item: item[1][0])
        if not ranked:
            return None
        label, (distance, template) = ranked[0]
        runner_up = min(best_reject, ranked[1][1][0] if len(ranked) > 1 else float("inf"))
        if distance > self.threshold or runner_up - distance < self.margin:
            return None
        # A partial trajectory can resemble a faster full clip. Do not start the
        # dwell confirmation until movement settles at the gesture's endpoint.
        if template.requires_motion and self.motion_active:
            return None
        # This is a match-strength score, never a calibrated correctness claim.
        confidence = float(np.clip(0.80 + 0.19 * (1.0 - distance / self.threshold), 0.8, 0.99))
        self.last_match = {"label": label, "distance": distance, "runner_up_distance": runner_up,
                           "template": template.path, "timestamp_ms": now}
        self._confirmed = (label, confidence, now + self.confirmation_ms)
        self._cooldown_until = now + self.confirmation_ms + self.cooldown_ms
        self._held_label, self._held_feature = label, feature.copy()
        self._history.clear()
        self._last_motion_ms = None
        self.motion_active = False
        return Prediction(label, confidence, now, "temporal-calibrated",
                          (time.perf_counter() - started) * 1000)
