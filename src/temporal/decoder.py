"""Commit stable canonical predictions once per deliberate gesture."""
from __future__ import annotations

import math
import time
from numbers import Integral

from src.contracts import ALPHABET, CLASS_TO_ID, Prediction, Token


def _timestamp(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("timestamp_ms must be a nonnegative monotonic integer")
    return int(value)


class TemporalDecoder:
    """Dwell gate with an independent held-gesture lock.

    None or a Prediction whose label is None is an explicit release observation.
    A named, low-confidence prediction breaks pending dwell/release/change runs
    but never unlocks the last committed label. Gaps greater than max_gap_ms
    also break those runs: unobserved time does not count as deliberate signing.
    """

    def __init__(self, confidence_threshold: float = 0.8, dwell_ms: int = 350,
                 release_ms: int = 200, change_ms: int = 150,
                 max_gap_ms: int = 250, max_prediction_age_ms: int = 250):
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        for name, value in (("dwell_ms", dwell_ms), ("release_ms", release_ms),
                            ("change_ms", change_ms), ("max_gap_ms", max_gap_ms),
                            ("max_prediction_age_ms", max_prediction_age_ms)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if max_gap_ms == 0:
            raise ValueError("max_gap_ms must be positive")
        self.confidence_threshold = confidence_threshold
        self.dwell_ms, self.release_ms, self.change_ms = dwell_ms, release_ms, change_ms
        self.max_gap_ms = max_gap_ms
        self.max_prediction_age_ms = max_prediction_age_ms
        self.reset()

    def reset(self) -> None:
        self.last_timestamp_ms: int | None = None
        self.locked_label: str | None = None
        self._clear_runs()

    def _clear_runs(self) -> None:
        self._candidate: str | None = None
        self._candidate_since: int | None = None
        self._candidate_confidence = 1.0
        self._release_since: int | None = None
        self._change_since: int | None = None

    def snapshot(self) -> dict:
        """Report observed dwell without advancing the recognition clock.

        ``candidate`` is the pending, above-threshold gesture being timed, not
        the classifier's current candidate. After a commit it becomes None;
        ``locked_label`` identifies the held gesture that cannot commit again.
        UI refreshes must not turn time without observations into stable dwell.
        """
        stable_duration_ms = 0
        if self._candidate_since is not None and self.last_timestamp_ms is not None:
            stable_duration_ms = self.last_timestamp_ms - self._candidate_since
        return {
            "candidate": self._candidate,
            "stable_duration_ms": stable_duration_ms,
            "dwell_ms": self.dwell_ms,
            "confidence_threshold": self.confidence_threshold,
            "locked_label": self.locked_label,
            "last_timestamp_ms": self.last_timestamp_ms,
        }

    def reject(self, timestamp_ms: int) -> None:
        """Observe unusable input without treating it as a hand release.

        Invalid features and inference failures break pending dwell and release
        runs, but cannot unlock a gesture that was already committed.
        """
        now = _timestamp(timestamp_ms)
        if self.last_timestamp_ms is not None and now < self.last_timestamp_ms:
            raise ValueError("TemporalDecoder timestamps moved backwards; reset before a new stream")
        self.last_timestamp_ms = now
        self._clear_runs()

    def commit_completed(self, prediction: Prediction, *, repeat: bool = False) -> Token | None:
        """Commit an independently completed trajectory; ordinary poses use update.

        The motion/repeat detector owns its duration, completion and cooldown.
        Retain the existing held-label lock so the following static pose cannot
        immediately append the same letter again.
        """
        now = _timestamp(prediction.timestamp_ms)
        if self.last_timestamp_ms is not None and now < self.last_timestamp_ms:
            raise ValueError('Completed motion timestamps moved backwards.')
        if prediction.label not in ALPHABET or not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1:
            raise ValueError('Completed motion requires a canonical letter and finite confidence.')
        self.last_timestamp_ms = now
        self._clear_runs()
        if prediction.confidence < self.confidence_threshold or (prediction.label == self.locked_label and not repeat):
            return None
        self.locked_label = prediction.label
        return Token(prediction.label, 'letter', now, prediction.confidence, prediction.source)

    def update(self, prediction: Prediction | None,
               timestamp_ms: int | None = None) -> Token | None:
        now = _timestamp(timestamp_ms if timestamp_ms is not None else
                         prediction.timestamp_ms if prediction is not None else
                         int(time.monotonic() * 1000))
        if self.last_timestamp_ms is not None and now < self.last_timestamp_ms:
            raise ValueError("TemporalDecoder timestamps moved backwards; reset before a new stream")
        if prediction is not None:
            pred_time = _timestamp(prediction.timestamp_ms)
            if pred_time > now:
                raise ValueError("prediction timestamp cannot be in the future")
            if not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1:
                raise ValueError("prediction confidence must be finite and in [0, 1]")
            if prediction.label is not None and prediction.label not in CLASS_TO_ID:
                raise ValueError(f"unknown canonical label: {prediction.label!r}")
        if self.last_timestamp_ms is not None and now - self.last_timestamp_ms > self.max_gap_ms:
            self._clear_runs()
        self.last_timestamp_ms = now
        if prediction is not None and now - prediction.timestamp_ms > self.max_prediction_age_ms:
            self._clear_runs()
            return None
        if prediction is None or prediction.label is None:
            self._candidate = self._candidate_since = self._change_since = None
            if self._release_since is None:
                self._release_since = now
            if now - self._release_since >= self.release_ms:
                self.locked_label = None
            return None
        if prediction.confidence < self.confidence_threshold:
            self._clear_runs()
            return None
        self._release_since = None
        label = prediction.label
        if label == self.locked_label:
            self._candidate = self._candidate_since = self._change_since = None
            return None
        if label != self._candidate:
            self._candidate = label
            self._candidate_since = now
            self._candidate_confidence = prediction.confidence
            self._change_since = now if self.locked_label is not None else None
        else:
            self._candidate_confidence = min(self._candidate_confidence, prediction.confidence)
        if self.locked_label is not None:
            if self._change_since is None:
                self._change_since = now
            if now - self._change_since < self.change_ms:
                return None
            self.locked_label = None
        if self._candidate_since is not None and now - self._candidate_since >= self.dwell_ms:
            token = Token(label, "letter" if label in ALPHABET else "sign", now,
                          self._candidate_confidence, prediction.source)
            self.locked_label = label
            self._clear_runs()
            return token
        return None
