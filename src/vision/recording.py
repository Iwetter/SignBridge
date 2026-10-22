"""Camera-independent, safe-NPZ recording for CLI and in-app calibration."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import re
import uuid
import time

import numpy as np

from src.contracts import CLASS_TO_ID, FEATURE_DIM, SCHEMA_VERSION, STATIC_LABELS, FrameObservation
from .features import primary_features, sequence_features, validated_landmarks
from .dataset_summary import NORMALIZATION_ID


class DatasetRecorder:
    """One recorder spans one capture session and may save several labels/clips.

    Call start(label), add(observation) on the camera worker and finish() to save.
    Static captures sample valid primary hands at a fixed interval. Temporal
    captures preserve every frame after the first hand detection, including
    brief tracking gaps; finish trims trailing empty frames. release_ready lets
    the caller end a clip after hands leave view. The class never reads a camera,
    downloads a model, starts a thread, or writes raw camera images.

    Keep the same session_id across captures from one sitting. Start a different
    session after a break for honest held-out evaluation; adjacent frames from
    one capture are not independent validation examples.
    """
    def __init__(self, data_dir="data/samples", session_id: str | None = None,
                 primary: str = "Right", static_interval_ms: int = 100,
                 release_ms: int = 350, min_clip_ms: int = 250,
                 min_clip_frames: int = 6):
        if primary not in ("Right", "Left"):
            raise ValueError("primary must be Left or Right")
        if min(static_interval_ms, release_ms, min_clip_ms, min_clip_frames) <= 0:
            raise ValueError("Recording intervals and minimum clip length must be positive")
        self.data_dir = Path(data_dir)
        self.session_id = session_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        )
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be nonempty text")
        self.primary = primary
        self.static_interval_ms = int(static_interval_ms)
        self.release_ms = int(release_ms)
        self.min_clip_ms = int(min_clip_ms)
        self.min_clip_frames = int(min_clip_frames)
        self.last_path: Path | None = None
        self.last_message = "Ready to record"
        self.receipt = dict(saved_path=None, saved_samples=0, saved_label=None, saved_session_id=None, save_ms=None, save_error='')
        self.cancel()

    def snapshot(self):
        return dict(label=self.label or self.receipt['saved_label'], session_id=self.session_id,
                    frames_seen=self.frames_seen, samples=self.count, valid_frames=self.valid_count,
                    rejected_frames=self.rejected_frames, skipped_frames=self.skipped_frames,
                    elapsed_ms=(time.perf_counter()-self._record_started)*1000 if self.active else 0,
                    active=self.active, **self.receipt)

    @property
    def active(self) -> bool:
        return self.label is not None

    @property
    def is_temporal(self) -> bool:
        return self.active and self.label not in STATIC_LABELS

    @property
    def count(self) -> int:
        return len(self._timestamps)

    @property
    def valid_count(self) -> int:
        return sum(self._present)

    @property
    def duration_ms(self) -> int:
        return self._timestamps[-1] - self._timestamps[0] if self.count > 1 else 0

    @property
    def release_ready(self) -> bool:
        return bool(self.is_temporal and self.count and self._missing_since is not None
                    and self._last_seen_timestamp - self._missing_since >= self.release_ms)

    def start(self, label: str, timestamp_ms: int | None = None) -> None:
        if self.active:
            raise RuntimeError("Finish or cancel the current recording before starting another")
        label = " ".join(str(label).upper().split())
        if label not in CLASS_TO_ID:
            raise ValueError(f"Unknown capture label: {label!r}")
        self.cancel()
        self.label = label
        self._record_started = time.perf_counter()
        self._start_timestamp = None if timestamp_ms is None else int(timestamp_ms)
        self.last_message = f"Recording {label}"

    def add(self, observation: FrameObservation) -> bool:
        """Append when appropriate; return whether a frame was buffered.

        Non-increasing timestamps are rejected, so duplicate camera events cannot
        quietly create zero-duration examples. A release-ready clip stops accepting
        returning hands until finish/cancel, preventing joined performances.
        """
        if not self.active:
            return False
        self.frames_seen += 1
        timestamp = int(observation.timestamp_ms)
        if self._start_timestamp is not None and timestamp < self._start_timestamp:
            self.skipped_frames += 1
            return False
        if self._last_seen_timestamp is not None and timestamp <= self._last_seen_timestamp:
            self.rejected_frames += 1
            raise ValueError("Recording timestamps must strictly increase")
        if self.release_ready:
            self.skipped_frames += 1
            return False
        self._last_seen_timestamp = timestamp
        try:
            feature = primary_features(observation, self.primary)
            sequence = sequence_features(observation)
        except (ValueError, TypeError):
            self.rejected_frames += 1
            return False
        hand_present = bool(sequence[132:].any())
        if not hand_present:
            self.rejected_frames += 1
        if self.is_temporal:
            if not hand_present:
                if not self.count:
                    return False
                if self._missing_since is None:
                    self._missing_since = timestamp
            else:
                self._missing_since = None
            feature = np.zeros(FEATURE_DIM, dtype=np.float32) if feature is None else feature
        else:
            if feature is None:
                return False
            if self.count and timestamp - self._timestamps[-1] < self.static_interval_ms:
                self.skipped_frames += 1
                return False
        self._features.append(feature.copy())
        self._sequence.append(sequence)
        self._timestamps.append(timestamp)
        self._present.append(hand_present)
        if self.label in ('J', 'Z'):
            raw = np.zeros((2, 21, 3), dtype=np.float32)
            for slot, side in enumerate(('Right', 'Left')):
                matches = [h for h in observation.hands if h.handedness == side]
                if matches and sequence[132 + slot]:
                    raw[slot] = validated_landmarks(max(matches, key=lambda h: h.score).landmarks)
            self._raw_landmarks.append(raw)
            self._frame_sizes.append((observation.width, observation.height))
        return True

    def finish(self) -> Path | None:
        """Save a nonempty capture; reject insufficient temporal clips explicitly.

        Empty recordings return None. Too-short temporal recordings raise
        ValueError after discarding their buffer, allowing an immediate retry.
        Write failures leave the buffer intact so the caller can retry saving.
        """
        if not self.active:
            return None
        started = time.perf_counter()
        stop = self.count
        if self.is_temporal:
            while stop and not self._present[stop - 1]:
                stop -= 1
        if not stop:
            self.cancel()
            self.last_message = "No valid hands captured; nothing saved"
            return None
        if self.is_temporal:
            duration = self._timestamps[stop - 1] - self._timestamps[0]
            if sum(self._present[:stop]) < self.min_clip_frames or duration < self.min_clip_ms:
                self.cancel()
                self.last_message = "Clip too short; record one complete performance, then release"
                raise ValueError(self.last_message)
        arrays = {
            "features": np.asarray(self._features[:stop], dtype=np.float32),
            "sequence": np.asarray(self._sequence[:stop], dtype=np.float32),
            "timestamps_ms": np.asarray(self._timestamps[:stop], dtype=np.int64),
            "label": np.asarray(self.label),
            "session_id": np.asarray(self.session_id),
            "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
            "normalization_id": np.asarray(NORMALIZATION_ID),
        }
        if self.label in ('J', 'Z'):
            # Optional Dataset v1 fields: leave every static archive unchanged.
            # Raw unmirrored landmarks retain fingertip travel and hand size;
            # the existing sequence retains normalized pose and global wrists.
            arrays.update(raw_landmarks=np.asarray(self._raw_landmarks[:stop], dtype=np.float32),
                          frame_sizes=np.asarray(self._frame_sizes[:stop], dtype=np.int32),
                          motion_schema=np.asarray('raw-unmirrored-two-hand-v1'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        safe_session = re.sub(r"[^A-Za-z0-9_-]", "_", self.session_id)[:80]
        safe_label = self.label.replace(" ", "_")
        output = self.data_dir / f"{safe_session}__{safe_label}__{uuid.uuid4().hex[:12]}.npz"
        temporary = output.with_suffix(".npz.tmp")
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            with np.load(temporary, allow_pickle=False) as check:
                if set(check.files) != set(arrays) or any(not np.array_equal(check[k],v) for k,v in arrays.items()):
                    raise IOError('Saved archive failed read-back verification')
            os.replace(temporary, output)
        except Exception as exc:
            self.receipt['save_error'] = str(exc)
            self.last_message = f'Save failed: {exc}'
            raise
        finally:
            if temporary.exists():
                temporary.unlink()
        self.last_path = output
        valid_saved = sum(self._present[:stop])
        self.receipt = dict(saved_path=str(output.resolve()), saved_samples=valid_saved,
                            saved_label=self.label, saved_session_id=self.session_id,
                            save_ms=(time.perf_counter()-started)*1000, save_error='')
        self.last_message = f"Saved successfully: {self.label}, {valid_saved} valid samples. {output.resolve()}"
        self.cancel()
        return output

    def cancel(self) -> None:
        self.frames_seen = self.rejected_frames = self.skipped_frames = 0
        self._record_started = time.perf_counter()
        self.label: str | None = None
        self._features: list[np.ndarray] = []
        self._sequence: list[np.ndarray] = []
        self._timestamps: list[int] = []
        self._present: list[bool] = []
        self._raw_landmarks: list[np.ndarray] = []
        self._frame_sizes: list[tuple[int, int]] = []
        self._start_timestamp: int | None = None
        self._last_seen_timestamp: int | None = None
        self._missing_since: int | None = None
