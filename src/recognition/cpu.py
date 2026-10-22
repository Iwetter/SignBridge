"""Versioned feature adapter and ReLU MLP, serialized as numeric NPZ + JSON."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import numpy as np

from src.contracts import ALPHABET, CLASS_TO_ID, Prediction, STATIC_LABELS
from .geometric import RAW, GEOMETRIC, feature_names, model_features


class ModelError(ValueError):
    """Invalid or incompatible model; never silently replace it with fake weights."""


def validate_features(features: np.ndarray) -> np.ndarray:
    raw = np.asarray(features)
    if raw.shape != (63,) or raw.dtype.kind not in 'fiu':
        raise ValueError('Expected a real numeric normalized feature vector with shape (63,).')
    x = raw.astype(np.float32)
    if not np.isfinite(x).all() or np.any(np.abs(x) > 4.000001):
        raise ValueError('Features must be finite and wrist/scale normalized to [-4, 4].')
    return np.clip(x, -4, 4)


def validate_metadata(raw: np.ndarray) -> dict:
    if raw.shape != () or raw.dtype.kind != 'U':
        raise ModelError('Model metadata must be a scalar Unicode JSON string.')
    try:
        meta = json.loads(str(raw.item()))
    except (ValueError, TypeError) as exc:
        raise ModelError('Model metadata is not valid JSON.') from exc
    if not isinstance(meta, dict) or meta.get('schema_version') != 1:
        raise ModelError('Unsupported model schema; retrain with training/train.py.')
    labels = meta.get('supported_labels')
    if (not isinstance(labels, list) or len(labels) < 2 or
            any(not isinstance(x, str) or x not in STATIC_LABELS for x in labels) or
            len(set(labels)) != len(labels)):
        raise ModelError('A model needs at least two unique static classes; J/Z require motion data.')
    if labels != sorted(labels, key=CLASS_TO_ID.get):
        raise ModelError('supported_labels must be in canonical A..Z order.')
    if meta.get('class_mapping') != {x: CLASS_TO_ID[x] for x in ALPHABET}:
        raise ModelError('Model class_mapping must match canonical IDs 0..25 = A..Z.')
    if not isinstance(meta.get('training_provenance'), dict):
        raise ModelError('Model is missing training provenance.')
    feature_set=meta.get('feature_set',RAW)
    try:names=feature_names(feature_set)
    except ValueError as exc:raise ModelError(str(exc)) from exc
    if feature_set==GEOMETRIC and meta.get('feature_names')!=names:
        raise ModelError('Geometric model feature names/version do not match this runtime.')
    if 'architecture' in meta and meta['architecture']!=[len(names),32,26]:
        raise ModelError('Model architecture does not match its feature set.')
    temperature=meta.get('temperature',1.)
    if isinstance(temperature,bool) or not isinstance(temperature,(float,int)) or not np.isfinite(temperature) or not .05<=temperature<=20:
        raise ModelError('Invalid global softmax temperature.')
    return meta


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f'No trained static-letter model at {path}. Collect at least two letters '
            '(start with A and B) in two separate sessions, then click Train or run '
            'python -m training.train --data-dir data/samples --labels A B --output models/alphabet.npz. '
            'Tracking, collection, and labeled replay remain available.')
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ModelError('Model is unexpectedly large (limit 32 MiB).')
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise ModelError(f'Cannot read safe NPZ model {path}: {exc}') from exc


def load_float_model(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    arrays = read_npz(path)
    if not {'W1', 'b1', 'W2', 'b2', 'metadata'} <= arrays.keys():
        raise ModelError('Float model requires W1, b1, W2, b2, and metadata.')
    meta = validate_metadata(arrays['metadata'])
    dimension=len(feature_names(meta.get('feature_set',RAW)))
    for key, shape in [('W1', (dimension, 32)), ('b1', (32,)), ('W2', (32, 26)), ('b2', (26,))]:
        arr = arrays[key]
        if arr.shape != shape or arr.dtype != np.dtype('float32'):
            raise ModelError(f'{key} must have dtype float32 and shape {shape}.')
        if not np.isfinite(arr).all() or np.max(np.abs(arr)) > 1e6:
            raise ModelError(f'{key} contains non-finite or out-of-bounds parameters.')
    return {k: arrays[k] for k in ('W1', 'b1', 'W2', 'b2')}, meta


def masked_softmax(logits: np.ndarray, supported_labels: list[str] | tuple[str, ...], temperature: float = 1.) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (26,) or not np.isfinite(logits).all():
        raise ValueError('Expected 26 finite logits.')
    ids = np.array([CLASS_TO_ID[x] for x in supported_labels], dtype=np.intp)
    if len(ids) < 2 or any(x not in STATIC_LABELS for x in supported_labels):
        raise ValueError('Softmax needs at least two supported static letters.')
    if not np.isfinite(temperature) or not .05<=temperature<=20:raise ValueError('Invalid global softmax temperature.')
    selected = logits[ids]/temperature
    exponent = np.exp(selected - selected.max())
    scores = np.zeros(26, dtype=np.float32)
    scores[ids] = exponent / exponent.sum()
    return scores


class CPURecognizer:
    name = 'CPU'

    def __init__(self, model_path: str | Path = 'models/alphabet.npz'):
        self.model_path = Path(model_path)
        self.arrays, self.metadata = load_float_model(model_path)
        self.supported_labels = tuple(self.metadata['supported_labels'])
        self.feature_set=self.metadata.get('feature_set',RAW)
        self.temperature=float(self.metadata.get('temperature',1.))

    def logits(self, features: np.ndarray) -> np.ndarray:
        x = model_features(validate_features(features),self.feature_set)
        a = self.arrays
        return np.maximum(x @ a['W1'] + a['b1'], 0) @ a['W2'] + a['b2']

    def predict(self, features: np.ndarray, timestamp_ms: int) -> Prediction:
        start = perf_counter()
        scores = masked_softmax(self.logits(features), self.supported_labels,self.temperature)
        class_id = int(scores.argmax())
        return Prediction(ALPHABET[class_id], float(scores[class_id]), int(timestamp_ms),
                          source='cpu', latency_ms=(perf_counter() - start) * 1000, scores=scores)

    def close(self) -> None:
        pass
