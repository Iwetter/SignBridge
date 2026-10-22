"""Strict Dataset v1 loading and session-isolated evaluation helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from itertools import combinations
import csv
import json
from zipfile import BadZipFile
import numpy as np

from src.contracts import CLASS_TO_ID, LABELS, STATIC_LABELS


@dataclass
class StaticDataset:
    features: np.ndarray
    labels: np.ndarray
    sessions: np.ndarray
    files: list[dict]
    excluded_files: list[dict]
    warnings: list[str] = field(default_factory=list)

    @property
    def supported_labels(self) -> list[str]:
        present = set(self.labels.tolist())
        return [label for label in STATIC_LABELS if CLASS_TO_ID[label] in present]

    def summary(self, indices: np.ndarray | None = None) -> dict:
        selected = np.arange(len(self.labels)) if indices is None else indices
        labels, sessions = self.labels[selected], self.sessions[selected]
        return {'samples': len(selected), 'classes': self.supported_labels,
                'class_count': len(self.supported_labels),
                'samples_per_class': {label: int(np.sum(labels == CLASS_TO_ID[label]))
                                      for label in self.supported_labels},
                'sessions_per_class': {label: sorted(set(sessions[labels == CLASS_TO_ID[label]].tolist()))
                                       for label in self.supported_labels},
                'session_ids': sorted(set(sessions.tolist()))}


def _text_scalar(a: np.ndarray, key: str, path: Path) -> str:
    if a.shape != () or a.dtype.kind != 'U' or not str(a.item()).strip():
        raise ValueError(f'{path}: {key} must be a nonempty Unicode scalar.')
    return str(a.item())


def load_dataset(data_dir: str | Path, require_normalization: str | None = None,
                 labels: list[str] | tuple[str, ...] | None = None) -> StaticDataset:
    data_dir = Path(data_dir)
    requested = set(labels) if labels is not None else None
    if requested is not None and (not requested or requested - set(STATIC_LABELS)):
        raise ValueError('Requested labels must be static canonical letters (J/Z require motion data).')
    paths = sorted(data_dir.rglob('*.npz'))
    if not paths:
        raise ValueError(f'No sample NPZ files in {data_dir}. Record A and B in at least two '
                         'separate sessions with python -m training.collect before training.')
    features, target_ids, sessions, manifest, excluded, warnings = [], [], [], [], [], []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                required = {'features', 'sequence', 'timestamps_ms', 'label', 'session_id', 'schema_version'}
                if not required <= set(data.files):
                    raise ValueError(f'{path}: missing dataset keys {sorted(required - set(data.files))}.')
                schema = data['schema_version']
                if schema.shape != () or schema.dtype.kind not in 'iu' or int(schema) != 1:
                    raise ValueError(f'{path}: expected dataset schema_version=1.')
                label = _text_scalar(data['label'], 'label', path)
                session = _text_scalar(data['session_id'], 'session_id', path)
                if label not in LABELS:
                    raise ValueError(f'{path}: unknown canonical label {label!r}.')
                normalization = (_text_scalar(data['normalization_id'], 'normalization_id', path)
                                 if 'normalization_id' in data else None)
                x, seq, ts = data['features'], data['sequence'], data['timestamps_ms']
                if x.ndim != 2 or x.shape[1] != 63 or x.dtype != np.dtype('float32') or len(x) == 0:
                    raise ValueError(f'{path}: features must be nonempty float32 (T,63).')
                if not np.isfinite(x).all() or np.any(np.abs(x) > 4.000001):
                    raise ValueError(f'{path}: static features must be finite and in [-4,4].')
                if seq.shape != (len(x), 134) or seq.dtype != np.dtype('float32') or not np.isfinite(seq).all():
                    raise ValueError(f'{path}: sequence must be finite float32 (T,134).')
                if ts.shape != (len(x),) or ts.dtype != np.dtype('int64') or np.any(np.diff(ts) <= 0):
                    raise ValueError(f'{path}: timestamps_ms must be strictly increasing int64 (T,).')
                entry = {'path': str(path.relative_to(data_dir)), 'sha256': sha256(path.read_bytes()).hexdigest(),
                         'session_id': session, 'label': label, 'samples': len(x),
                         'normalization_id': normalization}
                reason = ('temporal class' if label not in STATIC_LABELS else
                          'outside requested training classes' if requested is not None and label not in requested else
                          'input convention does not match required normalization' if require_normalization and normalization != require_normalization else None)
                if reason:
                    excluded.append({**entry, 'reason': reason})
                    continue
                features.append(x.copy())
                target_ids.extend([CLASS_TO_ID[label]] * len(x))
                sessions.extend([session] * len(x))
                manifest.append(entry)
        except (OSError, EOFError, BadZipFile) as exc:
            raise ValueError(f'Cannot read dataset {path}: {exc}') from exc
    if not features:
        raise ValueError('No eligible static-letter examples. Existing legacy recordings are preserved; record with the current input convention before training.' if require_normalization else 'No static-letter examples found; J/Z and signs are trained as temporal clips.')
    present = {item['label'] for item in manifest}
    if requested and requested - present:
        raise ValueError('No persisted eligible samples for requested class(es): ' + ', '.join(sorted(requested - present)) + '.')
    legacy = [item for item in manifest if item['normalization_id'] is None]
    if legacy:
        warnings.append(f'Included {sum(item["samples"] for item in legacy)} legacy samples without an input-convention tag. '
                        'They are real persisted recordings; physical handedness/input orientation was not verified. '
                        'Validate A/B with the current webcam before accepting this model.')
    conventions = {item['normalization_id'] for item in manifest if item['normalization_id'] is not None}
    if len(conventions) > 1:
        warnings.append('Multiple recorded input conventions are included: ' + ', '.join(sorted(conventions)) + '.')
    return StaticDataset(np.concatenate(features), np.asarray(target_ids, dtype=np.int64),
                         np.asarray(sessions, dtype=np.str_), manifest, excluded, warnings)


def session_split(dataset: StaticDataset, seed: int = 42, validation_fraction: float = .25,
                  validation_sessions: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1.')
    unique = sorted(set(dataset.sessions.tolist()))
    classes = set(dataset.labels.tolist())
    if len(classes) < 2:
        raise ValueError('At least two static letters are required; one-class confidence would be meaningless.')
    if len(unique) < 2:
        raise ValueError('No honest holdout: record each letter in at least two different session IDs. '
                         'Adjacent frames from one session cannot be split into train/validation.')
    short = {label: len(set(dataset.sessions[dataset.labels == CLASS_TO_ID[label]].tolist()))
             for label in dataset.supported_labels}
    short = {label: count for label, count in short.items() if count < 2}
    if short:
        detail = ', '.join(f'{label}: {count} session' for label, count in short.items())
        raise ValueError(f'No honest holdout: {detail}. Record each listed letter in another session using New session. '
                         'Every class needs at least two distinct session IDs; adjacent frames cannot be split.')
    def valid(selected):
        mask = np.isin(dataset.sessions, selected)
        return (set(dataset.labels[mask].tolist()) == classes and
                set(dataset.labels[~mask].tolist()) == classes)
    if validation_sessions:
        if not set(validation_sessions) <= set(unique) or not valid(validation_sessions):
            raise ValueError('Explicit validation sessions must exist and leave every class in BOTH splits.')
        chosen = validation_sessions
    else:
        nval = min(len(unique)-1, max(1, round(len(unique) * validation_fraction)))
        rng = np.random.default_rng(seed)
        # Search metadata only; labels ensure coverage, features/accuracy never drive the split.
        chosen = None
        # A/B recorded in separate sessions can require TWO validation sessions even
        # when the target fraction rounds to one. Try other sizes before rejecting.
        sizes = sorted(range(1, len(unique)), key=lambda size: (abs(size-nval), size))
        for size in sizes:
            if len(unique) <= 16:
                candidates = list(combinations(unique, size))
                rng.shuffle(candidates)
            else:
                candidates = (rng.choice(unique, size, replace=False).tolist() for _ in range(2048))
            chosen = next((list(c) for c in candidates if valid(c)), None)
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError('No honest class-covered session split found. Record every static letter in '
                             'each of two or more common session IDs, or pass --validation-sessions explicitly.')
    val = np.isin(dataset.sessions, chosen)
    return np.flatnonzero(~val), np.flatnonzero(val)


def probability_matrix(features: np.ndarray, arrays: dict, supported: list[str], temperature: float = 1.) -> np.ndarray:
    ids = np.asarray([CLASS_TO_ID[x] for x in supported])
    logits = np.maximum(features @ arrays['W1'] + arrays['b1'], 0) @ arrays['W2'] + arrays['b2']
    if not np.isfinite(temperature) or not .05<=temperature<=20:raise ValueError('Invalid global softmax temperature.')
    selected = logits[:, ids].astype(np.float64)/temperature
    selected -= selected.max(axis=1, keepdims=True)
    probs = np.exp(selected)
    probs /= probs.sum(axis=1, keepdims=True)
    result = np.zeros((len(features), 26), dtype=np.float64)
    result[:, ids] = probs
    return result


def metrics(features: np.ndarray, labels: np.ndarray, arrays: dict, supported: list[str], temperature: float = 1.) -> dict:
    if len(features) == 0:
        raise ValueError('Cannot evaluate an empty split.')
    probabilities = probability_matrix(features, arrays, supported,temperature)
    predictions = probabilities.argmax(axis=1)
    ids = [CLASS_TO_ID[x] for x in supported]
    cm = np.zeros((len(ids), len(ids)), dtype=np.int64)
    lookup = {value: i for i, value in enumerate(ids)}
    for actual, predicted in zip(labels, predictions):
        if int(actual) not in lookup:
            raise ValueError('Evaluation contains an unsupported class.')
        cm[lookup[int(actual)], lookup[int(predicted)]] += 1
    per_class = {}
    for i, label in enumerate(supported):
        tp, actual, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        precision, recall = tp / max(1, predicted), tp / max(1, actual)
        per_class[label] = {'samples': actual, 'precision': precision, 'recall': recall,
                            'f1': 2*precision*recall/max(1e-15, precision+recall)}
    return {'samples': len(labels), 'accuracy': float(np.mean(predictions == labels)),
            'cross_entropy': float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-15)).mean()),
            'macro_f1': float(np.mean([m['f1'] for m in per_class.values()])),
            'labels': supported, 'confusion_matrix': cm.tolist(), 'per_class': per_class}


def save_report(report: dict, output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for split in ('train', 'validation', 'evaluation'):
        if split in report and isinstance(report[split], dict) and 'confusion_matrix' in report[split]:
            metric = report[split]
            with output.with_name(output.stem + f'_{split}_confusion.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['actual / predicted'] + metric['labels'])
                for label, row in zip(metric['labels'], metric['confusion_matrix']):
                    writer.writerow([label] + row)
