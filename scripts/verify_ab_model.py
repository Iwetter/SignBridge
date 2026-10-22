"""Verify the loaded A/B CPU classifier against persisted landmark vectors.

This is offline classification evidence, never proof of live webcam recognition.
Production inputs default to this repository's model and saved sample directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from src.contracts import CLASS_TO_ID
from src.recognition.cpu import CPURecognizer
from src.recognition.dataset import load_dataset

LABELS = ['A', 'B']


def _summarize(actual: np.ndarray, predicted: np.ndarray, scores: np.ndarray,
               confidence: np.ndarray, threshold: float) -> dict:
    accepted = confidence >= threshold
    per_class = {}
    for label in LABELS:
        class_id = CLASS_TO_ID[label]
        selected = actual == class_id
        total = int(selected.sum())
        correct = int(np.count_nonzero(selected & (actual == predicted)))
        accepted_count = int(np.count_nonzero(selected & accepted))
        accepted_correct = int(np.count_nonzero(selected & accepted & (actual == predicted)))
        per_class[label] = {
            'samples': total,
            'correct': correct,
            'predicted': {name: int(np.count_nonzero(selected & (predicted == CLASS_TO_ID[name])))
                          for name in LABELS},
            'raw_candidate_accuracy': correct / total if total else None,
            'predominantly_correct': correct > total / 2 if total else None,
            'mean_actual_class_score': float(scores[selected, class_id].mean()) if total else None,
            'mean_candidate_confidence': float(confidence[selected].mean()) if total else None,
            'accepted_samples': accepted_count,
            'accepted_correct': accepted_correct,
            'accepted_fraction': accepted_count / total if total else None,
            'accepted_accuracy': accepted_correct / accepted_count if accepted_count else None,
        }
    count = len(actual)
    accepted_count = int(accepted.sum())
    return {
        'samples': count, 'labels': LABELS,
        'raw_candidate_accuracy': float(np.mean(actual == predicted)) if count else None,
        'confusion_matrix': [[int(np.count_nonzero((actual == CLASS_TO_ID[a]) &
                                                  (predicted == CLASS_TO_ID[p])))
                              for p in LABELS] for a in LABELS],
        'confusion_orientation': 'rows = actual, columns = predicted; labels A, B',
        'acceptance_threshold': threshold,
        'accepted_samples': accepted_count,
        'accepted_fraction': accepted_count / count if count else None,
        'accepted_accuracy': float(np.mean(actual[accepted] == predicted[accepted])) if accepted_count else None,
        'both_classes_present': all(per_class[label]['samples'] > 0 for label in LABELS),
        'both_predominantly_correct': all(per_class[label]['predominantly_correct'] is True for label in LABELS),
        'per_class': per_class,
    }


def verify(model_path: str | Path, data_dir: str | Path, *, threshold: float = .8,
           allow_synthetic_fixture: bool = False) -> dict:
    if not np.isfinite(threshold) or not .5 < threshold <= 1:
        raise ValueError('A/B acceptance threshold must be greater than .5 and at most 1.')
    model_path, data_dir = Path(model_path).resolve(), Path(data_dir).resolve()
    recognizer = CPURecognizer(model_path)
    try:
        if list(recognizer.supported_labels) != LABELS:
            raise ValueError('A/B smoke verification requires model supported_labels exactly [A, B].')
        metadata = recognizer.metadata
        provenance = metadata['training_provenance']
        origin = provenance.get('data_origin')
        synthetic = bool(metadata.get('synthetic_fixture')) or origin == 'synthetic_fixture'
        if synthetic and not allow_synthetic_fixture:
            raise ValueError('Synthetic fixture refused as recognition evidence. For numerical tests only, pass --allow-synthetic-fixture.')
        if not synthetic and origin != 'local_capture':
            raise ValueError('Recognition evidence requires training provenance data_origin=local_capture.')
        train_sessions = provenance.get('train_sessions', [])
        validation_sessions = provenance.get('validation_sessions', [])
        for name, values in [('train_sessions', train_sessions), ('validation_sessions', validation_sessions)]:
            if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f'Model provenance {name} must be a list of nonempty session IDs.')
        if set(train_sessions) & set(validation_sessions):
            raise ValueError('Model provenance overlaps training and validation sessions; holdout evidence is invalid.')
        dataset = load_dataset(data_dir, labels=LABELS)
        selected = np.isin(dataset.labels, [CLASS_TO_ID[label] for label in LABELS])
        features, actual, sessions = (dataset.features[selected], dataset.labels[selected], dataset.sessions[selected])
        if set(actual.tolist()) != {CLASS_TO_ID[label] for label in LABELS}:
            raise ValueError('Saved feature vectors for BOTH A and B are required.')
        # Exercise exactly the same loaded model/predict method as the live path.
        predictions = [recognizer.predict(vector, index) for index, vector in enumerate(features)]
        scores = np.asarray([prediction.scores for prediction in predictions], dtype=np.float32)
        predicted = np.asarray([CLASS_TO_ID[prediction.label] for prediction in predictions], dtype=np.int64)
        confidence = np.asarray([prediction.confidence for prediction in predictions], dtype=np.float64)
        masks = {'all_saved': np.ones(len(actual), dtype=bool),
                 'training': np.isin(sessions, train_sessions),
                 'validation': np.isin(sessions, validation_sessions),
                 'unseen_sessions': ~np.isin(sessions, train_sessions + validation_sessions)}
        scopes = {name: _summarize(actual[mask], predicted[mask], scores[mask], confidence[mask], threshold)
                  for name, mask in masks.items()}
        manifest = [entry for entry in dataset.files if entry['label'] in LABELS]
        trained_hashes = {entry.get('sha256') for entry in provenance.get('files', []) if isinstance(entry, dict)}
        for entry in manifest:
            entry['in_model_dataset_manifest'] = entry['sha256'] in trained_hashes
            entry['session_scope'] = ('training' if entry['session_id'] in train_sessions else
                                      'validation' if entry['session_id'] in validation_sessions else 'unseen_sessions')
        return {
            'schema_version': 1,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'model': str(model_path), 'model_sha256': sha256(model_path.read_bytes()).hexdigest(),
            'data_dir': str(data_dir), 'classifier': 'CPURecognizer.predict',
            'model_loaded': True, 'supported_labels': LABELS,
            'synthetic_fixture': synthetic, 'training_data_origin': origin,
            'recognition_evidence': not synthetic,
            'live_recognition_validated': False,
            'evidence_scope': ('SYNTHETIC NUMERICAL FIXTURE ONLY: no ASL recognition evidence.' if synthetic else
                               'Offline predictions on persisted local feature vectors. Real origin is declared in model provenance; this does not validate live signing.'),
            'scope_notes': {
                'all_saved': 'Includes training data; never call this held-out accuracy.',
                'training': 'Sessions used for fitting; these are in-sample results.',
                'validation': 'Session holdout used for model selection/early stopping; not an untouched final test.',
                'unseen_sessions': 'Sessions absent from model split metadata; recording quality and signer independence are unverified.',
                'scores': 'Actual masked A/B softmax scores from CPURecognizer; high confidence alone does not prove an arbitrary pose is A/B.',
            },
            'train_sessions': train_sessions, 'validation_sessions': validation_sessions,
            'saved_sessions': sorted(set(sessions.tolist())), 'files': manifest,
            'dataset_warnings': dataset.warnings,
            'scopes': scopes,
        }
    finally:
        recognizer.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(ROOT / 'models' / 'alphabet.npz'))
    parser.add_argument('--data-dir', default=str(ROOT / 'data' / 'samples'))
    parser.add_argument('--output', default=str(ROOT / 'work' / 'ab-verification.json'))
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--allow-synthetic-fixture', action='store_true',
                        help='Numerical test fixtures only; report will explicitly deny recognition evidence.')
    args = parser.parse_args(argv)
    try:
        report = verify(args.model, args.data_dir, threshold=args.threshold,
                        allow_synthetic_fixture=args.allow_synthetic_fixture)
        output = Path(args.output).resolve()
        model_path, data_dir = Path(args.model).resolve(), Path(args.data_dir).resolve()
        if output == model_path or output.suffix.lower() != '.json' or output == data_dir or data_dir in output.parents:
            raise ValueError('Report must be a .json file outside the sample directory and must not replace the model.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(report['evidence_scope'])
        for name, scope in report['scopes'].items():
            print(f'{name}: {scope["samples"]} samples; raw accuracy={scope["raw_candidate_accuracy"]}; accepted fraction={scope["accepted_fraction"]}')
        print(f'Saved {output}')
        return 0 if report['scopes']['all_saved']['both_predominantly_correct'] else 1
    except (ValueError, OSError) as exc:
        print(f'A/B verification unavailable: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
