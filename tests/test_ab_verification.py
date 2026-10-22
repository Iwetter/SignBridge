"""Synthetic fitted-model fixtures verify the offline audit, not ASL accuracy."""
from hashlib import sha256
import json

import numpy as np
import pytest

from scripts.verify_ab_model import main, verify
from src.contracts import ALPHABET, CLASS_TO_ID
from src.recognition.cpu import CPURecognizer
from src.recognition.dataset import load_dataset
from training.train import fit_mlp


@pytest.fixture
def fitted_ab(tmp_path):
    samples = tmp_path / 'samples'
    samples.mkdir()
    rng = np.random.default_rng(3)
    for session in ('train-one', 'validation-two', 'unseen-three'):
        for label, sign in [('A', -1), ('B', 1)]:
            vectors = rng.normal(0, .015, (12, 63)).astype(np.float32)
            vectors[:, 3:15] += sign
            np.savez_compressed(samples / f'{session}-{label}.npz',
                                features=vectors, sequence=np.zeros((12, 134), dtype=np.float32),
                                timestamps_ms=np.arange(12, dtype=np.int64) * 100,
                                label=np.asarray(label), session_id=np.asarray(session), schema_version=np.asarray(1))
    dataset = load_dataset(samples)
    train, val = dataset.sessions == 'train-one', dataset.sessions == 'validation-two'
    arrays, _ = fit_mlp(dataset.features[train], dataset.labels[train], dataset.features[val], dataset.labels[val],
                        ['A', 'B'], seed=9, epochs=80, patience=30, learning_rate=.01)
    meta = {'schema_version': 1, 'supported_labels': ['A', 'B'],
            'class_mapping': {label: CLASS_TO_ID[label] for label in ALPHABET},
            'synthetic_fixture': True,
            'training_provenance': {'data_origin': 'synthetic_fixture',
                                    'train_sessions': ['train-one'], 'validation_sessions': ['validation-two'],
                                    'files': [entry for entry in dataset.files if entry['session_id'] != 'unseen-three']}}
    model = tmp_path / 'synthetic-model.npz'
    np.savez_compressed(model, **arrays, metadata=np.asarray(json.dumps(meta)))
    return model, samples, meta, arrays


def test_actual_fitted_cpu_predictions_and_split_scope(fitted_ab):
    model, samples, _, _ = fitted_ab
    original = {path: sha256(path.read_bytes()).hexdigest() for path in [model, *samples.glob('*.npz')]}
    report = verify(model, samples, allow_synthetic_fixture=True)
    assert report['synthetic_fixture'] and not report['recognition_evidence']
    assert not report['live_recognition_validated']
    assert 'SYNTHETIC' in report['evidence_scope']
    for name, count in [('all_saved', 72), ('training', 24), ('validation', 24), ('unseen_sessions', 24)]:
        scope = report['scopes'][name]
        assert scope['samples'] == count
        assert scope['confusion_matrix'] == [[count // 2, 0], [0, count // 2]]
        assert scope['both_predominantly_correct'] and scope['accepted_fraction'] == 1
        assert scope['per_class']['A']['mean_actual_class_score'] > .8
    recognizer, dataset = CPURecognizer(model), load_dataset(samples)
    first = recognizer.predict(dataset.features[0], 0)
    assert first.source == 'cpu' and first.label in ['A', 'B'] and .8 < first.confidence <= 1
    assert {entry['session_scope'] for entry in report['files']} == {'training', 'validation', 'unseen_sessions'}
    assert sum(entry['in_model_dataset_manifest'] for entry in report['files']) == 4
    assert {path: sha256(path.read_bytes()).hexdigest() for path in original} == original


def test_synthetic_requires_explicit_permission_and_cli_writes_report_only(fitted_ab, tmp_path):
    model, samples, _, _ = fitted_ab
    with pytest.raises(ValueError, match='Synthetic fixture refused'):
        verify(model, samples)
    output = tmp_path / 'report.json'
    assert main(['--model', str(model), '--data-dir', str(samples), '--output', str(output)]) == 2
    assert not output.exists()
    assert main(['--model', str(model), '--data-dir', str(samples), '--output', str(output), '--allow-synthetic-fixture']) == 0
    assert json.loads(output.read_text())['scopes']['validation']['samples'] == 24
    assert main(['--model', str(model), '--data-dir', str(samples), '--output', str(samples / 'report.json'), '--allow-synthetic-fixture']) == 2
    assert not (samples / 'report.json').exists()


def test_rejects_extra_classes_missing_class_and_overlapping_sessions(fitted_ab):
    model, samples, meta, arrays = fitted_ab
    meta['supported_labels'] = ['A', 'B', 'C']
    np.savez_compressed(model, **arrays, metadata=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError, match='exactly'):
        verify(model, samples, allow_synthetic_fixture=True)
    meta['supported_labels'] = ['A', 'B']
    meta['training_provenance']['validation_sessions'] = ['train-one']
    np.savez_compressed(model, **arrays, metadata=np.asarray(json.dumps(meta)))
    with pytest.raises(ValueError, match='overlaps'):
        verify(model, samples, allow_synthetic_fixture=True)
    meta['training_provenance']['validation_sessions'] = ['validation-two']
    np.savez_compressed(model, **arrays, metadata=np.asarray(json.dumps(meta)))
    for path in samples.glob('*-B.npz'):
        path.unlink()
    with pytest.raises(ValueError, match='No persisted eligible samples|BOTH'):
        verify(model, samples, allow_synthetic_fixture=True)


def test_low_confidence_is_not_counted_as_accepted(fitted_ab):
    model, samples, meta, arrays = fitted_ab
    # A deliberately flattened fitted model tests score gating, not a claimed classifier.
    arrays['W2'] *= 0
    arrays['b2'] *= 0
    np.savez_compressed(model, **arrays, metadata=np.asarray(json.dumps(meta)))
    report = verify(model, samples, allow_synthetic_fixture=True)
    scope = report['scopes']['all_saved']
    assert scope['raw_candidate_accuracy'] == .5 and scope['accepted_fraction'] == 0
    assert scope['accepted_accuracy'] is None and not scope['both_predominantly_correct']
    assert scope['per_class']['B']['mean_actual_class_score'] == .5
