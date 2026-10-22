"""Readiness and incremental training with temporary numeric test fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import CLASS_TO_ID, STATIC_LABELS, STATIC_CLASS_TO_ID
from src.recognition import CPURecognizer
from src.recognition.readiness import alphabet_readiness
from src.vision.dataset_summary import dataset_summary
from training import train


def write_samples(directory, label, session, count, *, suffix='', legacy=False, no_hand=0, zero=0):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sum(map(ord, label + session + suffix)))
    features = rng.normal(0, .01, (count, 63)).astype(np.float32)
    features[:, CLASS_TO_ID[label] + 3] += 1
    features[:zero] = 0
    sequence = np.zeros((count, 134), dtype=np.float32)
    sequence[:, 132] = 1
    if no_hand:
        sequence[-no_hand:, 132] = 0
    arrays = dict(features=features, sequence=sequence,
                  timestamps_ms=np.arange(count, dtype=np.int64) * 100,
                  label=np.asarray(label), session_id=np.asarray(session), schema_version=np.asarray(1))
    if not legacy:
        arrays['normalization_id'] = np.asarray('physical-unmirrored-v2')
    path = directory / f'{session}-{label}{suffix}.npz'
    np.savez_compressed(path, **arrays)
    return path


def ready_class(directory, label, *, legacy=False):
    for session in ('one', 'two'):
        write_samples(directory, label, session, 50, legacy=legacy)


def args(directory, model):
    return ['--data-dir', str(directory), '--output', str(model), '--ready-only',
            '--epochs', '8', '--data-origin', 'synthetic_fixture',
            '--provenance', 'Temporary numeric training regression fixtures; no live recognition claim.']


def test_readiness_requires_100_and_two_sessions_with_50_each():
    summary = {'files': [
        {'label': 'A', 'session_id': 'one', 'samples': 30},
        {'label': 'A', 'session_id': 'one', 'samples': 20},
        {'label': 'A', 'session_id': 'two', 'samples': 50},
        {'label': 'B', 'session_id': 'one', 'samples': 200},
        {'label': 'C', 'session_id': 'one', 'samples': 90},
        {'label': 'C', 'session_id': 'two', 'samples': 10},
        {'label': 'D', 'session_id': 'one', 'samples': 49},
        {'label': 'D', 'session_id': 'two', 'samples': 49},
        {'label': 'J', 'session_id': 'one', 'samples': 100},
        {'label': 'J', 'session_id': 'two', 'samples': 100},
        {'label': 'Z', 'session_id': 'one', 'samples': 100},
        {'label': 'Z', 'session_id': 'two', 'samples': 100},
    ]}
    result = alphabet_readiness(summary)
    assert list(result['static_progress']) == list(STATIC_LABELS)
    assert len(result['static_progress']) == 24
    assert result['ready_labels'] == ['A']
    assert result['static_progress']['A']['sessions'] == 2
    assert result['static_progress']['A']['session_counts'] == {'one': 50, 'two': 50}
    assert result['static_progress']['B']['qualified_sessions'] == 1
    assert '1/2 sessions' in result['static_progress']['C']['reason']
    assert result['static_progress']['D']['samples'] == 98
    assert not result['static_progress']['D']['ready']
    assert 'J' not in result['static_progress'] and 'Z' not in result['static_progress']


def test_summary_keeps_legacy_and_counts_only_valid_static_rows(tmp_path):
    for session in ('one', 'two'):
        write_samples(tmp_path, 'A', session, 55, legacy=True, zero=2, no_hand=3)
    summary = dataset_summary(tmp_path)
    assert summary['counts']['A'] == summary['legacy_samples'] == 100
    assert summary['static_progress']['A']['ready']
    assert summary['ready_labels'] == ['A']
    assert sum(entry['total_frames'] for entry in summary['files']) == 110
    assert sum(entry['rejected_frames'] for entry in summary['files']) == 10
    assert not summary['errors']


@pytest.mark.parametrize('invalid', ['range', 'dtype', 'schema', 'timestamp', 'nan'])
def test_invalid_archive_cannot_make_class_ready(tmp_path, invalid):
    ready_class(tmp_path, 'A')
    path = tmp_path / 'two-A.npz'
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if invalid == 'range':
        arrays['features'][0, 0] = 4.5
    elif invalid == 'dtype':
        arrays['features'] = arrays['features'].astype(np.float64)
    elif invalid == 'schema':
        arrays['schema_version'] = np.asarray(2)
    elif invalid == 'timestamp':
        arrays['timestamps_ms'][2] = arrays['timestamps_ms'][1]
    else:
        arrays['features'][0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    summary = dataset_summary(tmp_path)
    assert summary['counts']['A'] == 50
    assert not summary['static_progress']['A']['ready']
    assert len(summary['errors']) == 1
    model = tmp_path / 'models' / 'model.npz'
    assert train.main(args(tmp_path, model)) == 2
    assert not model.exists()


def test_ready_training_expands_ab_to_abcd_and_preserves_old_artifacts(tmp_path):
    directory, model_path = tmp_path / 'samples', tmp_path / 'models' / 'alphabet.npz'
    ready_class(directory, 'A')
    ready_class(directory, 'B')
    write_samples(directory, 'C', 'one', 50)
    write_samples(directory, 'D', 'one', 100)
    write_samples(directory, 'E', 'one', 49)
    write_samples(directory, 'E', 'two', 49)
    ready_class(directory, 'J')
    ready_class(directory, 'Z')
    originals = {path: path.read_bytes() for path in directory.iterdir()}
    assert train.main(args(directory, model_path)) == 0
    assert CPURecognizer(model_path).supported_labels == ('A', 'B')
    first_report = json.loads(model_path.with_name('alphabet_metrics.json').read_text())
    assert first_report['ready_labels'] == ['A', 'B']
    assert not first_report['previous_artifacts_history']
    assert 'C' in first_report['skipped_static_classes']
    assert 'J' not in first_report['static_progress']
    first_artifacts = {path.name: path.read_bytes() for path in model_path.parent.iterdir() if path.is_file()}
    write_samples(directory, 'C', 'two', 50)
    write_samples(directory, 'D', 'two', 50)
    assert train.main(args(directory, model_path)) == 0
    updated = CPURecognizer(model_path)
    assert updated.supported_labels == ('A', 'B', 'C', 'D')
    report = json.loads(model_path.with_name('alphabet_metrics.json').read_text())
    assert report['ready_labels'] == ['A', 'B', 'C', 'D']
    assert report['static_class_mapping'] == STATIC_CLASS_TO_ID
    assert len(report['class_mapping']) == 26
    assert report['samples_per_class'] == {'A': 100, 'B': 100, 'C': 100, 'D': 150}
    assert 'E' in report['skipped_static_classes']
    assert len(report['validation']['confusion_matrix']) == 4
    assert set(report['training_provenance']['train_sessions']).isdisjoint(report['training_provenance']['validation_sessions'])
    history = Path(report['previous_artifacts_history'])
    assert history.parent == model_path.parent / 'history'
    for name, content in first_artifacts.items():
        assert (history / name).read_bytes() == content
    assert CPURecognizer(history / model_path.name).supported_labels == ('A', 'B')
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert json.loads(model_path.with_name('alphabet_classes.json').read_text())['static_class_mapping'] == STATIC_CLASS_TO_ID


def test_ready_training_filters_no_hand_and_zero_rows_consistently(tmp_path):
    directory, model = tmp_path / 'samples', tmp_path / 'model.npz'
    for label in ('A', 'B'):
        for session in ('one', 'two'):
            write_samples(directory, label, session, 55, zero=2, no_hand=3)
    assert train.main(args(directory, model)) == 0
    report = json.loads(model.with_name('model_metrics.json').read_text())
    assert report['samples_per_class'] == {'A': 100, 'B': 100}
    assert report['train']['samples'] + report['validation']['samples'] == 200
    assert sum(entry['rejected_frames'] for entry in report['training_provenance']['files']) == 20


def test_ready_training_requires_two_ready_classes_and_flags_are_exclusive(tmp_path, capsys):
    ready_class(tmp_path / 'samples', 'A')
    write_samples(tmp_path / 'samples', 'B', 'one', 100)
    model = tmp_path / 'model.npz'
    assert train.main(args(tmp_path / 'samples', model)) == 2
    assert 'At least two READY' in capsys.readouterr().err
    assert not model.exists()
    with pytest.raises(SystemExit) as exc:
        train.main(args(tmp_path / 'samples', model) + ['--labels', 'A', 'B'])
    assert exc.value.code == 2


def test_failed_retrain_keeps_current_model_and_earlier_history(tmp_path, monkeypatch):
    directory, model = tmp_path / 'samples', tmp_path / 'models' / 'alphabet.npz'
    ready_class(directory, 'A')
    ready_class(directory, 'B')
    assert train.main(args(directory, model)) == 0
    assert train.main(args(directory, model)) == 0
    previous = {str(path.relative_to(model.parent)): path.read_bytes()
                for path in model.parent.rglob('*') if path.is_file()}
    replace = train.os.replace

    def fail_model(source, target):
        if Path(target) == model:
            raise OSError('locked model test fixture')
        return replace(source, target)

    monkeypatch.setattr(train.os, 'replace', fail_model)
    assert train.main(args(directory, model)) == 2
    assert {str(path.relative_to(model.parent)): path.read_bytes()
            for path in model.parent.rglob('*') if path.is_file()} == previous


def test_all_24_static_classes_keep_canonical_ids_and_mask_j_z(tmp_path):
    directory,model_path=tmp_path/'samples',tmp_path/'models'/'alphabet.npz'
    for label in STATIC_LABELS:ready_class(directory,label)
    assert train.main(args(directory,model_path))==0
    model=CPURecognizer(model_path)
    assert model.supported_labels==STATIC_LABELS
    assert model.metadata['class_mapping']['K']==STATIC_CLASS_TO_ID['K']==10
    assert model.metadata['class_mapping']['Y']==24
    report=json.loads(model_path.with_name('alphabet_metrics.json').read_text())
    assert report['class_count']==24 and len(report['validation']['confusion_matrix'])==24
    assert report['ready_labels']==list(STATIC_LABELS)
    for label in STATIC_LABELS:
        with np.load(directory/f'one-{label}.npz',allow_pickle=False) as archive:
            prediction=model.predict(archive['features'][0],0)
        assert prediction.scores.shape==(26,)
        assert prediction.scores[9]==prediction.scores[25]==0
        assert prediction.label in STATIC_LABELS
        assert CLASS_TO_ID[prediction.label]==int(prediction.scores.argmax())
