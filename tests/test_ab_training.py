"""A/B training contracts, using temporary numeric fixtures, never live evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import CLASS_TO_ID
from src.recognition import CPURecognizer
from src.recognition.dataset import load_dataset, session_split, probability_matrix
from training import evaluate, train


def sample(directory, label, session, count=12, *, tagged=True):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sum(map(ord, label + session)))
    features = rng.normal(0, .015, (count, 63)).astype(np.float32)
    features[:, CLASS_TO_ID[label] + 3] += 1
    arrays = {'features': features, 'sequence': np.zeros((count, 134), dtype=np.float32),
              'timestamps_ms': np.arange(count, dtype=np.int64) * 100,
              'label': np.asarray(label), 'session_id': np.asarray(session),
              'schema_version': np.asarray(1)}
    if tagged:
        arrays['normalization_id'] = np.asarray('physical-unmirrored-v2')
    path = directory / f'{label}-{session}.npz'
    np.savez_compressed(path, **arrays)
    return path


def ab_data(directory):
    for label in ('A', 'B'):
        for session in ('one', 'two'):
            sample(directory, label, f'{label}-{session}', tagged=(label == 'B'))
    sample(directory, 'C', 'preserved')


def train_args(directory, output):
    return ['--data-dir', str(directory), '--output', str(output), '--labels', 'A', 'B',
            '--epochs', '45', '--data-origin', 'synthetic_fixture', '--provenance',
            'Temporary synthetic unit-test vectors, not human recognition evidence.']


def test_ab_filter_validates_persisted_files_and_reports_legacy_and_c(tmp_path):
    directory = tmp_path / 'samples'
    ab_data(directory)
    original = {path: path.read_bytes() for path in directory.glob('*.npz')}
    dataset = load_dataset(directory, labels=['A', 'B'])
    assert dataset.supported_labels == ['A', 'B']
    assert dataset.summary()['samples_per_class'] == {'A': 24, 'B': 24}
    assert dataset.excluded_files[0]['label'] == 'C'
    assert dataset.excluded_files[0]['reason'] == 'outside requested training classes'
    assert dataset.excluded_files[0]['samples'] == 12
    assert 'Included 24 legacy samples' in dataset.warnings[0]
    assert all(path.read_bytes() == content for path, content in original.items())
    corrupted = directory / 'C-preserved.npz'
    with np.load(corrupted, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['features'][0, 0] = np.nan
    np.savez_compressed(corrupted, **arrays)
    with pytest.raises(ValueError, match='finite'):
        load_dataset(directory, labels=['A', 'B'])


def test_session_split_handles_classes_recorded_in_different_sessions(tmp_path):
    ab_data(tmp_path)
    dataset = load_dataset(tmp_path, labels=['A', 'B'])
    training, validation = session_split(dataset)
    assert len(training) == 24 and len(validation) == 24
    assert set(dataset.sessions[training]).isdisjoint(dataset.sessions[validation])
    assert set(dataset.labels[training]) == set(dataset.labels[validation]) == {0, 1}
    again = session_split(dataset)
    np.testing.assert_array_equal(training, again[0])
    np.testing.assert_array_equal(validation, again[1])


def test_single_b_session_is_actionable_and_does_not_overwrite_model(tmp_path, capsys):
    directory = tmp_path / 'samples'
    sample(directory, 'A', 'A-one')
    sample(directory, 'A', 'A-two')
    sample(directory, 'B', 'B-one')
    model = tmp_path / 'model.npz'
    model.write_bytes(b'previous-model-must-remain')
    assert train.main(train_args(directory, model)) == 2
    captured = capsys.readouterr()
    assert 'No honest holdout: B: 1 session' in captured.err
    assert 'New session' in captured.err
    assert model.read_bytes() == b'previous-model-must-remain'
    assert not model.with_name('model_metrics.json').exists()


def test_training_saves_counts_mapping_scope_and_session_metrics(tmp_path, capsys, monkeypatch):
    directory, output = tmp_path / 'samples', tmp_path / 'models' / 'ab.npz'
    ab_data(directory)
    fitted={}
    real_fit=train.fit_mlp
    def capture_fit(*args,**kwargs):
        arrays,history=real_fit(*args,**kwargs)
        fitted.update({key:value.copy() for key,value in arrays.items()})
        return arrays,history
    monkeypatch.setattr(train,'fit_mlp',capture_fit)
    assert train.main(train_args(directory, output)) == 0
    model = CPURecognizer(output)
    assert model.supported_labels == ('A', 'B')
    metadata = model.metadata
    assert metadata['synthetic_fixture'] is True
    assert metadata['live_recognition_validated'] is False
    report = json.loads(output.with_name('ab_metrics.json').read_text())
    mapping = json.loads(output.with_name('ab_classes.json').read_text())
    assert mapping['model_id'] == report['model_id'] == metadata['model_id']
    assert mapping['supported_labels'] == ['A', 'B']
    assert mapping['class_mapping']['A'] == 0 and mapping['class_mapping']['B'] == 1
    assert len(mapping['class_mapping']) == 26
    assert report['class_count'] == 2 and report['samples_per_class'] == {'A': 24, 'B': 24}
    assert report['train']['samples'] == report['validation']['samples'] == 24
    assert len(report['validation']['confusion_matrix']) == 2
    assert set(report['training_provenance']['train_sessions']).isdisjoint(report['training_provenance']['validation_sessions'])
    assert report['excluded_files'][0]['label'] == 'C'
    assert metadata['training_results']['validation_accuracy'] == report['validation']['accuracy']
    assert metadata['artifact_paths']['metrics'] == str(output.with_name('ab_metrics.json'))
    dataset=load_dataset(directory,labels=['A','B'])
    _,held_out=session_split(dataset)
    expected=probability_matrix(dataset.features[held_out],fitted,['A','B'])
    reloaded=CPURecognizer(output)
    actual=np.stack([reloaded.predict(vector,n).scores for n,vector in enumerate(dataset.features[held_out])])
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
    for label in ('A','B'):
        assert report['validation']['per_class'][label]['recall']>.5
    assert 'Model trained successfully' in capsys.readouterr().out
    assert not list(output.parent.glob('*.tmp'))


def test_model_commit_failure_restores_previous_model_and_sidecars(tmp_path, monkeypatch, capsys):
    directory, output = tmp_path / 'samples', tmp_path / 'models' / 'ab.npz'
    ab_data(directory)
    assert train.main(train_args(directory, output)) == 0
    originals = {path: path.read_bytes() for path in output.parent.iterdir()}
    real_replace = train.os.replace

    def failing_replace(source, target):
        if Path(target) == output:
            raise OSError('test model target is locked')
        return real_replace(source, target)

    monkeypatch.setattr(train.os, 'replace', failing_replace)
    assert train.main(train_args(directory, output)) == 2
    assert 'test model target is locked' in capsys.readouterr().err
    assert {path: path.read_bytes() for path in output.parent.iterdir()} == originals


def test_evaluation_uses_ab_mapping_and_preserves_excluded_c(tmp_path):
    directory, output = tmp_path / 'samples', tmp_path / 'models' / 'ab.npz'
    ab_data(directory)
    c_path = directory / 'C-preserved.npz'
    original = c_path.read_bytes()
    assert train.main(train_args(directory, output)) == 0
    report_path = tmp_path / 'evaluation.json'
    assert evaluate.main(['--data-dir', str(directory), '--model', str(output),
                          '--output', str(report_path), '--benchmark-samples', '5']) == 0
    report = json.loads(report_path.read_text())
    assert report['coverage'] == ['A', 'B']
    assert report['evaluation']['samples'] == 24
    assert report['excluded_files'][0]['label'] == 'C'
    assert c_path.read_bytes() == original
