"""Regression checks using synthetic inputs only, never recognition evidence.

Run with --basetemp=work/pytest so every generated dataset/model stays in work/.
The physical camera and FPGA are deliberately outside these software tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.application import ApplicationController, run_headless_replay
from src.contracts import ALPHABET, CLASS_TO_ID, FrameObservation, HandObservation, Prediction, Token
from src.conversation import ConversationEngine
from src.recognition import CPURecognizer, QuantizedRecognizer
from src.recognition.dataset import load_dataset, session_split
from src.recognition.quantized import integer_forward, load_quantized_model
from src.temporal import MotionRecognizer, TemporalDecoder
from src.vision import DatasetRecorder, normalize_landmarks, primary_features, sequence_features

ROOT = Path(__file__).resolve().parents[1]


def hand():
    points = np.zeros((21, 3), dtype=np.float32)
    points[:, 0] = np.linspace(.4, .6, 21)
    points[:, 1] = np.linspace(.7, .3, 21)
    points[:, 2] = np.linspace(0, -.04, 21)
    return points


def observation(timestamp, points=None):
    return FrameObservation(timestamp, [HandObservation(hand() if points is None else points)])


def cli(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True,
                          text=True, timeout=45, encoding='utf-8', errors='replace')


def hold(decoder, label, start, duration=600, confidence=.95):
    emitted = []
    for stamp in range(start, start + duration, 40):
        prediction = None if label is None else Prediction(label, confidence, stamp, 'synthetic-test')
        token = decoder.update(prediction, stamp)
        if token:
            emitted.append(token)
    return emitted, start + duration


def test_normalization_translation_scale_handedness_and_invalid_inputs():
    points = hand()
    expected = normalize_landmarks(points)
    np.testing.assert_allclose(normalize_landmarks(points * 2 + [.2, -.1, .03]), expected, atol=2e-6)
    reflected = points.copy()
    reflected[:, 0] *= -1
    np.testing.assert_array_equal(normalize_landmarks(reflected, 'Left'), expected)
    assert expected.dtype == np.float32 and expected.shape == (63,)
    np.testing.assert_array_equal(expected[:3], 0)
    np.testing.assert_array_equal(normalize_landmarks(np.zeros((21, 3))), 0)
    for invalid in (np.full((21, 3), np.nan), np.zeros((20, 3))):
        with pytest.raises(ValueError):
            normalize_landmarks(invalid)


def test_two_hand_sequence_and_primary_hand_share_normalization():
    right, left = hand(), hand() + [.2, 0, 0]
    obs = FrameObservation(20, [HandObservation(left, 'Left'), HandObservation(right, 'Right')])
    sequence = sequence_features(obs)
    np.testing.assert_array_equal(sequence[:63], primary_features(obs))
    np.testing.assert_array_equal(sequence[63:126], normalize_landmarks(left, 'Left'))
    np.testing.assert_array_equal(sequence[126:129], right[0])
    np.testing.assert_array_equal(sequence[129:132], left[0].astype(np.float32))
    np.testing.assert_array_equal(sequence[132:], [1, 1])
    assert sequence.dtype == np.float32 and sequence.shape == (134,)
    assert primary_features(FrameObservation(21)) is None


def test_static_recorder_roundtrips_shared_schema_and_rejects_bad_timestamps(tmp_path):
    recorder = DatasetRecorder(tmp_path, session_id='synthetic-session')
    recorder.start('I')
    assert not recorder.add(FrameObservation(0))
    assert recorder.add(observation(100))
    assert not recorder.add(observation(140))
    with pytest.raises(ValueError, match='strictly increase'):
        recorder.add(observation(140))
    assert recorder.add(observation(200))
    output = recorder.finish()
    assert output and not recorder.active
    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {'features', 'sequence', 'timestamps_ms', 'label', 'session_id', 'schema_version', 'normalization_id'}
        np.testing.assert_array_equal(saved['features'][0], primary_features(observation(100)))
        np.testing.assert_array_equal(saved['sequence'][0], sequence_features(observation(100)))
        assert saved['timestamps_ms'].dtype == np.int64
        assert str(saved['session_id']) == 'synthetic-session'
    dataset = load_dataset(tmp_path)
    assert dataset.features.shape == (2, 63) and dataset.supported_labels == ['I']
    recorder.start('I')
    assert recorder.finish() is None


def test_temporal_recorder_preserves_interior_gaps_trims_release_and_rejects_frozen_j(tmp_path):
    recorder = DatasetRecorder(tmp_path, session_id='synthetic-clip')
    recorder.start('J')
    for stamp in range(0, 440, 40):
        recorder.add(FrameObservation(stamp) if stamp == 160 else observation(stamp))
    for stamp in range(440, 880, 40):
        recorder.add(FrameObservation(stamp))
    assert recorder.release_ready
    output = recorder.finish()
    with np.load(output, allow_pickle=False) as saved:
        assert saved['sequence'].shape == (11, 134)
        assert saved['sequence'][4, 132] == 0
        assert saved['timestamps_ms'][-1] == 400
    motion = MotionRecognizer(tmp_path)
    assert motion.supported_labels == ()
    assert any('real trajectory' in message for message in motion.diagnostics)
    assert all(motion.update(observation(t)) is None for t in range(0, 1000, 40))


@pytest.mark.parametrize('word', ['IAN', 'HELLO'])
def test_temporal_word_replay_commits_held_letters_once_and_requires_release_for_repeat(word):
    decoder, engine, timestamp = TemporalDecoder(), ConversationEngine('free_spell'), 0
    committed = []
    for label in word:
        tokens, timestamp = hold(decoder, label, timestamp, 800)
        committed.extend(token.label for token in tokens)
        for token in tokens:
            engine.consume(token)
        _, timestamp = hold(decoder, None, timestamp, 320)
    assert committed == list(word)
    assert engine.snapshot()['word'] == word


def test_temporal_flicker_low_confidence_and_unobserved_time_cannot_repeat():
    decoder = TemporalDecoder()
    tokens, stamp = hold(decoder, 'I', 0)
    assert [token.label for token in tokens] == ['I']
    assert not hold(decoder, 'A', stamp, 80)[0]
    stamp += 80
    assert not hold(decoder, 'I', stamp, 600, .1)[0]
    stamp += 600
    assert not hold(decoder, 'I', stamp, 600)[0]
    stamp += 600
    tokens, stamp = hold(decoder, 'N', stamp)
    assert [token.label for token in tokens] == ['N']
    fresh = TemporalDecoder()
    assert fresh.update(Prediction('A', .99, 0), 0) is None
    assert fresh.update(Prediction('A', .99, 1000), 1000) is None
    assert not hold(fresh, 'A', 1040, 200)[0]


def test_temporal_rejects_backward_future_invalid_and_stale_predictions():
    decoder = TemporalDecoder()
    decoder.update(Prediction('A', .9, 100), 100)
    with pytest.raises(ValueError, match='backwards'):
        decoder.update(None, 99)
    with pytest.raises(ValueError, match='future'):
        decoder.update(Prediction('A', .9, 201), 200)
    with pytest.raises(ValueError, match='confidence'):
        decoder.update(Prediction('A', float('nan'), 200), 200)
    with pytest.raises(ValueError, match='canonical'):
        decoder.update(Prediction('not-a-class', .9, 200), 200)
    assert decoder.update(Prediction('A', .9, 100), 500) is None


def test_arbitrary_names_and_signed_hello_remain_distinct_from_spelled_hello():
    result = run_headless_replay('EMMA')
    assert result['committed'] == ['MY', 'NAME', 'E', 'M', 'M', 'A', 'YOUR', 'NAME', 'WHAT']
    assert result['word'] == 'EMMA'
    assert result['interpretation'] == 'My name is Emma. What is your name?'
    assert result['response'] == 'My name is Astra. Nice to meet you, Emma.'
    assert 'synthetic' in result['source']
    engine = ConversationEngine('free_spell')
    engine.consume(Token('HELLO', 'sign', 0, .9, 'synthetic-test'))
    assert engine.snapshot()['word'] == ''
    for i, label in enumerate('HELLO'):
        engine.consume(Token(label, 'letter', i + 1, .9, 'synthetic-test'))
    assert engine.snapshot()['word'] == 'HELLO'


def test_tutor_keeps_correct_target_visible_then_advances_on_clock():
    clock = [0.0]
    engine = ConversationEngine('alphabet_tutor', clock=lambda: clock[0])
    state = engine.consume(Token('A', 'letter', 0, .95, 'synthetic-test'))
    assert state['correct'] is True and state['target'] == 'A'
    clock[0] = 1.5
    state = engine.snapshot()
    assert state['correct'] is None and state['target'] == 'B'


@pytest.fixture(scope='module')
def synthetic_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp('synthetic_ml')
    samples = directory / 'samples'
    samples.mkdir()
    rng = np.random.default_rng(991)
    for session in ('synthetic-s1', 'synthetic-s2'):
        for i, label in enumerate(('A', 'I', 'N')):
            features = rng.normal(0, .03, (16, 63)).astype(np.float32)
            features[:, i * 3:i * 3 + 3] += 1
            np.savez_compressed(samples / f'{session}-{label}.npz', features=features,
                                sequence=np.zeros((16, 134), dtype=np.float32),
                                timestamps_ms=np.arange(16, dtype=np.int64) * 100,
                                label=np.asarray(label), session_id=np.asarray(session), schema_version=np.asarray(1))
    model = directory / 'synthetic_alphabet.npz'
    result = cli('-m', 'training.train', '--data-dir', samples, '--output', model,
                 '--epochs', 50, '--data-origin', 'synthetic_fixture', '--provenance',
                 'Generated numeric unit-test vectors; no real signing or accuracy evidence.')
    assert result.returncode == 0, result.stderr
    assert 'SYNTHETIC FIXTURE' in result.stdout
    return directory, samples, model


def test_cpu_training_session_split_supported_mask_and_input_safety(synthetic_model):
    directory, samples, model_path = synthetic_model
    dataset, model = load_dataset(samples), CPURecognizer(model_path)
    train, validation = session_split(dataset)
    assert set(dataset.sessions[train]).isdisjoint(dataset.sessions[validation])
    assert model.metadata['synthetic_fixture'] is True
    assert model.metadata['live_recognition_validated'] is False
    assert model.supported_labels == ('A', 'I', 'N')
    for vector in dataset.features:
        prediction = model.predict(vector, 100)
        assert prediction.label in model.supported_labels
        assert prediction.source == 'cpu' and prediction.latency_ms >= 0
        assert np.isclose(prediction.scores.sum(), 1)
        for i, label in enumerate(ALPHABET):
            if label not in model.supported_labels:
                assert prediction.scores[i] == 0
    for vector in (np.ones(62), np.full(63, np.nan), np.full(63, 4.1), np.ones(63, dtype=complex)):
        with pytest.raises(ValueError):
            model.predict(vector, 0)
    with pytest.raises(FileNotFoundError, match='Collect'):
        CPURecognizer(directory / 'missing.npz')


def test_training_cli_refuses_missing_data_and_single_session_without_model(tmp_path):
    output = tmp_path / 'must_not_exist.npz'
    result = cli('-m', 'training.train', '--data-dir', tmp_path / 'empty', '--output', output)
    assert result.returncode == 2 and 'No sample NPZ' in result.stderr
    assert not output.exists()
    recorder = DatasetRecorder(tmp_path / 'single', session_id='synthetic-one-session')
    for label in ('A', 'I'):
        recorder.start(label)
        recorder.add(observation(0))
        recorder.finish()
    result = cli('-m', 'training.train', '--data-dir', tmp_path / 'single', '--output', output)
    assert result.returncode == 2 and 'No honest holdout' in result.stderr
    assert not output.exists()


def test_quantized_export_matches_independent_integer_oracle_and_masks_classes(synthetic_model):
    directory, samples, model = synthetic_model
    quantized_path, rom = directory / 'synthetic_quantized.npz', directory / 'rom'
    result = cli('-m', 'training.export_quantized', '--model', model, '--output', quantized_path, '--rom-dir', rom)
    assert result.returncode == 0, result.stderr
    arrays, metadata = load_quantized_model(quantized_path)
    with np.load(rom / 'test_vectors.npz', allow_pickle=False) as vectors:
        assert json.loads(str(vectors['metadata']))['recognition_evidence'] is False
        for inputs, expected_hidden, expected_logits in zip(vectors['inputs'], vectors['hidden'], vectors['logits']):
            hidden = [min(32767, max(0, (sum(int(inputs[i]) * int(arrays['w1'][i, j]) for i in range(63))
                                        + int(arrays['b1'][j])) >> int(arrays['shift1']))) for j in range(32)]
            logits = [(sum(hidden[j] * int(arrays['w2'][j, k]) for j in range(32))
                       + int(arrays['b2'][k])) >> int(arrays['shift2']) for k in range(26)]
            assert hidden == expected_hidden.tolist()
            assert logits == expected_logits.tolist()
            np.testing.assert_array_equal(integer_forward(inputs, arrays), expected_logits)
    cpu, quantized = CPURecognizer(model), QuantizedRecognizer(quantized_path)
    for features in load_dataset(samples).features:
        prediction = quantized.predict(features, 100)
        assert prediction.label == cpu.predict(features, 100).label
        assert prediction.scores[9] == prediction.scores[25] == 0
        assert prediction.source == 'cpu-quantized'


def test_evaluation_refuses_reusing_sessions_as_new_test_and_reports_synthetic_scope(synthetic_model):
    directory, samples, model = synthetic_model
    output = directory / 'evaluation.json'
    result = cli('-m', 'training.evaluate', '--model', model, '--data-dir', samples,
                 '--output', output, '--benchmark-samples', 8)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert 'Synthetic' in report['evidence_scope'] and report['cpu_benchmark']['sample_count'] == 8
    result = cli('-m', 'training.evaluate', '--model', model, '--data-dir', samples,
                 '--output', directory / 'no_new_test.json', '--split', 'new')
    assert result.returncode == 2 and 'No new examples' in result.stderr


def controller_args(tmp_path):
    return SimpleNamespace(dwell_ms=350, data_dir=tmp_path / 'empty-data', mute=True,
                           replay=False, model=tmp_path / 'missing-model.npz', camera=0,
                           landmarker=tmp_path / 'unused.task', port=None,
                           quantized_model=tmp_path / 'unused.npz')


def test_controller_normalization_classifier_decoder_word_buffer_events(tmp_path):
    from src.vision.camera import CapturedFrame
    events=[]
    class FakeSource:
        frame=None
        def snapshot(self): return dict(state='LIVE',generation=1,error='',capture_fps=25.,capture_ms=2.,input_mirrored=False)
        def latest(self,after_id=None): return self.frame
        def close(self): pass
    source=FakeSource()
    controller=ApplicationController(controller_args(tmp_path),events.append,source_controller=source)
    controller.mode='free_spell'
    controller.conversation.set_mode('free_spell')
    current=['I']
    class Classifier:
        supported_labels=('A','I','N')
        def predict(self,features,timestamp_ms):
            np.testing.assert_array_equal(features,normalize_landmarks(hand()))
            return Prediction(current[0],.99,timestamp_ms,'synthetic-test',.125)
    controller.cpu=Classifier()
    stamp=0
    for letter in 'IAN':
        current[0]=letter
        for _ in range(16):
            source.frame=CapturedFrame(stamp,stamp,np.zeros((32,48,3),np.uint8),2.,1)
            controller._results.put((source.frame,observation(stamp),4.,None))
            controller._consume_result()
            controller._event()
            stamp+=40
        for _ in range(8):
            source.frame=CapturedFrame(stamp,stamp,np.zeros((32,48,3),np.uint8),2.,1)
            controller._results.put((source.frame,FrameObservation(stamp),4.,None))
            controller._consume_result()
            stamp+=40
    assert events[-1]['word']=='IAN'
    assert events[-1]['tokens']==['I','A','N']
    assert all(e['engine']=='CPU' and e['source']=='live' for e in events)
    assert events[-1]['model_classes']==['A','I','N']
    assert events[-1]['coverage']==['A','I','N','J','Z']
    controller.action('engine','FPGA')
    controller._commands()
    assert controller.engine=='CPU' and 'not implemented' in controller.status
    controller.close()


def test_replay_is_labeled_and_cannot_record_samples(tmp_path):
    events = []
    controller = ApplicationController(controller_args(tmp_path), events.append)
    controller.action('replay')
    controller.action('record', 'I')
    controller._commands()
    controller._event()
    event = events[-1]
    assert event['source'] == 'replay'
    assert event['cpu_latency_ms'] is event['fpga_latency_ms'] is None
    assert not controller.recorder.active and 'live camera' in event['calibration_status']


def test_controller_evaluate_runs_only_new_sessions_and_surfaces_result(tmp_path, monkeypatch):
    import src.application as application
    controller = ApplicationController(controller_args(tmp_path), lambda event: None)
    captured = {}
    class FakeProcess:
        result = None
        def poll(self):
            return self.result
    process = FakeProcess()
    def popen(command, **kwargs):
        captured['command'] = command
        return process
    monkeypatch.setattr(application, 'ROOT', tmp_path)
    monkeypatch.setattr(application.subprocess, 'Popen', popen)
    controller.action('evaluate')
    controller._commands()
    command = captured['command']
    assert command[command.index('-m')+1] == 'training.evaluate'
    assert command[command.index('--split') + 1] == 'new'
    assert controller.job_kind == 'evaluation'
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/evaluation.json').write_text(json.dumps({'evaluation': {'accuracy': .75, 'samples': 12}}))
    process.result = 0
    controller._poll_training()
    assert controller.training_process is None and controller.training_log is None
    assert '75.0%' in controller.calibration_status and '12 samples' in controller.calibration_status
    assert 'Live word/sign acceptance is still required' in controller.calibration_status


def test_qt_latest_event_updates_visible_buffer_and_replay_status_without_action_feedback():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from src.ui.desktop import SignBridgeUI
    actions = []
    ui = SignBridgeUI(lambda action, value=None: actions.append((action, value)))
    ui.update({'word': 'OLD'})
    ui.update({'source': 'replay', 'word': 'IAN', 'stable': 'N', 'engine': 'CPU',
               'candidate': 'N', 'confidence': .95, 'cpu_latency_ms': .125,
               'response': 'My name is Astra. Nice to meet you, Ian.', 'coverage': ['I', 'A', 'N']})
    ui._drain()
    assert ui.word.text() == 'IAN' and ui.stable.text() == 'Committed: N'
    assert ui.response.text() == 'My name is Astra. Nice to meet you, Ian.'
    assert 'NOT LIVE' in ui.source_badge.text() and ui.cpu_ms.text() == '—'
    assert not ui.record.isEnabled() and not actions and ui._latest is None
    ui.window.close()
    assert actions == [('stop', None)]
