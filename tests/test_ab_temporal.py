"""A/B commit-gate behavior; supplied predictions are synthetic test inputs.

These checks verify timing and repeat suppression, not recognition accuracy.
"""
import pytest

from src.contracts import Prediction
from src.temporal import TemporalDecoder


def predict(decoder, label, timestamp_ms, confidence=.96):
    prediction = None if label is None else Prediction(
        label, confidence, timestamp_ms, 'synthetic-temporal-test')
    return decoder.update(prediction, timestamp_ms)


def hold(decoder, label, start, stop, confidence=.96):
    return [token for stamp in range(start, stop, 50)
            if (token := predict(decoder, label, stamp, confidence)) is not None]


def test_a_then_b_commits_observed_dwell_once_per_held_letter():
    decoder = TemporalDecoder()
    assert hold(decoder, 'A', 0, 350) == []
    assert decoder.snapshot() == {
        'candidate': 'A', 'stable_duration_ms': 300, 'dwell_ms': 350,
        'confidence_threshold': .8, 'locked_label': None,
        'last_timestamp_ms': 300,
    }
    first = predict(decoder, 'A', 350)
    assert first is not None and first.label == 'A'
    assert first.confidence == .96 and first.timestamp_ms == 350
    assert decoder.snapshot()['locked_label'] == 'A'
    assert decoder.snapshot()['candidate'] is None
    assert decoder.snapshot()['stable_duration_ms'] == 0
    assert hold(decoder, 'A', 400, 3000) == []
    second = hold(decoder, 'B', 3000, 6000)
    assert [(token.label, token.timestamp_ms) for token in second] == [('B', 3350)]
    assert decoder.snapshot()['locked_label'] == 'B'


def test_deliberate_release_allows_a_second_a_without_repeating_held_a():
    decoder = TemporalDecoder()
    assert [token.label for token in hold(decoder, 'A', 0, 1000)] == ['A']
    assert hold(decoder, None, 1000, 1200) == []
    assert decoder.snapshot()['locked_label'] == 'A'  # Only 150 ms observed.
    assert predict(decoder, None, 1200) is None
    assert decoder.snapshot()['locked_label'] is None
    assert hold(decoder, 'A', 1250, 1600) == []
    token = predict(decoder, 'A', 1600)
    assert token is not None and token.label == 'A'
    assert hold(decoder, 'A', 1650, 3000) == []


def test_low_confidence_and_short_other_pose_do_not_unlock_a_held_letter():
    decoder = TemporalDecoder()
    assert [token.label for token in hold(decoder, 'A', 0, 600)] == ['A']
    assert hold(decoder, 'B', 600, 700) == []  # 50 ms, below change threshold.
    assert hold(decoder, 'A', 700, 900) == []
    assert hold(decoder, 'B', 900, 1400, confidence=.43) == []
    state = decoder.snapshot()
    assert state['candidate'] is None and state['stable_duration_ms'] == 0
    assert state['locked_label'] == 'A'
    assert hold(decoder, 'A', 1400, 2000) == []


def test_low_confidence_breaks_pending_dwell_and_preserves_real_token_confidence():
    decoder = TemporalDecoder()
    assert hold(decoder, 'B', 0, 300) == []
    assert predict(decoder, 'B', 300, confidence=.43) is None
    assert decoder.snapshot()['stable_duration_ms'] == 0
    assert hold(decoder, 'B', 350, 700, confidence=.89) == []
    token = predict(decoder, 'B', 700, confidence=.98)
    assert token is not None and token.label == 'B'
    assert token.confidence == .89  # Minimum observed confidence across dwell.


def test_snapshot_cannot_accumulate_unobserved_time_and_long_gap_resets_dwell(monkeypatch):
    decoder = TemporalDecoder()
    assert hold(decoder, 'A', 0, 300) == []
    state = decoder.snapshot()
    monkeypatch.setattr('src.temporal.decoder.time.monotonic', lambda: 999999.)
    for _ in range(10):
        assert decoder.snapshot() == state
    assert predict(decoder, 'A', 2000) is None
    assert decoder.snapshot()['stable_duration_ms'] == 0
    assert decoder.snapshot()['candidate'] == 'A'
    assert hold(decoder, 'A', 2050, 2350) == []
    token = predict(decoder, 'A', 2350)
    assert token is not None and token.label == 'A'


def test_no_hand_and_stale_prediction_clear_pending_dwell_without_committing():
    decoder = TemporalDecoder()
    assert hold(decoder, 'A', 0, 300) == []
    assert predict(decoder, None, 300) is None
    assert decoder.snapshot()['candidate'] is None
    assert decoder.snapshot()['stable_duration_ms'] == 0
    assert hold(decoder, 'B', 350, 600) == []
    assert decoder.update(Prediction('B', .99, 100, 'synthetic-temporal-test'), 600) is None
    assert decoder.snapshot()['candidate'] is None
    assert decoder.snapshot()['stable_duration_ms'] == 0
    assert decoder.snapshot()['locked_label'] is None


def test_reset_clears_observable_state_for_next_camera_or_model_stream():
    decoder = TemporalDecoder()
    assert [token.label for token in hold(decoder, 'A', 0, 600)] == ['A']
    decoder.reset()
    assert decoder.snapshot() == {
        'candidate': None, 'stable_duration_ms': 0, 'dwell_ms': 350,
        'confidence_threshold': .8, 'locked_label': None,
        'last_timestamp_ms': None,
    }
    assert [token.label for token in hold(decoder, 'A', 0, 600)] == ['A']


def test_rejected_observations_preserve_lock_and_break_release_and_change_runs():
    decoder = TemporalDecoder()
    assert [token.label for token in hold(decoder, 'A', 0, 600)] == ['A']
    assert hold(decoder, None, 600, 800) == []
    decoder.reject(800)  # Not the final 50 ms of the preceding release run.
    assert predict(decoder, None, 850) is None
    assert decoder.snapshot()['locked_label'] == 'A'
    assert hold(decoder, 'B', 900, 1050) == []
    decoder.reject(1050)  # A failed inference cannot finish the change run.
    for stamp in range(1100, 2000, 50):
        decoder.reject(stamp)
    assert decoder.snapshot()['locked_label'] == 'A'
    assert decoder.snapshot()['candidate'] is None
    assert decoder.snapshot()['last_timestamp_ms'] == 1950
    assert hold(decoder, 'A', 2000, 2600) == []


def test_rejected_observations_clear_dwell_and_validate_timestamp_without_mutating_on_error():
    decoder = TemporalDecoder()
    assert hold(decoder, 'A', 0, 350) == []
    decoder.reject(350)
    assert decoder.snapshot()['candidate'] is None
    assert decoder.snapshot()['stable_duration_ms'] == 0
    state = decoder.snapshot()
    for invalid in (349, -1, True, 1.5):
        with pytest.raises(ValueError):
            decoder.reject(invalid)
        assert decoder.snapshot() == state
    assert hold(decoder, 'A', 400, 750) == []
    token = predict(decoder, 'A', 750)
    assert token is not None and token.label == 'A'
