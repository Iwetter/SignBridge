"""UI event regressions for static-letter collection; no camera or production data."""
import os
from pathlib import Path

import pytest

from src.contracts import ALPHABET, STATIC_LABELS


@pytest.fixture
def ui():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from src.ui.desktop import SignBridgeUI
    view = SignBridgeUI(lambda *args: None)
    yield view
    view.window.close()


def summary(*entries):
    """entries are (label, session, valid samples, persisted path)."""
    counts = {label: 0 for label in ALPHABET}
    files = []
    for label, session, count, path in entries:
        counts[label] += count
        files.append(dict(label=label, session_id=session, samples=count,
                          total_frames=count, path=str(Path(path).resolve()), legacy=False))
    return dict(counts=counts, files=files, total_samples=sum(counts.values()),
                legacy_samples=0, errors=[])


def event(view, dataset=None, **changes):
    value = dict(source='live', camera_state='LIVE', recording=False, saving=False,
                 save_failed=False, recording_stats={}, dataset=dataset or summary())
    value.update(changes)
    view.update(value)
    view._drain()


def receipt(path, *, label='A', samples=50, session='new-session'):
    return dict(label=label, saved_path=str(Path(path).resolve()), saved_label=label,
                saved_samples=samples, saved_session_id=session, save_error='')


def start_and_save(view, dataset, label='A'):
    view.labels.setCurrentText(label)
    event(view, dataset, recording=True, recording_stats={'label': label, 'active': True})
    event(view, dataset, saving=True, recording_stats={'label': label, 'active': True})


def test_static_label_navigation_wraps_and_excludes_motion_letters(ui):
    event(ui)
    assert [ui.labels.itemText(i) for i in range(ui.labels.count())] == list(STATIC_LABELS)
    assert ui.labels.count() == 24 and 'J' not in STATIC_LABELS and 'Z' not in STATIC_LABELS
    assert not ui.auto_advance.isChecked()
    ui.labels.setCurrentText('I')
    ui.next_label.click()
    assert ui.labels.currentText() == 'K'
    ui.previous_label.click()
    assert ui.labels.currentText() == 'I'
    ui.labels.setCurrentText('Y')
    ui.next_label.click()
    assert ui.labels.currentText() == 'A'
    ui.previous_label.click()
    assert ui.labels.currentText() == 'Y'


def test_all_static_progress_counts_and_current_letter_update(ui, tmp_path):
    # Multiple files in one session count as one session, not several.
    data = summary(('A', 's1', 25, tmp_path/'a1.npz'),
                   ('A', 's1', 25, tmp_path/'a2.npz'),
                   ('A', 's2', 50, tmp_path/'a3.npz'),
                   ('B', 's1', 99, tmp_path/'b1.npz'),
                   ('B', 's2', 1, tmp_path/'b2.npz'))
    event(ui, data)
    table = ui.alphabet_status
    assert table.rowCount() == 24 and table.columnCount() == 4
    assert [table.item(i, 0).text() for i in range(24)] == list(STATIC_LABELS)
    assert table.item(0, 1).text() == '100'
    assert table.item(0, 2).text() == '2'
    assert table.item(0, 3).text().upper() == 'READY'
    assert table.item(1, 3).text().upper() != 'READY'  # Second session too short.
    assert table.item(2, 1).text() == '0'
    ui.labels.setCurrentText('B')
    assert 'B' in ui.selected_label_status.text() and '100' in ui.selected_label_status.text()
    ui.alphabet_progress_button.click()
    assert ui.alphabet_progress.isVisible()
    ui.alphabet_progress.close()


@pytest.mark.parametrize('busy', ['recording', 'saving', 'save_failed'])
def test_label_navigation_is_locked_while_samples_are_pending(ui, busy):
    ui.labels.setCurrentText('I')
    event(ui, **{busy: True}, recording_stats={'label': 'I', 'active': True})
    assert not ui.labels.isEnabled()
    assert not ui.previous_label.isEnabled() and not ui.next_label.isEnabled()
    ui.next_label.click()
    assert ui.labels.currentText() == 'I'
    event(ui)
    assert ui.labels.isEnabled() and ui.next_label.isEnabled()


def test_auto_advance_waits_for_persisted_receipt_and_fresh_summary_once(ui, tmp_path):
    old = summary(('A', 'old-session', 50, tmp_path/'old.npz'),
                  ('B', 'b1', 50, tmp_path/'b1.npz'), ('B', 'b2', 50, tmp_path/'b2.npz'))
    path = tmp_path/'new.npz'
    saved = receipt(path)
    fresh = summary(*[(item['label'], item['session_id'], item['samples'], item['path']) for item in old['files']],
                    ('A', 'new-session', 50, path))
    event(ui, old)
    ui.auto_advance.setChecked(True)
    start_and_save(ui, old)
    event(ui, old, recording_stats=saved)
    assert ui.labels.currentText() == 'A'  # Disk receipt alone is not a fresh count.
    event(ui, fresh, recording_stats=saved)
    assert ui.labels.currentText() == 'C'  # B is already ready.
    for _ in range(3):
        event(ui, fresh, recording_stats=saved)
    assert ui.labels.currentText() == 'C'
    assert 'Saved successfully' in ui.receipt.text() and '50' in ui.receipt.text()


def test_auto_advance_off_startup_receipt_replay_and_failed_save_do_not_move(ui, tmp_path):
    path = tmp_path/'old.npz'
    saved = receipt(path)
    data = summary(('A', 'new-session', 50, path))
    # Opt-in is required, even for a new successful save.
    event(ui)
    start_and_save(ui, summary())
    event(ui, data, recording_stats=saved)
    assert ui.labels.currentText() == 'A'
    ui.auto_advance.setChecked(True)
    event(ui, data, recording_stats=saved)  # Existing receipt does not become new.
    assert ui.labels.currentText() == 'A'
    # Failed save keeps the old receipt and buffered label.
    start_and_save(ui, data)
    failed = dict(saved, save_error='Disk is full', active=True)
    event(ui, data, recording_stats=failed, save_failed=True)
    assert ui.labels.currentText() == 'A'
    # Replay and clearing/no-hand with no newly saved path never advance.
    event(ui, data, source='replay', camera_state='REPLAY', recording_stats=saved)
    assert ui.labels.currentText() == 'A'
    event(ui, data, recording_stats=saved, hand_detected=False)
    assert ui.labels.currentText() == 'A'


def test_initial_existing_receipt_never_advances_when_option_enabled(ui, tmp_path):
    ui.auto_advance.setChecked(True)
    path = tmp_path/'previous-run.npz'
    data = summary(('A', 'new-session', 50, path))
    event(ui, data, recording_stats=receipt(path))
    assert ui.labels.currentText() == 'A'


def test_pending_advance_never_overrides_a_new_manual_label_or_recording(ui,tmp_path):
    path=tmp_path/'a-saved.npz'
    saved=receipt(path)
    fresh=summary(('A','new-session',50,path))
    ui.auto_advance.setChecked(True)
    start_and_save(ui,summary())
    event(ui,summary(),recording_stats=saved)
    ui.labels.setCurrentText('X')
    event(ui,fresh,recording_stats=saved)
    assert ui.labels.currentText()=='X'
    # A new recording before the old count refresh supersedes the old advance.
    start_and_save(ui,summary())
    newer=receipt(tmp_path/'a-newer.npz')
    event(ui,summary(),recording_stats=newer)
    ui.labels.setCurrentText('D')
    event(ui,summary(),recording=True,recording_stats=dict(newer,label='D'))
    newest=summary(('A','new-session',50,tmp_path/'a-newer.npz'))
    event(ui,newest,recording_stats=newer)  # Empty D capture; no new receipt.
    assert ui.labels.currentText()=='D'


def test_pending_advance_requires_matching_persisted_label_count_session(ui, tmp_path):
    event(ui)
    ui.auto_advance.setChecked(True)
    start_and_save(ui, summary(), label='I')
    path = tmp_path/'i.npz'
    saved = receipt(path, label='I')
    for label, session, count in [('I', 'new-session', 49), ('K', 'new-session', 50),
                                  ('I', 'wrong-session', 50)]:
        event(ui, summary((label, session, count, path)), recording_stats=saved)
        assert ui.labels.currentText() == 'I'
    event(ui, summary(('I', 'new-session', 50, path)), recording_stats=saved)
    assert ui.labels.currentText() == 'K'


def test_empty_capture_and_replay_cancel_pending_auto_advance(ui, tmp_path):
    event(ui)
    ui.auto_advance.setChecked(True)
    start_and_save(ui, summary())
    event(ui, hand_detected=False, recording_stats={'samples': 0, 'saved_path': None})
    assert ui.labels.currentText() == 'A'
    start_and_save(ui, summary())
    path = tmp_path/'new.npz'
    saved = receipt(path)
    event(ui, recording_stats=saved)  # Receipt exists; refreshed summary has not arrived.
    event(ui, source='replay', camera_state='REPLAY', recording_stats=saved)
    event(ui, summary(('A', 'new-session', 50, path)), recording_stats=saved)
    assert ui.labels.currentText() == 'A'
