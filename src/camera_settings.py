"""Explicit camera selection; only successfully read configurations are remembered."""
import json
import os
from pathlib import Path
import threading
import uuid

BACKENDS = ('DSHOW', 'MSMF', 'ANY')
_lock = threading.RLock()


def validate_selection(index, backend):
    if type(index) is not int or index < 0 or backend not in BACKENDS:
        raise ValueError('Choose the tested camera index and backend in Settings / Diagnostics.')
    return index, backend


def read_settings(path):
    with _lock:
        try:
            result = json.loads(Path(path).read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {}
        if not isinstance(result, dict):
            raise ValueError('Camera settings must contain a JSON object')
        return result


def update_settings(path, changes):
    path = Path(path)
    with _lock:
        values = read_settings(path)
        values.update(changes)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as handle:
                json.dump(values, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()


def resolve_selection(settings, index=None, backend=None):
    # A partial command-line override must never borrow half of a saved pair.
    if index is not None or backend is not None:
        return validate_selection(index, backend)
    previous = settings.get('last_known_working')
    if previous is None:
        return None, None
    if not isinstance(previous, dict):
        raise ValueError('Invalid saved camera selection; choose the tested index and backend in Settings / Diagnostics.')
    return validate_selection(previous.get('index'), previous.get('backend'))
