"""Fingerspelling preferences kept separate from the working camera settings."""
import json
import math
import os
from pathlib import Path
import uuid

DEFAULTS = dict(repeat_threshold=.65, repeat_cooldown_ms=700,
                auto_space=True, space_pause_ms=1200, space_shortcut='Ctrl+Space')


def validate_settings(values):
    if not isinstance(values, dict) or set(values) - DEFAULTS.keys():
        raise ValueError('Unknown fingerspelling setting.')
    result = {**DEFAULTS, **values}
    for key, low, high in (('repeat_threshold', .25, 2.), ('repeat_cooldown_ms', 300, 3000),
                           ('space_pause_ms', 800, 5000)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be between {low} and {high}.')
    if not isinstance(result['auto_space'], bool):
        raise ValueError('Automatic spacing must be enabled or disabled.')
    shortcut = result['space_shortcut']
    if not isinstance(shortcut, str) or not shortcut.strip() or len(shortcut) > 64:
        raise ValueError('Choose a valid space shortcut.')
    result['space_shortcut'] = shortcut.strip()
    for key in ('repeat_cooldown_ms', 'space_pause_ms'):
        result[key] = int(result[key])
    return result


def read_settings(path):
    path = Path(path)
    return validate_settings(json.loads(path.read_text(encoding='utf-8'))) if path.exists() else dict(DEFAULTS)


def write_settings(path, values):
    settings = validate_settings(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(settings, indent=2), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return settings
