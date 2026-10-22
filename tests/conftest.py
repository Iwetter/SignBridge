"""Keep generated test recordings under a unique project scratch directory."""
from pathlib import Path
import uuid


def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(__file__).resolve().parents[1] / 'work' / ('pytest-' + uuid.uuid4().hex[:12]))
