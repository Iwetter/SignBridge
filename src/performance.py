"""Bounded measurements; null means not observed, never an invented rate."""
from collections import defaultdict, deque
from pathlib import Path
import json
import time
import numpy as np


class PerformanceLog:
    def __init__(self):
        self.values = defaultdict(lambda: deque(maxlen=3600))
        self.started = time.time()
        self.states = defaultdict(int)

    def add(self, key, value):
        if isinstance(value, (int,float)) and np.isfinite(value) and value >= 0:
            self.values[key].append(float(value))

    def save(self, path):
        summary = {key: {'samples':len(v), 'median':float(np.median(v)), 'p95':float(np.percentile(v,95))}
                   for key,v in self.values.items() if v}
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(started_unix=self.started, ended_unix=time.time(),
            scope='Actual observed application stages only. Replay contributes no camera or model timings.',
            units='FPS for *_fps; milliseconds for *_ms', camera_states=dict(self.states), measurements=summary),indent=2),encoding='utf-8')
        return output
