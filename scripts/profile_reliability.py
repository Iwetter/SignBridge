"""Matched, deterministic CPU/UI benchmark. No webcam FPS is inferred from this."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))

import numpy as np


def stats(values):
    return {'count': len(values), 'median_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'mean_ms': float(np.mean(values))}


def load_ui(path):
    spec = importlib.util.spec_from_file_location('profile_ui', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['before', 'after'], required=True)
    parser.add_argument('--frames', type=int, default=300)
    args = parser.parse_args()
    path = ROOT / ('work/reliability-before/src/ui/desktop.py' if args.version == 'before' else 'src/ui/desktop.py')
    ui = load_ui(path).SignBridgeUI(lambda *a: None)
    ui.window.resize(1400, 900)
    ui.window.show()
    ui._app.processEvents()
    rng = np.random.default_rng(72)
    frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
    base = dict(kind='frame', source='live', camera_state='LIVE', candidate='A', stable='A',
                confidence=.92, word='A', recognized='A', interpretation='', response='',
                engine='CPU', coverage=['A'], status='', sample_count=0, recording=False,
                mode='free_spell', calibration_status='', fps=30., capture_fps=30.,
                cpu_latency_ms=.1, model_ms=.1, physical_handedness='RIGHT', raw_handedness='LEFT')
    elapsed = []
    for i in range(args.frames + 25):
        event = {**base, 'image': frames[i%4], 'frame_id': i, 'timestamp_ms': i*33}
        start = time.perf_counter()
        ui.update(event)
        ui._drain()
        ui._app.processEvents()
        # Force one complete camera paint consistently in both versions.
        ui.camera.repaint()
        ui._app.processEvents()
        if i >= 25:
            elapsed.append((time.perf_counter()-start)*1000)
    ui.window.close()
    report = {'version': args.version, 'scope': 'Synthetic 640x480 BGR events, offscreen Qt, 1400x900 window; UI update + paint only. Not camera throughput.',
              'ui_update_and_paint': stats(elapsed), 'camera_capture_fps': None, 'physical_webcam': False,
              'frames': args.frames, 'warmup_frames': 25, 'python': sys.version, 'numpy': np.__version__}
    output = ROOT / f'work/performance-{args.version}-ui.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
