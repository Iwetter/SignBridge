"""Evaluate recorded session holdout, or new sessions untouched by model selection."""
from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import CLASS_TO_ID
from src.recognition import CPURecognizer
from src.recognition.dataset import load_dataset, metrics, save_report
from src.recognition.geometric import model_features


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/samples')
    parser.add_argument('--model', default='models/alphabet.npz')
    parser.add_argument('--output', default='models/evaluation.json')
    parser.add_argument('--split', choices=['validation', 'new'], default='validation',
                        help='validation repeats saved holdout; new uses only sessions never used for training/selection')
    parser.add_argument('--benchmark-samples', type=int, default=500)
    args = parser.parse_args(argv)
    try:
        if args.benchmark_samples < 1:
            raise ValueError('--benchmark-samples must be positive.')
        model = CPURecognizer(args.model)
        provenance = model.metadata['training_provenance']
        dataset = load_dataset(args.data_dir, provenance.get('normalization_requirement'),
                               labels=list(model.supported_labels))
        training = set(provenance.get('train_sessions', []))
        validation = set(provenance.get('validation_sessions', []))
        if not training or not validation or training & validation:
            raise ValueError('Model has no valid disjoint session split provenance. Retrain it with training/train.py.')
        selected_sessions = (validation if args.split == 'validation' else
                             set(dataset.sessions.tolist()) - training - validation)
        selected = np.isin(dataset.sessions, sorted(selected_sessions))
        if not selected.any():
            raise ValueError(f'No {args.split} examples found. For a new-session test, record a fresh session ID.')
        if set(dataset.labels[selected].tolist()) - {CLASS_TO_ID[x] for x in model.supported_labels}:
            raise ValueError('Selected evaluation sessions include unsupported static classes; train coverage first.')
        # Saved validation identity must be unchanged; otherwise it is not the reported holdout.
        if args.split == 'validation':
            expected = {f['path']: f['sha256'] for f in provenance.get('files', [])
                        if f['session_id'] in validation}
            actual = {f['path']: f['sha256'] for f in dataset.files if f['session_id'] in validation}
            if actual != expected:
                raise ValueError('Validation files changed since training. Use fresh session IDs and --split new, or retrain.')
        x, y = dataset.features[selected], dataset.labels[selected]
        for i in range(min(20, len(x))):
            model.predict(x[i], i)
        timings = []
        for i in range(args.benchmark_samples):
            start = perf_counter()
            model.predict(x[i % len(x)], i)
            timings.append((perf_counter()-start)*1000)
        report = {'model': str(Path(args.model).resolve()), 'model_sha256': sha256(Path(args.model).read_bytes()).hexdigest(),
                  'coverage': list(model.supported_labels), 'split': args.split,
                  'dataset_summary': dataset.summary(selected.nonzero()[0]),
                  'excluded_files': dataset.excluded_files, 'warnings': dataset.warnings,
                  'session_ids': sorted(set(dataset.sessions[selected].tolist())),
                  'training_provenance': provenance,
                  'evidence_scope': ('Synthetic fixtures only; not ASL recognition evidence.' if model.metadata.get('synthetic_fixture') else
                                     'Untouched session test; does not establish live or signer-independent performance.' if args.split == 'new' else
                                     'Saved session validation reused for model selection; not an untouched test.'),
                  'evaluation': metrics(model_features(x,model.feature_set), y, model.arrays, list(model.supported_labels),model.temperature),
                  'cpu_benchmark': {'backend': model.name, 'sample_count': len(timings), 'input_source': 'selected recorded landmark vectors',
                                    'scope': 'predict only; no camera, landmark extraction, UI or FPGA',
                                    'median_ms': float(np.median(timings)), 'p95_ms': float(np.percentile(timings, 95)),
                                    'mean_ms': float(np.mean(timings))}}
        save_report(report, args.output)
        print(f'Saved {args.output}; {args.split} accuracy={report["evaluation"]["accuracy"]:.3f}; '
              f'CPU median={report["cpu_benchmark"]["median_ms"]:.4f} ms over {len(timings)} calls.')
        print(report['evidence_scope'])
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'Evaluation unavailable: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
