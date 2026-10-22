"""Train reproducible NumPy MLP from session-separated real landmark recordings.

Run from repository root: python -m training.train --data-dir data/samples
Use --ready-only to train collected static letters that meet the shared readiness policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import csv
import io
import json
import os
from pathlib import Path
import sys
import uuid
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import ALPHABET, CLASS_TO_ID, STATIC_LABELS, STATIC_CLASS_TO_ID
from src.recognition.dataset import load_dataset, session_split, metrics
from src.recognition.cpu import load_float_model
from src.recognition.geometric import RAW, FEATURE_SETS, feature_names, model_features
from src.recognition.readiness import alphabet_readiness, valid_static_rows
from src.vision.dataset_summary import dataset_summary


def fit_mlp(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
            supported: list[str], *, seed: int = 42, epochs: int = 500, batch_size: int = 128,
            learning_rate: float = .002, l2: float = .0001, patience: int = 60) -> tuple[dict, dict]:
    if epochs < 1 or batch_size < 1 or patience < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('epochs, batch size, patience and learning rate must be positive.')
    if not np.isfinite(l2) or l2 < 0:
        raise ValueError('l2 must be finite and nonnegative.')
    rng = np.random.default_rng(seed)
    ids = np.asarray([CLASS_TO_ID[label] for label in supported])
    if len(ids) < 2 or set(y_train.tolist()) != set(ids.tolist()) or set(y_val.tolist()) != set(ids.tolist()):
        raise ValueError('Training and validation must both contain every supported class (at least two).')
    local = {int(label): i for i, label in enumerate(ids)}
    y = np.asarray([local[int(label)] for label in y_train])
    weights = {
        'W1': (rng.standard_normal((x_train.shape[1], 32)) * np.sqrt(2/x_train.shape[1])).astype(np.float32),
        'b1': np.zeros(32, dtype=np.float32),
        'W2': (rng.standard_normal((32, len(ids))) * np.sqrt(2/32)).astype(np.float32),
        'b2': np.zeros(len(ids), dtype=np.float32),
    }
    moments = {k: np.zeros_like(v) for k, v in weights.items()}
    variances = {k: np.zeros_like(v) for k, v in weights.items()}
    counts = np.bincount(y, minlength=len(ids))
    class_weights = (len(y) / (len(ids)*counts)).astype(np.float32)
    best, best_loss, best_epoch, stale, step = None, float('inf'), 0, 0, 0
    history = []
    def expanded():
        result = {'W1': weights['W1'].copy(), 'b1': weights['b1'].copy(),
                  'W2': np.zeros((32,26), dtype=np.float32), 'b2': np.zeros(26, dtype=np.float32)}
        result['W2'][:, ids] = weights['W2']
        result['b2'][ids] = weights['b2']
        return result
    for epoch in range(1, epochs+1):
        if epoch == 1 or epoch % 25 == 0:
            print(f'Epoch {epoch}/{epochs}', flush=True)
        permutation = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            batch = permutation[start:start+batch_size]
            xb, yb = x_train[batch], y[batch]
            hpre = xb @ weights['W1'] + weights['b1']
            h = np.maximum(hpre, 0)
            logits = h @ weights['W2'] + weights['b2']
            logits -= logits.max(axis=1, keepdims=True)
            delta = np.exp(logits)
            delta /= delta.sum(axis=1, keepdims=True)
            delta[np.arange(len(batch)), yb] -= 1
            sample_weights = class_weights[yb]
            delta *= (sample_weights / sample_weights.sum())[:, None]
            dh = (delta @ weights['W2'].T) * (hpre > 0)
            gradient = {'W2': h.T @ delta + l2*weights['W2'], 'b2': delta.sum(axis=0),
                        'W1': xb.T @ dh + l2*weights['W1'], 'b1': dh.sum(axis=0)}
            step += 1
            for key in weights:
                np.clip(gradient[key], -5, 5, out=gradient[key])
                moments[key] *= .9
                moments[key] += .1 * gradient[key]
                variances[key] *= .999
                variances[key] += .001 * gradient[key]**2
                weights[key] -= learning_rate * (moments[key]/(1-.9**step)) / (np.sqrt(variances[key]/(1-.999**step))+1e-8)
        arrays = expanded()
        measured = metrics(x_val, y_val, arrays, supported)
        loss = measured['cross_entropy']
        if not np.isfinite(loss) or any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Training became non-finite; check inputs or reduce --learning-rate.')
        history.append({'epoch': epoch, 'validation_loss': loss, 'validation_accuracy': measured['accuracy']})
        if loss < best_loss - 1e-6:
            best, best_loss, best_epoch, stale = arrays, loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best, {'selected_epoch': best_epoch, 'epochs_run': epoch, 'history': history,
                  'selection_rule': 'lowest validation cross-entropy; validation is used for early stopping, not an untouched test'}


def save_training_artifacts(output: Path, arrays: dict, metadata: dict, report: dict,
                            report_path: Path, mapping_path: Path) -> None:
    """Stage and read-validate the model; publish it atomically after its reports.

    A failed write leaves the previous model intact. If publishing an artifact
    fails, already published sidecars are restored before surfacing the error.
    """
    history_path = (output.parent / 'history' /
                    (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8])
                    if output.exists() else None)
    report['previous_artifacts_history'] = str(history_path) if history_path else None
    metadata['previous_artifacts_history'] = report['previous_artifacts_history']
    model_bytes = io.BytesIO()
    np.savez_compressed(model_bytes, **arrays,
                        metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
    mapping = {'schema_version': 1, 'supported_labels': metadata['supported_labels'],
               'class_mapping': metadata['class_mapping'], 'model_id': metadata['model_id'],
               'static_class_mapping': dict(STATIC_CLASS_TO_ID),
               'model': str(output)}
    targets = [report_path, mapping_path, output]
    if len(set(targets)) != len(targets):
        raise ValueError('Model, report and class-mapping paths must be different.')
    payloads = {report_path: json.dumps(report, indent=2, allow_nan=False).encode('utf-8'),
                mapping_path: json.dumps(mapping, indent=2, allow_nan=False).encode('utf-8')}
    for split in ('train', 'validation'):
        metric = report[split]
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['actual / predicted'] + metric['labels'])
        for label, row in zip(metric['labels'], metric['confusion_matrix']):
            writer.writerow([label] + row)
        path = report_path.with_name(report_path.stem + f'_{split}_confusion.csv')
        if path in targets:
            raise ValueError('Confusion-matrix and model/report paths must be different.')
        payloads[path] = stream.getvalue().encode('utf-8')
    payloads[output] = model_bytes.getvalue()  # Commit model last, never a partial NPZ.
    staged, backups, committed = {}, {}, []
    history_files = []
    history_parent_existed = history_path.parent.exists() if history_path else False

    def write_temporary(target, content):
        target.parent.mkdir(parents=True, exist_ok=True)
        path = target.with_name('.' + target.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with path.open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    try:
        for target, content in payloads.items():
            staged[target] = write_temporary(target, content)
            backups[target] = write_temporary(target, target.read_bytes()) if target.exists() else None
        verified_arrays, verified_metadata = load_float_model(staged[output])
        if verified_metadata != metadata or any(not np.array_equal(verified_arrays[key], value)
                                               for key, value in arrays.items()):
            raise ValueError('Saved model readback differs from trained parameters.')
        if history_path:
            # Every overwritten artifact is safely copied before publishing any
            # replacement. Keep model bytes and report/mapping/CSV evidence together.
            history_path.mkdir(parents=True, exist_ok=False)
            manifest = []
            for target, backup in backups.items():
                if backup is None:
                    continue
                preserved = history_path / target.name
                if preserved in history_files:
                    preserved = history_path / (str(len(history_files)) + '-' + target.name)
                content = backup.read_bytes()
                history_files.append(preserved)
                with preserved.open('xb') as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                if preserved.read_bytes() != content:
                    raise ValueError(f'History readback failed: {preserved}')
                manifest.append({'original_path': str(target), 'preserved_name': preserved.name,
                                 'sha256': sha256(content).hexdigest()})
            history_manifest = history_path / 'manifest.json'
            history_files.append(history_manifest)
            with history_manifest.open('x', encoding='utf-8') as handle:
                json.dump({'created_utc': datetime.now(timezone.utc).isoformat(), 'files': manifest}, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
        for target, temporary in staged.items():
            os.replace(temporary, target)
            committed.append(target)
    except BaseException:
        for target in reversed(committed):
            backup = backups[target]
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                os.replace(backup, target)
        # A failed transaction keeps the original artifacts. Remove only this
        # attempt's explicitly created history files, not earlier saved versions.
        for path in history_files:
            path.unlink(missing_ok=True)
        if history_path and history_path.exists():
            history_path.rmdir()
        if history_path and not history_parent_existed and history_path.parent.exists():
            history_path.parent.rmdir()
        raise
    finally:
        for temporary in [*staged.values(), *backups.values()]:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/samples')
    parser.add_argument('--require-normalization', default=None)
    parser.add_argument('--feature-set',choices=FEATURE_SETS,default=RAW)
    parser.add_argument('--calibrate-temperature',action='store_true',help='Fit one global softmax temperature on validation NLL; no per-class overrides.')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--labels', nargs='+', choices=STATIC_LABELS,
                           help='Train only these static classes; preserve and report other recorded files.')
    selection.add_argument('--ready-only', action='store_true',
                           help='Train all static classes with 100 valid samples and two sessions of at least 50 each.')
    parser.add_argument('--output', default='models/alphabet.npz')
    parser.add_argument('--metrics', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.002)
    parser.add_argument('--l2', type=float, default=.0001)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--validation-fraction', type=float, default=.25)
    parser.add_argument('--validation-sessions', nargs='+')
    parser.add_argument('--data-origin', choices=['local_capture', 'external', 'synthetic_fixture'], default='local_capture')
    parser.add_argument('--provenance', default='Local user-labeled camera sessions; labels and signing quality have not been independently audited.')
    args = parser.parse_args(argv)
    try:
        readiness = None
        selected_labels = args.labels
        if args.ready_only:
            persisted = dataset_summary(args.data_dir)
            if persisted['errors']:
                raise ValueError('Dataset validation failed: ' + '; '.join(persisted['errors']))
            readiness = alphabet_readiness(persisted)
            selected_labels = readiness['ready_labels']
            if len(selected_labels) < 2:
                raise ValueError('At least two READY static classes are required. Each needs 100 valid samples '
                                 'and at least two sessions with 50 valid samples each. '
                                 'Currently ready: ' + (', '.join(selected_labels) or 'none') + '.')
        dataset = load_dataset(args.data_dir, args.require_normalization, selected_labels)
        if args.ready_only:
            # The shared summary counts only observed, nonzero valid vectors.
            # Use exactly the same mask for training while retaining every file.
            masks = []
            for entry in dataset.files:
                path = Path(args.data_dir) / entry['path']
                if sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError(f'Dataset changed while preparing training: {path}. Retry after recording stops.')
                with np.load(path, allow_pickle=False) as archive:
                    mask = valid_static_rows(archive['features'], archive['sequence'])
                masks.append(mask)
                entry['total_frames'] = entry['samples']
                entry['samples'] = int(mask.sum())
                entry['rejected_frames'] = int((~mask).sum())
            selected = np.concatenate(masks)
            dataset.features = dataset.features[selected]
            dataset.labels = dataset.labels[selected]
            dataset.sessions = dataset.sessions[selected]
            eligible = alphabet_readiness({'files': dataset.files})
            no_longer_ready = set(selected_labels) - set(eligible['ready_labels'])
            if no_longer_ready:
                raise ValueError('Eligible data no longer meets readiness for: ' + ', '.join(sorted(no_longer_ready)) +
                                 '. Check input-convention filters and record more valid samples.')
            print('Ready static classes: ' + ', '.join(selected_labels), flush=True)
        print('Persisted dataset: ' + json.dumps(dataset.summary(), sort_keys=True), flush=True)
        for warning in dataset.warnings:
            print('Warning: ' + warning, flush=True)
        if dataset.excluded_files:
            print(f'Preserved/excluded {len(dataset.excluded_files)} files outside the requested data scope.', flush=True)
        train, val = session_split(dataset, args.seed, args.validation_fraction, args.validation_sessions)
        supported = dataset.supported_labels
        print(f'Training samples: {len(train)}; validation samples: {len(val)}; split by session_id.', flush=True)
        prepared=model_features(dataset.features,args.feature_set)
        arrays, fit = fit_mlp(prepared[train], dataset.labels[train], prepared[val], dataset.labels[val],
                             supported, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
                             learning_rate=args.learning_rate, l2=args.l2, patience=args.patience)
        output = Path(args.output).resolve()
        report_path = Path(args.metrics).resolve() if args.metrics else output.with_name(output.stem + '_metrics.json')
        mapping_path = output.with_name(output.stem + '_classes.json')
        artifact_paths = {'model': str(output), 'metrics': str(report_path), 'class_mapping': str(mapping_path),
                          'validation_confusion': str(report_path.with_name(report_path.stem + '_validation_confusion.csv'))}
        training_results = {'dataset': dataset.summary(), 'train': dataset.summary(train), 'validation': dataset.summary(val),
                            'validation_accuracy': None, 'warnings': dataset.warnings}
        temperature=1.;calibration=None
        if args.calibrate_temperature:
            trials=[{'temperature':float(t),'validation_nll':metrics(prepared[val],dataset.labels[val],arrays,supported,float(t))['cross_entropy']}
                    for t in sorted(set([1.,*np.geomspace(.5,2.,41).tolist()]))]
            best_temperature=min(trials,key=lambda item:item['validation_nll'])
            temperature=best_temperature['temperature']
            calibration={'method':'global-temperature-validation-nll-v1','selection_scope':'Same validation sessions as early stopping; not an untouched final test.',
                         'trials':trials,'temperature':temperature}
        train_metrics = metrics(prepared[train], dataset.labels[train], arrays, supported,temperature)
        val_metrics = metrics(prepared[val], dataset.labels[val], arrays, supported,temperature)
        training_results['validation_accuracy'] = val_metrics['accuracy']
        training_results['confusion_matrix'] = val_metrics['confusion_matrix']
        provenance = {'data_origin': args.data_origin, 'description': args.provenance,
                      'dataset_root': str(Path(args.data_dir).resolve()), 'files': dataset.files,
                      'requested_labels': selected_labels, 'selection_mode': 'ready_only' if args.ready_only else 'explicit' if args.labels else 'all',
                      'normalization_requirement': args.require_normalization,
                      'warnings': dataset.warnings,
                      'train_sessions': sorted(set(dataset.sessions[train].tolist())),
                      'validation_sessions': sorted(set(dataset.sessions[val].tolist())),
                      'split_unit': 'session_id', 'seed': args.seed, 'numpy_version': np.__version__,
                      'training_config': {k: getattr(args, k) for k in ('epochs','batch_size','learning_rate','l2','patience')},
                      'selected_epoch': fit['selected_epoch'], 'selection_rule': fit['selection_rule'],
                      'created_utc': datetime.now(timezone.utc).isoformat()}
        metadata = {'schema_version': 1, 'model_id': uuid.uuid4().hex,
                    'architecture': [prepared.shape[1], 32, 26], 'activation': 'relu',
                    'feature_set':args.feature_set,'feature_names':feature_names(args.feature_set),
                    'temperature':temperature,'calibration':calibration,
                    'normalization': 'wrist / max MCP distance; Left x reflect; clip [-4,4]; xyz interleaved',
                    'supported_labels': supported, 'class_mapping': {x: CLASS_TO_ID[x] for x in ALPHABET},
                    'static_class_mapping': dict(STATIC_CLASS_TO_ID),
                    'training_provenance': provenance, 'partial_coverage': len(supported) != len(STATIC_LABELS),
                    'synthetic_fixture': args.data_origin == 'synthetic_fixture',
                    'dataset_summary': dataset.summary(), 'training_results': training_results,
                    'artifact_paths': artifact_paths,
                    'live_recognition_validated': False}
        report = {'model': str(output), 'model_id': metadata['model_id'], 'coverage': supported,
                  'feature_set':args.feature_set,
                  'temperature':temperature,'calibration':calibration,
                  'classes': supported, 'class_count': len(supported),
                  'samples_per_class': dataset.summary()['samples_per_class'],
                  'dataset_summary': dataset.summary(), 'training_results': training_results,
                  'artifact_paths': artifact_paths, 'class_mapping': metadata['class_mapping'],
                  'static_class_mapping': dict(STATIC_CLASS_TO_ID),
                  'missing_static_letters': [x for x in STATIC_LABELS if x not in supported],
                  'excluded_files': dataset.excluded_files,
                  'excluded_temporal_files': [item for item in dataset.excluded_files if item['reason'] == 'temporal class'],
                  'warnings': dataset.warnings, 'training_provenance': provenance,
                  'evidence_scope': 'Synthetic numerical pipeline test only; no recognition evidence.' if args.data_origin == 'synthetic_fixture' else
                                    'Session-held-out local landmark classification; validation used for model selection. Live signing and signer independence are unverified.',
                  'train': train_metrics, 'validation': val_metrics, 'fit': fit}
        if readiness is not None:
            skipped = {label: progress['reason'] for label, progress in readiness['static_progress'].items()
                       if not progress['ready']}
            report.update(readiness)
            report['skipped_static_classes'] = skipped
            metadata['readiness'] = readiness
            metadata['skipped_static_classes'] = skipped
        save_training_artifacts(output, arrays, metadata, report, report_path, mapping_path)
        print('Model trained successfully', flush=True)
        print('Classes: ' + ', '.join(supported), flush=True)
        print(f'Validation accuracy: {val_metrics["accuracy"]:.1%}', flush=True)
        print('Validation confusion matrix (rows actual / columns predicted): ' + json.dumps(val_metrics['confusion_matrix']), flush=True)
        print(f'Saved {output}; static coverage: {", ".join(supported)} ({len(supported)}/24). J/Z use motion clips.')
        print(f'Session-held-out validation accuracy: {report["validation"]["accuracy"]:.3f}; see {report_path} for scope/provenance.')
        if args.data_origin == 'synthetic_fixture':
            print('SYNTHETIC FIXTURE: these results do not validate ASL recognition.')
        return 0
    except (ValueError, OSError) as exc:
        print(f'Training unavailable: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
