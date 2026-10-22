"""Recheck the targeted experiment against unchanged data before atomic promotion."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.targeted_recognition_audit import (
    POLICY, class_rows, probabilities, validation_checks,
)
from src.contracts import STATIC_LABELS
from src.recognition import CPURecognizer
from src.recognition.dataset import load_dataset, session_split
from training.train import save_training_artifacts


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def promote(root=ROOT):
    root = Path(root).resolve()
    work = root / 'work/targeted-recognition'
    evidence = json.loads((work / 'comparison.json').read_text())
    if evidence['policy'] != POLICY:
        raise ValueError('Validation policy changed; rerun the comparison.')
    selected = evidence['selected']
    if not selected or not evidence['candidates'][selected]['passed']:
        raise ValueError('No passing candidate; production remains unchanged.')
    result = evidence['candidates'][selected]
    candidate_path = Path(result['path'])
    output = root / 'models/alphabet.npz'
    fallback = work / 'fallback/alphabet.npz'
    if digest(output) != evidence['production_sha256'] or digest(fallback) != digest(output):
        raise ValueError('Production or fallback changed since comparison; rerun the audit.')
    if digest(candidate_path) != result['sha256']:
        raise ValueError('Candidate changed since comparison; rerun the audit.')
    dataset = load_dataset(root / 'data/samples', labels=list(STATIC_LABELS))
    if dataset.files != evidence['dataset_files']:
        raise ValueError('Dataset changed since comparison; rerun the audit.')
    old, candidate = CPURecognizer(output), CPURecognizer(candidate_path)
    train, val = session_split(dataset, validation_sessions=evidence['split']['validation_sessions'])
    provenance = candidate.metadata['training_provenance']
    for split, indices in (('train', train), ('validation', val)):
        actual = sorted(set(dataset.sessions[indices].tolist()))
        if actual != evidence['split'][split + '_sessions'] or actual != provenance[split + '_sessions']:
            raise ValueError('Candidate/session provenance mismatch.')
    if set(dataset.sessions[train]) & set(dataset.sessions[val]):
        raise ValueError('Training/validation sessions overlap.')
    scores = probabilities(candidate, dataset.features[val])
    checks = validation_checks(
        class_rows(dataset, val, probabilities(old, dataset.features[val])),
        class_rows(dataset, val, scores), old.supported_labels,
    )
    checks['coverage'] = candidate.supported_labels == STATIC_LABELS
    checks['reload_parity'] = all(
        np.allclose(candidate.predict(x, i).scores, scores[i], atol=2e-5)
        for i, x in enumerate(dataset.features[val])
    )
    if not all(checks.values()):
        raise ValueError('Promotion rejected: ' + ', '.join(k for k, passed in checks.items() if not passed))

    metadata = deepcopy(candidate.metadata)
    report = json.loads(candidate_path.with_name(candidate_path.stem + '_metrics.json').read_text())
    report_path = output.with_name('alphabet_metrics.json')
    mapping_path = output.with_name('alphabet_classes.json')
    artifacts = {
        'model': str(output), 'metrics': str(report_path), 'class_mapping': str(mapping_path),
        'validation_confusion': str(report_path.with_name('alphabet_metrics_validation_confusion.csv')),
    }
    receipt = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'selected': selected,
        'old_sha256': digest(output), 'candidate_sha256': digest(candidate_path),
        'comparison': str(work / 'comparison.json'), 'policy': POLICY, 'checks': checks,
        'live_confirmed': False,
        'evidence_scope': 'Recording-session holdout used for stopping, calibration and selection; not an untouched test or webcam acceptance.',
    }
    metadata.update(artifact_paths=artifacts, deployment_validation=deepcopy(receipt), live_recognition_validated=False)
    report.update(model=str(output), artifact_paths=artifacts, deployment_validation=deepcopy(receipt))
    save_training_artifacts(output, candidate.arrays, metadata, report, report_path, mapping_path)
    installed = CPURecognizer(output)
    if not np.array_equal(probabilities(installed, dataset.features[val]), scores):
        raise RuntimeError('Installed score verification failed; restore the preserved fallback before use.')
    history = Path(metadata['previous_artifacts_history'])
    if digest(history / output.name) != receipt['old_sha256']:
        raise RuntimeError('Previous production history verification failed.')
    receipt.update(production_sha256=digest(output), previous_artifacts_history=str(history),
                   production_replaced=True, model_id=installed.metadata['model_id'],
                   classes=list(installed.supported_labels), temperature=installed.temperature,
                   installed_score_parity=True)
    (work / 'promotion.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    promote()
