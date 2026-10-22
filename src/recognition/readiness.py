"""Shared readiness policy for the 24 static fingerspelling classes."""
from __future__ import annotations

import numpy as np
from src.contracts import STATIC_LABELS


READINESS_POLICY = {'minimum_samples': 100, 'minimum_sessions': 2,
                    'minimum_samples_per_session': 50, 'split_unit': 'session_id',
                    'static_labels': list(STATIC_LABELS), 'excluded_motion_labels': ['J', 'Z']}


def valid_static_rows(features: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """Mask finite, normalized, nonzero static vectors with an observed hand."""
    return (np.isfinite(features).all(axis=1) &
            (np.abs(features) <= 4.000001).all(axis=1) &
            np.any(np.abs(features) > 1e-8, axis=1) &
            np.isfinite(sequence).all(axis=1) &
            np.any(sequence[:, 132:] > .5, axis=1))


def alphabet_readiness(summary: dict) -> dict:
    """Aggregate persisted valid file counts, including legacy recordings.

    Multiple recordings with the same session ID count as one session. This is
    collection readiness, not a claim about model quality or live recognition.
    """
    by_label = {label: {} for label in STATIC_LABELS}
    for entry in summary.get('files', []):
        label = entry.get('label')
        session = str(entry.get('session_id', '')).strip()
        count = int(entry.get('samples', 0))
        if label in by_label and session and count > 0:
            sessions = by_label[label]
            sessions[session] = sessions.get(session, 0) + count
    progress = {}
    for label, session_counts in by_label.items():
        samples = sum(session_counts.values())
        qualified = sum(count >= READINESS_POLICY['minimum_samples_per_session']
                        for count in session_counts.values())
        ready = (samples >= READINESS_POLICY['minimum_samples'] and
                 qualified >= READINESS_POLICY['minimum_sessions'])
        reasons = []
        if samples < READINESS_POLICY['minimum_samples']:
            reasons.append(f'{samples}/100 valid samples')
        if qualified < READINESS_POLICY['minimum_sessions']:
            reasons.append(f'{qualified}/2 sessions with at least 50 valid samples each')
        progress[label] = {'samples': samples, 'sessions': len(session_counts),
                           'qualified_sessions': qualified,
                           'session_counts': dict(sorted(session_counts.items())),
                           'ready': ready, 'reason': 'Ready' if ready else '; '.join(reasons)}
    return {'static_progress': progress,
            'ready_labels': [label for label in STATIC_LABELS if progress[label]['ready']],
            'readiness_policy': {**READINESS_POLICY,
                                 'static_labels': list(STATIC_LABELS),
                                 'excluded_motion_labels': ['J', 'Z']}}
