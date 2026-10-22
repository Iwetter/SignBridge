"""Read persisted numeric archives; no cached or in-memory sample claims."""
from pathlib import Path
import numpy as np
from src.contracts import LABELS, STATIC_LABELS
from src.recognition.readiness import alphabet_readiness, valid_static_rows

NORMALIZATION_ID = 'physical-unmirrored-v2'


def dataset_summary(data_dir):
    result = dict(counts=dict.fromkeys(LABELS, 0), total_samples=0, files=[], errors=[], legacy_samples=0)
    for path in sorted(Path(data_dir).rglob('*.npz')):
        try:
            with np.load(path, allow_pickle=False) as data:
                for key in ('label', 'session_id'):
                    if data[key].shape != () or data[key].dtype.kind != 'U' or not str(data[key].item()).strip():
                        raise ValueError(f'{key} must be a nonempty Unicode scalar')
                schema = data['schema_version']
                if schema.shape != () or schema.dtype.kind not in 'iu' or int(schema) != 1:
                    raise ValueError('Expected dataset schema_version=1')
                label, session = str(data['label'].item()), str(data['session_id'].item())
                x, seq, stamps = data['features'], data['sequence'], data['timestamps_ms']
                if label not in LABELS or x.ndim != 2 or x.shape[1] != 63 or not len(x) or seq.shape != (len(x),134):
                    raise ValueError('Invalid label or feature dimensions')
                if x.dtype != np.dtype('float32') or seq.dtype != np.dtype('float32') or stamps.dtype != np.dtype('int64'):
                    raise ValueError('Expected float32 features/sequence and int64 timestamps')
                if not np.isfinite(x).all() or not np.isfinite(seq).all() or np.any(np.abs(x) > 4.000001):
                    raise ValueError('Invalid or non-finite samples')
                if stamps.shape != (len(x),) or np.any(np.diff(stamps) <= 0):
                    raise ValueError('Timestamps must strictly increase')
                if 'normalization_id' in data and (data['normalization_id'].shape != () or data['normalization_id'].dtype.kind != 'U'):
                    raise ValueError('normalization_id must be a Unicode scalar')
                valid = int(np.count_nonzero(valid_static_rows(x, seq) if label in STATIC_LABELS else
                                            np.any(seq[:,132:] > .5, axis=1)))
                legacy = 'normalization_id' not in data or str(data['normalization_id'].item()) != NORMALIZATION_ID
                result['counts'][label] += valid
                result['total_samples'] += valid
                result['legacy_samples'] += valid if legacy else 0
                result['files'].append(dict(path=str(path.resolve()), label=label, session_id=session,
                                            samples=valid, total_frames=len(x), rejected_frames=len(x)-valid,
                                            legacy=legacy))
        except Exception as exc:
            result['errors'].append(f'{path.name}: {exc}')
    result.update(alphabet_readiness(result))
    return result
