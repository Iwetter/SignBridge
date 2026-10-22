"""Reversible removal of a labeled class from the active training dataset."""
from pathlib import Path
import json
import uuid
import numpy as np
from src.contracts import LABELS


def archive_class(data_dir, label):
    if label not in LABELS:
        raise ValueError('Choose a valid class to reset.')
    root=Path(data_dir).resolve()
    paths=[]
    for path in sorted(root.rglob('*.npz')):
        resolved=path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f'Refusing dataset path outside {root}: {path}')
        with np.load(resolved,allow_pickle=False) as data:
            if str(data['label'].item())==label:paths.append(resolved)
    if not paths:
        return dict(label=label,files=0,archive=None)
    archive=root.parent/'removed_samples'/f'{label}-{uuid.uuid4().hex[:12]}'
    archive.mkdir(parents=True,exist_ok=False)
    moved=[]
    try:
        for source in paths:
            target=archive/source.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            source.replace(target);moved.append((source,target))
        (archive/'manifest.json').write_text(json.dumps(dict(label=label,original_root=str(root),
            files=[dict(original=str(a),archived=str(b)) for a,b in moved]),indent=2),encoding='utf-8')
    except Exception:
        for source,target in reversed(moved):target.replace(source)
        raise
    return dict(label=label,files=len(moved),archive=str(archive.resolve()))
