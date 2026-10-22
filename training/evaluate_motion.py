"""Evaluate geometric J/Z completion on recorded raw landmark clips; no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.contracts import FrameObservation, HandObservation
from src.temporal.jz import JZRecognizer


def evaluate_clips(data_dir, sessions=None, calibration=None):
    profile=None
    if calibration is not None:
        from src.temporal.motion_calibration import load_profile
        profile=load_profile(calibration)
    rows=[]
    for path in sorted(Path(data_dir).rglob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            label=str(data['label'].item())
            if label not in ('J','Z'):
                continue
            session=str(data['session_id'].item())
            if sessions and session not in sessions:
                continue
            row=dict(path=str(path.resolve()),label=label,session_id=session)
            required={'raw_landmarks','frame_sizes','timestamps_ms','sequence','motion_schema'}
            if not required <= set(data.files):
                rows.append({**row,'error':'Legacy clip has no raw fingertip trajectories; preserved, not reconstructed.'})
                continue
            raw=data['raw_landmarks'];sizes=data['frame_sizes'];times=data['timestamps_ms'];seq=data['sequence']
            if str(data['motion_schema'].item())!='raw-unmirrored-two-hand-v1' or raw.shape!=(len(times),2,21,3) or sizes.shape!=(len(times),2) or seq.shape!=(len(times),134):
                raise ValueError(f'Invalid motion archive: {path}')
            if len(times)<2 or not np.isfinite(raw).all() or not np.isfinite(seq).all() or np.any(np.diff(times)<=0) or np.any(sizes<=0):
                raise ValueError(f'Invalid motion samples: {path}')
            if profile is None:recognizer=JZRecognizer()
            else:
                from src.temporal.motion_calibration import CalibratedGeometry
                recognizer=CalibratedGeometry(profile)
            detected=[]
            for points,size,timestamp,features in zip(raw,sizes,times,seq):
                hands=[HandObservation(points[i],side) for i,side in enumerate(('Right','Left')) if features[132+i]>.5]
                result=recognizer.update(FrameObservation(int(timestamp),hands,int(size[0]),int(size[1])))
                if result:detected.append(dict(label=result.label,timestamp_ms=result.timestamp_ms,score=result.confidence))
            rows.append({**row,'frames':len(times),'duration_ms':int(times[-1]-times[0]),'detections':detected,
                         'correct': [item['label'] for item in detected]==[label]})
    eligible=[r for r in rows if 'error' not in r]
    per_class={}
    for label in ('J','Z'):
        found=[r for r in eligible if r['label']==label]
        correct=sum(r['correct'] for r in found)
        per_class[label]=dict(clips=len(found),correct=correct,accuracy=correct/len(found) if found else None)
    return dict(method='geometric-jz-v1' if profile is None else profile['schema'],calibration_id=profile['id'] if profile else None,scope='Recorded trajectory completion, no training or probability calibration. Session independence and live use require user confirmation.',
                clips=rows,sessions=sorted({r['session_id'] for r in rows}),
                per_class=per_class)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/samples'))
    parser.add_argument('--sessions',nargs='+')
    parser.add_argument('--calibration',type=Path,help='Evaluate this saved personal geometry profile; omit for original default geometry')
    parser.add_argument('--output',type=Path,default=Path('work/motion-evaluation.json'))
    args=parser.parse_args(argv)
    result=evaluate_clips(args.data_dir,args.sessions,args.calibration)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['per_class'],indent=2))
    print('Saved',args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
