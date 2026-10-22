"""Fit personal geometric acceptance thresholds from saved raw J/Z strokes."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import uuid
import numpy as np
from src.temporal.motion_calibration import (CalibratedGeometry, LIMITS, SCHEMA, observations,
                                             save_json, validate_profile)


def replay(frames, profile):
    detector=CalibratedGeometry(profile);found=[]
    for frame in frames:
        result=detector.update(frame)
        if result:found.append(dict(label=result.label,**detector.last_measurement))
    return found


def fit(rows):
    profile=dict(schema=SCHEMA,id=uuid.uuid4().hex,labels={c:dict(LIMITS) for c in ('J','Z')})
    for label in ('J','Z'):
        measurements=[m for r in rows if r['label']==label for m in r.get('strokes',[]) if m['label']==label]
        if not measurements:raise ValueError(f'No complete usable {label} stroke. See the calibration report; existing files are preserved.')
        profile['labels'][label]['min_extent']=float(np.clip(np.percentile([m['extent'] for m in measurements],10)*.35,.55,1.))
        profile['labels'][label]['residual']=float(np.clip(np.percentile([m['residual'] for m in measurements],95)+.04,.14,.24))
    return profile


def calibrate(data_dir, output):
    root=Path(data_dir);output=Path(output);rows=[];frames={}
    broad=dict(schema=SCHEMA,id='bounded-segmentation',labels={c:dict(LIMITS) for c in ('J','Z')})
    for path in sorted(root.rglob('*.npz')):
        with np.load(path,allow_pickle=False) as data:
            label=str(data['label'].item())
            if label not in ('J','Z'):continue
            session=str(data['session_id'].item())
        row=dict(path=str(path.relative_to(root)),label=label,session_id=session,sha256=sha256(path.read_bytes()).hexdigest())
        try:
            clip=observations(path);frames[row['path']]=clip
            row['frames']=len(clip);row['strokes']=replay(clip,broad)
            row['usable']=bool(row['strokes']) and all(s['label']==label for s in row['strokes'])
            row['reason']=('Complete directed stroke(s) with observed endpoint hold' if row['usable'] else
                           'No complete directed stroke plus endpoint hold survived tracking; not used for calibration')
        except (ValueError,KeyError,OSError) as exc:row.update(usable=False,reason=str(exc),strokes=[])
        rows.append(row)
    report=dict(method=SCHEMA,clips=rows,sessions=sorted({r['session_id'] for r in rows}),
                scope='Personal geometric calibration. Same-session replay is not independent-session or live accuracy.')
    report_path=output.with_name(output.stem+'_report.json')
    save_json(output.with_name(output.stem+'_attempt.json'),report)
    profile=fit([r for r in rows if r['usable']])
    profile.update(created_at=datetime.now(timezone.utc).isoformat(),files=[{k:r[k] for k in ('path','label','session_id','sha256')} for r in rows if r['usable']],
                   sessions=report['sessions'],scope=report['scope'])
    for row in rows:
        if row['path'] in frames:
            row['calibrated_detections']=replay(frames[row['path']],profile)
    # Leave one clip out: never use that clip's measurements to fit its thresholds.
    # This remains same-session evidence; an unsupported held-out label is explicit.
    for row in rows:
        if row['path'] not in frames:continue
        try:
            held=fit([r for r in rows if r['usable'] and r['path']!=row['path']])
            row['leave_one_clip_out']=replay(frames[row['path']],held)
        except ValueError:row['leave_one_clip_out']=None
    report['per_class']={c:dict(saved_clips=sum(r['label']==c for r in rows),
            usable_clips=sum(r['label']==c and r['usable'] for r in rows),
            detected_strokes=sum(len(r.get('calibrated_detections',[])) for r in rows if r['label']==c),
            sessions=len({r['session_id'] for r in rows if r['label']==c and r['usable']})) for c in ('J','Z')}
    report['profile_id']=profile['id'];report['thresholds']=profile['labels']
    if len(report['sessions'])<2:report['warning']='One recording session only; independent-session validation unavailable.'
    for row in rows:
        if row['usable'] and not any(m['label']==row['label'] for m in row.get('calibrated_detections',[])):
            save_json(output.with_name(output.stem+'_attempt.json'),report);raise ValueError('Fitted thresholds rejected a usable calibration clip; prior profile retained.')
    validate_profile(profile)
    history=output.parent/'motion_history';history.mkdir(parents=True,exist_ok=True)
    if output.exists():
        old=output.read_bytes();archive=history/(output.stem+'-'+sha256(old).hexdigest()[:16]+'.json')
        if not archive.exists():archive.write_bytes(old)
    # A complete versioned report is durable before the active config is replaced.
    save_json(history/(profile['id']+'_report.json'),report)
    save_json(history/(profile['id']+'.json'),profile)
    save_json(report_path,report);save_json(output.with_name(output.stem+'_attempt.json'),report);save_json(output,profile)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/samples'))
    parser.add_argument('--output',type=Path,default=Path('models/motion_calibration.json'))
    args=parser.parse_args(argv)
    try:
        report=calibrate(args.data_dir,args.output)
        print('J/Z calibration saved. '+str(report['per_class']))
        print(report.get('warning',report['scope']))
        return 0
    except (ValueError,OSError,KeyError) as exc:
        print('Calibration failed: '+str(exc));return 2


if __name__=='__main__':raise SystemExit(main())
