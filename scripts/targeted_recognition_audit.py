"""Reproducible targeted audit/ablation; does not install a production model."""
from pathlib import Path
import argparse,json,sys,shutil,subprocess
from hashlib import sha256
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.contracts import STATIC_LABELS,CLASS_TO_ID
from src.recognition import CPURecognizer
from src.recognition.dataset import load_dataset,probability_matrix,metrics,session_split
from src.recognition.geometric import RAW,GEOMETRIC,model_features
from src.vision.dataset_summary import dataset_summary

TARGETS=('T','U','V','S','Q')
AUDIT_LABELS=('T','N','U','R','V','S','E','Q','H')
POLICY={'target_min_recall':.90,'target_min_confident_correct_fraction':.75,
        'maximum_existing_class_drop':.05,'maximum_existing_macro_drop':.01,
        'maximum_existing_confident_correct_drop':.05,'maximum_existing_confident_error_increase':.05,
        'confidence_threshold':.8,'coverage':list(STATIC_LABELS),
        'live_acceptance':'Required from user; session validation is not live confirmation.'}


def extended_holdout(dataset,old):
    """Solve session membership using metadata only; retain every old assignment."""
    provenance=old.metadata['training_provenance']
    fixed={s:False for s in provenance['train_sessions']}
    fixed.update({s:True for s in provenance['validation_sessions']})
    groups=[set(dataset.sessions[dataset.labels==CLASS_TO_ID[c]].tolist()) for c in dataset.supported_labels]
    def search(state):
        state=dict(state)
        while True:
            changed=False
            for group in groups:
                values={state[s] for s in group if s in state};missing=group-state.keys()
                if not missing and len(values)<2:return None
                if len(values)==1 and len(missing)==1:
                    state[next(iter(missing))]=not next(iter(values));changed=True
            if not changed:break
        remaining=set(dataset.sessions.tolist())-state.keys()
        if not remaining:return state
        item=sorted(remaining,key=lambda s:(-sum(s in g for g in groups),s))[0]
        for value in (False,True):
            result=search({**state,item:value})
            if result is not None:return result
        return None
    solved=search(fixed)
    if solved is None:raise ValueError('Cannot extend the old split without leakage; collect another independent session.')
    selected=sorted(s for s,v in solved.items() if v and s in set(dataset.sessions.tolist()))
    return session_split(dataset,validation_sessions=selected)


def probabilities(model,x):
    return probability_matrix(model_features(x,model.feature_set),model.arrays,list(model.supported_labels),model.temperature)


def class_rows(dataset,indices,scores):
    predicted=scores.argmax(1);confidence=scores.max(1)
    rows={}
    for c in dataset.supported_labels:
        cid=CLASS_TO_ID[c];mask=dataset.labels[indices]==cid;n=int(mask.sum())
        wrong=predicted[mask & (predicted!=cid)]
        confusion={chr(65+i):int(np.sum(wrong==i)) for i in sorted(set(wrong.tolist()))}
        rows[c]={'samples':n,'accuracy':float(np.mean(predicted[mask]==cid)),
                 'confident_correct_fraction':float(np.mean((predicted[mask]==cid)&(confidence[mask]>=.8))),
                 'confident_error_fraction':float(np.mean((predicted[mask]!=cid)&(confidence[mask]>=.8))),
                 'below_threshold_fraction':float(np.mean(confidence[mask]<.8)),
                 'most_frequent_confusion':max(confusion,key=confusion.get) if confusion else None,
                 'confusions':confusion,'validation_sessions':sorted(set(dataset.sessions[indices][mask].tolist()))}
    return rows


def validation_checks(old_rows,new_rows,old_classes):
    checks={}
    for c in TARGETS:
        checks[c+'_recall']=new_rows[c]['accuracy']>=POLICY['target_min_recall']
        checks[c+'_confident_correct']=new_rows[c]['confident_correct_fraction']>=POLICY['target_min_confident_correct_fraction']
    for c in old_classes:
        checks[c+'_regression']=new_rows[c]['accuracy']>=old_rows[c]['accuracy']-POLICY['maximum_existing_class_drop']-1e-12
        checks[c+'_confident_correct_regression']=new_rows[c]['confident_correct_fraction']>=old_rows[c]['confident_correct_fraction']-POLICY['maximum_existing_confident_correct_drop']-1e-12
        checks[c+'_confident_error_regression']=new_rows[c]['confident_error_fraction']<=old_rows[c]['confident_error_fraction']+POLICY['maximum_existing_confident_error_increase']+1e-12
    old_macro=np.mean([old_rows[c]['accuracy'] for c in old_classes])
    new_macro=np.mean([new_rows[c]['accuracy'] for c in old_classes])
    checks['existing_macro']=bool(new_macro>=old_macro-POLICY['maximum_existing_macro_drop'])
    return {k:bool(v) for k,v in checks.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/samples')
    parser.add_argument('--model',type=Path,default=ROOT/'models/alphabet.npz')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'work/targeted-recognition')
    parser.add_argument('--epochs',type=int,default=500)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    old=CPURecognizer(args.model);dataset=load_dataset(args.data_dir,labels=list(STATIC_LABELS))
    summary=dataset_summary(args.data_dir)
    if summary['errors']:raise ValueError(summary['errors'])
    train,val=extended_holdout(dataset,old)
    # Preserve the actual starting artifact and reports before any experiments.
    fallback=out/'fallback';fallback.mkdir(exist_ok=True)
    for path in args.model.parent.glob(args.model.stem+'*'):
        if path.is_file():
            target=fallback/path.name
            if target.exists() and target.read_bytes()!=path.read_bytes():raise ValueError('Fallback already contains a different model; choose a new output directory.')
            if not target.exists():shutil.copy2(path,target)
    old_scores=probabilities(old,dataset.features[val]);old_rows=class_rows(dataset,val,old_scores)
    split={'train_sessions':sorted(set(dataset.sessions[train].tolist())),
           'validation_sessions':sorted(set(dataset.sessions[val].tolist()))}
    plan={'policy':POLICY,'split':split,'production_sha256':sha256(args.model.read_bytes()).hexdigest(),
          'model_classes':list(old.supported_labels),'dataset_files':dataset.files,
          'experiments':[RAW,GEOMETRIC,'coordinates-calibrated-v1'],'epochs':args.epochs,'seed':42,
          'scope':'Same recording-session holdout. Validation selects stopping epoch and candidate; no untouched final test.'}
    (out/'comparison-plan.json').write_text(json.dumps(plan,indent=2))
    candidates={}
    for experiment in (RAW,GEOMETRIC,'coordinates-calibrated-v1'):
        feature_set=GEOMETRIC if experiment==GEOMETRIC else RAW
        model_path=out/(experiment+'.npz')
        command=[sys.executable,'-u','-m','training.train','--data-dir',str(args.data_dir),'--ready-only',
                 '--output',str(model_path),'--feature-set',feature_set,'--epochs',str(args.epochs),
                 '--validation-sessions',*split['validation_sessions']]
        if experiment=='coordinates-calibrated-v1':command+=['--calibrate-temperature']
        with (out/(experiment+'.log')).open('w',encoding='utf-8') as handle:
            result=subprocess.run(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError((out/(experiment+'.log')).read_text()[-2000:])
        candidate=CPURecognizer(model_path);scores=probabilities(candidate,dataset.features[val]);rows=class_rows(dataset,val,scores)
        checks=validation_checks(old_rows,rows,old.supported_labels)
        checks['coverage']=candidate.supported_labels==STATIC_LABELS
        # Compare saved/loaded single-frame adapter with batch experiment results.
        checks['reload_parity']=all(np.allclose(candidate.predict(dataset.features[val[i]],i).scores,scores[i],atol=2e-5) for i in range(0,len(val),max(1,len(val)//20)))
        candidates[experiment]={'path':str(model_path),'feature_set':feature_set,'experiment':experiment,'temperature':candidate.temperature,'rows':rows,
            'macro_accuracy':float(np.mean([r['accuracy'] for r in rows.values()])),
            'checks':checks,'passed':all(checks.values()),'sha256':sha256(model_path.read_bytes()).hexdigest()}
        print(experiment,json.dumps({'passed':all(checks.values()),'failed':[k for k,v in checks.items() if not v],
            'targets':{c:rows[c]['accuracy'] for c in TARGETS}}),flush=True)
    # Prefer the unchanged feature pipeline when its measured accuracy is effectively tied.
    passing=[v for v in candidates.values() if v['passed']]
    selected=max(passing,key=lambda c:c['macro_accuracy']) if passing else None
    if selected and candidates[RAW]['passed'] and candidates[RAW]['macro_accuracy']>=selected['macro_accuracy']-.005:selected=candidates[RAW]
    file_audit=[];offset=0
    chosen=selected if selected else max(candidates.values(),key=lambda c:c['macro_accuracy'])
    best_model=CPURecognizer(chosen['path'])
    for entry in dataset.files:
        count=entry['samples'];x=dataset.features[offset:offset+count];offset+=count
        p=x.reshape(-1,21,3);direction=p[:,9]-p[:,0];unit=direction/np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-6)
        scores=probabilities(best_model,x);pred=scores.argmax(1);correct=pred==CLASS_TO_ID[entry['label']]
        flags=[]
        accuracy=float(np.mean(correct));spread=float(np.sqrt(np.mean(np.var(x,axis=0))))
        direction_spread=float(np.degrees(np.arccos(np.clip(np.linalg.norm(unit.mean(0)),0,1))))
        degenerate=np.zeros(count,bool)
        for a,b in ((0,5),(5,6),(6,7),(7,8),(9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)):
            degenerate|=np.linalg.norm(p[:,a]-p[:,b],axis=1)<1e-5
        if accuracy<.9:flags.append('low agreement with recorded label; review pose/label/tracker, not proof of mislabeling')
        if spread<.02:flags.append('very low within-file variation')
        if np.any(degenerate):flags.append('degenerate coincident landmarks')
        file_audit.append({**entry,'split':'validation' if entry['session_id'] in split['validation_sessions'] else 'training',
            'candidate_label_agreement':accuracy,'rms_feature_spread':spread,'orientation_spread_degrees':direction_spread,
            'mean_wrist_to_middle_direction':unit.mean(0).tolist(),'degenerate_frames':int(degenerate.sum()),
            'flags':flags,'top_predictions':{chr(65+i):int(np.sum(pred==i)) for i in sorted(set(pred.tolist()))}})
    report={**plan,'created_utc':datetime.now(timezone.utc).isoformat(),'summary':summary,'old_rows':old_rows,
            'candidates':candidates,'selected':selected['experiment'] if selected else None,'production_replaced':False,
            'file_audit':file_audit,'live_confirmed':False,
            'limitations':['No raw webcam images or per-landmark visibility/confidence were stored; mislabeling or occlusion cannot be proved from these vectors.',
                          'Different session IDs do not prove separate sittings; the user must confirm collection independence.']}
    (out/'comparison.json').write_text(json.dumps(report,indent=2))
    print('Saved',out/'comparison.json',flush=True)


if __name__=='__main__':main()
