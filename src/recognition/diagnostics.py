"""Explain scores and commit gates without changing classifier/decoder behavior."""
import numpy as np
from src.contracts import CLASS_TO_ID


def prediction_diagnostics(prediction,gate,classes,*,fresh=True,reason='',paused=False,replay=False):
    result={'classifier_candidate':None,'top_five':[],
            'classifier_decision':'No current prediction','temporal_decision':'Not evaluated'}
    if replay:
        result['classifier_decision']='Synthetic replay — no classifier scores'
        return result
    if not fresh or prediction is None:
        result['classifier_decision']=reason or ('Stale / no current hand observation' if not fresh else 'No model prediction')
        return result
    result['classifier_candidate']=prediction.label
    if prediction.scores is not None:
        scores=np.asarray(prediction.scores)
        if scores.shape==(26,) and np.isfinite(scores).all():
            result['top_five']=[{'label':c,'score':float(scores[CLASS_TO_ID[c]])}
                for c in sorted(classes,key=lambda c:(-scores[CLASS_TO_ID[c]],CLASS_TO_ID[c]))[:5]]
    if prediction.confidence<gate['confidence_threshold']:
        result['classifier_decision']='Rejected: below confidence threshold'
        result['temporal_decision']='Not evaluated: classifier confidence rejected'
    else:
        result['classifier_decision']='Accepted by confidence threshold'
        if paused:result['temporal_decision']='Paused: recording, saving, maintenance or motion'
        elif gate['locked_label']==prediction.label:result['temporal_decision']='Already committed; held-letter repeat blocked'
        elif gate['candidate']==prediction.label:
            result['temporal_decision']=f'Waiting for stable pose: {gate["stable_duration_ms"]}/{gate["dwell_ms"]} ms'
        else:result['temporal_decision']='Not committed: dwell interrupted or prediction not yet observed'
    return result
