"""Personal geometric J/Z calibration. No neural training or probability claims.

The same bounded trajectory search is used for fitting, replay and live inference.
Starting handshape is required; wrist-turn handshape flicker cannot erase a stroke.
"""
from collections import deque
import json
import os
from pathlib import Path
import uuid
import numpy as np
from src.contracts import FrameObservation, HandObservation, Prediction
from src.vision.features import primary_hand
from .jz import _shape, _resample

SCHEMA = 'signbridge-geometric-calibration-v1'
LIMITS = dict(min_extent=.55, residual=.24, drift=.65, slope=.65)


def geometry(label, path, limits):
    """Return measured geometry only for a complete directed hook/three-stroke Z."""
    if len(path) < 7:
        return None
    x, y = path.T
    extent = float(max(np.ptp(x), np.ptp(y)))
    if extent < limits['min_extent']:
        return None
    if label == 'J':
        bottom = int(np.argmax(y))
        depth = float(y[bottom])
        width = float(-x[-1])
        if (bottom < 2 or bottom >= len(path)-2 or depth < .5 or width < .3
                or depth-y[-1] < .2 or x.max() > limits['drift']*depth
                or abs(x[np.argmax(y >= depth*.5)]) > limits['drift']*depth):
            return None
        templates = [np.array([[0,0],[0,.6],[-.08,.84],[-.35,1],[-.72,.95],[-1,end]])*[width,depth]
                     for end in (.1,.3,.55,.7,.85)]
        error = min(float(np.sqrt(np.mean((_resample(path)-_resample(t))**2)))/extent for t in templates)
    else:
        # An actual final rightward stroke is mandatory, including in calibration.
        first = None
        for i in range(1,len(path)-3):
            if x[i] >= .4 and x[i]-x[i+1:].min() >= .4:
                first=i
                while first+1 < len(path) and x[first+1] > x[first]: first+=1
                break
        if first is None or first >= len(path)-3: return None
        second=first+1+int(np.argmin(x[first+1:]))
        a,b,c=path[first],path[second],path[-1]
        if (a[0]<.4 or abs(a[1])>limits['slope']*a[0] or a[0]-b[0]<.4 or b[1]-a[1]<.35
                or c[0]-b[0]<.4 or abs(c[1]-b[1])>limits['slope']*(c[0]-b[0])): return None
        error=float(np.sqrt(np.mean((_resample(path)-_resample(np.array([[0,0],a,b,c])))**2)))/extent
    if error > limits['residual']: return None
    return dict(extent=extent,residual=error)


def observations(path):
    with np.load(path,allow_pickle=False) as data:
        required={'raw_landmarks','frame_sizes','timestamps_ms','sequence','motion_schema'}
        if not required <= set(data.files): raise ValueError('No raw trajectories in this legacy clip (preserved).')
        raw=data['raw_landmarks']; sizes=data['frame_sizes']; times=data['timestamps_ms']; seq=data['sequence']
        if (str(data['motion_schema'].item())!='raw-unmirrored-two-hand-v1' or raw.shape!=(len(times),2,21,3)
                or sizes.shape!=(len(times),2) or seq.shape!=(len(times),134) or len(times)<2
                or not np.isfinite(raw).all() or not np.isfinite(seq).all()
                or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or np.any(sizes<=0)):
            raise ValueError('Invalid raw trajectory archive')
        return [FrameObservation(int(t),[HandObservation(p[i],side) for i,side in enumerate(('Right','Left')) if f[132+i]>.5],int(s[0]),int(s[1]))
                for p,s,t,f in zip(raw,sizes,times,seq)]


def validate_profile(value):
    if not isinstance(value,dict) or value.get('schema')!=SCHEMA: raise ValueError('Unsupported J/Z calibration schema')
    if set(value.get('labels',{})) != {'J','Z'}: raise ValueError('Calibration must contain J and Z')
    for label,limits in value['labels'].items():
        if set(limits)!=set(LIMITS): raise ValueError(f'Invalid {label} geometry parameters')
        for key,low,high in [('min_extent',.5,3.),('residual',.05,.24),('drift',.1,.65),('slope',.15,.65)]:
            v=limits[key]
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not low<=v<=high:
                raise ValueError(f'Invalid {label} {key}')
    if not isinstance(value.get('id'),str) or not value['id']: raise ValueError('Missing calibration identity')
    return value


def load_profile(path):
    return validate_profile(json.loads(Path(path).read_text(encoding='utf-8')))


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temporary,path)
    finally: temporary.unlink(missing_ok=True)


class CalibratedGeometry:
    supported_labels=('J','Z')
    max_gap_ms=250
    def __init__(self,profile):
        self.profile=validate_profile(profile);self.reset()
    def reset(self):
        self.history=deque(maxlen=100);self._last_time=None;self._last_seen=None;self._hand=None
        self._label=None;self._shape_since=None;self._endpoint=None;self._committed_at=None
        self.pending=self.motion_active=False;self.state='idle';self.reason='Waiting for starting handshape';self.progress=0.
        self.last_measurement=None;self._wrist=None;self._flip_since=None;self._shape_confirmations=0;self._incompatible_since=None
    def snapshot(self):
        return dict(mode='MOTION' if self.pending or self.motion_active else 'STATIC',state=self.state,
                    label=self._label,pending=self.pending,reason=self.reason,progress=self.progress,
                    trajectory=[list(r[5]) for r in self.history],score_kind='personal geometric match, not probability',
                    calibration_id=self.profile['id'],thresholds=self.profile['labels'])
    def update(self,observation,static_prediction=None):
        now=observation.timestamp_ms
        if self._last_time is not None and now<self._last_time: raise ValueError('Motion timestamps moved backwards')
        if now==self._last_time:return None
        if self._last_time is not None and now-self._last_time>250:self.reset()
        self._last_time=now
        hand=primary_hand(observation)
        if hand is None:
            if self._last_seen is not None and now-self._last_seen>300:
                self.reset();self._last_time=now
            return None
        # A wrist turn can briefly flip MediaPipe's handedness. Preserve the track
        # only with spatial continuity; a genuinely different hand resets it.
        if self._hand is not None and self._wrist is not None and hand.handedness!=self._hand:
            distance=np.linalg.norm((np.asarray(hand.landmarks)[0,:2]-self._wrist)*[observation.width/observation.height,1])
            if self._flip_since is None:self._flip_since=now
            if distance>.25 or now-self._flip_since>200:
                self.reset();self._last_time=now
        else:self._flip_since=None
        if self._hand is None:self._hand=hand.handedness
        self._wrist=np.asarray(hand.landmarks)[0,:2].copy();self._last_seen=now
        raw=np.asarray(hand.landmarks,float);points=raw.copy();points[:,(0,2)]*=observation.width/max(1,observation.height)
        scale=float(np.mean(np.linalg.norm(points[[5,9,13,17]]-points[0],axis=1)))
        if scale<.015:return None
        shape=_shape(points)
        # Release an attempt when the drawing finger is clearly folded. Brief
        # occlusion in the hook is tolerated; a new static fist is not held hostage.
        if self._label is not None:
            base=17 if self._label=='J' else 5
            joints=points[base:base+4]
            straight=np.linalg.norm(joints[-1]-joints[0])/max(1e-8,np.linalg.norm(np.diff(joints,axis=0),axis=1).sum())
            if straight<.65:
                if self._incompatible_since is None:self._incompatible_since=now
                if now-self._incompatible_since>=180:
                    self.reset();self._last_time=now;return None
            else:self._incompatible_since=None
        if self._endpoint is not None:
            label,position,previous_scale=self._endpoint
            tip=20 if label=='J' else 8
            current=points[tip,:2].copy();current[0]*=-1 if self._hand=='Right' else 1
            if shape not in (None,label) or (now-self._committed_at>350 and np.linalg.norm(current-position)/previous_scale>.65):
                self.reset();self._last_time=now;self._hand=hand.handedness;self._last_seen=now;self._wrist=raw[0,:2].copy()
            else:
                self.pending=self.motion_active=False;self.state='cooldown';return None
        if self._label is None:
            if shape is None:return None
            self._label=shape;self._shape_since=now
        # A changed pose can end a settled/idle attempt, but not a wrist-turn frame.
        if shape not in (None,self._label) and not self.motion_active:
            self.history.clear();self._label=shape;self._shape_since=now
        if shape==self._label:self._shape_confirmations+=1
        tip=20 if self._label=='J' else 8
        position=points[tip,:2].copy();position[0]*=-1 if self._hand=='Right' else 1
        self.history.append((now,position,scale,self._label if self._shape_confirmations>=2 else shape,points.copy(),tuple(raw[tip,:2])))
        while self.history and now-self.history[0][0]>3500:self.history.popleft()
        rows=list(self.history)
        if now-self._shape_since>4000:
            self.reset();self._last_time=now;return None
        self.pending=now-self._shape_since<650
        self.progress=0.
        # Endpoint stability uses a window, not an exact last-frame movement threshold.
        tail=[r for r in rows if now-r[0]<=200]
        settled=(len(tail)>=3 and now-tail[0][0]>=140 and
                 max(np.linalg.norm(r[1]-position) for r in tail)/scale<.20)
        if settled:
            for start in range(max(0,len(rows)-75),len(rows)-6):
                part=rows[start:]
                if part[0][3]!=self._label or now-part[0][0]<300:continue
                if sum(r[3]==self._label for r in part[:3])<2:continue
                if any(b[0]-a[0]>300 for a,b in zip(part,part[1:])):continue
                radius=float(np.median([r[2] for r in part]))
                path=(np.array([r[1] for r in part])-part[0][1])/radius
                # Drop endpoint duplicates; preserve all genuine movement and direction changes.
                moving=np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>.015]
                measured=geometry(self._label,path[moving],self.profile['labels'][self._label])
                if measured:
                    label=self._label;self.last_measurement={**measured,'start_ms':part[0][0],'end_ms':now}
                    self._endpoint=(label,position.copy(),scale);self._committed_at=now
                    self.pending=self.motion_active=False;self.state='completed';self.progress=1.
                    self.reason=f'Completed {label} using personal geometry calibration'
                    score=float(np.clip(1-measured['residual'],.8,.99))
                    return Prediction(label,score,now,'motion-calibrated',scores={label:score})
        # Only a directed prefix suppresses static recognition. Use recent starts,
        # so raising the hand before the stroke is not treated as part of J/Z.
        active=False
        for r in rows:
            if r[3]!=self._label or now-r[0]>2000:continue
            delta=(position-r[1])/r[2]
            if (self._label=='J' and delta[1]>.25 and -.8<delta[0]<.8) or (self._label=='Z' and delta[0]>.4 and abs(delta[1])<.65*delta[0]):active=True;break
        if active:self.motion_active=True
        self.state='tracking' if self.motion_active else ('pending' if self.pending else 'idle')
        self.progress=.5 if self.motion_active else 0.
        self.reason=f'Tracking {self._label}; finish and hold briefly' if self.motion_active else 'Waiting for directed movement'
        if settled and not self.motion_active:
            self.history=deque(rows[-4:],maxlen=100)
        # A changed handshape held for 200ms releases an incomplete attempt.
        if shape not in (None,self._label) and settled:
            self.reset();self._last_time=now
        return None
