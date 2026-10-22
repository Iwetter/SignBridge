"""Calibration, artifact loading, buffer coordination and negative-motion regressions."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.contracts import Prediction
from src.temporal.motion_calibration import (CalibratedGeometry, LIMITS, SCHEMA, load_profile,
                                             observations, save_json, validate_profile)
from src.temporal.fingerspelling import FingerspellingCoordinator
from src.temporal.boundaries import RepeatDetector
from src.vision import DatasetRecorder
from src.conversation import ConversationEngine
from training.calibrate_motion import calibrate
from test_continuous_fingerspelling import hand, obs, stroke


def profile():
    return dict(schema=SCHEMA,id='test-only',labels={c:dict(LIMITS) for c in ('J','Z')})


def record(root,label,session='fixture'):
    recorder=DatasetRecorder(root,session_id=session);recorder.start(label,0)
    for frame in stroke(label):recorder.add(frame)
    recorder.finish()


@pytest.mark.parametrize('label',['J','Z'])
@pytest.mark.parametrize('side',['Right','Left'])
@pytest.mark.parametrize('dt',[25,40,65])
def test_calibrated_completion_is_one_shot_and_survives_turn_flicker(label,side,dt):
    frames=stroke(label,side=side,dt=dt);detector=CalibratedGeometry(profile());found=[]
    for i,frame in enumerate(frames):
        # One lost observation and one handedness flicker in the moving stroke.
        if i==17:frame=obs(frame.timestamp_ms)
        if i==20:frame.hands[0].handedness='Left' if side=='Right' else 'Right'
        result=detector.update(frame)
        if result:found.append(result.label)
    assert found==[label]
    last=frames[-1]
    for t in range(last.timestamp_ms+dt,last.timestamp_ms+1000,dt):
        assert detector.update(obs(t,last.hands[0].landmarks,side)) is None


@pytest.mark.parametrize('label',['J','Z'])
@pytest.mark.parametrize('case',['stationary','swipe','incomplete','reverse','gap'])
def test_calibrated_geometry_rejects_nonletters(label,case):
    paths={'stationary':[(0,0),(0,0)],'swipe':[(0,0),(1,0)],
           'incomplete':[(0,0),(1,0),(0,.8)] if label=='Z' else [(0,0),(0,1)]}
    frames=stroke(label,path=paths.get(case))
    if case=='reverse':
        positions=[f.hands[0].landmarks.copy() for f in frames][::-1]
        frames=[obs(f.timestamp_ms,p) for f,p in zip(frames,positions)]
    if case=='gap':
        mid=len(frames)//2
        for f in frames[mid:]:f.timestamp_ms+=600
    detector=CalibratedGeometry(profile())
    assert not [r for f in frames if (r:=detector.update(f))]


def test_calibrate_reload_preserves_recordings_and_failed_attempt_preserves_config(tmp_path):
    data=tmp_path/'samples';output=tmp_path/'motion.json'
    record(data,'J');record(data,'Z')
    originals={p:p.read_bytes() for p in data.glob('*.npz')}
    report=calibrate(data,output);loaded=load_profile(output)
    assert report['per_class']['J']['usable_clips']==1
    assert report['per_class']['Z']['usable_clips']==1
    assert loaded['files'] and loaded['labels']['J']['min_extent']>=.55
    assert all(p.read_bytes()==content for p,content in originals.items())
    active=output.read_bytes();active_report=output.with_name('motion_report.json').read_bytes()
    with pytest.raises(ValueError,match='No complete usable'):
        calibrate(tmp_path/'empty',output)
    assert output.read_bytes()==active
    assert output.with_name('motion_report.json').read_bytes()==active_report
    assert len(list((tmp_path/'motion_history').glob('*.json')))==2


def test_profile_validation_rejects_invalid_or_nonfinite_parameters():
    p=profile();p['labels']['J']['residual']=float('nan')
    with pytest.raises(ValueError):validate_profile(p)


def test_completed_j_promotes_its_starting_i_but_withdrawal_preserves_intentional_ij():
    for separate in (False,True):
        c=FingerspellingCoordinator(motion_profile=profile());engine=ConversationEngine('free_spell')
        frames=[obs(t,hand('J')) for t in range(0,1000,40)]
        if separate:frames += [obs(t) for t in range(1000,1400,40)]
        frames += stroke('J',start=1400 if separate else 1000)
        for f in frames:
            pred=Prediction('I',.99,f.timestamp_ms) if f.hands else None
            token=c.update(f,pred,has_text=bool(engine.snapshot()['word']))
            if c.replace_last_static:engine.backspace()
            if token:engine.consume(token)
        assert engine.snapshot()['word']==('IJ' if separate else 'J')


def test_static_pose_after_motion_and_after_incomplete_attempt_is_not_blocked():
    for complete in (True,False):
        c=FingerspellingCoordinator(motion_profile=profile());found=[]
        frames=stroke('J',path=None if complete else [(0,0),(0,1)])
        for f in frames:
            t=c.update(f,Prediction('I',.99,f.timestamp_ms))
            if t:found.append(t.label)
        start=frames[-1].timestamp_ms+40
        for now in range(start,start+1000,40):
            t=c.update(obs(now,hand('A')),Prediction('A',.99,now))
            if t:found.append(t.label)
        assert found[-1]=='A'


def test_repeat_arms_at_commit_and_survives_brief_confidence_dip():
    c=FingerspellingCoordinator();p=hand('L');found=[]
    for now in range(0,400,40):
        t=c.update(obs(now,p),Prediction('L',.99,now))
        if t:found.append(t.label)
    # Start moving immediately after the first commit, without an extra origin hold.
    for i,dx in enumerate(np.r_[np.linspace(0,-.18,12),np.repeat(-.18,10)]):
        now=400+i*40;q=p.copy();q[:,0]+=dx
        t=c.update(obs(now,q),Prediction('L',.5 if i in (5,6) else .99,now))
        if t:found.append(t.label)
    assert found==['L','L']


def test_default_hands_free_spacing_is_enabled():
    c=FingerspellingCoordinator();engine=ConversationEngine('free_spell')
    for now in range(0,800,40):
        t=c.update(obs(now,hand()),Prediction('A',.99,now),has_text=bool(engine.snapshot()['word']))
        if t:engine.consume(t)
    for now in range(800,2600,40):
        c.update(obs(now),None,has_text=bool(engine.snapshot()['word']))
        if c.space_requested:engine.insert_space()
    assert engine.snapshot()['word']=='A '


def test_application_loads_calibration_routes_into_buffer_and_ui_actions(tmp_path):
    import os
    from src.application import ApplicationController
    from src.vision.camera import CapturedFrame
    from src.contracts import STATIC_LABELS
    data=tmp_path/'samples';record(data,'J');record(data,'Z')
    report=calibrate(data,tmp_path/'motion_calibration.json')
    class Source:
        frame=None
        def snapshot(self):return dict(state='LIVE',generation=1,error='',capture_fps=30,capture_ms=1,input_mirrored=False)
        def latest(self,*args):return self.frame
        def close(self):pass
    class Classifier:
        supported_labels=STATIC_LABELS
        def predict(self,x,t):return Prediction('I',.99,t,'fixture')
    args=SimpleNamespace(camera=None,dwell_ms=350,data_dir=data,landmarker=tmp_path/'unused',model=tmp_path/'alphabet.npz',mute=True,replay=False)
    source=Source();events=[];app=ApplicationController(args,events.append,source_controller=source)
    app.cpu=Classifier();app.mode='free_spell';app.conversation.set_mode('free_spell')
    try:
        assert app._load_motion_profile()
        frames=[obs(t,hand('J')) for t in range(0,1000,40)]+stroke('J',start=1000)
        for frame in frames:
            source.frame=CapturedFrame(frame.timestamp_ms,frame.timestamp_ms,np.zeros((48,64,3),np.uint8),1.,1)
            app._results.put((source.frame,frame,1.,None));app._consume_result()
        assert app.conversation.snapshot()['word']=='J'
        app.action('fingerspelling_settings',dict(auto_space=True));app._commands()
        assert app.continuous.motion.profile['id']==report['profile_id']
        app._event();assert events[-1]['motion_calibration_report']['profile_id']==report['profile_id']
        # Invalid replacement config must not discard an already working runtime.
        app.motion_profile_path.write_text('{}',encoding='utf-8')
        assert not app._load_motion_profile()
        assert app.continuous.motion.profile['id']==report['profile_id']
        assert 'load failed' in app.motion_profile_status
        os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
        from src.ui.desktop import SignBridgeUI
        calls=[];ui=SignBridgeUI(lambda *a:calls.append(a))
        try:
            ui.update(events[-1]);ui._drain()
            ui.calibrate_motion.click();ui.reload_motion.click()
            assert ('calibrate_motion',None) in calls and ('reload_motion',None) in calls
            assert '1/1 usable' in ui.motion_profile_status.text()
            assert 'USED' in ui.motion_result_text.toPlainText()
            assert ui.train.text()=='Train static'
        finally:ui.window.close()
    finally:app.close()
