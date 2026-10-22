"""Synthetic trajectory/coordination tests, not webcam recognition evidence."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.contracts import FrameObservation, HandObservation, Prediction, STATIC_LABELS, Token
from src.conversation import ConversationEngine
from src.temporal import TemporalDecoder
from src.temporal.boundaries import RepeatDetector, AutoSpaceDetector
from src.temporal.fingerspelling import FingerspellingCoordinator
from src.temporal.jz import JZRecognizer, _shape
from src.vision import DatasetRecorder
from src.vision.dataset_summary import dataset_summary


def hand(shape='A', side='Right', scale=1., offset=(0., 0.)):
    p=np.zeros((21,3),np.float32);p[0]=(.5,.75,0)
    for base,x,y in ((5,.43,.60),(9,.48,.57),(13,.53,.59),(17,.58,.62)):
        p[base]=(x,y,0)
        extended=(shape in ('Z','L') and base==5) or (shape=='J' and base==17)
        for n in range(1,4):
            p[base+n]=(x,y+(-.055*n if extended else (-.05,.01,.025)[n-1]),0)
    p[1:5]=[(.45,.70,0),(.44,.66,0),(.46,.63,0),(.48,.61,0)]
    if shape=='L':p[4]=(.25,.62,0)
    if side=='Left':p[:,0]=1-p[:,0]
    p[:,:2]=(p[:,:2]-p[0,:2])*scale+p[0,:2]+offset
    return p


def obs(t, p=None, side='Right', width=640,height=480):
    return FrameObservation(t, [] if p is None else [HandObservation(p,side)],width,height)


def geometry_path(label):
    return np.array([(0,0),(0,.65),(-.08,.9),(-.35,1.1),(-.65,1.05),(-.8,.8)] if label=='J'
                    else [(0,0),(1.,0),(0,.8),(1.,.8)],float)


def stroke(label, *, start=0, dt=40, side='Right', size=1., position=(0,0), amplitude=1., path=None):
    p=hand(label,side,size,position);aspect=640/480
    pixel=p.copy();pixel[:,(0,2)]*=aspect
    radius=np.linalg.norm(pixel[[5,9,13,17]]-pixel[0],axis=1).mean()
    path=geometry_path(label) if path is None else np.asarray(path,float)
    dense=np.concatenate([np.linspace(a,b,9,endpoint=False) for a,b in zip(path,path[1:])]+[path[-1:]])*amplitude
    positions=np.concatenate([np.repeat(dense[:1],5,axis=0),dense,np.repeat(dense[-1:],10,axis=0)])
    result=[]
    for i,(x,y) in enumerate(positions):
        shifted=p.copy();shifted[:,0]+=x*radius/aspect*(-1 if side=='Right' else 1);shifted[:,1]+=y*radius
        result.append(obs(start+i*dt,shifted,side))
    return result


class Signing:
    def __init__(self, settings=None):
        self.recognizer=FingerspellingCoordinator(settings=settings)
        self.engine=ConversationEngine('free_spell');self.t=0;self.commits=[]

    def feed(self, observation, letter=None, confidence=.98):
        pred=Prediction(letter,confidence,observation.timestamp_ms,'numeric-fixture') if letter else None
        token=self.recognizer.update(observation,pred,has_text=bool(self.engine.snapshot()['word'].strip()))
        if token:self.commits.append(token.label);self.engine.consume(token)
        if self.recognizer.space_requested:self.engine.insert_space()
        self.t=observation.timestamp_ms+40

    def hold(self, letter, ms=800, p=None):
        p=hand('L' if letter=='L' else 'A') if p is None else p
        for t in range(self.t,self.t+ms,40):self.feed(obs(t,p),letter)

    def repeat(self, letter):
        p=hand('L' if letter=='L' else 'A')
        radius=np.linalg.norm((p[[5,9,13,17],:2]-p[0,:2])*[640,480],axis=1).max()/640
        for displacement in np.r_[np.linspace(0,-.9*radius,12),np.repeat(-.9*radius,12)]:
            q=p.copy();q[:,0]+=displacement;self.feed(obs(self.t,q),letter)

    def motion(self, letter, **kwargs):
        for frame in stroke(letter,start=self.t,**kwargs):self.feed(frame,'I' if letter=='J' else 'D')


@pytest.mark.parametrize('word',['IAN','HELLO','ANNA','LETTER','MY NAME','MY NAME IS IAN'])
def test_continuous_words_use_different_letter_dwell_repeat_movements_and_spaces(word):
    signing=Signing();previous=None
    for c in word:
        if c==' ':signing.engine.insert_space();previous=None
        elif c==previous:signing.repeat(c)
        else:signing.hold(c)
        if c!=' ':previous=c
    assert signing.engine.snapshot()['word']==word


@pytest.mark.parametrize('letter',['A','L'])
def test_five_second_hold_has_one_commit(letter):
    signing=Signing();signing.hold(letter,5000)
    assert signing.commits==[letter]


def test_lateral_repeat_jitter_return_cooldown_and_left_handed_coordinates():
    for side in ('Right','Left'):
        detector=RepeatDetector();p=hand('L',side);events=[];t=0
        rng=np.random.default_rng(20)
        for _ in range(80):
            q=p.copy();q[:,:2]+=rng.normal(0,.0006,(21,2))
            events.append(detector.update(obs(t,q,side),Prediction('L',.97,t),'L'));t+=40
        assert not any(events)
        for displacement in np.r_[np.linspace(0,-.18,14),np.repeat(-.18,12)]:
            q=p.copy();q[:,0]+=displacement
            events.append(detector.update(obs(t,q,side),Prediction('L',.97,t),'L'));t+=40
        assert sum(events)==1
        for displacement in np.r_[np.linspace(-.18,0,14),np.zeros(40)]:
            q=p.copy();q[:,0]+=displacement
            events.append(detector.update(obs(t,q,side),Prediction('L',.97,t),'L'));t+=40
        assert sum(events)==1


@pytest.mark.parametrize('case',['low_confidence','other_letter','vertical','wrong_direction','hand_switch','gap'])
def test_repeat_requires_continuous_same_confident_shape_and_lateral_movement(case):
    detector=RepeatDetector();p=hand('L');t=0;events=[]
    for _ in range(10):
        detector.update(obs(t,p),Prediction('L',.99,t),'L');t+=40
    for i,displacement in enumerate(np.r_[np.linspace(0,-.18,12),np.repeat(-.18,14)]):
        q=p.copy();q[:,0 if case!='vertical' else 1]+=(-displacement if case=='wrong_direction' else displacement)
        if case=='gap' and i==8:t+=600
        label='N' if case=='other_letter' else 'L';confidence=.5 if case=='low_confidence' else .99
        events.append(detector.update(obs(t,q,'Left' if case=='hand_switch' and i>=8 else 'Right'),Prediction(label,confidence,t),'L'));t+=40
    assert not any(events)


def test_brief_missing_hand_low_confidence_and_pauses_do_not_emit_space_or_letter():
    signing=Signing(dict(auto_space=True));signing.hold('A')
    for _ in range(3):signing.feed(obs(signing.t))
    signing.hold('A');assert signing.engine.snapshot()['word']=='A'
    for _ in range(30):signing.feed(obs(signing.t,hand()),'N',.4)
    assert signing.engine.snapshot()['word']=='A'
    for _ in range(40):signing.feed(obs(signing.t))
    assert signing.engine.snapshot()['word']=='A '
    for _ in range(40):signing.feed(obs(signing.t))
    assert signing.engine.snapshot()['word']=='A '
    # Time without observations cannot finish an automatic space interval.
    spacing=AutoSpaceDetector(True,1200)
    assert not spacing.update(0,True,True)
    assert not spacing.update(40,False,True)
    assert not spacing.update(5000,False,True)


@pytest.mark.parametrize('label',['J','Z'])
@pytest.mark.parametrize('side',['Right','Left'])
@pytest.mark.parametrize('dt,size,amplitude',[(25,.65,.8),(40,1.,1.),(65,1.25,1.2)])
def test_complete_motion_speed_size_position_handedness_and_one_shot(label,side,dt,size,amplitude):
    detector=JZRecognizer();frames=stroke(label,side=side,dt=dt,size=size,position=(.03,-.04),amplitude=amplitude)
    output=[p for f in frames if (p:=detector.update(f))]
    assert [p.label for p in output]==[label]
    last=frames[-1]
    for t in range(last.timestamp_ms+dt,last.timestamp_ms+5000,dt):
        assert detector.update(obs(t,last.hands[0].landmarks,side)) is None


@pytest.mark.parametrize('label',['J','Z'])
@pytest.mark.parametrize('path',[[[0,0],[.8,0]],[[0,0],[0,1]],[[0,0],[.1,.1]],[[0,0],[-.7,.7]]])
def test_ordinary_tip_movement_is_not_a_motion_letter(label,path):
    detector=JZRecognizer()
    assert not any(detector.update(f) for f in stroke(label,path=path))


def test_motion_shape_loss_tracking_gap_reverse_path_and_no_starting_static_letter():
    for label in ('J','Z'):
        detector=JZRecognizer();frames=stroke(label);middle=len(frames)//2
        frames[middle]=obs(frames[middle].timestamp_ms)
        assert not any(detector.update(f) for f in frames)
        detector.reset();frames=stroke(label)
        for f in frames[middle:]:f.timestamp_ms+=600
        assert not any(detector.update(f) for f in frames)
        detector.reset()
        reverse=geometry_path(label)[::-1];reverse=reverse-reverse[0]
        assert not any(detector.update(f) for f in stroke(label,path=reverse))
        signing=Signing();signing.motion(label)
        assert signing.commits==[label]


def test_jazz_includes_new_z_stroke_without_lowering_hand():
    signing=Signing();signing.motion('J');signing.hold('A');signing.motion('Z')
    last=stroke('Z')[-1].hands[0].landmarks;first=hand('Z')
    # Wait past cooldown then return; reverse movement is rearming only.
    for _ in range(12):signing.feed(obs(signing.t,last),'D')
    for ratio in np.linspace(0,1,12):signing.feed(obs(signing.t,last*(1-ratio)+first*ratio),'D')
    signing.motion('Z')
    assert signing.engine.snapshot()['word']=='JAZZ'


def test_incomplete_motion_does_not_leak_starting_static_pose_and_recovers_on_shape_change():
    for label,static,path in [('J','I',[(0,0),(0,1)]),('Z','D',[(0,0),(1,0)])]:
        signing=Signing();frames=stroke(label,path=path)
        for frame in frames:signing.feed(frame,static)
        last=frames[-1].hands[0].landmarks
        signing.hold(static,1000,last)
        assert signing.commits==[] and signing.recognizer.snapshot()['state']=='retry'
        signing.hold('A')
        assert signing.commits==['A']


def test_all_static_classes_use_unchanged_dwell_and_no_motion_from_stationary_pose():
    for label in STATIC_LABELS:
        signing=Signing();signing.hold(label,1400)
        assert signing.commits==[label]
    for shape,letter in [('J','I'),('Z','D')]:
        signing=Signing();signing.hold(letter,2500,hand(shape))
        assert signing.commits==[letter]  # Starting pose alone cannot emit J/Z.


def test_i_to_a_transition_retains_static_dwell_during_motion_reservation():
    signing=Signing()
    signing.hold('I',400,hand('J'))
    signing.hold('A',400)
    signing.hold('N',400)
    assert signing.engine.snapshot()['word']=='IAN'
    signing=Signing();signing.hold('I',400,hand('J'))
    signing.t+=1000;signing.hold('A',400)
    assert signing.commits==['A']  # Stalled video must not publish a stale provisional I.
    signing=Signing(dict(auto_space=True))
    signing.hold('A',400)
    for _ in range(36):signing.feed(obs(signing.t))
    assert signing.engine.snapshot()['word']=='A '


def test_buffer_editing_no_leading_duplicate_spaces_and_speech_text():
    engine=ConversationEngine('free_spell');engine.insert_space();engine.backspace()
    assert engine.snapshot()['word']==''
    for c in 'MY':engine.consume(Token(c,'letter',0,.99,'fixture'))
    engine.insert_space();engine.insert_space()
    for c in 'NAME':engine.consume(Token(c,'letter',0,.99,'fixture'))
    assert engine.speak_text()=='MY NAME'
    engine.backspace();assert engine.snapshot()['word']=='MY NAM'
    engine.reset();assert engine.speak_text()==''


def test_motion_recording_roundtrip_preserves_raw_timing_and_static_training_exclusion(tmp_path):
    from src.recognition.dataset import load_dataset
    recorder=DatasetRecorder(tmp_path,session_id='independent-one')
    for label in ('A','B','J','Z'):
        recorder.start(label)
        frames=stroke(label) if label in ('J','Z') else [obs(t,hand()) for t in range(0,1000,100)]
        for frame in frames:recorder.add(frame)
        path=recorder.finish()
        with np.load(path,allow_pickle=False) as saved:
            assert len(saved['features'])==len(saved['timestamps_ms'])
            if label in ('J','Z'):
                assert saved['raw_landmarks'].shape==(len(frames),2,21,3)
                assert saved['frame_sizes'].shape==(len(frames),2)
                np.testing.assert_array_equal(saved['raw_landmarks'][0,0],frames[0].hands[0].landmarks)
            else:assert 'raw_landmarks' not in saved.files
    assert len(dataset_summary(tmp_path)['files'])==4
    dataset=load_dataset(tmp_path)
    assert dataset.supported_labels==['A','B'] and len(dataset.excluded_files)==2
    from training.evaluate_motion import evaluate_clips
    evaluation=evaluate_clips(tmp_path)
    assert evaluation['per_class']['J']['correct']==evaluation['per_class']['Z']['correct']==1
    assert evaluate_clips(tmp_path,['another-sitting'])['clips']==[]


def test_settings_roundtrip_and_ui_actions_use_existing_controls(tmp_path):
    from src.fingerspelling_settings import read_settings,write_settings
    path=tmp_path/'settings.json';settings=read_settings(path)
    settings.update(auto_space=True,space_shortcut='Alt+Space',repeat_threshold=.9)
    write_settings(path,settings);assert read_settings(path)==settings
    with pytest.raises(ValueError):write_settings(path,dict(repeat_threshold=-1))
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from src.ui.desktop import SignBridgeUI
    calls=[];ui=SignBridgeUI(lambda *args:calls.append(args))
    try:
        ui.update(dict(source='live',mode='free_spell',word='MY',fingerspelling_settings=settings,
                       capabilities={'record':dict(enabled=True,reason='')},dataset={}))
        ui._drain();ui.space.click();ui.backspace.click();ui.record_j.click();ui.record_z.click()
        assert ('space',None) in calls and ('backspace',None) in calls
        assert ('record','J') in calls and ('record','Z') in calls
        assert ui.space_shortcut.key().toString()=='Alt+Space'
        assert tuple(ui.labels.itemText(i) for i in range(ui.labels.count()))==STATIC_LABELS
    finally:ui.window.close()


def test_actual_controller_routes_motion_repeat_space_backspace_and_speech(tmp_path):
    from src.application import ApplicationController
    from src.vision.camera import CapturedFrame
    class Source:
        frame=None
        def snapshot(self):return dict(state='LIVE',generation=1,error='',capture_fps=30,capture_ms=1,input_mirrored=False)
        def latest(self,*args):return self.frame
        def close(self):pass
    class Classifier:
        supported_labels=STATIC_LABELS
        label='I'
        def predict(self,x,t):return Prediction(self.label,.99,t,'numeric-fixture')
    class Speech:
        error=None
        spoken=[]
        def speak(self,text):self.spoken.append(text)
        def close(self):pass
    args=SimpleNamespace(camera=None,dwell_ms=350,data_dir=tmp_path/'samples',landmarker=tmp_path/'unused',model=tmp_path/'unused.npz',mute=True,replay=False)
    source=Source();events=[];app=ApplicationController(args,events.append,source_controller=source)
    app.cpu=Classifier();app.mode='free_spell';app.conversation.set_mode('free_spell');app.speech=Speech()
    def feed(frame):
        source.frame=CapturedFrame(frame.timestamp_ms,frame.timestamp_ms,np.zeros((48,64,3),np.uint8),1.,1)
        app._results.put((source.frame,frame,1.,None));app._consume_result()
    try:
        frames=stroke('J')
        for frame in frames:feed(frame)
        assert app.conversation.snapshot()['word']=='J'
        app.action('space');app._commands();app.action('space');app._commands()
        assert app.conversation.snapshot()['word']=='J '
        app.cpu.label='L';t=frames[-1].timestamp_ms+40
        signing=Signing();signing.hold('L');signing.repeat('L')
        p=hand('L')
        for stamp in range(t,t+1000,40):feed(obs(stamp,p))
        t+=1000
        for dx in np.r_[np.linspace(0,-.18,12),np.repeat(-.18,14)]:
            q=p.copy();q[:,0]+=dx;feed(obs(t,q));t+=40
        assert app.conversation.snapshot()['word']=='J LL'
        app._event();assert events[-1]['fingerspelling']['repeat']['state']=='COOLDOWN'
        app.action('speak');app._commands();assert app.speech.spoken[-1]=='J LL'
        app.action('backspace');app._commands();assert app.conversation.snapshot()['word']=='J L'
        app.action('clear');app._commands();assert app.conversation.snapshot()['word']==''
        app.conversation.consume(Token('A','letter',t,.99,'numeric-fixture'))
        app.continuous.spacing.enabled=True
        feed(obs(t,p));t+=40
        for _ in range(20):feed(obs(t));t+=40
        source.frame=CapturedFrame(t,t,np.zeros((48,64,3),np.uint8),1.,1)
        app._results.put((source.frame,None,None,'Synthetic tracker failure'));app._consume_result();t+=40
        for _ in range(25):feed(obs(t));t+=40
        assert app.conversation.snapshot()['word']=='A'  # Errors cannot complete a withdrawal timer.
    finally:app.close()
