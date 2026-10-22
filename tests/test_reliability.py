"""Physical devices are replaced at I/O boundaries; real app/controller code runs."""
import time
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from src.source_controller import SourceController
from src.vision.camera import CapturedFrame,HandTracker
from src.vision import DatasetRecorder
from src.vision.dataset_summary import dataset_summary
from src.application import ApplicationController
from src.contracts import FrameObservation,HandObservation


def wait_for(check,timeout=3):
    deadline=time.perf_counter()+timeout
    while time.perf_counter()<deadline:
        if check(): return
        time.sleep(.01)
    assert check(),'Timed out'


class Capture:
    def __init__(self,index,delay=.01,**kwargs):
        self.delay=delay
        self.closed=False
        self.count=0
    def read(self):
        time.sleep(self.delay)
        self.count+=1
        return CapturedFrame(self.count,time.monotonic_ns()//1000000,np.zeros((48,64,3),np.uint8),self.delay*1000)
    def close(self): self.closed=True


def points():
    result=np.zeros((21,3),np.float32)
    result[:,0]=np.linspace(.3,.6,21)
    result[:,1]=np.linspace(.8,.3,21)
    return result


def obs(t): return FrameObservation(t,[HandObservation(points(),'Right',.99,'Right')])


def test_repeated_source_switches_release_and_never_duplicate_owner():
    captures=[]
    def factory(index,**kwargs):
        assert all(c.closed for c in captures)
        captures.append(Capture(index))
        return captures[-1]
    source=SourceController(camera_index=0,camera_backend='ANY',capture_factory=factory)
    source.start(); owner=source._thread; source.start()
    assert source._thread is owner
    previous=0
    for _ in range(10):
        source.request('live')
        wait_for(lambda:source.snapshot()['state']=='LIVE')
        assert source.latest().frame_id>previous
        previous=source.latest().frame_id
        source.request('live')
        source.request('replay')
        assert source.latest() is None
        wait_for(lambda:source.snapshot()['state']=='REPLAY')
        assert captures[-1].closed and source.latest() is None
    source.close()
    assert len(captures)==10 and not owner.is_alive()


def test_camera_error_requires_explicit_retry_and_old_generation_is_dropped():
    calls=[]
    def factory(index,**kwargs):
        calls.append(index)
        if len(calls)==1: raise OSError('injected unplugged camera')
        return Capture(index)
    source=SourceController(camera_index=0,camera_backend='ANY',capture_factory=factory)
    source.start();source.request('live')
    wait_for(lambda:source.snapshot()['state']=='ERROR')
    time.sleep(.08)
    assert len(calls)==1 and 'unplugged' in source.snapshot()['error']
    source.request('live')
    wait_for(lambda:source.snapshot()['state']=='LIVE')
    generation=source.latest().generation
    source.request('off')
    assert source.latest() is None
    wait_for(lambda:source.snapshot()['state']=='OFF')
    assert source.snapshot()['generation']>generation
    source.close()


def test_blocked_capture_does_not_spawn_second_thread_or_open_second_device():
    release=threading.Event();entered=threading.Event();opened=[]
    class Blocked(Capture):
        def read(self):
            entered.set();release.wait(2)
            return super().read()
    def factory(index,**kwargs): opened.append(Blocked(index));return opened[-1]
    source=SourceController(camera_index=0,camera_backend='ANY',capture_factory=factory)
    source.start();source.request('live');assert entered.wait(1)
    source.request('replay');source.request('live');source.start()
    source.close(join_timeout=.02)
    assert len(opened)==1 and source.snapshot()['state']=='ERROR'
    release.set();wait_for(lambda:not source._thread.is_alive())
    assert opened[0].closed


def test_tasks_categories_retained_and_driver_mirror_changes_pixels_once():
    images=[]
    categories=[SimpleNamespace(category_name='Right',score=.99)]
    result=SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=float(x),y=float(y),z=float(z)) for x,y,z in points()]],handedness=[categories])
    detector=SimpleNamespace(detect_for_video=lambda image,t:result,close=lambda:None)
    tracker=HandTracker(landmarker_factory=lambda:detector,image_factory=lambda rgb:images.append(rgb.copy()) or rgb)
    frame=np.zeros((2,4,3),np.uint8);frame[:,0,0]=100
    observation=tracker.detect(frame,1)
    assert observation.hands[0].handedness==observation.hands[0].raw_handedness=='Right'
    assert images[0][0,0,2]==100 and images[0][0,-1,2]==0
    tracker.close()


def test_source_driver_mirror_and_qt_preview_are_separate():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from src.ui.desktop import SignBridgeUI
    frame=np.zeros((40,60,3),np.uint8);frame[:,:30,2]=255
    class Pattern(Capture):
        def read(self):
            time.sleep(.01)
            return CapturedFrame(1,time.monotonic_ns()//1000000,frame.copy(),10.)
    source=SourceController(camera_index=0,camera_backend='ANY',capture_factory=Pattern,input_mirrored=True)
    source.start();source.request('live')
    wait_for(lambda:source.snapshot()['state']=='LIVE')
    pixels=source.latest().bgr
    assert pixels[10,-10,2]==255 and pixels[10,10,2]==0
    source.close()
    ui=SignBridgeUI(lambda *a:None)
    ui.window.show();ui._app.processEvents()
    ui.update(dict(source='live',camera_state='LIVE',generation=1,frame_id=1,image=frame))
    ui._drain();ui._app.processEvents()
    rendered=ui.camera.grab().toImage()
    # Sample inside the fitted video, not a letterbox margin after controls resize.
    video_width=min(rendered.width(),rendered.height()*frame.shape[1]/frame.shape[0])
    assert rendered.pixelColor(int(rendered.width()/2+video_width*.25),rendered.height()//2).red()>200
    assert rendered.pixelColor(int(rendered.width()/2-video_width*.25),rendered.height()//2).red()<30
    assert frame[10,10,2]==255  # Preview never mutates recognition pixels.
    assert not ui.engine.model().item(1).isEnabled()
    ui.window.close()


def test_recording_receipt_counts_restart_and_write_failure_retry(tmp_path,monkeypatch):
    recorder=DatasetRecorder(tmp_path,session_id='same-session')
    recorder.start('A')
    recorder.add(FrameObservation(0));recorder.add(obs(100));recorder.add(obs(140));recorder.add(obs(200))
    stats=recorder.snapshot()
    assert (stats['frames_seen'],stats['samples'],stats['valid_frames'],stats['rejected_frames'],stats['skipped_frames'])==(4,2,2,1,1)
    import src.vision.recording as recording
    original=recording.os.replace
    def fail(*a): raise PermissionError('injected disk denial')
    monkeypatch.setattr(recording.os,'replace',fail)
    with pytest.raises(PermissionError,match='disk denial'):recorder.finish()
    assert recorder.active and recorder.count==2 and 'disk denial' in recorder.snapshot()['save_error']
    monkeypatch.setattr(recording.os,'replace',original)
    path=recorder.finish();stats=recorder.snapshot()
    assert stats['saved_samples']==2 and stats['saved_path']==str(path.resolve()) and stats['save_ms']>=0
    assert not recorder.active and not stats['save_error']
    fresh=DatasetRecorder(tmp_path)
    assert not fresh.active and dataset_summary(tmp_path)['counts']['A']==2
    (tmp_path/'broken.npz').write_bytes(b'not an archive')
    summary=dataset_summary(tmp_path)
    assert summary['counts']['A']==2 and len(summary['errors'])==1


def args(tmp_path):
    return SimpleNamespace(camera=0,dwell_ms=350,data_dir=tmp_path/'samples',landmarker=tmp_path/'fake.task',
                           model=tmp_path/'model.npz',mute=True,replay=False,input_mirrored=False)


def test_actual_application_switch_record_save_and_newest_frame_under_slow_tracking(tmp_path,monkeypatch):
    import src.application as application
    monkeypatch.setattr(application,'ROOT',tmp_path)
    captures=[];events=[]
    def factory(index,**kwargs):
        c=Capture(index,1/30);captures.append(c);return c
    class SlowTracker:
        last_latency_ms=60.
        def __init__(self,*a):pass
        def detect(self,image,t):time.sleep(.06);return obs(t)
        def close(self):pass
    source=SourceController(camera_index=0,camera_backend='ANY',capture_factory=factory)
    app=ApplicationController(args(tmp_path),events.append,source_controller=source,tracker_factory=SlowTracker)
    app.start()
    wait_for(lambda:bool(events and events[-1]['physical_handedness']=='RIGHT'))
    app.action('record','A')
    wait_for(lambda:app.recorder.count>=4)
    app.action('finish_record')
    wait_for(lambda:bool(events[-1]['recording_stats']['saved_path']))
    saved=events[-1]['recording_stats']['saved_samples']
    assert saved>=4
    app.action('replay')
    wait_for(lambda:events[-1]['camera_state']=='REPLAY')
    assert captures[0].closed
    app.action('live')
    wait_for(lambda:events[-1]['camera_state']=='LIVE' and len(captures)==2)
    app.action('camera_off')
    wait_for(lambda:events[-1]['camera_state']=='OFF')
    app.close()
    assert all(c.closed for c in captures)
    assert dataset_summary(tmp_path/'samples')['counts']['A']==saved
    live=[e for e in events if e['camera_state']=='LIVE' and e['frame_id'] is not None]
    assert max(e['frame_age_ms'] for e in live)<120
    assert len(set(e['frame_id'] for e in live))>saved
    assert not app.worker.is_alive() and not app._inference.is_alive()
