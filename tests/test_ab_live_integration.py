"""Real trainer/classifier/decoder/UI; generated landmarks are numerical tests only."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.application import ApplicationController
from src.contracts import FrameObservation,HandObservation
from src.source_controller import SourceController
from src.vision.camera import CapturedFrame
from src.vision import DatasetRecorder,primary_features
from src.vision.dataset_summary import dataset_summary


def points(label):
    result=np.zeros((21,3),np.float32)
    result[:,0]=np.linspace(.3,.65,21)
    result[:,1]=np.linspace(.8,.3,21)
    if label=='B':
        result[[4,8,12,16,20],1]-=.5
        result[[4,8,12,16,20],2]=-.12
    return result


def test_train_button_reload_actual_classifier_live_decode_ui_and_class_reset(tmp_path,monkeypatch,capsys):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import src.application as application
    from src.ui.desktop import SignBridgeUI
    # Keep subprocess cwd at the project so module invocation stays identical to production.
    data=tmp_path/'samples'
    for session in ('numeric-s1','numeric-s2'):
        recorder=DatasetRecorder(data,session_id=session)
        for label in ('A','B'):
            recorder.start(label)
            for n in range(50):
                recorder.add(FrameObservation(n*100,[HandObservation(points(label))]))
            recorder.finish()
    # A partially collected next letter must not block or enter A/B retraining.
    recorder.start('C')
    for n in range(12):recorder.add(FrameObservation(n*100,[HandObservation(points('A'))]))
    c_path=recorder.finish();c_bytes=c_path.read_bytes()
    selection=[points('A')];opened=[]
    class Capture:
        def __init__(self,index,**kwargs):self.closed=False;self.n=0;opened.append(self)
        def read(self):
            time.sleep(.02);self.n+=1
            return CapturedFrame(self.n,time.monotonic_ns()//1000000,np.zeros((48,64,3),np.uint8),20.)
        def close(self):self.closed=True
    class Tracker:
        last_latency_ms=1.
        def __init__(self,*args):pass
        def detect(self,pixels,timestamp):
            return FrameObservation(timestamp,[] if selection[0] is None else [HandObservation(selection[0])])
        def close(self):pass
    args=SimpleNamespace(camera=0,dwell_ms=350,data_dir=data,landmarker=tmp_path/'unused',
        model=tmp_path/'ab.npz',mute=True,replay=False,input_mirrored=False)
    source=SourceController(camera_index=0,camera_backend='DSHOW',capture_factory=Capture)
    app=None
    ui=SignBridgeUI(lambda action,value:app.action(action,value) if app else None)
    app=ApplicationController(args,ui.update,source_controller=source,tracker_factory=Tracker)
    # Explicitly mark the generated fixture in the real training subprocess.
    popen=application.subprocess.Popen
    def fixture_popen(command,**kwargs):
        command=list(command)+['--data-origin','synthetic_fixture','--provenance','Generated A/B numeric integration fixture; not ASL recognition evidence.']
        return popen(command,**kwargs)
    monkeypatch.setattr(application.subprocess,'Popen',fixture_popen)
    app.profile_path=tmp_path/'profile.json'
    ui.window.show();app.start()
    def wait(check,timeout=15):
        until=time.perf_counter()+timeout
        while time.perf_counter()<until:
            ui._app.processEvents()
            if check():return
            time.sleep(.01)
        assert check(),app.calibration_status
    def pump(seconds):
        until=time.perf_counter()+seconds
        while time.perf_counter()<until:ui._app.processEvents();time.sleep(.01)
    try:
        wait(lambda:ui.train.isEnabled())
        assert not args.model.exists() and app.cpu is None
        ui.train.click()
        wait(lambda:app.cpu is not None and 'Model trained successfully' in app.calibration_status)
        assert app.cpu.metadata['synthetic_fixture'] is True
        assert app.cpu.supported_labels==('A','B') and app.mode=='free_spell'
        assert args.model.exists() and args.model.with_name('ab_classes.json').exists()
        wait(lambda:ui.word.text()=='A' and 'Candidate: A' in ui.live_candidate.text())
        assert 'Model loaded: YES' in ui.recognition_debug.text()
        assert 'Model classes: A, B' in ui.recognition_debug.text()
        assert app.prediction.source=='cpu' and np.isclose(app.prediction.confidence,max(app.prediction.scores))
        pump(.5);assert ui.word.text()=='A'  # Held pose does not repeat.
        selection[0]=points('B')
        wait(lambda:ui.word.text()=='AB' and 'Candidate: B' in ui.live_candidate.text())
        # Find an actual model-uncertain interpolation; do not inject predictions.
        candidates=[points('A')*(1-a)+points('B')*a for a in np.linspace(0,1,101)]
        weak=min(candidates,key=lambda p:app.cpu.predict(primary_features(FrameObservation(1,[HandObservation(p)])),1).confidence)
        assert app.cpu.predict(primary_features(FrameObservation(1,[HandObservation(weak)])),1).confidence<.8
        selection[0]=weak
        wait(lambda:'Candidate: —' in ui.live_candidate.text() and app.prediction is not None and app.prediction.confidence<.8)
        pump(.3);assert ui.word.text()=='AB'
        selection[0]=np.full((21,3),np.nan,np.float32)
        pump(.3);assert ui.word.text()=='AB' and app.worker.is_alive()
        assert 'paintEvent' not in capsys.readouterr().err
        selection[0]=points('B');pump(.5);assert ui.word.text()=='AB'
        selection[0]=None
        wait(lambda:'Hand detected: NO' in ui.recognition_debug.text());pump(.3)
        selection[0]=points('B');wait(lambda:ui.word.text()=='ABB')
        assert app.inference_count>20
        assert 'Validation confusion matrix' in ui.training_result_text.toPlainText()
        assert 'Training samples: 100' in ui.training_result_text.toPlainText()
        assert c_path.read_bytes()==c_bytes
        # Real reversible reset and stale-model rejection, confined to fixture data.
        app.action('reset_class','A')
        # Poll the controller's completed receipt, not files being moved on Windows.
        wait(lambda:app.dataset['counts'].get('A')==0 and app.cpu is None)
        assert len(list((tmp_path/'removed_samples').rglob('*.npz')))==2
        assert dataset_summary(data)['counts']['B']==100
        assert not app._load_models() and 'reset or changed' in app.model_error
    finally:
        ui.window.close();app.close()
    assert all(c.closed for c in opened)
    assert json.loads((tmp_path/'profile.json').read_text())['measurements']['model_ms']['samples']>20
