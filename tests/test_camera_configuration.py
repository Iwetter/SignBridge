"""Exercise exact backend selection at the OpenCV boundary; never open a device."""
import json
import os
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from src.camera_settings import read_settings, resolve_selection, update_settings
from src.source_controller import SourceController
from src.vision.camera import CameraCapture, CameraError


def wait_for(check):
    limit=time.perf_counter()+3
    while time.perf_counter()<limit:
        if check():return
        time.sleep(.01)
    assert check()


class Device:
    def __init__(self,opened=True,first=True,gate=None):
        self.opened,self.first,self.gate=opened,first,gate
        self.closed=False;self.sets=[];self.entered=threading.Event()
    def isOpened(self):return self.opened
    def getBackendName(self):return 'DSHOW'
    def set(self,*args):self.sets.append(args);return False
    def read(self):
        self.entered.set()
        if self.gate:self.gate.wait(3)
        time.sleep(.02)
        return (True,np.zeros((1080,1920,3),np.uint8)) if self.first else (False,None)
    def release(self):self.closed=True


CV=SimpleNamespace(CAP_DSHOW=700,CAP_MSMF=1400,CAP_ANY=0,
                   CAP_PROP_FRAME_WIDTH=3,CAP_PROP_FRAME_HEIGHT=4,CAP_PROP_BUFFERSIZE=38)


def factory_for(devices,calls):
    def factory(index,**kwargs):
        device=devices.pop(0)
        def open_device(selected,api):calls.append((selected,api));return device
        return CameraCapture(index,cv2_module=CV,capture_factory=open_device,**kwargs)
    return factory


def test_no_assumed_camera_and_saved_selection_preserves_mirror(tmp_path):
    path=tmp_path/'settings.json'
    assert resolve_selection(read_settings(path))==(None,None)
    for index,backend in [(1,None),(None,'DSHOW'),(-1,'ANY'),(1,'invalid')]:
        with pytest.raises(ValueError):resolve_selection({},index,backend)
    with pytest.raises(ValueError):resolve_selection({'last_known_working':[]})
    update_settings(path,{'input_mirrored':True})
    update_settings(path,{'last_known_working':{'index':1,'backend':'DSHOW'}})
    assert resolve_selection(read_settings(path))==(1,'DSHOW')
    assert read_settings(path)['input_mirrored'] is True
    assert resolve_selection(read_settings(path),2,'MSMF')==(2,'MSMF')
    source=SourceController(capture_factory=lambda *a,**kw:pytest.fail('Must not guess a device'))
    source.start();source.request('live')
    try:
        wait_for(lambda:source.snapshot()['state']=='ERROR')
        assert 'tested camera index' in source.snapshot()['error']
        assert source.latest() is None
    finally:source.close()


def test_exact_pair_first_frame_gate_persistence_and_failed_replacement(tmp_path):
    path=tmp_path/'settings.json';log=tmp_path/'camera.jsonl'
    update_settings(path,{'input_mirrored':True})
    gate=threading.Event();device=Device(gate=gate);failed=Device(opened=False);calls=[]
    source=SourceController(camera_index=1,camera_backend='DSHOW',settings_path=path,log_path=log,
        capture_factory=factory_for([device,failed],calls))
    source.start();source.request('live')
    try:
        assert device.entered.wait(1)
        assert source.snapshot()['state']=='STARTING' and source.latest() is None
        assert 'last_known_working' not in read_settings(path)
        gate.set();wait_for(lambda:source.snapshot()['state']=='LIVE')
        assert calls==[(1,CV.CAP_DSHOW)]
        assert not any(prop in (3,4) for prop,value in device.sets)  # Probe's native size retained.
        assert read_settings(path)['last_known_working']['frame_dimensions']==[1920,1080]
        assert read_settings(path)['input_mirrored'] is True
        source.configure(2,'MSMF')
        wait_for(lambda:source.snapshot()['state']=='ERROR')
        assert device.closed and failed.closed and source.latest() is None
        assert 'cap.isOpened() returned False' in source.snapshot()['error']
        assert resolve_selection(read_settings(path))==(1,'DSHOW')
        records=[json.loads(line) for line in log.read_text().splitlines()]
        assert any(r.get('cap_is_opened') is True and r['selected_index']==1 and r['selected_backend']=='DSHOW' for r in records)
        assert any(r.get('first_frame_success') is True and r['frame_dimensions']==[1920,1080] for r in records)
        assert any(r.get('cap_is_opened') is False and r['selected_backend']=='MSMF' for r in records)
    finally:gate.set();source.close()


def test_opened_device_without_a_first_frame_never_live_or_saved(tmp_path):
    path=tmp_path/'settings.json';log=tmp_path/'camera.jsonl';device=Device(first=False)
    source=SourceController(camera_index=1,camera_backend='DSHOW',settings_path=path,log_path=log,
        capture_factory=factory_for([device],[]))
    source.start();source.request('live')
    try:
        wait_for(lambda:source.snapshot()['state']=='ERROR')
        assert 'first frame failed' in source.snapshot()['error']
        assert not path.exists() and source.latest() is None and device.closed
        details=source.snapshot()['diagnostics']
        assert details['cap_is_opened'] is True and details['first_frame_success'] is False
    finally:source.close()


def test_app_settings_selector_stays_responsive_during_blocked_open(tmp_path,monkeypatch):
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    import src.application as application
    from src.ui.desktop import SignBridgeUI
    from src.contracts import FrameObservation
    monkeypatch.setattr(application,'ROOT',tmp_path)
    from src.camera_devices import device_record
    catalog=[device_record(1,'IC800 1080P HD','usb#ic800'),device_record(3,'Second webcam','usb#second')]
    gate=threading.Event();entered=threading.Event();calls=[]
    def factory(index,**kwargs):
        entered.set();gate.wait(3)
        return factory_for([Device()],calls)(index,**kwargs)
    source=SourceController(camera_index=1,camera_backend='DSHOW',capture_factory=factory,
        settings_path=tmp_path/'settings.json',log_path=tmp_path/'camera.jsonl',device_enumerator=lambda:catalog)
    args=SimpleNamespace(camera=1,camera_backend='DSHOW',dwell_ms=350,data_dir=tmp_path/'samples',
        landmarker=tmp_path/'unused.task',model=tmp_path/'unused.npz',mute=True,replay=False,input_mirrored=False)
    app=None
    ui=SignBridgeUI(lambda action,value:app.action(action,value) if app else None)
    class Tracker:
        last_latency_ms=1
        def __init__(self,*a):pass
        def detect(self,pixels,timestamp):return FrameObservation(timestamp)
        def close(self):pass
    app=application.ApplicationController(args,ui.update,source_controller=source,tracker_factory=Tracker)
    ui.window.show();app.start()
    try:
        assert entered.wait(1)
        deadline=time.perf_counter()+.2;updates=0
        while time.perf_counter()<deadline:
            ui._app.processEvents();updates+=1;time.sleep(.005)
        assert ui.source_badge.text()=='STARTING' and updates>10
        assert ui.camera_devices.currentText()=='IC800 1080P HD'
        assert 'Backend: DSHOW' in ui.camera_diagnostics.text()
        ui.camera_settings.show();ui._app.processEvents()
        assert ui.camera_settings.isVisible()
        ui.camera_devices.setCurrentIndex(ui.camera_devices.findData(catalog[1]['device_id']))
        ui.camera_apply.click()
        wait_for(lambda:source.snapshot()['selected_device'].get('device_id')==catalog[1]['device_id'])
        assert source.snapshot()['selected_backend']=='DSHOW'
        # Return to the working pair before releasing this intentionally blocked open.
        app.action('camera_config',{'device_id':catalog[0]['device_id']})
        wait_for(lambda:source.snapshot()['selected_device'].get('device_id')==catalog[0]['device_id'])
        gate.set();wait_for(lambda:source.snapshot()['state']=='LIVE')
        wait_for(lambda:ui._latest and ui._latest['camera_state']=='LIVE')
        ui._drain();ui._app.processEvents()
        assert ui.camera._image is not None
        assert ui.camera._image.width()==1920 and ui.camera._image.height()==1080
        assert 'First frame success: True' in ui.camera_diagnostics.text()
        assert 'Device: IC800 1080P HD' in ui.camera_diagnostics.text()
        assert 'Capture FPS:' in ui.camera_diagnostics.text() and 'Display FPS:' in ui.camera_diagnostics.text()
        assert app.source=='live'
    finally:
        gate.set();ui.window.close();app.close()
