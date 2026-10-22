"""Device identity tests use fake devices only; never open a physical webcam."""
import threading
import time
import numpy as np
import pytest
from src.camera_devices import device_record,identity,resolve_device
from src.camera_settings import read_settings
from src.source_controller import SourceController
from src.vision.camera import CapturedFrame


def catalog(ic_index=0,obs_index=1):
    return [device_record(ic_index,'IC800 1080P HD',r'\\?\usb#vid_0c45&pid_636b#serial'),
            device_record(obs_index,'OBS Virtual Camera'),device_record(2,'GoPro Webcam')]


def wait_for(check):
    until=time.perf_counter()+3
    while time.perf_counter()<until:
        if check():return
        time.sleep(.01)
    assert check()


class Capture:
    def __init__(self,index,*,backend,diagnostic):
        assert backend=='DSHOW'
        self.index,self.closed,self.count=index,False,0
        self.diagnostic=diagnostic
        diagnostic(event='opened',cap_is_opened=True,actual_backend=backend)
    def read(self):
        assert not self.closed
        time.sleep(.01);self.count+=1
        if self.count==1:self.diagnostic(event='first_frame',first_frame_success=True,frame_dimensions=[64,48])
        return CapturedFrame(self.count,time.monotonic_ns()//1000000,np.zeros((48,64,3),np.uint8),10.)
    def close(self):self.closed=True


def test_physical_default_and_identity_override_never_guess_virtual():
    devices=catalog(4,0)
    assert resolve_device(devices)['name']=='IC800 1080P HD'
    assert resolve_device(devices,identity(devices[1]))['name']=='IC800 1080P HD'
    assert resolve_device(devices,identity(devices[1]),explicit=True)['name']=='OBS Virtual Camera'
    with pytest.raises(ValueError,match='No physical'):
        resolve_device(devices[1:])
    preferred=identity(devices[0])
    others=[device_record(9,'A different physical webcam','usb#different')]+catalog(3,0)
    assert resolve_device(others,preferred)['index']==3
    with pytest.raises(ValueError,match='disconnected'):
        resolve_device(others[:1],preferred)


def test_duplicate_names_need_unique_identity():
    a=device_record(0,'USB Camera','usb#serial-one')
    b=device_record(1,'USB Camera','usb#serial-two')
    assert resolve_device([b,a],identity(b),explicit=True)['index']==1
    unknown=[device_record(0,'USB Camera'),device_record(1,'USB Camera')]
    with pytest.raises(ValueError,match='Multiple cameras'):
        resolve_device(unknown,identity(unknown[0]),explicit=True)


def test_identity_survives_index_reordering_replay_reopen_and_restart(tmp_path):
    devices=catalog();opened=[];settings=tmp_path/'camera.json'
    def factory(index,**kwargs):
        assert all(c.closed for c in opened)
        assert index==next(d['index'] for d in devices if d['name']=='IC800 1080P HD')
        capture=Capture(index,**kwargs);opened.append(capture);return capture
    source=SourceController(device_enumerator=lambda:list(devices),capture_factory=factory,
        selected_device=identity(devices[0]),settings_path=settings)
    source.start()
    try:
        for cycle in range(6):
            devices=catalog(cycle+3,0)
            source.request('live')
            wait_for(lambda:source.snapshot()['state']=='LIVE')
            assert source.snapshot()['selected_device']['name']=='IC800 1080P HD'
            assert source.snapshot()['selected_index']==cycle+3
            assert read_settings(settings)['selected_device']['device_id']==devices[0]['device_id']
            assert source.latest() is not None
            if cycle==0:
                previous=opened[-1];count=len(opened)
                source.request('live',restart=True)
                wait_for(lambda:len(opened)==count+1 and source.snapshot()['state']=='LIVE')
                assert previous.closed
            source.request('replay')
            wait_for(lambda:source.snapshot()['state']=='REPLAY')
            assert opened[-1].closed and source.latest() is None
        saved=read_settings(settings)['selected_device']
    finally:source.close()
    assert len(opened)==7 and all(c.closed for c in opened)
    devices=catalog(8,0)
    restarted=SourceController(device_enumerator=lambda:list(devices),capture_factory=factory,
        selected_device=saved,settings_path=settings)
    restarted.start();restarted.request('live')
    try:
        wait_for(lambda:restarted.snapshot()['state']=='LIVE')
        assert opened[-1].index==8
    finally:restarted.close()


def test_refresh_lists_new_devices_without_opening_or_changing_selection():
    devices=catalog();source=SourceController(device_enumerator=lambda:list(devices),
        capture_factory=lambda *a,**kw:pytest.fail('Rescan must not open a camera'))
    source.start()
    try:
        wait_for(lambda:len(source.snapshot()['devices'])==3)
        devices.append(device_record(3,'Integrated Camera','usb#integrated'))
        source.refresh_devices()
        wait_for(lambda:len(source.snapshot()['devices'])==4)
        assert source.snapshot()['state']=='OFF'
    finally:source.close()


def test_unplugged_selected_device_cannot_fall_back_to_obs(tmp_path):
    selected=identity(catalog()[0]);source=SourceController(device_enumerator=lambda:catalog()[1:],
        selected_device=selected,settings_path=tmp_path/'settings.json',
        capture_factory=lambda *a,**kw:pytest.fail('Must not fall back to another source'))
    source.start();source.request('live')
    try:
        wait_for(lambda:source.snapshot()['state']=='ERROR')
        assert 'disconnected' in source.snapshot()['error']
        assert source.latest() is None and not (tmp_path/'settings.json').exists()
    finally:source.close()


def test_switch_during_final_identity_scan_does_not_persist_abandoned_device(tmp_path):
    entered=threading.Event();gate=threading.Event();scans=[];opened=[]
    def enumerate_devices():
        scans.append(1)
        if len(scans)==3:entered.set();gate.wait(3)
        return catalog()
    def factory(index,**kwargs):
        capture=Capture(index,**kwargs);opened.append(capture);return capture
    settings=tmp_path/'settings.json'
    source=SourceController(device_enumerator=enumerate_devices,capture_factory=factory,settings_path=settings)
    source.start();source.request('live')
    try:
        assert entered.wait(1)
        source.request('replay');gate.set()
        wait_for(lambda:source.snapshot()['state']=='REPLAY')
        assert not settings.exists() and opened[0].closed and source.latest() is None
    finally:gate.set();source.close()


def test_device_order_change_during_open_is_rejected(tmp_path):
    scans=[];opened=[]
    def enumerate_devices():
        scans.append(1)
        return catalog() if len(scans)<3 else catalog(1,0)
    def factory(index,**kwargs):
        capture=Capture(index,**kwargs);opened.append(capture);return capture
    source=SourceController(device_enumerator=enumerate_devices,capture_factory=factory,
        settings_path=tmp_path/'settings.json')
    source.start();source.request('live')
    try:
        wait_for(lambda:source.snapshot()['state']=='ERROR')
        assert 'device order changed' in source.snapshot()['error']
        assert opened[0].closed and source.latest() is None
        assert not (tmp_path/'settings.json').exists()
    finally:source.close()
