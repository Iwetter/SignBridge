"""One capture owner and one latest-frame slot, independent of inference/UI."""
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import uuid
import numpy as np
from src.vision.camera import CameraCapture
from src.camera_settings import validate_selection, update_settings
from src.camera_devices import resolve_device, identity


class SourceController:
    def __init__(self, camera_index=None, capture_factory=None, input_mirrored=False,
                 camera_backend=None, settings_path=None, log_path=None,
                 device_enumerator=None, selected_device=None):
        self.camera_index = camera_index
        self.camera_backend = camera_backend
        self.settings_path = settings_path
        self.log_path = Path(log_path) if log_path else None
        self._factory = capture_factory or CameraCapture
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._desired = 'off'
        self._generation = 0
        self._state = 'OFF'
        self._error = ''
        self._latest = None
        self._counter = 0
        self._mirror = bool(input_mirrored)
        self._times = deque(maxlen=60)
        self._capture_ms = None
        self._diagnostics = {}
        self._settings_error = ''
        self._device_enumerator = device_enumerator
        self._requested_device = dict(selected_device or {})
        self._resolved_device = {}
        self._explicit_device = False
        self._devices = []
        self._device_error = ''
        self._rescan_requested = bool(device_enumerator)
        self._scanning = False

    def refresh_devices(self):
        with self._lock:
            self._rescan_requested = True
        self._wake.set()

    def select_device(self, device_id):
        if not isinstance(device_id,str) or not device_id:
            raise ValueError('Choose a camera by name.')
        with self._lock:
            if self._stop.is_set():return
            matching=[d for d in self._devices if d['device_id']==device_id]
            self._requested_device = identity(matching[0]) if matching else {'device_id':device_id}
            self._explicit_device = True
            self._resolved_device = {}
            self.camera_index,self.camera_backend = None,'DSHOW'
            self._desired = 'live'
            self._generation += 1
            self._latest = None
            self._times.clear()
            self._capture_ms = None
            self._state,self._error = 'STARTING',''
            self._diagnostics = {}
        self._wake.set()

    def _scan_devices(self):
        with self._lock:self._scanning=True
        try:
            devices=list(self._device_enumerator())
            with self._lock:
                self._devices,self._device_error=devices,''
            return devices
        except Exception as exc:
            with self._lock:
                self._devices=[]
                self._device_error=f'Camera enumeration failed: {exc}'
            raise
        finally:
            with self._lock:self._scanning=False

    def configure(self, index, backend):
        if self._device_enumerator:
            raise ValueError('Select a named camera in Settings / Diagnostics; numeric selection is disabled.')
        validate_selection(index, backend)
        with self._lock:
            if self._stop.is_set():
                return
            self.camera_index, self.camera_backend = index, backend
            self._desired = 'live'
            self._generation += 1
            self._latest = None
            self._times.clear()
            self._capture_ms = None
            self._state, self._error = 'STARTING', ''
            self._settings_error = ''
            self._diagnostics = {}
        self._wake.set()

    def _diagnostic(self, generation, attempt, index, backend, **fields):
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), attempt=attempt,
                      selected_index=index, selected_backend=backend, **fields)
        with self._lock:
            if generation == self._generation:
                self._diagnostics.update(record)
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record) + '\n')
            except OSError as exc:
                with self._lock:
                    self._settings_error = f'Camera log could not be written: {exc}'

    def start(self):
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name='SignBridge capture', daemon=True)
            self._thread.start()

    def request(self, mode, *, restart=False):
        if mode not in ('live', 'replay', 'off'):
            raise ValueError('Source must be live, replay or off')
        with self._lock:
            if self._stop.is_set():
                return
            if mode == self._desired and self._state != 'ERROR' and not restart:
                return
            self._desired = mode
            self._generation += 1
            self._latest = None
            self._times.clear()
            self._capture_ms = None
            self._error = ''
            self._settings_error = ''
            self._state = 'STARTING'
            self._diagnostics = {}
        self._wake.set()

    def set_input_mirrored(self, value):
        with self._lock:
            if self._mirror == bool(value):
                return
            self._mirror = bool(value)
            self._generation += 1
            self._latest = None
            self._times.clear()
            self._state = 'STARTING'
        self._wake.set()

    def snapshot(self):
        with self._lock:
            fps = ((len(self._times)-1)/(self._times[-1]-self._times[0])
                   if len(self._times) >= 2 and self._times[-1] > self._times[0] else None)
            return dict(state=self._state, error=self._error, generation=self._generation,
                        capture_fps=fps, capture_ms=self._capture_ms, input_mirrored=self._mirror,
                        selected_index=self.camera_index, selected_backend=self.camera_backend,
                        selected_device=dict(self._resolved_device or self._requested_device),
                        devices=[dict(d) for d in self._devices],scanning=self._scanning,
                        device_error=self._device_error,
                        diagnostics=dict(self._diagnostics), settings_error=self._settings_error,
                        log_path=str(self.log_path.resolve()) if self.log_path else '')

    def latest(self, after_id=None):
        with self._lock:
            return self._latest if self._latest and self._latest.frame_id != after_id else None

    def _publish_state(self, state, generation, error=''):
        with self._lock:
            if generation == self._generation:
                self._state, self._error = state, error
                if state != 'LIVE':
                    self._latest = None

    def _run(self):
        capture, opened_generation, report, remembered = None, None, None, False
        opened_device = None
        try:
            while not self._stop.is_set():
                with self._lock:
                    mode, generation, mirror = self._desired, self._generation, self._mirror
                    index, backend = self.camera_index, self.camera_backend
                    preferred,explicit = dict(self._requested_device),self._explicit_device
                    rescan,self._rescan_requested=self._rescan_requested,False
                if rescan and self._device_enumerator:
                    try:self._scan_devices()
                    except Exception:pass  # Keep a running capture; expose refresh errors separately.
                if capture is not None and (mode != 'live' or generation != opened_generation):
                    capture.close()
                    capture = None
                if mode != 'live':
                    self._publish_state('REPLAY' if mode == 'replay' else 'OFF', generation)
                    self._wake.wait(.03)
                    self._wake.clear()
                    continue
                if capture is None:
                    # Error is sticky until an explicit new request; no reopen loop.
                    if self.snapshot()['state'] == 'ERROR':
                        self._wake.wait(.03)
                        self._wake.clear()
                        continue
                    try:
                        attempt = uuid.uuid4().hex[:12]
                        opened_device = None
                        # Define a reporter before discovery so discovery failures are logged too.
                        def report(_generation=generation, _attempt=attempt, _index=index, _backend=backend, **fields):
                            self._diagnostic(_generation, _attempt, _index, _backend, **fields)
                        if self._device_enumerator:
                            opened_device = resolve_device(self._scan_devices(),preferred,explicit=explicit)
                            index,backend=opened_device['index'],'DSHOW'
                            with self._lock:
                                if generation!=self._generation:continue
                                self.camera_index,self.camera_backend=index,backend
                                self._resolved_device=dict(opened_device)
                            def report(_generation=generation, _attempt=attempt, _index=index, _backend=backend,
                                       _device=dict(opened_device), **fields):
                                self._diagnostic(_generation,_attempt,_index,_backend,
                                    device_name=_device['name'],device_id=_device['device_id'],**fields)
                        report(event='opening', cap_is_opened=None, first_frame_success=None, frame_dimensions=None)
                        validate_selection(index, backend)
                        capture = self._factory(index, backend=backend, diagnostic=report)
                        opened_generation = generation
                        remembered = False
                    except Exception as exc:
                        report(event='startup_error', error=str(exc))
                        self._publish_state('ERROR', generation, str(exc))
                        continue
                try:
                    frame = capture.read()
                    pixels = np.ascontiguousarray(frame.bgr[:, ::-1]) if mirror else frame.bgr
                    pixels.flags.writeable = False
                    # Only this worker persists a selection, and only after a real read.
                    # A result from an abandoned selection must never replace the saved pair.
                    with self._lock:
                        current = generation == self._generation and self._desired == 'live'
                    if not current:
                        continue
                    if not remembered:
                        if opened_device:
                            # Detect device-list changes during OpenCV's index-based open.
                            after = self._scan_devices()
                            at_index = [d for d in after if d['index']==index]
                            if len(at_index)!=1 or at_index[0]['device_id']!=opened_device['device_id']:
                                raise RuntimeError('Camera device order changed during startup. Capture released; refresh and retry the named camera.')
                        remembered = True
                        if self.settings_path:
                            try:
                                saved = dict(
                                    index=index, backend=backend, frame_dimensions=[int(pixels.shape[1]),int(pixels.shape[0])],
                                    verified_at=datetime.now(timezone.utc).isoformat(), evidence='SignBridge first frame')
                                changes={'last_known_working':saved}
                                if opened_device:
                                    saved.update(identity(opened_device))
                                    changes['selected_device']=identity(opened_device)
                                # Selection changes wait for this small atomic settings update;
                                # an abandoned generation cannot overwrite the saved identity.
                                with self._lock:
                                    if generation!=self._generation or self._desired!='live':
                                        continue
                                    update_settings(self.settings_path,changes)
                            except Exception as exc:
                                with self._lock:
                                    self._settings_error = f'Camera works, but its settings could not be saved: {exc}'
                                report(event='settings_error', error=str(exc))
                            else:
                                report(event='configuration_saved')
                        if opened_device:
                            with self._lock:
                                if generation==self._generation:
                                    self._requested_device=identity(opened_device)
                    with self._lock:
                        if generation != self._generation or self._desired != 'live':
                            continue
                        self._counter += 1
                        self._latest = replace(frame, frame_id=self._counter, generation=generation, bgr=pixels)
                        self._times.append(time.perf_counter())
                        self._capture_ms = frame.capture_ms
                        self._state, self._error = 'LIVE', ''
                except Exception as exc:
                    capture.close()
                    capture = None
                    report(event='capture_error', error=str(exc))
                    self._publish_state('ERROR', generation, str(exc))
        finally:
            if capture is not None:
                capture.close()
            with self._lock:
                self._latest = None
                self._state = 'OFF'

    def close(self, join_timeout=.5):
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(join_timeout)
        if thread and thread.is_alive():
            with self._lock:
                self._state = 'ERROR'
                self._error = 'Camera driver is still closing. No second capture will be opened.'
