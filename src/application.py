"""App integration: capture, inference, UI and persistence have separate owners."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import json
from hashlib import sha256
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

from src.contracts import ALPHABET, STATIC_LABELS, Prediction
from src.performance import PerformanceLog
from src.source_controller import SourceController
from src.camera_settings import update_settings
from src.camera_devices import enumerate_dshow_devices

ROOT = Path(__file__).resolve().parents[1]


def rehearsal_predictions(word='IAN'):
    timestamp = 0
    for label in ['MY', 'NAME', *word, 'YOUR', 'NAME', 'WHAT']:
        for _ in range(16):
            yield Prediction(label,.95,timestamp,source='replay'), timestamp
            timestamp += 40
        for _ in range(8):
            yield None,timestamp
            timestamp += 40


def run_headless_replay(word='IAN'):
    from src.temporal import TemporalDecoder
    from src.conversation import ConversationEngine
    decoder, conversation = TemporalDecoder(), ConversationEngine()
    committed = []
    for prediction,timestamp in rehearsal_predictions(word):
        token = decoder.update(prediction,timestamp)
        if token:
            committed.append(token.label)
            conversation.consume(token)
    return dict(source='synthetic prediction replay; no camera/model/FPGA used',committed=committed,**conversation.snapshot())


class ApplicationController:
    def __init__(self,args,emit,*,source_controller=None,tracker_factory=None):
        from src.conversation import ConversationEngine, SpeechWorker
        from src.temporal import TemporalDecoder, MotionRecognizer
        from src.vision import DatasetRecorder
        from src.vision.camera import HandTracker
        self.args,self.emit = args,emit
        self.commands = queue.Queue(maxsize=64)
        self.stopping = threading.Event()
        self.conversation = ConversationEngine()
        self.decoder = TemporalDecoder(dwell_ms=args.dwell_ms)
        self.motion = MotionRecognizer(args.data_dir)
        from src.fingerspelling_settings import read_settings, DEFAULTS
        from src.temporal.fingerspelling import FingerspellingCoordinator
        self.fingerspelling_settings_path = getattr(args, 'fingerspelling_settings_path', Path(args.data_dir).parent / 'fingerspelling_settings.json')
        self.fingerspelling_settings_error = ''
        try:
            self.fingerspelling_settings = read_settings(self.fingerspelling_settings_path)
        except (OSError, ValueError) as exc:
            self.fingerspelling_settings = dict(DEFAULTS)
            self.fingerspelling_settings_error = str(exc)
        self.motion_profile_path = Path(args.model).with_name('motion_calibration.json')
        self.motion_profile = None
        self.motion_profile_status = 'J/Z uses default geometry. Save clips, then Calibrate J/Z.'
        self.motion_calibration_report = {}
        self.continuous = FingerspellingCoordinator(self.decoder, self.fingerspelling_settings)
        self.speech = SpeechWorker() if not args.mute else None
        self.recorder = DatasetRecorder(args.data_dir,session_id=uuid.uuid4().hex[:12])
        self.camera_settings_path = getattr(args,'camera_settings_path',ROOT/'data/camera_settings.json')
        self.sources = source_controller or SourceController(args.camera,input_mirrored=getattr(args,'input_mirrored',False),
            camera_backend='DSHOW',settings_path=self.camera_settings_path,
            log_path=ROOT/'work/camera-diagnostics.jsonl',device_enumerator=enumerate_dshow_devices,
            selected_device=getattr(args,'camera_device',None))
        self._tracker_factory = tracker_factory or HandTracker
        self._results = queue.Queue(maxsize=1)
        self._inference = threading.Thread(target=self._infer_loop,name='SignBridge landmarks',daemon=True)
        self.worker = threading.Thread(target=self._run,name='SignBridge coordinator',daemon=True)
        self._executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='SignBridge storage')
        self._saving = None
        self._save_failed = False
        self._frozen_stats = None
        self._dataset_future = None
        self._maintenance_future = None
        self.dataset = dict(counts={},files=[],total_samples=0,legacy_samples=0,errors=[])
        self.cpu = self.fpga = None
        self.engine,self.source,self.mode = 'CPU','replay' if args.replay else 'live','conversation'
        self.status,self.calibration_status = '', 'Choose a label and record your hand.'
        self.last_response,self.stable = '',None
        self.prediction = None
        self.recognition_error = ''
        self.recognition_reason = 'Model not loaded'
        self.training_report = {}
        self.model_error = ''
        self.inference_count = 0
        self.last_observation = None
        self.observation_frame_id = None
        self.observation_generation = None
        self.cpu_latency = self.fpga_latency = self.landmark_ms = self.motion_ms = None
        self.tracker_error = ''
        self.replay_iter = None
        self._next_replay = 0
        self.training_process = self.training_log = None
        self.job_kind,self.job_started,self.job_progress = '',None,''
        self.ui_metrics = {}
        self.metrics_visible = True
        self.performance = PerformanceLog()
        self.profile_path = ROOT / 'work' / f'performance-live-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}.json'
        self._last_metric_sample = 0
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self.sources.start()
        self.sources.request(self.source)
        self._inference.start()
        self.worker.start()

    def action(self,name,value=None):
        if name == 'stop':
            self.stopping.set()
            return
        try:
            self.commands.put_nowait((name,value))
        except queue.Full:
            # GUI input remains responsive; repeated nonessential clicks cannot grow memory.
            pass

    def close(self):
        self.stopping.set()
        self.sources.close()
        if self._started:
            self.worker.join(timeout=5)
            self._inference.join(timeout=1)
        self._executor.shutdown(wait=True)
        if self.speech:
            self.speech.close()

    def _load_models(self):
        from src.recognition import CPURecognizer
        from src.temporal import MotionRecognizer
        try:
            model = CPURecognizer(self.args.model)
            # Resetting or replacing training files invalidates the old model in the app.
            for entry in model.metadata.get('training_provenance',{}).get('files',[]):
                path=Path(self.args.data_dir)/entry['path']
                if not path.is_file() or sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
                    raise ValueError('Training samples were reset or changed. Train again before live recognition.')
            self.cpu = model
            self.model_error = ''
            self.recognition_error = ''
            self.recognition_reason = 'Waiting for live hand tracking'
            self.status = 'Model loaded: '+', '.join(self.cpu.supported_labels)
            report_path=Path(self.args.model).with_name(Path(self.args.model).stem+'_metrics.json')
            try:self.training_report=json.loads(report_path.read_text(encoding='utf-8'))
            except (OSError,ValueError):self.training_report={}
        except (ValueError,OSError) as exc:
            self.cpu = None
            self.model_error = str(exc)
            self.recognition_reason = 'Model not loaded'
            self.status = 'No trained model. Collect at least two READY letters, then Train.' if isinstance(exc,FileNotFoundError) else f'Model unavailable: {exc}'
        self.motion = MotionRecognizer(self.args.data_dir)
        self._load_motion_profile()
        self.continuous.reset()
        self.prediction = None
        if self.cpu and set(self.cpu.supported_labels)<=set(STATIC_LABELS) and self.source!='replay':
            self.mode='free_spell'
            self.conversation.set_mode(self.mode)
        return self.cpu is not None

    def _load_motion_profile(self):
        from src.temporal.motion_calibration import load_profile
        from src.temporal.fingerspelling import FingerspellingCoordinator
        if not self.motion_profile_path.exists():
            self.motion_profile_status = ('J/Z calibration file is missing. Previous loaded calibration retained.' if self.motion_profile else
                'No saved J/Z calibration. Default geometry is active; save clips, then Calibrate J/Z.')
            return False
        try:
            profile = load_profile(self.motion_profile_path)
            for entry in profile.get('files', []):
                path = Path(self.args.data_dir) / entry['path']
                if not path.is_file() or sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                    raise ValueError('A calibrated recording changed or is missing. Calibrate J/Z again.')
            report_path = self.motion_profile_path.with_name(self.motion_profile_path.stem + '_report.json')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report.get('profile_id') != profile['id']:
                raise ValueError('J/Z report and configuration do not match. Calibrate again.')
            self.motion_profile = profile
            self.motion_calibration_report = report
            locked = self.decoder.locked_label
            self.continuous = FingerspellingCoordinator(self.decoder, self.fingerspelling_settings, profile)
            self.decoder.locked_label = locked
            counts = report['per_class']
            self.motion_profile_status = ('Loaded personal J/Z calibration · ' + ', '.join(
                f"{c}: {counts[c]['usable_clips']}/{counts[c]['saved_clips']} usable clips" for c in ('J','Z')) +
                ' · ' + report.get('warning', report['scope']))
            return True
        except (ValueError, OSError, KeyError) as exc:
            self.motion_profile_status = f'J/Z calibration load failed: {exc}. ' + (
                'Previous loaded calibration retained.' if self.motion_profile else 'Default geometry remains active.')
            return False

    def _refresh_dataset(self):
        from src.vision.dataset_summary import dataset_summary
        from src.temporal import MotionRecognizer
        def refresh():
            return dataset_summary(self.args.data_dir),MotionRecognizer(self.args.data_dir)
        self._dataset_future = self._executor.submit(refresh)

    def _reset(self):
        self.continuous.reset()
        self.motion.reset()
        self.conversation.reset()
        self.prediction = self.stable = self.last_observation = None
        self.recognition_error = ''
        self.recognition_reason = ''
        self.last_response = ''
        self.observation_frame_id = self.observation_generation = None

    def _transition(self,mode):
        if self.recorder.active and not self._saving:
            self._finish_record()
        self.source = mode
        self._reset()
        self.cpu_latency = self.fpga_latency = self.landmark_ms = None
        self.sources.request(mode,restart=(mode=='live'))
        if mode == 'replay':
            self.mode = 'conversation'
            self.conversation.set_mode(self.mode)
            self.replay_iter = iter(rehearsal_predictions())
            self._next_replay = time.perf_counter()
        else:
            self.replay_iter = None
            if mode=='live' and self.cpu and set(self.cpu.supported_labels)<=set(STATIC_LABELS):
                self.mode='free_spell'
                self.conversation.set_mode(self.mode)

    def _commands(self):
        while True:
            try:
                name,value = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                if name in ('live','replay','camera_off'):
                    self._transition('off' if name == 'camera_off' else name)
                elif name == 'clear':
                    self._reset()
                elif name == 'space':
                    self.continuous.cancel_motion_promotion()
                    self.conversation.insert_space()
                elif name == 'backspace':
                    self.continuous.cancel_motion_promotion()
                    self.conversation.backspace()
                elif name == 'fingerspelling_settings':
                    from src.fingerspelling_settings import write_settings
                    from src.temporal.fingerspelling import FingerspellingCoordinator
                    updated = write_settings(self.fingerspelling_settings_path, value)
                    locked = self.decoder.locked_label
                    self.continuous = FingerspellingCoordinator(self.decoder, updated, self.motion_profile)
                    self.decoder.locked_label = locked
                    self.fingerspelling_settings = updated
                    self.fingerspelling_settings_error = ''
                    self.calibration_status = 'Fingerspelling settings saved.'
                elif name == 'mode':
                    self.mode = str(value)
                    self.conversation.set_mode(self.mode)
                    self._reset()
                elif name == 'record':
                    if self.sources.snapshot()['state'] != 'LIVE' or self._saving or self.recorder.active or self.tracker_error or self.training_process or self._maintenance_future:
                        self.calibration_status = 'Recording requires a live camera and hand tracker; finish any pending save first.'
                    else:
                        self._reset()
                        self.recorder.start(str(value),int(time.monotonic()*1000))
                        self._save_failed = False
                        self.calibration_status = f'Recording {value}. Stop saves the samples.'
                elif name == 'finish_record':
                    self._finish_record()
                elif name == 'new_session':
                    if self.recorder.active or self._saving or self.training_process or self._maintenance_future:
                        self.calibration_status = 'Stop and finish saving before starting a new session.'
                    else:
                        self.recorder.session_id = uuid.uuid4().hex[:12]
                        self.calibration_status = 'New session. Record each label again after a short break.'
                elif name == 'input_mirrored':
                    if self.recorder.active or self._saving:
                        self.calibration_status = 'Finish saving before changing the camera input convention.'
                    else:
                        self.sources.set_input_mirrored(bool(value))
                        update_settings(self.camera_settings_path,{'input_mirrored':bool(value)})
                        self._reset()
                elif name == 'camera_config':
                    if self.recorder.active or self._saving:
                        self.calibration_status = 'Finish saving before changing the camera.'
                    else:
                        if value.get('device_id'):
                            self.sources.select_device(value['device_id'])
                        else:
                            self.sources.configure(value['index'],value['backend'])
                        self.source = 'live'
                        self.replay_iter = None
                        self._reset()
                elif name == 'camera_refresh':
                    self.sources.refresh_devices()
                elif name == 'calibrate_motion':
                    self._job('motion calibration')
                elif name == 'reload_motion':
                    if not (self.recorder.active or self._saving or self.training_process):
                        self._load_motion_profile()
                        self.calibration_status = self.motion_profile_status
                elif name == 'train':
                    self._train()
                elif name == 'evaluate':
                    self._evaluate()
                elif name == 'reset_class':
                    if self.recorder.active or self._saving or self.training_process or self._maintenance_future:
                        raise ValueError('Finish the current recording, save or training before resetting a class.')
                    from src.vision.dataset_maintenance import archive_class
                    self._maintenance_future=self._executor.submit(archive_class,self.args.data_dir,str(value))
                    self.calibration_status=f'Archiving {value} recordings…'
                elif name == 'reload' and not self._saving:
                    self._load_models()
                    self._refresh_dataset()
                elif name == 'engine':
                    self.engine = 'CPU'
                    if str(value).upper() != 'CPU':
                        self.status = 'FPGA is not implemented for this demo.'
                elif name == 'toggle_metrics':
                    self.metrics_visible = bool(value)
                elif name == 'ui_metrics':
                    self.ui_metrics = dict(value)
                elif name == 'speak' and self.speech:
                    text = self.conversation.speak_text()
                    if text:
                        self.speech.speak(text)
            except Exception as exc:
                self.calibration_status = f'{name}: {exc}'

    def _finish_record(self):
        if not self.recorder.active or self._saving:
            return
        self._frozen_stats = self.recorder.snapshot()
        self._saving = self._executor.submit(self.recorder.finish)
        self.calibration_status = 'Saving and verifying the recording…'
        self.continuous.reset()

    def _poll_storage(self):
        if self._maintenance_future and self._maintenance_future.done():
            try:
                result=self._maintenance_future.result()
                self.calibration_status=f'Reset {result["label"]}: {result["files"]} files archived at {result["archive"] or "—"}. Record this class again and retrain.'
                if result['files']:
                    self._load_models()
                    self._reset()
            except Exception as exc:self.calibration_status=f'Class reset failed: {exc}'
            self._maintenance_future=None
            self._refresh_dataset()
        if self._saving and self._saving.done():
            try:
                self._saving.result()
                self.calibration_status = self.recorder.last_message
                self.performance.add('disk_write_ms',self.recorder.receipt['save_ms'])
            except Exception as exc:
                self.calibration_status = f'Save failed: {exc}. Samples remain in memory; press Stop to retry.'
                self._save_failed = True
            else:
                self._save_failed = False
            self._saving = self._frozen_stats = None
            self._refresh_dataset()
        if self._dataset_future and self._dataset_future.done():
            try:
                self.dataset,self.motion = self._dataset_future.result()
            except Exception as exc:
                self.dataset['errors'] = [str(exc)]
            self._dataset_future = None

    def _job(self,kind):
        if self.training_process or self.recorder.active or self._saving or self._maintenance_future:
            self.calibration_status = 'Finish recording/saving and wait for the current operation.'
            return
        log_path = ROOT / 'work/training.log'
        log_path.parent.mkdir(exist_ok=True)
        self.training_log = log_path.open('w',encoding='utf-8')
        self.job_kind,self.job_started = kind,time.perf_counter()
        module = {'training':'training.train', 'evaluation':'training.evaluate', 'motion calibration':'training.calibrate_motion'}[kind]
        command = [sys.executable,'-u','-m',module,
                   '--data-dir',str(self.args.data_dir)]
        if kind=='motion calibration':
            command += ['--output',str(self.motion_profile_path)]
        elif kind=='training':
            model_path=Path(self.args.model)
            command += ['--output',str(model_path),'--ready-only',
                        '--feature-set',self.cpu.metadata.get('feature_set','coordinates-v1') if self.cpu else 'coordinates-v1',
                        '--metrics',str(model_path.with_name(model_path.stem+'_metrics.json'))]
            if self.cpu and self.cpu.metadata.get('calibration'):command+=['--calibrate-temperature']
        else:
            command += ['--model',str(self.args.model),'--split','new','--output',str(ROOT/'models/evaluation.json')]
        try:
            self.training_process = subprocess.Popen(command,cwd=ROOT,stdout=self.training_log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:
            self.training_log.close()
            self.training_log = None
            raise
        self.calibration_status = 'Calibrating saved J/Z trajectories…' if kind=='motion calibration' else ('Training…' if kind=='training' else 'Evaluating new sessions…')

    def _train(self): self._job('training')
    def _evaluate(self): self._job('evaluation')

    def _poll_training(self):
        if self.training_process is None:
            return
        log = (ROOT/'work/training.log').read_text(encoding='utf-8',errors='replace')
        self.job_progress = log.strip().splitlines()[-1] if log.strip() else 'Starting…'
        result = self.training_process.poll()
        if result is None:
            return
        self.training_log.close()
        self.training_log = self.training_process = None
        if result == 0 and self.job_kind == 'motion calibration':
            self._load_motion_profile()
            self.calibration_status = self.motion_profile_status
        elif result == 0 and self.job_kind == 'training':
            if self._load_models():
                self.mode='free_spell'
                self.conversation.set_mode(self.mode)
                self._reset()
                accuracy=self.training_report.get('validation',{}).get('accuracy')
                measured=f'{accuracy:.1%}' if isinstance(accuracy,(int,float)) else 'Report unavailable'
                self.calibration_status=f'Model trained successfully. Classes: {", ".join(self.cpu.supported_labels)}. Validation accuracy: {measured}. Model loaded — show a trained letter.'
            else:self.calibration_status=f'Training finished, but model loading failed: {self.model_error}'
        elif result == 0:
            measured = json.loads((ROOT/'models/evaluation.json').read_text(encoding='utf-8'))['evaluation']
            self.calibration_status = f'New-session accuracy {measured["accuracy"]:.1%} on {measured["samples"]} samples. Live word/sign acceptance is still required.'
        else:
            self.calibration_status = self.job_kind.capitalize() + ' failed: ' + self.job_progress[-500:]
            if self.job_kind == 'motion calibration':
                try:
                    self.motion_calibration_report = json.loads(self.motion_profile_path.with_name(self.motion_profile_path.stem+'_attempt.json').read_text(encoding='utf-8'))
                except (OSError, ValueError): pass
                self.motion_profile_status = self.calibration_status + ' Previous calibration and recordings preserved.'

    def _infer_loop(self):
        tracker,last_id = None,None
        try:
            while not self.stopping.is_set():
                if self.sources.snapshot()['state'] != 'LIVE':
                    self.stopping.wait(.015)
                    continue
                frame = self.sources.latest(last_id)
                if frame is None:
                    self.stopping.wait(.003)
                    continue
                last_id = frame.frame_id
                try:
                    if tracker is None:
                        tracker = self._tracker_factory(self.args.landmarker)
                    observation = tracker.detect(frame.bgr,frame.timestamp_ms)
                    result = (frame,observation,tracker.last_latency_ms,None)
                except Exception as exc:
                    result = (frame,None,None,str(exc))
                    self.stopping.wait(.1)
                if frame.generation == self.sources.snapshot()['generation']:
                    try: self._results.get_nowait()
                    except queue.Empty: pass
                    self._results.put_nowait(result)
        finally:
            if tracker:
                tracker.close()

    def _accept(self,prediction,timestamp):
        token = self.decoder.update(prediction,timestamp)
        self._consume_token(token)

    def _consume_token(self, token):
        if token:
            self.stable = token.label
            if self.mode=='conversation' and token.label in ALPHABET:
                tokens=self.conversation.snapshot()['tokens']
                if not (tokens[:2]==['MY','NAME'] and 'YOUR' not in tokens):
                    return  # Candidate/commit remains visible; phrase grammar is independent.
            state = self.conversation.consume(token)
            response = state.get('response','')
            if response and response != self.last_response:
                self.last_response = response
                if self.speech: self.speech.speak(response)

    def _consume_result(self):
        try: frame,observation,latency,error = self._results.get_nowait()
        except queue.Empty: return
        snapshot = self.sources.snapshot()
        if snapshot['state'] != 'LIVE' or frame.generation != snapshot['generation']:
            return
        self.tracker_error = error or ''
        if error:
            self.prediction=None
            self.recognition_error=error
            self.decoder.reject(frame.timestamp_ms)
            self.continuous.motion.reset()
            self.continuous.repeat.reset()
            self.continuous.spacing.reset()
            return
        self.last_observation,self.observation_frame_id,self.observation_generation = observation,frame.frame_id,frame.generation
        self.landmark_ms = latency
        self.performance.add('landmark_ms',latency)
        if self.recorder.active and not self._saving and not self._save_failed:
            self.recorder.add(observation)
        from src.vision import primary_features
        self.recognition_error = ''
        self.recognition_reason = ''
        paused = bool(self.recorder.active or self._saving or self._save_failed or self._maintenance_future)
        try:
            features = primary_features(observation)
            prediction = self.cpu.predict(features, observation.timestamp_ms) if self.cpu and features is not None and features.any() else None
            if prediction:
                self.inference_count += 1
            self.cpu_latency = prediction.latency_ms if prediction else None
            self.performance.add('model_ms', self.cpu_latency)
            start = time.perf_counter()
            if self.mode in ('free_spell', 'alphabet_tutor'):
                token = self.continuous.update(observation, prediction, paused=paused,
                    has_text=bool(self.conversation.snapshot()['word'].strip()) and self.mode == 'free_spell')
                self.prediction = self.continuous.prediction
                self.recognition_reason = self.continuous.reason
                if self.continuous.space_requested:
                    self.continuous.cancel_motion_promotion()
                    self.conversation.insert_space()
                if token and self.continuous.replace_last_static and self.conversation.snapshot()['word'].endswith(self.continuous.replace_last_static):
                    self.conversation.backspace()
                self._consume_token(token)
            else:
                motion_prediction = self.motion.update(observation)
                self.prediction = motion_prediction or prediction
                if paused or (self.motion.motion_active and not motion_prediction):
                    self.recognition_reason = 'Motion in progress' if self.motion.motion_active else 'Recording paused'
                    self.decoder.reject(observation.timestamp_ms)
                elif features is None and observation.hands:
                    self.decoder.reject(observation.timestamp_ms)
                    self.recognition_reason = 'Invalid or ambiguous landmarks'
                else:
                    self._accept(self.prediction, observation.timestamp_ms)
            self.motion_ms = (time.perf_counter() - start) * 1000
            if not observation.hands:
                self.recognition_reason = 'No hand'
            elif self.prediction is None and not self.recognition_reason:
                self.recognition_reason = 'Model not loaded' if self.cpu is None else 'Invalid or ambiguous landmarks'
            elif self.prediction and self.prediction.confidence < self.decoder.confidence_threshold:
                self.recognition_reason = 'Low confidence'
        except (ValueError, TypeError, OverflowError) as exc:
            self.prediction = None
            self.recognition_error = f'Recognition error: {exc}'
            self.recognition_reason = 'Invalid features or model output'
            self.decoder.reject(observation.timestamp_ms)
            self.continuous.motion.reset()
            self.continuous.repeat.reset()
            self.continuous.spacing.reset()

    def _capabilities(self,snapshot):
        busy = bool(self._saving or self.recorder.active or self.training_process or self._maintenance_future)
        live = snapshot['state']=='LIVE' and not self.tracker_error and self.last_observation is not None
        def cap(enabled,reason): return dict(enabled=bool(enabled),reason='' if enabled else reason)
        return dict(record=cap(live and not busy,'Start live hand tracking and finish any pending operation.'),
            finish_record=cap(self.recorder.active and not self._saving,'No recording to save.'),
            new_session=cap(not busy,'Finish the current operation.'), train=cap(not busy,'Finish the current operation.'),
            reset_class=cap(not busy,'Finish the current operation.'),
            evaluate=cap(self.cpu is not None and not busy,'Train a model first; finish any pending operation.'),
            reload=cap(not busy,'Finish the current operation.'),fpga=cap(False,'FPGA inference is not implemented.'),
            speak=cap(self.speech and not self.speech.error and self.conversation.speak_text(),'No response or speech unavailable.'))

    def _event(self):
        snapshot = self.sources.snapshot()
        frame = self.sources.latest() if snapshot['state']=='LIVE' else None
        now = time.monotonic()*1000
        observation = self.last_observation
        fresh = bool(frame and observation and self.observation_generation==frame.generation and now-observation.timestamp_ms < 180)
        hands = observation if fresh else None
        pred = self.prediction if (fresh or self.source=='replay') else None
        continuous_active = self.mode in ('free_spell', 'alphabet_tutor') and self.source != 'replay'
        motion_state = self.continuous.snapshot() if continuous_active else {}
        motion_paused = bool(continuous_active and motion_state.get('suppress_static'))
        accepted=bool(pred and pred.label and pred.confidence>=self.decoder.confidence_threshold and
                      (self.source=='replay' or (not motion_paused and self.recognition_reason!='Motion in progress')))
        gate=self.decoder.snapshot()
        stats = dict(self._frozen_stats) if self._saving else self.recorder.snapshot()
        coverage = list(self.cpu.supported_labels) if self.cpu else []
        from src.recognition.diagnostics import prediction_diagnostics
        diagnostic=prediction_diagnostics(pred,gate,coverage,fresh=fresh,
            reason=self.recognition_error or self.recognition_reason,
            paused=bool(self.recorder.active or self._saving or self._save_failed or self._maintenance_future or motion_paused or self.recognition_reason=='Motion in progress'),
            replay=self.source=='replay')
        state = self.conversation.snapshot()
        event = dict(kind='frame',source=self.source,camera_state=snapshot['state'],camera_error=snapshot['error'],
            camera_selection=dict(index=snapshot.get('selected_index'),backend=snapshot.get('selected_backend')),
            camera_device=snapshot.get('selected_device',{}),camera_devices=snapshot.get('devices',[]),
            camera_scanning=snapshot.get('scanning',False),camera_device_error=snapshot.get('device_error',''),
            camera_diagnostics=snapshot.get('diagnostics',{}),camera_log_path=snapshot.get('log_path',''),
            camera_settings_error=snapshot.get('settings_error') or getattr(self.args,'camera_settings_error',''),
            image=frame.bgr if frame else None,frame_id=frame.frame_id if frame else None,generation=snapshot['generation'],
            observation_frame_id=self.observation_frame_id,hands=hands,landmarks_fresh=fresh,
            timestamp_ms=int(now),candidate=pred.label if accepted else None,stable=self.stable,confidence=pred.confidence if pred else None,
            hand_detected=bool(hands and hands.hands),model_loaded=self.cpu is not None,model_classes=coverage,
            recognition_reason=self.recognition_error or self.recognition_reason,model_error=self.model_error,
            stable_duration_ms=gate['stable_duration_ms'] if fresh else 0,confidence_threshold=self.decoder.confidence_threshold,
            dwell_ms=self.decoder.dwell_ms,locked_label=gate['locked_label'],inference_count=self.inference_count,
            training_report=self.training_report,prediction_diagnostics=diagnostic,
            untrained_ready_labels=[c for c in self.dataset.get('ready_labels',[]) if c not in coverage],
            engine='CPU',mode=self.mode,status=self.tracker_error or self.status,coverage=coverage+list(self.continuous.motion.supported_labels if continuous_active else self.motion.supported_labels),
            fingerspelling=motion_state,fingerspelling_settings=self.fingerspelling_settings,
            fingerspelling_settings_error=self.fingerspelling_settings_error,
            motion_profile_status=self.motion_profile_status,motion_calibration_report=self.motion_calibration_report,
            recording=self.recorder.active and not self._saving and not self._save_failed,saving=bool(self._saving),save_failed=self._save_failed,sample_count=stats['samples'],
            recording_stats=stats,dataset=self.dataset,calibration_status=self.calibration_status,
            physical_handedness=', '.join(h.handedness.upper() for h in hands.hands) if hands else '—',
            raw_handedness=', '.join((h.raw_handedness or h.handedness).upper() for h in hands.hands) if hands else '—',
            input_mirrored=snapshot['input_mirrored'],normalization_note='ML: unmirrored · Preview: selfie mirror',
            capture_fps=snapshot['capture_fps'],fps=snapshot['capture_fps'],capture_ms=snapshot['capture_ms'],
            landmark_ms=self.landmark_ms if frame else None,model_ms=self.cpu_latency if frame else None,
            cpu_latency_ms=self.cpu_latency if frame else None,fpga_latency_ms=None,motion_ms=self.motion_ms,
            disk_write_ms=stats['save_ms'],frame_age_ms=now-frame.timestamp_ms if frame else None,
            speech_status=self.speech.status if self.speech and self.speech.error else '',metrics_visible=self.metrics_visible,
            job=dict(kind=self.job_kind,state='running' if self.training_process else 'idle',
                     elapsed_s=time.perf_counter()-self.job_started if self.training_process else 0,progress=self.job_progress),
            capabilities=self._capabilities(snapshot),**state)
        self.emit(event)
        if time.perf_counter()-self._last_metric_sample >= 1:
            self._last_metric_sample = time.perf_counter()
            self.performance.states[snapshot['state']] += 1
            if frame:
                for key in ('capture_fps','capture_ms','frame_age_ms'):
                    self.performance.add(key,event[key])
                for key,value in self.ui_metrics.items(): self.performance.add(key,value)

    def _run(self):
        try:
            self._load_models()
            self._refresh_dataset()
            if self.source=='replay': self._transition('replay')
            last_job = 0
            while not self.stopping.is_set():
                loop_start = time.perf_counter()
                self._commands()
                self._poll_storage()
                if loop_start-last_job>.25:
                    self._poll_training()
                    last_job=loop_start
                if self.sources.snapshot()['state']=='REPLAY' and self.replay_iter and loop_start>=self._next_replay:
                    self._next_replay+=.04
                    try:
                        self.prediction,timestamp=next(self.replay_iter)
                        self._accept(self.prediction,timestamp)
                    except StopIteration:
                        self.replay_iter=None
                self._consume_result()
                self._event()
                self.stopping.wait(max(.001,1/30-(time.perf_counter()-loop_start)))
        except Exception as exc:
            self.status=f'Application error: {exc}'
            self._event()
        finally:
            self.sources.close()
            if self.recorder.active and not self._saving: self._finish_record()
            if self._saving:
                try: self._saving.result(timeout=10)
                except Exception as exc: self.status=f'Final save failed: {exc}'
                self.performance.add('disk_write_ms',self.recorder.receipt['save_ms'])
            if self.training_process and self.training_process.poll() is None: self.training_process.terminate()
            if self.training_log: self.training_log.close()
            self.performance.save(self.profile_path)
