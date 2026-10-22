"""Minimal desktop workspace, with latest-frame rendering and measured paint cost."""
from collections import deque
import math
import os
from pathlib import Path
import sys
import threading
import time
import statistics
from PySide6.QtCore import Qt,QTimer,QRectF
from PySide6.QtGui import QColor,QFont,QFontDatabase,QImage,QPainter,QPen,QKeySequence,QShortcut
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,
    QLabel,QPushButton,QComboBox,QCheckBox,QFrame,QSizePolicy,QScrollArea,QLayout,QDialog,QMessageBox,QPlainTextEdit,
    QTableWidget,QTableWidgetItem,QHeaderView,QAbstractItemView,QDoubleSpinBox,QSpinBox,QKeySequenceEdit,QProgressBar)
from src.contracts import ALPHABET,LABELS,STATIC_LABELS
from src.recognition.readiness import alphabet_readiness
from src.vision.camera import HAND_CONNECTIONS
from src.vision.features import validated_landmarks

STYLE='''
QWidget {color:#eceef1;font-family:"Segoe UI";font-size:13px;}
QMainWindow, QWidget#root {background:#1a1c20;}
QLabel {background:transparent;}
QLabel#title {font-size:25px;font-weight:600;}
QLabel#muted {color:#a2a6af;}
QLabel#word {font-size:36px;font-weight:600;}
QLabel#response {font-size:21px;line-height:1.4;}
QPushButton {border:1px solid #41454e;border-radius:5px;background:#292c32;padding:8px 14px;}
QPushButton:hover {background:#353941;border-color:#727b8c;}
QPushButton:pressed {background:#414650;}
QPushButton:disabled {color:#686d78;background:#22252a;border-color:#30343b;}
QPushButton#primary {background:#87a8ed;color:#141c2b;border-color:#87a8ed;font-weight:600;}
QPushButton#primary:hover {background:#a3bdf3;}
QPushButton#primary:disabled {color:#6e7687;background:#303849;border-color:#303849;}
QPushButton#tab {background:transparent;border:0;border-bottom:2px solid transparent;border-radius:0;color:#9fa5af;}
QPushButton#tab:checked {color:#f0f2f5;border-bottom:2px solid #87a8ed;}
QPushButton#subtle {border:0;background:transparent;color:#b4bac6;}
QPushButton#subtle:hover {background:#292d34;}
QPushButton#subtle:disabled {color:#686d78;background:transparent;}
QComboBox {background:#24272d;border:1px solid #41454e;border-radius:5px;padding:7px 10px;min-width:65px;}
QComboBox QAbstractItemView {background:#24272d;color:#eceef1;selection-background-color:#3a4a69;}
QSpinBox, QDoubleSpinBox, QKeySequenceEdit, QLineEdit {background:#24272d;color:#eceef1;border:1px solid #41454e;border-radius:4px;padding:5px;}
QCheckBox {color:#aab1bc;spacing:7px;}
QCheckBox::indicator {width:14px;height:14px;}
QFrame#line {background:#353941;max-height:1px;min-height:1px;border:0;}
QScrollArea {border:0;background:transparent;}
'''


def label(text='',name=None,wrap=False):
    widget=QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(wrap)
    if name:widget.setObjectName(name)
    return widget


def metric(value,suffix='',precision=1):
    try:
        number=float(value)
        if math.isfinite(number) and number>=0:return f'{number:.{precision}f}{suffix}'
    except (ValueError,TypeError):pass
    return '—'


def set_text(widget,text):
    text=str(text)
    if widget.text()!=text:widget.setText(text)


def line():
    widget=QFrame();widget.setObjectName('line');return widget


class CameraCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400,280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding)
        self._array=self._image=None
        self._key=None
        self._frame_event={}
        self.paint_ms=deque(maxlen=120)
        self.display_times=deque(maxlen=90)
        self.last_painted=None

    def set_event(self,event):
        key=(event.get('generation',0),event.get('frame_id'))
        pixels=event.get('image')
        if pixels is None or (self._key is not None and key[0]!=self._key[0]):
            self.display_times.clear()
            self.last_painted=None
        changed=key!=self._key or (pixels is None)!=(self._image is None)
        if changed:
            self._key=key
            self._array=pixels
            if pixels is None:self._image=None
            else:
                h,w,_=pixels.shape
                self._image=QImage(pixels.data,w,h,pixels.strides[0],QImage.Format.Format_BGR888)
        overlay_changed=any(event.get(k)!=self._frame_event.get(k) for k in ('camera_state','candidate','confidence','observation_frame_id','metrics_visible'))
        self._frame_event=event
        if changed or overlay_changed:self.update()

    def paintEvent(self,event):
        started=time.perf_counter()
        painter=QPainter(self)
        painter.fillRect(self.rect(),QColor('#111315'))
        state=self._frame_event.get('camera_state','OFF')
        if self._image is not None:
            ratio=min(self.width()/self._image.width(),self.height()/self._image.height())
            w,h=self._image.width()*ratio,self._image.height()*ratio
            rect=QRectF((self.width()-w)/2,(self.height()-h)/2,w,h)
            painter.save()
            painter.translate(rect.x()+w,rect.y());painter.scale(-1,1)
            # Draw source directly into target; do not allocate a scaled image every frame.
            painter.drawImage(QRectF(0,0,w,h),self._image)
            observation=self._frame_event.get('hands')
            if observation and self._frame_event.get('landmarks_fresh',False):
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                pen=QPen(QColor('#9ab7f0'),2);pen.setCosmetic(True);painter.setPen(pen)
                for hand in observation.hands:
                    try:points=validated_landmarks(hand.landmarks)
                    except (TypeError,ValueError,OverflowError):continue
                    for a,b in HAND_CONNECTIONS:
                        p,q=points[a],points[b]
                        painter.drawLine(int(p[0]*w),int(p[1]*h),int(q[0]*w),int(q[1]*h))
                    painter.setBrush(QColor('#ecf0f8'))
                    for p in points:painter.drawEllipse(QRectF(p[0]*w-2,p[1]*h-2,4,4))
            trail = (self._frame_event.get('fingerspelling') or {}).get('trajectory', [])
            if observation and self._frame_event.get('landmarks_fresh', False) and len(trail) > 1:
                painter.setPen(QPen(QColor('#87a8ed'), 3))
                for a, b in zip(trail, trail[1:]):
                    painter.drawLine(int(a[0]*w), int(a[1]*h), int(b[0]*w), int(b[1]*h))
            painter.restore()
            if self._key!=self.last_painted:
                self.last_painted=self._key
                self.display_times.append(time.perf_counter())
            candidate=self._frame_event.get('candidate')
            if candidate or self._frame_event.get('model_loaded'):
                painter.fillRect(QRectF(16,16,140,74),QColor(24,27,32,220))
                painter.setPen(QColor('#f0f2f6'));painter.setFont(QFont('Segoe UI',26,QFont.Weight.DemiBold))
                painter.drawText(QRectF(28,20,110,40),str(candidate or '—'))
                painter.setFont(QFont('Segoe UI',10))
                confidence=self._frame_event.get('confidence')
                painter.drawText(QRectF(28,63,110,20),metric(confidence*100 if confidence is not None else None,'% score',0))
        else:
            title={'OFF':'Camera off','STARTING':'Starting camera…','LIVE':'Waiting for camera frame',
                   'REPLAY':'Synthetic replay','ERROR':'Camera unavailable'}.get(state,'Camera off')
            painter.setPen(QColor('#dde1e7'));painter.setFont(QFont('Segoe UI',22,QFont.Weight.DemiBold))
            painter.drawText(QRectF(24,self.height()/2-32,self.width()-48,40),Qt.AlignmentFlag.AlignCenter,title)
            painter.setPen(QColor('#8e96a2'));painter.setFont(QFont('Segoe UI',11))
            hint='Scripted tokens. No live recognition.' if state=='REPLAY' else 'Select Live camera to begin.'
            painter.drawText(QRectF(24,self.height()/2+17,self.width()-48,35),Qt.AlignmentFlag.AlignCenter,hint)
        painter.end()
        self.paint_ms.append((time.perf_counter()-started)*1000)

    def metrics(self):
        times=self.display_times
        fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] and time.perf_counter()-times[-1]<1 else None
        return dict(display_fps=fps,ui_render_ms=statistics.median(self.paint_ms) if self.paint_ms else None)


class _Window(QMainWindow):
    def __init__(self,owner):super().__init__();self.owner=owner
    def closeEvent(self,event):
        self.owner._timer.stop();self.owner._dispatch('stop');super().closeEvent(event)


class SignBridgeUI:
    def __init__(self,on_action):
        self._on_action=on_action;self._latest=None;self._lock=threading.Lock()
        self._app=QApplication.instance() or QApplication(sys.argv[:1]);self._app.setStyle('Fusion')
        for name in ('segoeui.ttf','segoeuib.ttf','seguisb.ttf','seguisym.ttf'):
            path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
            if path.is_file():QFontDatabase.addApplicationFont(str(path))
        self.window=_Window(self);self.window.setWindowTitle('SignBridge');self.window.resize(1400,900)
        self.window.setMinimumSize(1100,720);self.window.setStyleSheet(STYLE)
        root=QWidget();root.setObjectName('root');self.window.setCentralWidget(root)
        layout=QVBoxLayout(root);layout.setContentsMargins(24,16,24,16);layout.setSpacing(16)
        top=QHBoxLayout();top.setSpacing(24);top.addWidget(label('SignBridge','title'));top.addSpacing(16)
        self.modes={}
        for key,title in [('conversation','Conversation'),('free_spell','Fingerspelling'),('alphabet_tutor','Tutor')]:
            b=self._button(title,'mode',value=key);b.setObjectName('tab');b.setCheckable(True);b.setChecked(key=='conversation')
            self.modes[key]=b;top.addWidget(b)
        top.addStretch();self.source_badge=label('OFF','muted');top.addWidget(self.source_badge)
        self.engine=QComboBox();self.engine.addItems(['CPU','FPGA'])
        self.engine.model().item(1).setEnabled(False)
        self.engine.model().item(1).setToolTip('Not implemented. CPU remains the active engine.')
        top.addWidget(self.engine);layout.addLayout(top);layout.addWidget(line())
        controls=QHBoxLayout();controls.setSpacing(8)
        controls.addWidget(self._button('Live camera','live',primary=True))
        controls.addWidget(self._button('Synthetic replay','replay'))
        controls.addWidget(self._button('Camera off','camera_off',subtle=True))
        settings_button=QPushButton('Settings / Diagnostics');settings_button.setObjectName('subtle')
        top.insertWidget(top.count()-2,settings_button)
        self.camera_settings=QDialog(self.window);self.camera_settings.setWindowTitle('Camera settings / Diagnostics')
        self.camera_settings.setMinimumWidth(560);self.camera_settings.setStyleSheet('QDialog {background:#1a1c20;}')
        settings_layout=QVBoxLayout(self.camera_settings)
        settings_layout.addWidget(label('Choose your webcam by name. Physical cameras are preferred automatically.',wrap=True))
        settings_row=QHBoxLayout();settings_row.addWidget(label('Camera'))
        self.camera_devices=QComboBox();self.camera_devices.setMinimumWidth(260)
        self.camera_devices.addItem('Scanning cameras…',None);settings_row.addWidget(self.camera_devices,1)
        self.camera_refresh=self._button('Refresh cameras','camera_refresh');settings_row.addWidget(self.camera_refresh)
        settings_layout.addLayout(settings_row)
        self.camera_apply=self._button('Apply and open camera','camera_config',value=lambda:dict(device_id=self.camera_devices.currentData()))
        settings_layout.addWidget(self.camera_apply)
        self.camera_diagnostics=label('No camera attempt yet.',wrap=True)
        self.camera_diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        settings_layout.addWidget(self.camera_diagnostics)
        settings_layout.addWidget(line())
        settings_layout.addWidget(label('Fingerspelling', wrap=True))
        settings_layout.addWidget(label('Repeat a letter by moving your hand to your right, then holding briefly. The preview is mirrored.', 'muted', True))
        repeat_row=QHBoxLayout()
        repeat_row.addWidget(label('Repeat distance (palm lengths)'))
        self.repeat_threshold=QDoubleSpinBox();self.repeat_threshold.setRange(.25,2.);self.repeat_threshold.setSingleStep(.05);self.repeat_threshold.setValue(.65)
        repeat_row.addWidget(self.repeat_threshold)
        repeat_row.addWidget(label('Cooldown (ms)'))
        self.repeat_cooldown=QSpinBox();self.repeat_cooldown.setRange(300,3000);self.repeat_cooldown.setSingleStep(100);self.repeat_cooldown.setValue(700)
        repeat_row.addWidget(self.repeat_cooldown);settings_layout.addLayout(repeat_row)
        spacing_row=QHBoxLayout()
        self.auto_spacing=QCheckBox('Automatic space after hand withdrawal');spacing_row.addWidget(self.auto_spacing)
        self.spacing_pause=QSpinBox();self.spacing_pause.setRange(800,5000);self.spacing_pause.setSingleStep(100);self.spacing_pause.setValue(1200);self.spacing_pause.setSuffix(' ms')
        spacing_row.addWidget(self.spacing_pause);settings_layout.addLayout(spacing_row)
        shortcut_row=QHBoxLayout();shortcut_row.addWidget(label('Space shortcut'))
        self.space_key_edit=QKeySequenceEdit(QKeySequence('Ctrl+Space'));shortcut_row.addWidget(self.space_key_edit);settings_layout.addLayout(shortcut_row)
        self.spelling_apply=self._button('Save fingerspelling settings','fingerspelling_settings',value=lambda:dict(
            repeat_threshold=self.repeat_threshold.value(),repeat_cooldown_ms=self.repeat_cooldown.value(),
            auto_space=self.auto_spacing.isChecked(),space_pause_ms=self.spacing_pause.value(),
            space_shortcut=self.space_key_edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText)))
        settings_layout.addWidget(self.spelling_apply)
        self._spelling_settings_signature=None
        self.space_shortcut=QShortcut(QKeySequence('Ctrl+Space'),self.window)
        self.space_shortcut.activated.connect(lambda:self._dispatch('space') if self._spelling_mode and not self.camera_settings.isVisible() else None)
        self._spelling_mode=False
        self._camera_selection=None;self._camera_settings_allowed=True;self._device_list_signature=None
        self.camera_devices.currentIndexChanged.connect(self._camera_controls)
        self._camera_controls()
        settings_button.clicked.connect(lambda:(self.camera_settings.show(),self.camera_settings.raise_()))
        controls.addSpacing(16);self.handedness=label('Physical hand: —  ·  Model label: —','muted');controls.addWidget(self.handedness)
        controls.addStretch();self.metrics_toggle=QCheckBox('Developer metrics');self.metrics_toggle.setChecked(True)
        self.metrics_toggle.toggled.connect(lambda value:self._dispatch('toggle_metrics',value));controls.addWidget(self.metrics_toggle)
        layout.addLayout(controls)
        main=QHBoxLayout();main.setSpacing(32)
        self.camera=CameraCanvas();main.addWidget(self.camera,65)
        right=QWidget();dialogue=QVBoxLayout(right);dialogue.setContentsMargins(0,4,0,4);dialogue.setSpacing(8)
        title_row=QHBoxLayout();self.panel_title=label('Conversation');self.panel_title.setStyleSheet('font-size:19px;font-weight:600;')
        title_row.addWidget(self.panel_title);title_row.addStretch();title_row.addWidget(self._button('Clear','clear',subtle=True));dialogue.addLayout(title_row)
        self.live_candidate=label('Candidate: —   Confidence: —',wrap=True)
        self.live_candidate.setStyleSheet('font-size:18px;font-weight:600;');dialogue.addWidget(self.live_candidate)
        self.recognition_debug=label('Hand detected: NO · Model loaded: NO','muted',True);dialogue.addWidget(self.recognition_debug)
        self.recognition_debug.setMinimumHeight(self.recognition_debug.fontMetrics().lineSpacing()*4)
        self.stable=label('Committed: —','muted');dialogue.addWidget(self.stable)
        self.motion_status=label('Recognition: STATIC','muted',True);dialogue.addWidget(self.motion_status)
        self.motion_progress=QProgressBar();self.motion_progress.setRange(0,100);self.motion_progress.setFixedHeight(5);self.motion_progress.setTextVisible(False)
        self.motion_progress.setStyleSheet('QProgressBar {border:0;background:#292c32;} QProgressBar::chunk {background:#87a8ed;}')
        self.motion_progress.hide();dialogue.addWidget(self.motion_progress)
        self.prediction_diagnostics_toggle=QCheckBox('Prediction diagnostics');self.prediction_diagnostics_toggle.setChecked(True)
        dialogue.addWidget(self.prediction_diagnostics_toggle)
        self.prediction_details=label('Waiting for classifier scores.','muted',True);dialogue.addWidget(self.prediction_details)
        self.prediction_diagnostics_toggle.toggled.connect(self.prediction_details.setVisible)
        self.tutor=label('',wrap=True);self.tutor.hide();dialogue.addWidget(self.tutor)
        self.word=label('','word',True);dialogue.insertWidget(2,self.word)
        edit_row=QHBoxLayout();self.space=self._button('Space','space',subtle=True);self.backspace=self._button('Backspace','backspace',subtle=True)
        edit_row.addWidget(self.space);edit_row.addWidget(self.backspace);edit_row.addStretch();dialogue.insertLayout(3,edit_row)
        dialogue.removeWidget(self.stable);dialogue.removeWidget(self.motion_status);dialogue.removeWidget(self.motion_progress)
        recognition_state=QHBoxLayout();recognition_state.addWidget(self.stable);recognition_state.addWidget(self.motion_status,1)
        dialogue.insertLayout(4,recognition_state);dialogue.insertWidget(5,self.motion_progress)
        dialogue.addWidget(label('Signed','muted'));self.recognized=label('—',wrap=True);self.recognized.setStyleSheet('font-size:17px;');dialogue.addWidget(self.recognized)
        dialogue.addSpacing(8);dialogue.addWidget(label('Interpreted','muted'));self.interpretation=label('—',wrap=True);self.interpretation.setStyleSheet('font-size:17px;');dialogue.addWidget(self.interpretation)
        dialogue.addSpacing(8);dialogue.addWidget(label('Astra','muted'));self.response=label('','response',True);dialogue.addWidget(self.response)
        dialogue.addStretch()
        self.speech_status=label('','muted',True);dialogue.addWidget(self.speech_status)
        self.speak=self._button('Speak response','speak',subtle=True);edit_row.insertWidget(edit_row.count()-1,self.speak)
        # Let longer recognition/debug text scroll instead of compressing labels into overlap.
        recognition_scroll=QScrollArea();recognition_scroll.setWidgetResizable(True)
        recognition_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        recognition_scroll.setWidget(right);right.setAutoFillBackground(False)
        main.addWidget(recognition_scroll,35);layout.addLayout(main,1)
        self.metrics_label=label('Capture —   Display —   Landmarks —   Model —   UI —','muted');layout.addWidget(self.metrics_label)
        # Compatibility value labels used by regression inspection; not decorative cards.
        self.cpu_ms=label('—');self.cpu_ms.hide();self.candidate=label('—');self.candidate.hide()
        self.fps=self.cpu_ms;self.fpga_ms=label('—');self.fpga_ms.hide()
        layout.addWidget(line())
        dataset_header=QHBoxLayout();self.dataset_toggle=QPushButton('Hide dataset');self.dataset_toggle.setObjectName('subtle');self.dataset_toggle.setCheckable(True);self.dataset_toggle.setChecked(True)
        dataset_header.addWidget(self.dataset_toggle);self.dataset_total=label('Loading saved samples…','muted');dataset_header.addWidget(self.dataset_total);dataset_header.addStretch()
        self.coverage=label('No trained model','muted');dataset_header.addWidget(self.coverage);layout.addLayout(dataset_header)
        self.dataset_body=QWidget();self.dataset_body.setObjectName('datasetBody')
        self.dataset_body.setStyleSheet('QWidget#datasetBody {background:#1a1c20;}')
        body=QVBoxLayout(self.dataset_body);body.setContentsMargins(0,0,8,0);body.setSpacing(8)
        body.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.dataset_scroll=QScrollArea();self.dataset_scroll.setWidgetResizable(True);self.dataset_scroll.setWidget(self.dataset_body)
        self.dataset_scroll.setMinimumHeight(150);self.dataset_scroll.setMaximumHeight(310)
        self.dataset_toggle.toggled.connect(self.dataset_scroll.setVisible)
        self.dataset_toggle.toggled.connect(lambda checked:self.dataset_toggle.setText('Hide dataset' if checked else 'Show dataset'))
        row=QHBoxLayout();row.setSpacing(8);row.addWidget(label('Label','muted'));self.labels=QComboBox();self.labels.addItems(STATIC_LABELS);row.addWidget(self.labels)
        self.record=self._button('Record','record',primary=True,value=lambda:self.labels.currentText());row.addWidget(self.record)
        self.finish=self._button('Stop and save','finish_record');row.addWidget(self.finish)
        self.new_session=self._button('New session','new_session');row.addWidget(self.new_session)
        self.reset_class=QPushButton('Reset class');self.reset_class.clicked.connect(self._confirm_reset_class);row.addWidget(self.reset_class)
        row.addStretch();self.train=self._button('Train static','train');row.addWidget(self.train)
        self.evaluate=self._button('Evaluate','evaluate');row.addWidget(self.evaluate)
        self.reload=self._button('Reload','reload',subtle=True);row.addWidget(self.reload);body.addLayout(row)
        navigation=QHBoxLayout();navigation.setSpacing(8)
        self.previous_label=QPushButton('Previous label');self.previous_label.clicked.connect(lambda:self._step_label(-1));navigation.addWidget(self.previous_label)
        self.next_label=QPushButton('Next label');self.next_label.clicked.connect(lambda:self._step_label(1));navigation.addWidget(self.next_label)
        self.auto_advance=QCheckBox('After saving, go to next incomplete letter');navigation.addWidget(self.auto_advance);navigation.addStretch()
        self.calibration_group=QComboBox();self.calibration_group.addItems(['All static letters','T / N','U / R / V','S / E','Q / H'])
        self.calibration_group.setToolTip('Focused calibration: keep the same session for a sitting; take a break before New session.')
        navigation.addWidget(self.calibration_group)
        self.alphabet_progress_button=QPushButton('Alphabet progress');navigation.addWidget(self.alphabet_progress_button);body.addLayout(navigation)
        dynamic_row=QHBoxLayout()
        self.record_j=self._button('Record J motion','record',value='J');self.record_z=self._button('Record Z motion','record',value='Z')
        dynamic_row.addWidget(self.record_j);dynamic_row.addWidget(self.record_z)
        self.dynamic_samples=label('J / Z: record one complete motion per clip, then Stop and save.','muted',True)
        dynamic_row.addWidget(self.dynamic_samples,1);body.addLayout(dynamic_row)
        motion_tools=QHBoxLayout()
        self.calibrate_motion=self._button('Calibrate J/Z','calibrate_motion');motion_tools.addWidget(self.calibrate_motion)
        self.reload_motion=self._button('Load J/Z calibration','reload_motion');motion_tools.addWidget(self.reload_motion)
        self.motion_results_button=QPushButton('J/Z results');motion_tools.addWidget(self.motion_results_button);motion_tools.addStretch()
        body.addLayout(motion_tools)
        self.motion_profile_status=label('Save motion clips, then Calibrate J/Z. Static training does not use these clips.','muted',True)
        body.addWidget(self.motion_profile_status)
        self.motion_results=QDialog(self.window);self.motion_results.setWindowTitle('J/Z calibration results');self.motion_results.resize(800,550)
        self.motion_results.setStyleSheet('QDialog {background:#1a1c20;} QPlainTextEdit {background:#202329;color:#eceef1;border:1px solid #41454e;padding:10px;}')
        motion_layout=QVBoxLayout(self.motion_results);self.motion_result_text=QPlainTextEdit();self.motion_result_text.setReadOnly(True);motion_layout.addWidget(self.motion_result_text)
        self.motion_results_button.clicked.connect(lambda:(self.motion_results.show(),self.motion_results.raise_()))
        self._motion_report=None
        self.selected_label_status=label('','muted',True);body.addWidget(self.selected_label_status)
        self.training_readiness=label('Train uses all READY letters; at least two are needed.','muted',True);body.addWidget(self.training_readiness)
        self.alphabet_progress=QDialog(self.window);self.alphabet_progress.setWindowTitle('Static alphabet progress');self.alphabet_progress.resize(690,700)
        self.alphabet_progress.setStyleSheet('QDialog {background:#1a1c20;} QTableWidget {background:#202329;color:#eceef1;gridline-color:#41454e;} QHeaderView::section {background:#292c32;color:#eceef1;border:0;padding:6px;}')
        progress_layout=QVBoxLayout(self.alphabet_progress)
        progress_layout.addWidget(label('READY: 100 valid samples total, with at least 50 in each of two distinct sessions. Readiness is collection progress, not recognition accuracy.',wrap=True))
        self.alphabet_status=QTableWidget(len(STATIC_LABELS),4);self.alphabet_status.setHorizontalHeaderLabels(['Letter','Samples','Sessions','Status'])
        self.alphabet_status.verticalHeader().hide();self.alphabet_status.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.alphabet_status.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.alphabet_status.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        progress_layout.addWidget(self.alphabet_status)
        progress_layout.addWidget(label('J and Z require motion and are not part of static collection.','muted',True))
        self.alphabet_progress_button.clicked.connect(lambda:(self.alphabet_progress.show(),self.alphabet_progress.raise_()))
        self._static_progress={};self._progress_signature=None
        self._last_saved_path=None;self._save_armed=False;self._pending_advance=None
        self._model_classes=[]
        self.labels.currentTextChanged.connect(lambda *_:self._update_selected_label())
        self.calibration_group.currentTextChanged.connect(self._select_calibration_group)
        self.training_results=QDialog(self.window);self.training_results.setWindowTitle('Static alphabet training results');self.training_results.resize(720,540)
        self.training_results.setStyleSheet('QDialog {background:#1a1c20;} QPlainTextEdit {background:#202329;color:#eceef1;border:1px solid #41454e;padding:10px;}')
        results_layout=QVBoxLayout(self.training_results);self.training_result_text=QPlainTextEdit();self.training_result_text.setReadOnly(True)
        results_layout.addWidget(self.training_result_text)
        self.training_results_button=QPushButton('Training results');row.addWidget(self.training_results_button)
        self.training_results_button.clicked.connect(lambda:(self.training_results.show(),self.training_results.raise_()))
        self._training_report=None;self._dataset_counts={}
        self.samples=label('Not recording','muted',True);body.addWidget(self.samples)
        self.receipt=label('',wrap=True);self.receipt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse);body.addWidget(self.receipt)
        self.calibration_status=label('','muted',True);body.addWidget(self.calibration_status)
        self.job_status=label('','muted');body.addWidget(self.job_status)
        self.alphabet_counts=label('','muted',True);body.addWidget(self.alphabet_counts)
        self.alphabet_counts.setStyleSheet('font-family:"Segoe UI Symbol";')
        bottom=QHBoxLayout();self.input_mirror=QCheckBox('Camera driver already mirrors its image');self.input_mirror.setToolTip('Only enable if your physical right hand is reported as left. This de-mirrors camera input before tracking; the preview stays in selfie view.')
        self.input_mirror.toggled.connect(lambda value:self._dispatch('input_mirrored',value));bottom.addWidget(self.input_mirror)
        bottom.addStretch();bottom.addWidget(label('ML coordinates: unmirrored · Preview: selfie','muted'));body.addLayout(bottom)
        layout.addWidget(self.dataset_scroll)
        self.status=label('','muted',True);layout.addWidget(self.status)
        foot=label('Limited ASL vocabulary and personal fingerspelling calibration.','muted');foot.setStyleSheet('font-size:11px;color:#777f8c;');layout.addWidget(foot)
        self._ui_times=deque(maxlen=120);self._last_metrics=time.perf_counter();self._timer=QTimer(self.window)
        self._timer.setInterval(16);self._timer.timeout.connect(self._drain);self._timer.start()

    def _button(self,text,action,value=None,primary=False,subtle=False):
        b=QPushButton(text)
        if primary:b.setObjectName('primary')
        elif subtle:b.setObjectName('subtle')
        b.clicked.connect(lambda checked=False:self._dispatch(action,value() if callable(value) else value))
        return b

    def _dispatch(self,action,value=None):self._on_action(action,value)

    def _step_label(self,delta):
        if self.labels.isEnabled():self.labels.setCurrentIndex((self.labels.currentIndex()+delta)%self.labels.count())

    def _select_calibration_group(self,name):
        previous=self.labels.currentText()
        choices=list(STATIC_LABELS) if name=='All static letters' else name.split(' / ')
        self.auto_advance.setText('After saving, go to next incomplete letter' if name=='All static letters' else 'After saving, go to next letter in group')
        self.labels.blockSignals(True);self.labels.clear();self.labels.addItems(choices)
        self.labels.setCurrentText(previous if previous in choices else choices[0]);self.labels.blockSignals(False)
        self._pending_advance=None;self._update_selected_label()

    def _update_selected_label(self):
        name=self.labels.currentText();entry=self._static_progress.get(name,{})
        state='READY' if entry.get('ready') else 'NEED DATA'
        set_text(self.selected_label_status,f'{name} · {entry.get("samples",0)} samples · {entry.get("sessions",0)} sessions · {state}'+
            ((' · '+entry['reason']) if entry.get('reason') and not entry.get('ready') else '')+
            (' · Not in loaded model — Train to include this letter' if name not in self._model_classes else ''))

    def _update_alphabet_progress(self,dataset):
        progress=dataset.get('static_progress') or alphabet_readiness(dataset)['static_progress']
        signature=tuple((c,entry['samples'],entry['sessions'],entry['ready'],entry['reason']) for c,entry in progress.items())
        self._static_progress=progress
        if signature!=self._progress_signature:
            self._progress_signature=signature
            for row,c in enumerate(STATIC_LABELS):
                entry=progress[c]
                for column,value in enumerate((c,str(entry['samples']),str(entry['sessions']),'READY' if entry['ready'] else 'NEED DATA')):
                    item=QTableWidgetItem(value);item.setToolTip(entry['reason']);self.alphabet_status.setItem(row,column,item)
            set_text(self.alphabet_counts,'   '.join(c+(' ✓' if progress[c]['ready'] else ' ○') for c in STATIC_LABELS))
        self._update_selected_label()
        ready=[c for c in STATIC_LABELS if progress[c]['ready']]
        set_text(self.training_readiness,f'{len(ready)}/24 READY · '+('Train will use '+', '.join(ready) if len(ready)>=2 else 'Collect at least two READY letters to train')+
            ' · Minimum: 50 valid samples in each of two sessions')

    def _advance_after_save(self,event,dataset,stats):
        # Receipts persist across events. Only a new save observed in this UI can advance.
        if event.get('source')=='replay':
            self._save_armed=False;self._pending_advance=None
        elif event.get('recording') or event.get('saving'):
            if not self._save_armed:
                self._last_saved_path=stats.get('saved_path')
                self._pending_advance=None
            self._save_armed=True
        path=stats.get('saved_path')
        if path and path!=self._last_saved_path:
            self._last_saved_path=path
            if self._save_armed and self.auto_advance.isChecked() and stats.get('saved_samples',0)>0 and not event.get('save_failed'):
                self._pending_advance=dict(path=path,label=stats.get('saved_label'),samples=stats['saved_samples'],session=stats.get('saved_session_id'))
            self._save_armed=False
        if event.get('save_failed'):
            self._pending_advance=None
            return
        if not self.auto_advance.isChecked():self._pending_advance=None
        pending=self._pending_advance
        if not pending or event.get('recording') or event.get('saving'):return
        persisted=any(Path(f['path'])==Path(pending['path']) and f['label']==pending['label'] and
            f['samples']==pending['samples'] and f['session_id']==pending['session'] for f in dataset.get('files',[]))
        if not persisted:return
        self._pending_advance=None
        if pending['label'] not in STATIC_LABELS:return
        if self.labels.currentText()!=pending['label']:return  # Preserve a newer manual selection.
        choices=[self.labels.itemText(i) for i in range(self.labels.count())]
        if pending['label'] not in choices:return
        index=choices.index(pending['label'])
        for step in range(1,len(choices)):
            candidate=choices[(index+step)%len(choices)]
            if self.calibration_group.currentText()!='All static letters' or not self._static_progress.get(candidate,{}).get('ready',False):
                self.labels.setCurrentText(candidate)
                return

    def _confirm_reset_class(self):
        selected=self.labels.currentText();count=self._dataset_counts.get(selected,0)
        answer=QMessageBox.question(self.window,'Reset recorded class',
            f'Archive all {count} saved {selected} samples from every session? You can then re-record {selected}. '
            'The original files will be kept in data/removed_samples. A model trained on removed files must be retrained.',
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
        if answer==QMessageBox.StandardButton.Yes:self._dispatch('reset_class',selected)

    def _show_training_report(self,report):
        if report==self._training_report:return
        self._training_report=report
        self.training_results_button.setEnabled(bool(report))
        if not report:
            self.training_result_text.setPlainText('No completed training report. Collect at least two READY static letters, then Train.')
            return
        classes=report.get('coverage') or report.get('classes') or []
        train=report.get('train',{});validation=report.get('validation',{})
        counts=report.get('samples_per_class') or {c:train.get('per_class',{}).get(c,{}).get('samples',0)+validation.get('per_class',{}).get(c,{}).get('samples',0) for c in classes}
        provenance=report.get('training_provenance',{})
        accuracy=validation.get('accuracy')
        lines=['Classes: '+', '.join(classes),f'Number of classes: {len(classes)}',
            'Saved samples used: '+', '.join(f'{c}: {counts.get(c,0)}' for c in classes),
            f'Training samples: {train.get("samples",0)}',f'Validation samples: {validation.get("samples",0)}',
            'Validation accuracy: '+metric(accuracy*100 if accuracy is not None else None,'%',1),
            'Train sessions: '+', '.join(provenance.get('train_sessions',[])),
            'Validation sessions: '+', '.join(provenance.get('validation_sessions',[])),
            '', 'Validation confusion matrix (rows = actual; columns = predicted)', '     '+'   '.join(classes)]
        lines.extend(f'{c}:  '+ '   '.join(str(n) for n in row) for c,row in zip(classes,validation.get('confusion_matrix',[])))
        lines.extend(['','Model: '+str(report.get('model','')),report.get('evidence_scope','')])
        lines.extend(str(w) for w in report.get('warnings',[]))
        self.training_result_text.setPlainText('\n'.join(lines))

    def _camera_controls(self,*args):
        self.camera_apply.setEnabled(self._camera_settings_allowed and bool(self.camera_devices.currentData()))

    def _update_camera_settings(self,event):
        selection=event.get('camera_selection') or {}
        device=event.get('camera_device') or {};selected_id=device.get('device_id')
        devices=event.get('camera_devices') or []
        signature=tuple((d['device_id'],d['name'],d['index'],d['kind']) for d in devices)
        if signature!=self._device_list_signature:
            pending=self.camera_devices.currentData()
            self._device_list_signature=signature
            self.camera_devices.blockSignals(True);self.camera_devices.clear()
            self.camera_devices.addItem('Choose a camera' if devices else 'No cameras found — refresh',None)
            for item in sorted(devices,key=lambda d:(d['kind']!='physical',d['name'].casefold(),d['device_id'])):
                duplicate_name=sum(d['name']==item['name'] for d in devices)>1
                title=item['name']+(f' (DSHOW {item["index"]})' if duplicate_name else '')
                self.camera_devices.addItem(title,item['device_id'])
                self.camera_devices.setItemData(self.camera_devices.count()-1,item['path'] or 'Device name identity',Qt.ItemDataRole.ToolTipRole)
            self.camera_devices.setCurrentIndex(max(0,self.camera_devices.findData(pending or selected_id)))
            self.camera_devices.blockSignals(False)
        if selected_id!=self._camera_selection:
            self._camera_selection=selected_id
            self.camera_devices.setCurrentIndex(max(0,self.camera_devices.findData(selected_id)))
        self._camera_settings_allowed=not event.get('recording',False) and not event.get('saving',False)
        self.camera_devices.setEnabled(self._camera_settings_allowed)
        self.camera_refresh.setEnabled(not event.get('camera_scanning',False))
        set_text(self.camera_refresh,'Refreshing…' if event.get('camera_scanning',False) else 'Refresh cameras')
        self._camera_controls()
        details=event.get('camera_diagnostics') or {}
        def result(value):return 'Not reported / not attempted' if value is None else str(value)
        size=details.get('frame_dimensions')
        live=event.get('camera_state')=='LIVE'
        set_text(self.camera_diagnostics,'\n'.join([
            'Device: '+(device.get('name') or 'Not selected'),
            'Backend: DSHOW',
            'Resolution: '+(f'{size[0]} × {size[1]}' if size else 'Not available'),
            'Capture FPS: '+metric(event.get('capture_fps') if live else None),
            'Display FPS: '+metric(self.camera.metrics().get('display_fps') if live else None),
            'Landmark latency: '+metric(event.get('landmark_ms') if live else None,' ms'),
            'State: '+event.get('camera_state','OFF'),
            'cap.isOpened(): '+result(details.get('cap_is_opened')),
            'First frame success: '+result(details.get('first_frame_success')),
            'Resolved DSHOW index: '+str(selection.get('index') if selection.get('index') is not None else 'Not resolved'),
            'Actual OpenCV backend: '+(details.get('actual_backend') or 'Not available'),
            'Error: '+(event.get('camera_error') or 'None'),
            event.get('camera_device_error') or '',
            event.get('camera_settings_error') or '',
            'Device identity: '+(device.get('device_id') or 'Not selected'),
            'Log: '+event.get('camera_log_path','')]))

    def update(self,event):
        with self._lock:self._latest=event

    def metrics_snapshot(self):
        return dict(**self.camera.metrics(),ui_update_ms=statistics.median(self._ui_times) if self._ui_times else None)

    def _drain(self):
        with self._lock:event,self._latest=self._latest,None
        if event is None:return
        start=time.perf_counter();replay=event.get('source')=='replay'
        if 'camera_state' not in event:event=dict(event,camera_state='REPLAY' if replay else 'OFF')
        self.camera.set_event(event)
        self._update_camera_settings(event)
        set_text(self.source_badge,'REPLAY · NOT LIVE' if event['camera_state']=='REPLAY' else event['camera_state'])
        set_text(self.handedness,f'Physical hand: {event.get("physical_handedness","—")}  ·  Model label: {event.get("raw_handedness","—")}')
        mode=event.get('mode','conversation')
        self._spelling_mode=mode=='free_spell'
        self.space.setEnabled(self._spelling_mode and bool(event.get('word')) and not event.get('word','').endswith(' '))
        self.backspace.setEnabled(self._spelling_mode and bool(event.get('word')))
        self.space_shortcut.setEnabled(self._spelling_mode)
        set_text(self.speak,'Speak' if self._spelling_mode else 'Speak response')
        settings=event.get('fingerspelling_settings')
        if settings and settings!=self._spelling_settings_signature:
            self._spelling_settings_signature=dict(settings)
            self.repeat_threshold.setValue(settings['repeat_threshold']);self.repeat_cooldown.setValue(settings['repeat_cooldown_ms'])
            self.auto_spacing.setChecked(settings['auto_space']);self.spacing_pause.setValue(settings['space_pause_ms'])
            self.space_key_edit.setKeySequence(QKeySequence(settings['space_shortcut']))
            self.space_shortcut.setKey(QKeySequence(settings['space_shortcut']))
        motion=event.get('fingerspelling') or {}
        set_text(self.motion_status,'Recognition: '+motion.get('mode','STATIC')+(' · '+(motion.get('label') or '')+' · '+motion.get('state','') if motion.get('mode')=='MOTION' else ''))
        repeat=motion.get('repeat') or {};spacing=motion.get('spacing') or {}
        boundary=('Space in '+str(round(spacing.get('remaining_ms',0)/1000,1))+'s' if spacing.get('state')=='counting' else
                  ('Auto space: withdraw hand '+str(round(motion.get('space_pause_ms',1200)/1000,1))+'s' if motion.get('auto_space') else 'Auto space off'))
        if motion.get('mode')!='MOTION' and motion:
            set_text(self.motion_status,boundary+' · Repeat: '+repeat.get('state','WAITING')+
                     (' '+str(round(repeat.get('progress',0)*100))+'%' if repeat.get('progress') else ''))
        self.motion_status.setToolTip(motion.get('reason','')+' · '+repeat.get('reason',''))
        self.motion_progress.setVisible(motion.get('mode')=='MOTION')
        self.motion_progress.setValue(round(motion.get('progress',0)*100))
        if motion.get('state') in ('completed','cooldown'):
            self.motion_status.setToolTip(motion.get('reason','')+' · Geometric match score, not a calibrated probability')
        for key,button in self.modes.items():
            if button.isChecked()!=(mode==key):button.setChecked(mode==key)
        set_text(self.panel_title,{'conversation':'Conversation','free_spell':'Fingerspelling','alphabet_tutor':'Tutor'}.get(mode,'Conversation'))
        set_text(self.word,event.get('word') or '')
        self.word.setVisible(bool(event.get('word')))
        set_text(self.recognized,event.get('recognized') or '—');set_text(self.interpretation,event.get('interpretation') or '—')
        set_text(self.response,event.get('response') or '')
        set_text(self.stable,'Committed: '+(event.get('stable') or '—'));set_text(self.candidate,event.get('candidate') or '—')
        confidence=event.get('confidence')
        set_text(self.live_candidate,f'Candidate: {event.get("candidate") or "—"}   Confidence: '+metric(confidence*100 if confidence is not None else None,'%',0))
        classes=', '.join(event.get('model_classes') or []) or '—'
        stable=f'{event.get("stable_duration_ms",0)} / {event.get("dwell_ms",350)} ms'
        if event.get('locked_label'):stable+=' · held '+event['locked_label']
        set_text(self.recognition_debug,'\n'.join([
            f'Hand detected: {"YES" if event.get("hand_detected") else "NO"} · Model loaded: {"YES" if event.get("model_loaded") else "NO"}',
            f'Model classes: {classes} · Stable: {stable}',
            'Synthetic token replay' if replay else event.get('recognition_reason','')]))
        self.recognition_debug.setToolTip(event.get('model_error',''))
        detail=event.get('prediction_diagnostics') or {}
        top=' · '.join(f'{p["label"]} {p["score"]:.1%}' for p in detail.get('top_five',[])) or '—'
        self._model_classes=event.get('model_classes') or []
        missing=event.get('untrained_ready_labels') or []
        set_text(self.prediction_details,'\n'.join([
            'Raw candidate: '+(detail.get('classifier_candidate') or '—')+' · Threshold: '+metric(event.get('confidence_threshold',.8)*100,'%',0),
            'Top five: '+top,'Classifier: '+detail.get('classifier_decision','No current scores'),
            'Temporal: '+detail.get('temporal_decision','Not evaluated'),
            ('Ready but absent from model: '+', '.join(missing)+'. Train and reload.') if missing else '']))
        set_text(self.speech_status,event.get('speech_status') or '');self.speech_status.setVisible(bool(event.get('speech_status')))
        set_text(self.status,event.get('fingerspelling_settings_error') or event.get('camera_error') or event.get('camera_device_error') or event.get('camera_settings_error') or event.get('status') or '')
        target=event.get('target');self.tutor.setVisible(bool(target));set_text(self.tutor,('Correct. ' if event.get('correct') else 'Show ') + str(target or ''))
        metrics=self.metrics_snapshot();show=event.get('metrics_visible',True)
        self.metrics_label.setVisible(show)
        set_text(self.cpu_ms,'—' if replay else metric(event.get('model_ms',event.get('cpu_latency_ms')),' ms',3))
        set_text(self.metrics_label,'   ·   '.join([
            'Capture '+metric(event.get('capture_fps'),' fps'),'Display '+metric(metrics.get('display_fps'),' fps'),
            'Landmarks '+metric(event.get('landmark_ms'),' ms'),'Model '+self.cpu_ms.text(),
            'Motion '+metric(event.get('motion_ms'),' ms')+' '+motion.get('state',''),
            'UI update '+metric(metrics.get('ui_update_ms'),' ms',2),'Paint '+metric(metrics.get('ui_render_ms'),' ms',2),
            'Save '+metric(event.get('disk_write_ms'),' ms')]))
        stats=event.get('recording_stats') or {};saving=event.get('saving',False);active=event.get('recording',False)
        state='Save failed' if event.get('save_failed') else 'Saving' if saving else 'Recording' if active else 'Not recording'
        current_label=stats.get('label') if active or saving or event.get('save_failed') else self.labels.currentText()
        set_text(self.samples,f'{state}  ·  Label {current_label}  ·  Session {stats.get("session_id","—")}  ·  Captured {stats.get("samples",0)}  ·  Valid {stats.get("valid_frames",0)}  ·  Rejected {stats.get("rejected_frames",0)}  ·  Processed {stats.get("frames_seen",0)}  ·  Cadence skipped {stats.get("skipped_frames",0)}  ·  {metric(stats.get("elapsed_ms",0)/1000," s")}')
        if stats.get('saved_path'):
            set_text(self.receipt,f'Saved successfully · {stats.get("saved_label")} · {stats.get("saved_samples")} valid samples · Session {stats.get("saved_session_id")}\n{stats["saved_path"]}')
            self.receipt.setToolTip(stats['saved_path'])
        set_text(self.calibration_status,event.get('calibration_status') or '')
        dataset=event.get('dataset') or {};counts=dataset.get('counts') or {}
        self._dataset_counts=counts
        self._update_alphabet_progress(dataset)
        self._advance_after_save(event,dataset,stats)
        self._show_training_report(event.get('training_report') or {})
        dynamic=[]
        for letter in ('J','Z'):
            clips=[f for f in dataset.get('files',[]) if f.get('label')==letter]
            dynamic.append(f'{letter}: {len(clips)} clips / {len({f["session_id"] for f in clips})} sessions')
        set_text(self.dynamic_samples,' · '.join(dynamic)+' · Finish the whole stroke, hold briefly, then Stop and save.')
        set_text(self.motion_profile_status,event.get('motion_profile_status') or '')
        report=event.get('motion_calibration_report') or {}
        if report!=self._motion_report:
            self._motion_report=report
            lines=[report.get('scope','No calibration loaded.'),report.get('warning','')]
            for letter,row in report.get('per_class',{}).items():
                lines.append(f"{letter}: {row['usable_clips']}/{row['saved_clips']} usable clips; {row['detected_strokes']} replay detections; {row['sessions']} sessions")
            for row in report.get('clips',[]):
                lines.append(f"{row['path']}: {'USED' if row['usable'] else 'EXCLUDED (preserved)'} — {row['reason']}")
            self.motion_result_text.setPlainText('\n'.join(lines))
        set_text(self.dataset_total,f'{dataset.get("total_samples",0)} saved samples'+(f' · {dataset["legacy_samples"]} legacy samples preserved' if dataset.get('legacy_samples') else ''))
        self.alphabet_counts.setToolTip('\n'.join(dataset.get('errors',[])))
        if dataset.get('errors'):set_text(self.calibration_status,'Dataset error: '+'; '.join(dataset['errors']))
        coverage=event.get('coverage') or [];set_text(self.coverage,'Trained: '+', '.join(coverage) if coverage else 'No trained model')
        job=event.get('job') or {};running=job.get('state')=='running'
        set_text(self.job_status,f'{job.get("kind","").capitalize()} · {job.get("elapsed_s",0):.0f}s · {job.get("progress","")}' if running else '')
        self.job_status.setVisible(running)
        caps=event.get('capabilities') or {}
        for name,button in [('record',self.record),('finish_record',self.finish),('new_session',self.new_session),('train',self.train),('evaluate',self.evaluate),('reload',self.reload),('speak',self.speak),('reset_class',self.reset_class)]:
            cap=caps.get(name,dict(enabled=False,reason='Waiting for application state.'))
            enabled=bool(cap['enabled']) and not (replay and name=='record')
            if button.isEnabled()!=enabled:button.setEnabled(enabled)
            if button.toolTip()!=cap.get('reason',''):button.setToolTip(cap.get('reason',''))
        collecting=active or saving or event.get('save_failed',False)
        self.labels.setEnabled(not collecting)
        self.previous_label.setEnabled(not collecting);self.next_label.setEnabled(not collecting)
        self.calibration_group.setEnabled(not collecting)
        self.record_j.setEnabled(self.record.isEnabled());self.record_z.setEnabled(self.record.isEnabled())
        self.calibrate_motion.setEnabled(self.train.isEnabled());self.reload_motion.setEnabled(self.train.isEnabled())
        self.input_mirror.setEnabled(not active and not saving)
        self.input_mirror.blockSignals(True);self.input_mirror.setChecked(event.get('input_mirrored',False));self.input_mirror.blockSignals(False)
        self._ui_times.append((time.perf_counter()-start)*1000)
        if time.perf_counter()-self._last_metrics>=1:
            self._last_metrics=time.perf_counter();self._dispatch('ui_metrics',self.metrics_snapshot())

    def run(self):self.window.show();return self._app.exec()
