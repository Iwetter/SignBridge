"""Coordinate completed motions and the existing static dwell/held-letter gate."""
from src.contracts import Prediction
from src.fingerspelling_settings import validate_settings
from src.vision.features import primary_hand
from .decoder import TemporalDecoder
from .boundaries import RepeatDetector, AutoSpaceDetector
from .jz import JZRecognizer


class FingerspellingCoordinator:
    def __init__(self, decoder=None, settings=None, motion_profile=None):
        self.decoder = decoder or TemporalDecoder()
        self.settings = validate_settings(settings or {})
        if motion_profile is None:
            self.motion = JZRecognizer()
        else:
            from .motion_calibration import CalibratedGeometry
            self.motion = CalibratedGeometry(motion_profile)
        self.repeat = RepeatDetector(self.settings['repeat_threshold'], self.settings['repeat_cooldown_ms'],
                                     self.decoder.confidence_threshold)
        self.spacing = AutoSpaceDetector(self.settings['auto_space'], self.settings['space_pause_ms'])
        self.reset()

    def reset(self):
        self.decoder.reset()
        self.motion.reset()
        self.repeat.reset()
        self.spacing.reset()
        self.prediction = None
        self.space_requested = False
        self.reason = ''
        self._motion_result = None
        self._attempted_motion = None
        self._motion_absence_since = None
        self._reserving = False
        self._reserved_static = None
        self._reservation_lock = None
        self._reservation_shape = None
        self._last_time = None
        self.replace_last_static = None
        self._motion_start_static = None

    def cancel_motion_promotion(self):
        self._motion_start_static = None

    def _track_static(self, token):
        if token:
            self._motion_start_static = ('I' if token.label == 'I' and
                self.motion.snapshot().get('calibration_id') and self.motion.snapshot()['label'] == 'J' else None)
        return token

    def _discard_reservation(self, now):
        if self._reserving:
            self.decoder.reject(now)
            self.decoder.locked_label = self._reservation_lock
        self._reserving = False
        self._reserved_static = None

    def _release_reservation(self, prediction, now):
        reserved = self._reserved_static
        self._reserving = False
        self._reserved_static = None
        if reserved is None:
            return None
        token = self.decoder.commit_completed(Prediction(reserved.label, reserved.confidence, now, reserved.source))
        # Seed a different following letter's ordinary dwell on this same frame.
        self.decoder.update(prediction, now)
        return self._track_static(token)

    def update(self, observation, static_prediction, paused=False, has_text=False):
        now = observation.timestamp_ms
        self.replace_last_static = None
        if self._last_time is not None and now - self._last_time > self.decoder.max_gap_ms:
            self._discard_reservation(now)
            self.cancel_motion_promotion()
        self._last_time = now
        self.prediction = static_prediction
        self.space_requested = False
        self.reason = ''
        if paused:
            self._discard_reservation(now)
            self.decoder.reject(now)
            self.motion.reset()
            self.repeat.reset()
            self.spacing.reset()
            self._motion_result = None
            self._attempted_motion = None
            self._motion_absence_since = None
            self.reason = 'Recognition paused during collection or maintenance'
            return None
        hand = primary_hand(observation)
        # Multiple/invalid visible hands are not an intentional withdrawal.
        if hand is None and observation.hands:
            self._discard_reservation(now)
            self.decoder.reject(now)
            self.motion.reset()
            self.repeat.reset()
            self.spacing.reset()
            self._motion_result = None
            self.prediction = None
            self.reason = 'Invalid or ambiguous landmarks'
            return None
        self.space_requested = self.spacing.update(now, hand is not None, has_text)
        completed = self.motion.update(observation, static_prediction)
        motion = self.motion.snapshot()
        if motion['label'] != 'J':
            self.cancel_motion_promotion()
        if hand is not None:
            self._motion_absence_since = None
            if motion['label'] != self._attempted_motion:
                self._attempted_motion = None
        elif self._attempted_motion:
            if self._motion_absence_since is None:
                self._motion_absence_since = now
            if now - self._motion_absence_since >= 200:
                self._attempted_motion = None
        if self.motion.motion_active:
            self._attempted_motion = motion['label']
            self._discard_reservation(now)
        if completed:
            self._discard_reservation(now)
            self.prediction = self._motion_result = completed
            self.repeat.reset()
            self.spacing.update(now, True, True)
            self.reason = 'Completed motion; geometric score'
            token = self.decoder.commit_completed(completed, repeat=True)
            if token and completed.label == 'J' and self._motion_start_static == 'I':
                self.replace_last_static = 'I'
            self.cancel_motion_promotion()
            return token
        if motion['state'] == 'cooldown' and self._motion_result is not None:
            self.prediction = self._motion_result
            self.decoder.reject(now)
            self.reason = 'Motion committed; awaiting a new stroke or handshape'
            return None
        self._motion_result = None
        if self.motion.motion_active or self._attempted_motion:
            self._discard_reservation(now)
            self.decoder.reject(now)
            self.repeat.reset()
            self.reason = motion['reason'] if self.motion.pending or self.motion.motion_active else 'Incomplete motion; relax or change handshape to retry'
            return None
        if self.motion.pending:
            if self._reserving and self._reservation_shape != motion['label']:
                token = self._release_reservation(static_prediction, now)
                if token:
                    return token
            if not self._reserving:
                self._reserving = True
                self._reservation_lock = self.decoder.locked_label
                self._reservation_shape = motion['label']
            if self._reserved_static is None:
                self._reserved_static = self.decoder.update(static_prediction, now)
                if self._reserved_static:
                    self.decoder.locked_label = self._reservation_lock
                    self.spacing.update(now, True, True)
            else:
                self.decoder.reject(now)
            self.reason = 'Checking for motion; static dwell is retained'
            return None
        if self._reserving:
            token = self._release_reservation(static_prediction, now)
            if token:
                return token
        if self.repeat.update(observation, static_prediction, self.decoder.locked_label):
            repeated = Prediction(static_prediction.label, static_prediction.confidence, now,
                                  'repeat-motion', scores=static_prediction.scores)
            self.prediction = repeated
            self.reason = 'Intentional rightward repeat'
            return self.decoder.commit_completed(repeated, repeat=True)
        token = self.decoder.update(static_prediction, now)
        if token and hand is not None:
            self.repeat.prime(observation, static_prediction, self.decoder.locked_label)
            self.spacing.update(now, True, True)
        return self._track_static(token)

    def snapshot(self):
        motion = self.motion.snapshot()
        retry = bool(self._attempted_motion and self._motion_result is None and not self.motion.pending and not self.motion.motion_active)
        return {**motion, 'mode': 'MOTION' if self.motion.pending or self.motion.motion_active or
                retry or motion['state'] in ('completed', 'cooldown') else 'STATIC',
                'state': 'retry' if retry else motion['state'],
                'suppress_static': bool(self.motion.pending or self.motion.motion_active or retry),
                'reason': self.reason or motion['reason'], 'repeat': self.repeat.snapshot(),
                'spacing': self.spacing.snapshot(), 'auto_space': self.settings['auto_space'], 'space_pause_ms': self.settings['space_pause_ms']}
