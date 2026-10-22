"""Bounded, nonblocking speech worker; optional dependencies load on its thread."""
from __future__ import annotations

import queue
import threading


class SpeechWorker:
    def __init__(self, on_status=None):
        self.status = 'Starting local speech…'
        self.error = None
        self._on_status = on_status
        self._queue = queue.Queue(maxsize=3)
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._last_text = None
        self._thread = threading.Thread(target=self._run, name='SignBridge speech', daemon=True)
        self._thread.start()

    def _status(self, value):
        self.status = value
        if self._on_status:
            try:
                self._on_status(value)
            except Exception:
                pass

    def speak(self, text):
        text = str(text or '').strip()
        if not text or self._closed.is_set() or self.error:
            return
        with self._lock:
            if text == self._last_text:
                return
            self._last_text = text
            try:
                self._queue.put_nowait(text)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(text)

    def _run(self):
        engine = None
        com = None
        try:
            import sys
            if sys.platform == 'win32':
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    com = pythoncom
                except ImportError:
                    pass
            import pyttsx3
            engine = pyttsx3.init(driverName='sapi5' if sys.platform == 'win32' else None)
            engine.setProperty('rate', 172)
            self._status('Local speech ready')
            while not self._closed.is_set():
                try:
                    text = self._queue.get(timeout=0.15)
                except queue.Empty:
                    continue
                if text is None or self._closed.is_set():
                    break
                self._status('Speaking response…')
                engine.say(text)
                engine.runAndWait()
                with self._lock:
                    if self._last_text == text:
                        self._last_text = None
                self._status('Local speech ready')
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            self._status(f'Speech unavailable: {self.error}')
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if com is not None:
                com.CoUninitialize()

    def close(self):
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=0.6)
