"""Strict one-request serial backend. ApplicationController supplies CPU fallback.

Only select a port with a deployed SignBridge bridge. The DE10-Standard onboard
USB UART terminates at HPS and is normally a Linux console, not this endpoint.
"""
from __future__ import annotations

import struct
import threading
from time import perf_counter
import numpy as np

from src.contracts import ALPHABET, CLASS_TO_ID, Prediction
from src.recognition.cpu import masked_softmax
from src.recognition.quantized import load_quantized_model, quantize_features
from .protocol import INFER, RESULT, ProtocolError, encode_packet, decode_packet


class FPGABackend:
    name = 'FPGA'

    def __init__(self, port: str, model_path='models/alphabet_quantized.npz', *,
                 timeout_s: float = 0.25, serial_factory=None):
        if not isinstance(port, str) or not port.strip():
            raise ValueError('Specify a verified SignBridge serial endpoint explicitly.')
        if not 0 < timeout_s <= 0.25:
            raise ValueError('Hardware timeout must be in (0, 0.25] seconds.')
        self.arrays, self.metadata = load_quantized_model(model_path)
        self.supported_labels = tuple(self.metadata['supported_labels'])
        self.model_id = self.metadata['model_id']
        self.port, self.timeout_s = port, float(timeout_s)
        self._serial_factory, self._serial = serial_factory, None
        self._sequence = 0
        self._lock = threading.Lock()
        self.last_compute_ms = None
        self.last_cycles = None
        clock = self.metadata.get('clock_hz')
        self.clock_hz = float(clock) if isinstance(clock, (int, float)) and 0 < clock <= 1e9 else None
        # A model-specific envelope catches corrupt/impossible integer score values.
        w2 = self.arrays['w2'].astype(np.int64)
        shift = int(self.arrays['shift2'])
        self._minimum = (np.minimum(w2, 0).sum(axis=0)*32767 + self.arrays['b2']) >> shift
        self._maximum = (np.maximum(w2, 0).sum(axis=0)*32767 + self.arrays['b2']) >> shift

    def _open(self):
        if self._serial is None:
            factory = self._serial_factory
            if factory is None:
                import serial  # Optional and lazy; CPU use does not need pyserial.
                factory = serial.Serial
            self._serial = factory(port=self.port, baudrate=115200, bytesize=8,
                                   parity='N', stopbits=1, timeout=self.timeout_s,
                                   write_timeout=self.timeout_s)
        return self._serial

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError('FPGA response exceeded the bounded request deadline.')
        return remaining

    def _read_exact(self, stream, count, deadline):
        data = bytearray()
        while len(data) < count:
            stream.timeout = self._remaining(deadline)
            chunk = stream.read(count-len(data))
            if not chunk:
                raise TimeoutError('FPGA response incomplete or endpoint is not SignBridge.')
            data.extend(chunk)
        return bytes(data)

    def predict(self, features, timestamp_ms: int) -> Prediction:
        x = quantize_features(features)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Only one FPGA inference request may be in flight.')
        try:
            stream = self._open()
            sequence = self._sequence
            self._sequence = (sequence + 1) & 0xffff
            request = encode_packet(INFER, sequence, struct.pack('<I', self.model_id) + x.astype('<i2').tobytes())
            # Opening a driver is initialization; measured round trip begins at I/O.
            start = perf_counter()
            deadline = start + self.timeout_s
            stream.write_timeout = self._remaining(deadline)
            if stream.write(request) != len(request):
                raise ProtocolError('Serial request was only partially written.')
            # No flush(): some drivers can block it beyond the request deadline.
            header = self._read_exact(stream, 8, deadline)
            if header[:4] != b'SB\x01\x81':
                raise ProtocolError('Endpoint sent an invalid RESULT header; verify HPS bridge.')
            size = struct.unpack('<H', header[6:8])[0]
            if size != 114:
                raise ProtocolError('RESULT payload must contain exactly 26 signed scores.')
            packet = decode_packet(header + self._read_exact(stream, size+2, deadline))
            self._remaining(deadline)
            if packet.message_type != RESULT or packet.sequence != sequence:
                raise ProtocolError('Unexpected response type or sequence.')
            status, model_id, winner, cycles = struct.unpack('<BIBI', packet.payload[:10])
            if status != 0 or model_id != self.model_id or cycles == 0:
                raise ProtocolError('FPGA status, loaded model ID, or cycle count is invalid.')
            if winner >= 26 or ALPHABET[winner] not in self.supported_labels:
                raise ProtocolError('FPGA returned a masked or unsupported class.')
            logits = np.frombuffer(packet.payload[10:], dtype='<i4').astype(np.int64)
            if np.any(logits < self._minimum) or np.any(logits > self._maximum):
                raise ProtocolError('FPGA scores exceed the loaded model arithmetic bounds.')
            ids = [CLASS_TO_ID[label] for label in self.supported_labels]
            expected_winner = ids[int(np.argmax(logits[ids]))]
            if winner != expected_winner:
                raise ProtocolError('FPGA class ID disagrees with supported-score argmax.')
            scores = masked_softmax(logits.astype(np.float64)/1024, self.supported_labels)
            if not np.isfinite(scores).all() or not np.isclose(scores.sum(), 1):
                raise ProtocolError('FPGA scores did not produce a finite probability vector.')
            self.last_cycles = cycles
            self.last_compute_ms = cycles/self.clock_hz*1000 if self.clock_hz else None
            latency = (perf_counter()-start)*1000
            return Prediction(ALPHABET[winner], float(scores[winner]), int(timestamp_ms),
                              source='fpga', latency_ms=latency, scores=scores)
        except Exception:
            # Discard stale bytes / session on any failure before allowing retry.
            if self._serial is not None:
                self._serial.close()
                self._serial = None
            raise
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            if self._serial is not None:
                self._serial.close()
                self._serial = None
