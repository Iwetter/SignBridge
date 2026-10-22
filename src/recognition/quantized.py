"""Integer oracle for hardware. NumPy backend is explicitly not a physical FPGA."""
from pathlib import Path
from time import perf_counter
import zlib
import numpy as np

from src.contracts import ALPHABET, Prediction
from .cpu import ModelError, masked_softmax, read_npz, validate_features, validate_metadata


def quantize_features(features: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(validate_features(features) * 1024), -4096, 4096).astype(np.int16)


def model_crc32(arrays: dict) -> int:
    """CRC over C-order w1:i8,b1:<i64,w2:i8,b2:<i64,shift1:<i32,shift2:<i32."""
    crc = 0
    for key, dtype in [('w1', 'i1'), ('b1', '<i8'), ('w2', 'i1'), ('b2', '<i8'),
                       ('shift1', '<i4'), ('shift2', '<i4')]:
        crc = zlib.crc32(np.asarray(arrays[key], dtype=dtype).tobytes(order='C'), crc)
    return crc & 0xffffffff


def validate_quantized_arrays(arrays: dict) -> None:
    for key, shape, dtype in [('w1', (63, 32), 'int8'), ('b1', (32,), 'int64'),
                              ('w2', (32, 26), 'int8'), ('b2', (26,), 'int64')]:
        if key not in arrays or arrays[key].shape != shape or arrays[key].dtype != np.dtype(dtype):
            raise ModelError(f'{key} must have dtype {dtype} and shape {shape}.')
    for key in ('shift1', 'shift2'):
        if (key not in arrays or arrays[key].shape != () or arrays[key].dtype.kind not in 'iu' or
                not 0 <= int(arrays[key]) <= 30):
            raise ModelError(f'{key} must be an integer scalar in [0,30].')
    # Python integer math avoids overflow while validating even hostile int64 bias arrays.
    bound1 = [sum(abs(int(w)) * 4096 for w in arrays['w1'][:, j]) + abs(int(arrays['b1'][j]))
              for j in range(32)]
    bound2 = [sum(abs(int(w)) * 32767 for w in arrays['w2'][:, j]) + abs(int(arrays['b2'][j]))
              for j in range(26)]
    if max(bound1 + bound2) > np.iinfo(np.int64).max:
        raise ModelError('Quantized model can overflow signed 64-bit accumulation.')
    if (max(bound2) >> int(arrays['shift2'])) > np.iinfo(np.int32).max:
        raise ModelError('Quantized model can overflow signed 32-bit output logits.')


def load_quantized_model(path: str | Path) -> tuple[dict, dict]:
    arrays = read_npz(path)
    if 'metadata' not in arrays:
        raise ModelError('Quantized model metadata missing.')
    meta = validate_metadata(arrays['metadata'])
    validate_quantized_arrays(arrays)
    model_id = model_crc32(arrays)
    if meta.get('model_id') != model_id:
        raise ModelError('Quantized model ID does not match parameter bytes; export again.')
    return {k: arrays[k] for k in ('w1', 'b1', 'w2', 'b2', 'shift1', 'shift2')}, meta


def integer_forward(x: np.ndarray, arrays: dict, *, return_hidden: bool = False):
    """x is already quantized int16. Arithmetic right shift follows Python/NumPy."""
    x = np.asarray(x)
    if x.shape != (63,) or x.dtype != np.dtype('int16') or np.any(np.abs(x.astype(np.int32)) > 4096):
        raise ValueError('Integer features must be int16 shape (63,) in [-4096,4096].')
    h = np.clip((x.astype(np.int64) @ arrays['w1'].astype(np.int64) + arrays['b1']) >>
                int(arrays['shift1']), 0, 32767).astype(np.int16)
    result = (h.astype(np.int64) @ arrays['w2'].astype(np.int64) + arrays['b2']) >> int(arrays['shift2'])
    if np.any(result < -(2**31)) or np.any(result > 2**31 - 1):
        raise ModelError('Integer output overflow.')
    logits = result.astype(np.int32)
    return (h, logits) if return_hidden else logits


class QuantizedRecognizer:
    name = 'CPU (quantized reference)'

    def __init__(self, model_path: str | Path = 'models/alphabet_quantized.npz'):
        self.arrays, self.metadata = load_quantized_model(model_path)
        self.supported_labels = tuple(self.metadata['supported_labels'])
        self.model_id = self.metadata['model_id']

    def predict(self, features: np.ndarray, timestamp_ms: int) -> Prediction:
        start = perf_counter()
        logits = integer_forward(quantize_features(features), self.arrays)
        scores = masked_softmax(logits.astype(np.float64) / 1024, self.supported_labels)
        winner = int(scores.argmax())
        return Prediction(ALPHABET[winner], float(scores[winner]), int(timestamp_ms),
                          source='cpu-quantized', latency_ms=(perf_counter()-start)*1000, scores=scores)

    def close(self) -> None:
        pass
