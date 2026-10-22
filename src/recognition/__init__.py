"""Safe, portable static-letter inference (no training dependencies)."""
from .cpu import CPURecognizer, ModelError, load_float_model, masked_softmax, validate_features
from .quantized import QuantizedRecognizer, load_quantized_model, quantize_features, integer_forward, model_crc32

__all__ = ['CPURecognizer', 'QuantizedRecognizer', 'ModelError', 'load_float_model',
           'load_quantized_model', 'masked_softmax', 'validate_features', 'quantize_features',
           'integer_forward', 'model_crc32']
