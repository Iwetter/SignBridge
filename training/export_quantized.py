"""Export the exact integer MLP contract, ROMs, and reproducible oracle vectors."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import CLASS_TO_ID
from src.recognition import load_float_model, integer_forward, model_crc32, quantize_features
from src.recognition.quantized import validate_quantized_arrays


def quantize_model(arrays: dict) -> dict:
    result = {}
    for layer in (1, 2):
        weights = arrays[f'W{layer}'].astype(np.float64)
        bias = arrays[f'b{layer}'].astype(np.float64)
        maximum = float(np.abs(weights).max())
        if maximum > 127:
            raise ValueError('Weight magnitude exceeds signed int8 scale with nonnegative shifts; retrain with regularization.')
        shift = 7 if maximum == 0 else min(20, max(0, int(np.floor(np.log2(127 / maximum)))))
        qweight = np.rint(weights * (2**shift))
        qbias = np.rint(bias * (1024 * 2**shift))
        if np.max(np.abs(qweight)) > 127 or np.max(np.abs(qbias)) >= 2**63:
            raise ValueError('Parameter quantization would overflow.')
        result[f'w{layer}'] = qweight.astype(np.int8)
        result[f'b{layer}'] = qbias.astype(np.int64)
        result[f'shift{layer}'] = np.asarray(shift, dtype=np.int32)
    validate_quantized_arrays(result)
    return result


def write_hex(path: Path, values: np.ndarray, bits: int) -> None:
    mask = (1 << bits) - 1
    width = (bits+3)//4
    path.write_text(''.join(f'{int(value)&mask:0{width}x}\n' for value in np.asarray(values).ravel(order='C')), encoding='ascii')


def export_model(model_path: str | Path, output: str | Path, rom_dir: str | Path,
                 *, vector_count: int = 12, seed: int = 42) -> dict:
    if not 5 <= vector_count <= 4096:
        raise ValueError('vector_count must be 5..4096 (first five exercise zero and input boundaries).')
    arrays, metadata = load_float_model(model_path)
    if metadata.get('feature_set','coordinates-v1')!='coordinates-v1':
        raise ValueError('This feature set is CPU-only. The existing FPGA export accepts 63 raw coordinates only; refusing an incompatible interface.')
    if metadata.get('temperature',1.)!=1.:
        raise ValueError('Calibrated scores are CPU-only in this version. Existing FPGA export does not implement temperature; refusing mismatched confidence.')
    quantized = quantize_model(arrays)
    model_id = model_crc32(quantized)
    metadata = dict(metadata)
    metadata.update({'model_id': model_id, 'model_id_hex': f'{model_id:08x}',
                     'source_float_model_sha256': sha256(Path(model_path).read_bytes()).hexdigest(),
                     'quantization': {'input_fractional_bits': 10, 'hidden_fractional_bits': 10,
                                      'output_fractional_bits': 10, 'rounding': 'nearest ties to even for quantization; arithmetic right shift for layers',
                                      'shift1': int(quantized['shift1']), 'shift2': int(quantized['shift2']),
                                      'hidden_saturation': [0, 32767], 'accumulator_bits': 64,
                                      'crc_order': 'w1:i1 C, b1:<i8 C, w2:i1 C, b2:<i8 C, shift1:<i4 scalar, shift2:<i4 scalar'}})
    output, rom_dir = Path(output), Path(rom_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    rom_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **quantized, metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
    for name in ('w1', 'w2'):
        write_hex(rom_dir / f'{name}.hex', quantized[name], 8)
    for name in ('b1', 'b2'):
        write_hex(rom_dir / f'{name}.hex', quantized[name], 64)
    ids = np.asarray([CLASS_TO_ID[x] for x in metadata['supported_labels']])
    supported_mask = np.zeros(26, dtype=np.uint8)
    supported_mask[ids] = 1
    write_hex(rom_dir / 'supported_mask.hex', supported_mask, 1)
    rng = np.random.default_rng(seed)
    vectors = rng.uniform(-1, 1, (vector_count, 63)).astype(np.float32)
    vectors[0] = 0
    vectors[1] = 4
    vectors[2] = -4
    vectors[3] = np.where(np.arange(63) % 2, -4, 4)
    vectors[4] = np.linspace(-4, 4, 63)
    inputs = np.asarray([quantize_features(x) for x in vectors], dtype=np.int16)
    results = [integer_forward(x, quantized, return_hidden=True) for x in inputs]
    hidden = np.asarray([row[0] for row in results], dtype=np.int16)
    logits = np.asarray([row[1] for row in results], dtype=np.int32)
    float_hidden = np.maximum(vectors @ arrays['W1'] + arrays['b1'], 0)
    float_logits = float_hidden @ arrays['W2'] + arrays['b2']
    class_ids = ids[logits[:, ids].argmax(axis=1)].astype(np.uint8)
    float_class_ids = ids[float_logits[:, ids].argmax(axis=1)].astype(np.uint8)
    vector_meta = {'synthetic_numeric_vectors': True, 'recognition_evidence': False, 'seed': seed,
                   'model_id': model_id, 'vector_count': vector_count,
                   'description': 'First five: all-zero, +4, -4, alternating +/-4, linear ramp. Remaining: seeded uniform [-1,1].'}
    np.savez_compressed(rom_dir / 'test_vectors.npz', features=vectors, inputs=inputs, hidden=hidden,
                        logits=logits, class_ids=class_ids, float_logits=float_logits,
                        metadata=np.asarray(json.dumps(vector_meta)))
    for filename, data, bits in [('input_vectors.hex', inputs, 16), ('expected_hidden.hex', hidden, 16),
                                 ('expected_logits.hex', logits, 32), ('expected_class.hex', class_ids, 8)]:
        write_hex(rom_dir / filename, data, bits)
    supported_integer = sum(1 << int(i) for i in ids)
    config = {'model_id': model_id, 'model_id_hex': f'{model_id:08x}', 'shift1': int(quantized['shift1']),
              'shift2': int(quantized['shift2']), 'supported_mask_hex': f'{supported_integer:07x}',
              'supported_labels': metadata['supported_labels'], 'input_dim': 63, 'hidden_dim': 32,
              'output_dim': 26, 'test_vector_count': vector_count,
              'array_order': {'w1': 'input-major: address = input_index*32 + hidden_index',
                              'w2': 'hidden-major: address = hidden_index*26 + class_id',
                              'vectors': 'vector-major: address = vector_index*dimension + element'},
              'known_vectors_scope': vector_meta,
              'numeric_comparison': {'argmax_agreement': float(np.mean(class_ids == float_class_ids)),
                                     'max_abs_logit_error': float(np.max(np.abs(logits.astype(np.float64)/1024-float_logits))),
                                     'hidden_saturations': int(np.sum(float_hidden > 32767/1024)),
                                     'note': 'Includes extreme feature vectors; numerical comparison only, never recognition accuracy.'}}
    (rom_dir / 'model_config.json').write_text(json.dumps(config, indent=2, allow_nan=False), encoding='utf-8')
    (rom_dir / 'model_config.vh').write_text(
        f'// Generated by training/export_quantized.py. Re-export for every trained model.\n'
        f'`define SIGNBRIDGE_MODEL_ID 32\'h{model_id:08x}\n'
        f'`define SIGNBRIDGE_SHIFT1 {int(quantized["shift1"])}\n'
        f'`define SIGNBRIDGE_SHIFT2 {int(quantized["shift2"])}\n'
        f'`define SIGNBRIDGE_SUPPORTED_MASK 26\'h{supported_integer:07x}\n'
        f'`define SIGNBRIDGE_VECTOR_COUNT {vector_count}\n', encoding='ascii')
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/alphabet.npz')
    parser.add_argument('--output', default='models/alphabet_quantized.npz')
    parser.add_argument('--rom-dir', default='fpga/generated_weights')
    parser.add_argument('--vectors', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    try:
        config = export_model(args.model, args.output, args.rom_dir, vector_count=args.vectors, seed=args.seed)
        print(f'Exported {args.output}; model ID {config["model_id_hex"]}; ROMs and integer oracle vectors in {args.rom_dir}.')
        print(f'Numerical vector float/int argmax agreement: {config["numeric_comparison"]["argmax_agreement"]:.3f}. '
              'Synthetic vectors are not ASL recognition validation or proof of FPGA operation.')
        return 0
    except (ValueError, OSError) as exc:
        print(f'Quantized export unavailable: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
