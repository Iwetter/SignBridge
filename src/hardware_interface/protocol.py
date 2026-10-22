"""SignBridge protocol v1. Byte order is little endian, including CRC bytes."""
from dataclasses import dataclass
import struct

MAGIC = b'SB'
VERSION = 1
MAX_PAYLOAD = 512
INFER, RESULT = 1, 129


class ProtocolError(RuntimeError):
    pass


def crc16_ccitt(data: bytes) -> int:
    crc = 0xffff
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xffff
    return crc


@dataclass(frozen=True)
class Packet:
    message_type: int
    sequence: int
    payload: bytes


def encode_packet(message_type: int, sequence: int, payload: bytes) -> bytes:
    if not 0 <= message_type <= 255 or not 0 <= sequence <= 65535:
        raise ValueError('Invalid packet type or sequence.')
    if len(payload) > MAX_PAYLOAD:
        raise ValueError('Payload exceeds protocol maximum.')
    body = struct.pack('<BBHH', VERSION, message_type, sequence, len(payload)) + payload
    return MAGIC + body + struct.pack('<H', crc16_ccitt(body))


def decode_packet(raw: bytes) -> Packet:
    if len(raw) < 10 or raw[:2] != MAGIC:
        raise ProtocolError('Invalid packet magic or truncated packet.')
    version, kind, seq, size = struct.unpack('<BBHH', raw[2:8])
    if version != VERSION or size > MAX_PAYLOAD or len(raw) != size + 10:
        raise ProtocolError('Invalid protocol version, length, or trailing bytes.')
    if struct.unpack('<H', raw[-2:])[0] != crc16_ccitt(raw[2:-2]):
        raise ProtocolError('Response CRC mismatch.')
    return Packet(kind, seq, raw[8:-2])
