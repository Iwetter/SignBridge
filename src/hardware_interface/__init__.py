"""Optional real-hardware serial transport; importing does not open a port."""
from .backend import FPGABackend
from .protocol import ProtocolError, encode_packet, decode_packet, crc16_ccitt

__all__ = ['FPGABackend', 'ProtocolError', 'encode_packet', 'decode_packet', 'crc16_ccitt']
