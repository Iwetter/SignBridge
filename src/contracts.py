"""Canonical labels and shared values. See docs/interface-contract.md."""
from dataclasses import dataclass, field
from typing import Any, Protocol

ALPHABET = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
SIGNS = ('MY', 'NAME', 'YOUR', 'WHAT', 'YOU', 'HELLO', 'YES', 'NO', 'THANK YOU', 'PLEASE', 'GOOD', 'NICE', 'MEET')
LABELS = ALPHABET + SIGNS
CLASS_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_CLASS = dict(enumerate(LABELS))
STATIC_LABELS = tuple(x for x in ALPHABET if x not in ('J', 'Z'))
# Keep canonical A..Z IDs and existing 26-output models compatible; J/Z remain masked.
STATIC_CLASS_TO_ID = {label: CLASS_TO_ID[label] for label in STATIC_LABELS}
FEATURE_DIM = 63
SEQUENCE_DIM = 134
SCHEMA_VERSION = 1

@dataclass
class HandObservation:
    landmarks: Any
    handedness: str = 'Right'
    score: float = 1.0
    raw_handedness: str | None = None

@dataclass
class FrameObservation:
    timestamp_ms: int
    hands: list[HandObservation] = field(default_factory=list)
    width: int = 640
    height: int = 480

@dataclass
class Prediction:
    label: str | None
    confidence: float
    timestamp_ms: int
    source: str = 'cpu'
    latency_ms: float | None = None
    scores: Any = None

@dataclass
class Token:
    label: str
    kind: str
    timestamp_ms: int
    confidence: float
    source: str

class InferenceBackend(Protocol):
    name: str
    def predict(self, features: Any, timestamp_ms: int) -> Prediction: ...
    def close(self) -> None: ...
