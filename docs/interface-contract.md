# SignBridge interface contract — v1

Canonical Python types live in src/contracts.py. Import their mapping rather than recreating class IDs. Root coordinates any contract change and downstream callers.

September 17 reliability update: see reliability-contract.md and reliability-report.md for the implemented capture/inference split, source generations, new UI event metrics, recorder receipts and normalization_id tagging. CameraTracker remains the synchronous CLI compatibility adapter; the desktop app uses SourceController + CameraCapture + resident HandTracker. No class mapping or vector dimension changed.

September 18 A/B integration: the existing contract is retained. The desktop Train action now passes `--ready-only`; it selects the ready static subset and retains every recording. `STATIC_CLASS_TO_ID` lists 24 static letters with their original canonical IDs; saved models retain all 26 output slots. Dataset summaries include `static_progress`, `ready_labels`, and `readiness_policy`. READY means 100 valid samples total and two sessions with at least 50 each. Existing A/B models remain loadable. Retraining preserves old artifacts in `models/history`. The frame event also exposes `save_failed` so collection navigation remains locked while a save needs retry. Successful training automatically loads the model and enters `free_spell`. Additional frame-event fields are `model_loaded`, `model_classes`, `hand_detected`, `recognition_reason`, `model_error`, `stable_duration_ms`, `confidence_threshold`, `dwell_ms`, `locked_label`, `inference_count` and `training_report`. `candidate` is absent below threshold; `confidence` retains the actual score when available and is None with no usable observation. `stable` is the last committed token. `TemporalDecoder.snapshot()` reports only observed dwell; `reject(timestamp_ms)` clears pending runs without treating failed input as a hand release. The `reset_class(label)` action archives recordings only after explicit UI confirmation.

## Labels

IDs 0..25 are A..Z. Additional IDs in order: 26 MY, 27 NAME, 28 YOUR, 29 WHAT, 30 YOU, 31 HELLO, 32 YES, 33 NO, 34 THANK YOU, 35 PLEASE, 36 GOOD, 37 NICE, 38 MEET. Unknown is -1 and emits no token. Kinds are letter or sign. Static MLP outputs have exactly 26 scores; columns J=9 and Z=25 are masked and never recognized statically.

## Vision data

HandObservation: landmarks float32 (21,3), image-normalized unmirrored x,y,z in MediaPipe index order; handedness Left/Right; score float. FrameObservation: timestamp_ms monotonic integer, hands list, width/height. The primary hand is configured Right by default (signer's anatomical hand from the tracker); fall back to the sole detected hand. Camera inference is unmirrored; mirror only the displayed frame and skeleton together.

normalize_landmarks(landmarks, handedness='Right') → float32 (63,): subtract landmark 0, divide by max Euclidean distance of landmarks 5,9,13,17 from wrist (minimum 1e-6), reflect x for Left to canonical Right, clip each coordinate [-4,4], flatten xyz interleaved. Do not rotate the hand: orientation distinguishes letters. All finite checks are mandatory. Normalized features remove translation/scale but not orientation.

sequence_features(observation) → float32 (134,): right normalized63 + left normalized63 + right wrist xyz + left wrist xyz + right-present,left-present flags. An absent slot is zero. Wrist coordinates retain image position for trajectory and two-hand signs. Temporal recognition must subtract clip-start wrist position when appropriate for J/Z; vocabulary matching may retain framing context. Matching depends on signer/framing calibration.

CameraTracker(camera_index=0, model_path='models/hand_landmarker.task'): read() → (BGR ndarray, FrameObservation), close(). Context manager encouraged. draw_landmarks(frame, observation, mirror=False) returns display BGR frame. Lazy optional dependency imports allow headless tests.

## Dataset v1

One .npz per capture session or temporal clip; no pickle/object arrays. Keys: features float32 (T,63), sequence float32 (T,134), timestamps_ms int64 (T,), label scalar Unicode, session_id scalar Unicode, schema_version scalar int (=1). A static session may contain many independent sampled frames. A motion/sign file is one performance with consecutive frames. Session IDs group recording sessions for hold-out evaluation; clips from the same session must not leak into validation. Invalid/missing primary-hand frames are not saved as static samples. Data root: data/samples. J/Z and all signs must be recorded as clips.

## Prediction and backend

Prediction(label: str|None, confidence: float, timestamp_ms: int, source: str='cpu', latency_ms: float|None=None, scores: optional 1D array=None). Confidence [0,1] is model score, not calibrated correctness. Unknown label None confidence 0.

InferenceBackend.predict(features: ndarray shape(63,), timestamp_ms: int) → Prediction; name str; close(). CPURecognizer(model_path) implements this and exposes supported_labels. Missing model is a clear actionable exception, not random weights. App may operate in collection/tracking mode when no model exists.

Static model: safe NPZ W1 float32(63,32), b1(32,), W2(32,26), b2(26,), metadata scalar JSON with schema_version=1, supported_labels, class_mapping, training provenance. ReLU then softmax over supported static classes only. No external pickle artifacts. Training/evaluation split by session and fail clearly if there is no honest holdout. Partial models are allowed but visibly report coverage.

Quantized model NPZ: w1 int8(63,32), b1 int64(32,), w2 int8(32,26), b2 int64(26,), shift1 int scalar, shift2 int scalar; x=clip(rint(features*1024),-4096,4096) int16; h=clip((x@w1+b1)>>shift1,0,32767) int16; logits=(h@w2+b2)>>shift2 signed int32 representing Q10 scores before softmax. Arithmetic right shift; accumulators signed64. Export must check bounds and write metadata, ROM hex and known-vector expected integer results. Hardware model dimensions fixed. Model ID CRC32 of canonical quantized array bytes ties host to loaded weights.

## Temporal

TemporalDecoder(confidence_threshold=0.8, dwell_ms=350, release_ms=200, change_ms=150). update(prediction: Prediction|None, timestamp_ms: int|None=None) → Token|None. Holding produces one commit. A sustained different confident gesture permits next commit; a no-hand/unknown release lasting release_ms permits repeat. A low-confidence transient cannot unlock. Same-letter repeated hold alone does not repeat. reset().

Token(label: str, kind: str, timestamp_ms: int, confidence: float, source: str). Kind derives from canonical mapping. MotionRecognizer(data_dir='data/samples'): update(observation: FrameObservation) → Prediction|None; reset(); supported_labels. It recognizes J/Z and calibrated sign clips using trajectory/sequence history with thresholds, minimum duration and cooldown. No J/Z from a frozen frame. Static I/D may need deferred commitment while motion is active; expose motion_active bool so root can suppress static commits until motion settles.

### Continuous fingerspelling coordinator (September 19)

`FingerspellingCoordinator(decoder, settings).update(observation, static_prediction, paused=False, has_text=False) -> Token | None` is the live Fingerspelling/Tutor coordinator. It shares the existing `TemporalDecoder` instance. `space_requested` is a separate one-shot word-boundary event; `prediction` is the chosen current result; `snapshot()` supplies mode, state, suppression, progress, original-image fingertip trail and repeat diagnostics. `reset()` clears recognition state. Conversation mode retains its original template module.

`RepeatDetector.update(observation, prediction, locked_label) -> bool` signals a completed confident same-letter rightward/settled movement. `AutoSpaceDetector.update(timestamp_ms, hand_present, has_text) -> bool` signals an optional sustained withdrawal. `JZRecognizer.update(observation, static_prediction=None) -> Prediction | None` signals a **completed trajectory once**, with source `motion-geometric`; it is not a stream to dwell a second time. Geometric scores are not calibrated probabilities. The decoder's new `commit_completed(prediction, repeat=False)` consumes these independently timed events and keeps the held-label lock; ordinary static predictions still use the original `update` path.

Geometry receives the existing physical/unmirrored `FrameObservation`. Repeat direction is signer's right regardless of handedness; J/Z path direction is mirrored for the anatomical left hand. Use `width`/`height` for aspect-correct displacement. Never alter the static 63-coordinate normalization to retain global motion: read global wrist/palm/tip positions separately from the observation.

New J/Z recordings preserve all Dataset v1 required fields and add `raw_landmarks: float32 (T,2,21,3)` (Right, Left slots), `frame_sizes: int32 (T,2)` (width, height), and scalar Unicode `motion_schema="raw-unmirrored-two-hand-v1"`. Presence remains in sequence channels 132/133; absent raw slots are zero. One file is one complete performance; timestamps retain real capture timing. Old archives are not rewritten. Static training still excludes J/Z. `training.evaluate_motion` evaluates raw clips without fitting parameters and reports legacy clips without raw positions explicitly.

The existing text buffer gains `insert_space()`, `backspace()` and `speak_text()`; spaces are buffer edits rather than canonical alphabet classes. Preferences are in `data/fingerspelling_settings.json`, independently of the preserved camera settings.

## Conversation/UI

ConversationEngine(mode='conversation'): consume(token) → dict snapshot; reset(); set_mode(mode); snapshot(). Snapshot keys word, tokens (list[str]), recognized (str), interpretation (str), response (str), target (str), correct (bool|None). Modes free_spell, alphabet_tutor, conversation, benchmark. Sequence MY NAME + letters + YOUR NAME WHAT yields My name is Ian. What is your name? and My name is Astra. Nice to meet you, Ian. The name is inferred from buffer; never hard-code IAN. Distinct letters and signs remain distinct (HELLO sign vs spelled HELLO).

SpeechWorker.speak(text), close(): queue local/system TTS; never block video. Errors exposed as status, avoid automatic repeated response speech.

UI event dict (latest frame): kind='frame', image (BGR ndarray or None), timestamp_ms, candidate, stable, confidence, word, recognized, interpretation, response, target, correct, engine ('CPU'|'FPGA'), latency_ms, fps, status, coverage (list[str]), source ('live'|'replay'), cpu_latency_ms, fpga_latency_ms. Absent measurements are None displayed as —, not zero. App callbacks: mode(str), engine(str), clear(), replay(), stop(), live(), record(label), finish_record(), new_session(), train(), evaluate(), reload(), speak(). UI class SignBridgeUI(on_action callback) with update(event), run(); callback(action: str, value: any=None). update(event) must be thread-safe via latest-event queue polled by Qt timer (do not emit unbounded queued image signals). Root owns orchestration; UI contains no duplicate recognizer. UI is PySide6 after installed Tcl failed startup. Optional event fields recording(bool), sample_count(int), calibration_status(str). Model/temporal reload occurs after training without app restart. Evaluate uses only sessions not in the trained model's training/validation provenance.

## Serial protocol v1 (optional hardware)

UART 115200 8N1. Logical packet: magic bytes 53 42, version u8=1, type u8, seq u16 LE, payload_len u16 LE, payload, CRC16-CCITT-FALSE (poly1021 initFFFF, computed over version through payload), CRC u16 LE. Maximum payload 512 bytes; reject wrong version, length, CRC, seq, status or model ID. Serial timeouts bounded at 250ms; at most one request in flight. Do not auto-open guessed ports.

INFER type1 request payload: model_id u32 LE, 63 signed int16 LE Q10 features (130 bytes). RESULT type129 response payload: status u8 (0 success), model_id u32 LE, class_id u8, cycles u32 LE, 26 signed int32 LE Q10 logits (114 bytes). Host validates class/masked supported mapping and scores. Fixed board clock metadata required for cycles→compute time. Round-trip latency measured separately. A UART-transport or generic byte-stream wrapper is not evidence of board integration. CPU fallback is mandatory on any backend exception.
