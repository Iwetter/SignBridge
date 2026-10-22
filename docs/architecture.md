# SignBridge architecture

SignBridge is a local Python interactive ASL/fingerspelling tutor with a limited, calibrated sign vocabulary. It does not translate unrestricted ASL. The Friday September 18, 2026 demonstration must use the CPU path independently of FPGA availability.

## Vertical slice

OpenCV webcam → MediaPipe Tasks Hand Landmarker (up to two hands) → wrist/scale normalized features → NumPy static MLP and temporal clip recognizer → confidence/dwell/release gate → canonical tokens → spelling and limited conversation state machine → desktop UI and queued system speech.

The app uses PySide6 (Qt) for a local desktop interface. A SourceController capture thread exclusively owns the webcam and publishes one latest raw frame. A separate resident landmark worker consumes the newest frame; a single result slot discards stale results. The coordinator feeds preview events independently at up to 30 Hz, validates source generations, and handles temporal/conversation state. Qt paints on the main thread; storage uses a separate batch writer and TTS uses its own worker. No web service is required. Missing camera, model, or board is surfaced honestly. An explicitly labeled synthetic replay validates integration without masquerading as live recognition.

Static letters exclude J/Z, which require motion clips. A collector saves whole temporal clips for J/Z and vocabulary signs, preserving two-hand geometry. Session-calibrated sequence matching is an initial practical approach, not a claim of signer-independent accuracy. MY/YOUR also depend on body-relative context; hand-only templates require consistent framing and must be evaluated on held-out real performances.

## Optional hardware

Landmark extraction, temporal processing, interpretation, and UI remain on the PC. The board may run only the quantized 63→32→26 static MLP. The backend returns the same Prediction object as CPU; failures switch to CPU with an explicit status message. Board round-trip time and compute time are separate measurements. Simulated hardware is never labeled FPGA hardware.

No board pin mapping or GPIO wiring is assumed. Board integration must use a verified DE10-Standard reference project. The onboard HPS serial interface is not automatically a direct FPGA-fabric UART.

## Ownership and execution waves

1. VISION-DATA: src/vision and training/collect.py.
2. ALPHABET-ML: src/recognition, training/train.py, training/evaluate.py, training/export_quantized.py, models.
3. TEMPORAL-GESTURES: src/temporal.
4. CONVERSATION-UI: src/conversation and src/ui.
5. FPGA-RTL: fpga and src/hardware_interface.
6. VERIFICATION: tests and docs/test-results.md.

Three specialists can run concurrently with the root; later specialists start as slots free. Root owns shared contracts, entry points, dependency setup, integration, scripts, and project documentation. Every specialist reads this document and interface-contract.md before writing. Changes to shared interfaces require explicit coordination.

## Evidence policy

Do not claim accuracy from synthetic examples or adjacent-frame random splits. Training metrics must identify real dataset provenance, session split, supported classes, and exclusions. Missing data is a calibration requirement. Benchmarks are measured with perf_counter and identify input/model/backend and sample count. Live camera, audible speech, physical-board operation, and real signing need explicit validation, separate from software unit tests.
