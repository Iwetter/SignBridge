# Vision and real calibration

`CameraTracker` uses OpenCV video capture and the current MediaPipe **Tasks**
`HandLandmarker`. It does not use the removed `mp.solutions.hands` interface.
The pretrained landmark detector is distinct from the ASL recognizers: it finds
21 xyz points per hand and does not provide ASL labels. Collection needs a local
`models/hand_landmarker.task` bundle even before an ASL model is trained.

Implementation references are Google's [Python Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
and [model bundle listing](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models).
The project's setup provisions the model; these modules do not download it or
install packages. Synchronous VIDEO mode is called on the application's camera
worker so each returned image and its landmarks have the same increasing
timestamp. A missing dependency, model, or camera produces an actionable error.

The camera frame is passed to inference without flipping. `draw_landmarks(...,
mirror=True)` flips only the displayed image and drawn landmark positions. The
tracker's Left/Right category is retained; it is never inferred from which side
of the screen a hand occupies. Check displayed handedness with actual hands at
setup, especially if a camera driver itself mirrors its stream. Primary selection
prefers anatomical Right, or Left when explicitly configured, and falls back to
a sole valid detected hand. The numerical feature definition and hand slots are
specified in `docs/interface-contract.md`.

## Guided capture

From the repository root, with its Python environment activated:

```powershell
python -m training.collect --labels "A,I,N,J,Z,MY,NAME,YOUR,WHAT" --session-id practice-01
```

- Press **Space** to start. Static labels collect 60 valid frames at 100 ms
  intervals by default. Maintain the correct sign with small natural variations
  in distance, placement, and finger pose; preserve the sign's actual orientation.
- For **J, Z, and every word sign**, perform one complete gesture per capture.
  Space stops the clip; removing both hands from view also stops it after a
  brief release. A clip stops automatically after four seconds. Repeat at least
  5–10 separate performances per sign to start calibration.
- **N/P** changes labels, **R** discards the current capture, and **Q/Esc** saves
  any valid current capture and exits. Labels do not advance automatically.
- Repeat in another sitting with a different `--session-id`, such as
  `practice-02`, for validation. Keep all recordings from the same sitting under
  one session ID. Randomly dividing adjacent camera frames is not a meaningful
  accuracy test.

`--camera`, `--primary`, `--samples`, `--sample-interval-ms`, `--clip-seconds`,
`--model`, and `--data-dir` customize acquisition. `--no-mirror` affects preview
only. Saved `.npz` files contain landmark features, two-hand sequences,
timestamps, label, session ID, and schema version. They contain no raw camera
images and require no pickle. Do not mix synthetic integration fixtures into
the real calibration directory.

MY/YOUR and several other vocabulary signs depend on body context. Hand-only
calibration needs consistent camera framing and real held-out checks; it does
not establish signer-independent recognition or unrestricted ASL translation.

## Reuse without a camera

```python
from src.vision import DatasetRecorder, primary_features, sequence_features

recorder = DatasetRecorder("data/samples", session_id="practice-01")
recorder.start("J")
for observation in observations:       # FrameObservation objects, increasing time
    recorder.add(observation)
    if recorder.release_ready:
        break
saved_path = recorder.finish()
```

The same recorder can be reused for more labels while retaining its session ID.
`active`, `label`, `is_temporal`, `count`, `valid_count`, `duration_ms`,
`release_ready`, `last_path`, and `last_message` support in-app calibration.
`finish()` returns `None` for an empty capture and raises `ValueError` after
discarding a temporal clip shorter than six valid frames or 250 ms. On disk
write failure its buffer remains available for retry. `cancel()` discards only
the current buffer. Call all recorder methods from one worker thread.

Static recording skips invalid/missing primary hands. Temporal recording keeps
consecutive frames and brief tracking gaps, represented by zero features and
presence flags, then trims trailing empty frames when saving. A release-ready
clip will not accept a returning hand until it is finished or cancelled. This
prevents separate performances from being joined into a single trajectory.
