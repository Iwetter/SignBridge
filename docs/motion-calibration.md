# Personal J/Z calibration — September 27, 2026

## What the audit found

1. Fingerspelling previously used hardcoded geometric trajectory detection, not a trained temporal model. `src/temporal/jz.py` did not read recordings.
2. The 10 saved J clips and 4 saved Z clips were not used by that fingerspelling detector. They loaded into the older Conversation-mode template matcher and could be evaluated offline, but did not alter live fingerspelling geometry.
3. Recordings stored raw unmirrored two-hand landmarks, timestamps and image dimensions for replay/calibration. Saving alone did not train anything.
4. The old Train button handled only the 24 static letters. It is now labeled **Train static** and remains unchanged underneath.
5. A motion must have a valid starting handshape, a complete directed hook or three-stroke Z, adequate extent/duration, and a briefly held endpoint. A completed motion goes through `commit_completed` directly into the existing text buffer; it does not need another static dwell.
6. Static predictions did not feed the geometric detector, but coordination could commit I before J, or suppress static predictions after a failed motion. Both paths now have regression tests.

The original detector produced zero detections on all 14 saved clips. J strokes suffered shape and handedness flicker during wrist turns. Three Z clips end without a clear final rightward stroke: suffixes `42e7311deca7`, `b994d6842471`, `c84892064697`. The remaining Z clip (`ab4688bc655b`) contains a complete stroke with intermittent tracking. All files are preserved.

## Current workflow

1. Launch the same app and select **Fingerspelling**, then **Live camera**. IC800 selection/capture code is unchanged.
2. Show the starting handshape. Select **Record J motion** or **Record Z motion**. Perform the complete letter and hold its endpoint for roughly 0.2–0.3 seconds.
3. Select **Stop and save** and wait for the saved receipt. The recording is not automatically training data for the static model.
4. Select **Calibrate J/Z**. This background job scans saved clips, finds usable complete strokes, fits geometric minimum extent and residual tolerances, and saves a separate motion configuration. No neural network is trained.
5. Successful calibration loads automatically. **Load J/Z calibration** reloads the saved artifact; startup also loads it. Changes to fingerspelling preferences retain the loaded calibration.
6. **J/Z results** lists used/excluded files and counts. Exclusion never deletes a recording. The report explicitly distinguishes calibration replay from independent validation.
7. Resume signing without recording. Watch motion state, the trajectory overlay, and the existing Candidate/Confidence/Stable/Committed display. Motion scores are geometric match strengths, not probabilities.

Your existing recordings have already been calibrated. No additional recording is needed to try this configuration. The UI should say **Loaded personal J/Z calibration**, with **J: 10/10 usable clips** and **Z: 1/4 usable clips**. All clips came from one session; Z has no independent usable clip to hold out.

The artifact is `models/motion_calibration.json`; its report is `models/motion_calibration_report.json`. Successful revisions are retained in `models/motion_history`. Failed attempts write a separate `_attempt.json` report and do not replace a working config/report. Missing, corrupt, mismatched or changed-source calibration artifacts produce a visible load error; a previously loaded valid configuration is retained. Without one, the status explicitly identifies default geometry.

## Hands-free text boundaries

- **Space:** automatic spacing now defaults to enabled. After a word, withdraw the hand for 1.2 seconds of continuously received frames. The UI counts down; one space is inserted. Short tracking losses and stalled streams do not complete the countdown. Preferences remain under Settings / Diagnostics.
- **Repeated letter:** once the first letter commits, keep the same handshape, move to your own right, then hold briefly. The origin is armed at commitment; a confidence dip up to 160 ms can preserve it, but cannot commit. A confident endpoint, lateral movement, shape/scale continuity and cooldown are still required. The UI shows repeat state/progress and a reason in its tooltip.
- **J starting I:** completing J from its uninterrupted I pose promotes that I to J, rather than producing IJ. To deliberately spell **IJ**, briefly withdraw the hand for more than 0.3 seconds, but less than the automatic-space pause, between letters. Space/Backspace/Clear also cancel that association. Ambiguity between a held I and a J start cannot be solved from a still frame alone.
- Space, Backspace, Clear and Speak remain available.

## Verification and limits

The actual application controller, existing CPU model and current text buffer were replayed with saved raw motion landmarks:

- Nine single-J clips produced J; the clip ending `65679a52e535` contains two strokes and produced JJ.
- The complete Z clip produced Z. The three incomplete Z clips produced no character.
- Leave-one-clip-out calibration detected J in all 10 held-out J clips, but these are all from the same session and share the developer-audited data. This is not an untouched test. The only usable Z cannot validate itself by leave-one-out.
- One frozen, confidently classified recorded sample from each of 24 static classes still committed that letter through the integrated controller. These are routing checks, not new webcam recognition measurements.
- The unchanged static model's saved validation: overall 98.87%; G 144/145 (one P confusion); T 241/242; U/V/S/Q all correct. These are previously selected validation sessions, not evidence that the reported live G problem is fixed.
- Controller replay latency: median about 0.16 ms, p95 about 0.32 ms. Camera and landmark extraction are excluded.
- All 236 pre-existing files under data/models were hash-checked unchanged, including static weights, camera settings, recordings and model history. Only new calibration artifacts were added.

Evidence: `work/motion-calibration-20260927/baseline.json`, `integration.json`, `static-validation.json`, `preservation-check.json`, `tests.xml`, and `app.png`.

Live J/Z, repeated letters and hands-free spacing still require your webcam acceptance. G remains an open live-recognition issue; no unvalidated static retraining or hardcoded G override was installed.

## Manual acceptance

1. Start the app using the command below. Confirm IC800 video and Fingerspelling mode.
2. Confirm the loaded calibration status and open **J/Z results** to inspect the 14 original clips.
3. Sign J, hold the endpoint briefly, and verify one J; continue holding and verify no extra J or I. Relax/withdraw briefly and repeat.
4. Sign all three strokes of Z and settle. Verify one Z. Try only its first two strokes: it must not commit Z.
5. Hold L until committed, shift right and settle: verify LL. Continue holding: verify no third L. Repeat with A.
6. Sign a word, withdraw for 1.2 seconds, and verify one automatic space. Return and sign the next word.
7. Check G, T, U, V, S and Q; report live errors using the existing Candidate/top predictions diagnostic.
8. Save a new motion if desired, press Calibrate J/Z, inspect results, reload, close and reopen. Confirm the same profile loads and original recordings remain.

```powershell
& "C:/example-user\Documents\Codex\2026-09-16\files-pasted-by-the-user-you\outputs\SignBridge\.venv\Scripts\python.exe" "C:/example-user\Documents\Codex\2026-09-16\files-pasted-by-the-user-you\outputs\SignBridge\app.py"
```
