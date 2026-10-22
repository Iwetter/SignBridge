# Continuous A–Z fingerspelling: implementation and webcam acceptance

The existing SignBridge app now coordinates its 24-class static recognizer with deliberate repeated-letter movement, word spacing and geometric J/Z trajectories. This milestone is implemented and tested with synthetic sequences and saved static landmarks. **Real IC800 acceptance is pending the user's tests below.** No J/Z webcam performances have been collected or validated yet.

## Preserved state

The current static model remains `6bef0efc7e3b46babe482786e5dfd90b`, with the same weights, 24-class mapping, confidence calibration and 63-coordinate preprocessing. Existing datasets, model history, IC800 settings, camera device selection, capture threads and source controller are unchanged. No static training, downloads, FPGA integration or conversation vocabulary changes were performed.

The existing temporal decoder's ordinary `update` path is retained. Its confidence threshold remains 80%, static dwell 350 ms, different-letter change guard 150 ms and deliberate no-hand release 200 ms. One additional method accepts an independently completed repeat or J/Z trajectory while preserving its held-letter lock. The same conversation engine and text buffer remain in use.

## Baseline audit

These are supplied-prediction tests of the pre-existing static decoder path, not camera recognition claims:

| Input intent | Existing output without a release or new APIs | Finding |
|---|---|---|
| IAN | IAN | Different consecutive letters already worked. |
| HELLO | HELO | A second L needed a hand release; no lateral-motion input existed. |
| ANNA | ANA | A held N correctly avoided duplicates but could not signal a repeat by movement. |
| MY NAME | MYNAME | The old buffer had no space insertion API. |
| A or L held five seconds | One letter | Existing held-letter suppression is retained. |
| Short tracking loss below release interval | No second letter | Brief loss does not establish a deliberate release. |

The original template-based motion module required recorded J/Z clips; no usable clips are present. It remains available for the existing conversation path. The new fingerspelling coordinator uses a separate geometric J/Z detector, so it neither needs fake training samples nor changes the static classifier.

**Rest limitation:** the static model has no trained REST/unknown-handshape class. It can reject low confidence and the decoder filters short transitions, but a resting hand that confidently resembles A or another letter may still commit that letter. Held-hand suppression prevents repeated copies; it does not solve open-set recognition. Withdraw the hand while resting. A prolonged tracker dropout is indistinguishable from withdrawal to a landmark-only system; automatic spacing is therefore optional and off by default.

## Repeated letters

After a static letter commits, keep the same handshape and shift your hand **to your own right**, then settle briefly. In the selfie preview this is also rightward. This direction is the same for either signing hand.

The detector tracks original wrist and palm-center positions separately from normalized handshape features. Pixel aspect correction and wrist-to-knuckle size normalize the displacement. Both positions must move laterally, the same anatomical hand and confident label must remain present, and normalized handshape must remain similar. Jitter, vertical motion, a label/confidence change, hand switching or a long frame gap cancel an incomplete repeat.

Defaults: 0.65 palm lengths of rightward travel, 120 ms endpoint settling, 700 ms cooldown. A settled return to the original location rearms without appending anything. A further deliberate rightward stroke can then repeat again. The controls are under **Settings / Diagnostics → Fingerspelling**. New preferences are saved separately in `data/fingerspelling_settings.json`; saving them does not modify camera settings.

Wait for the first letter under **Committed**, hold it briefly so a position baseline is established, then move. Do not lower the hand between repeated letters.

## Spaces, editing and speech

- **Space** inserts one space into the current text. **Ctrl+Space** is the default shortcut; change it in Settings / Diagnostics and click **Save fingerspelling settings**.
- **Automatic space after hand withdrawal** defaults to off. When enabled, continuously observed absence for 1,200 ms after text has been entered inserts one space. Its duration is adjustable between 800 and 5,000 ms. Keeping a hand still does not insert a space.
- Leading and consecutive spaces are refused. Backspace removes the last character, including a space. It does not unlock a held pose and immediately reinsert the deleted letter.
- **Clear** resets the text and recognition state. **Speak** speaks the current fingerspelled text through the existing speech worker. Conversation mode retains its original response behavior.
- Brief no-hand events, stale or absent frame streams, ambiguous/invalid landmarks and recording/maintenance pauses do not build automatic word-boundary time. The existing deliberate letter-release mechanism remains separate from the longer word-space timer.

## J and Z

J tracks the pinky tip; Z tracks the index tip. Both require a compatible handshape and a completed multi-part trajectory. The detector uses original timestamped landmarks, palm-relative scale, starting-position subtraction and distance-resampled trajectories. It supports a range of signing speeds and moderate path variation without changing the static model.

For right-handed signing, J descends then hooks left/up from the signer's perspective; Z moves right, down-left, then right. Left-handed trajectories are mirrored. These conventions follow [ASL University’s handedness explanation](https://www.lifeprint.com/asl101/pages-layout/rightorlefthand.htm) and [Z reference](https://www.lifeprint.com/asl101/pages-signs/z/z.htm). The app receives unmirrored camera coordinates and mirrors only its preview, so those transforms are kept separate.

Recognition states:

1. **Pending:** a compatible starting handshape is briefly reserved, up to 650 ms, retaining the normal static dwell evidence during that check, so an intended J does not first append I, or Z first append a static pointing pose.
2. **Tracking:** a correctly directed path has started. The ordinary static dwell is paused only for this plausible motion handshape/path, not for arbitrary movement of every letter.
3. **Completed:** the full geometry matches and the endpoint settles for at least 140 ms. One J/Z is committed directly because trajectory completion already supplies temporal evidence. The displayed score is geometric match strength, not the static model's calibrated confidence.
4. **Cooldown / rearm:** holding the endpoint produces no additional letter. Move away after cooldown and establish the next stroke's start to repeat J/Z. For JAZZ, perform two distinct complete Z strokes with a brief return/settle between them; no hand lowering is required.
5. **Retry:** an attempted trajectory was incomplete. It does not leak I/D into the word; relax or change handshape and try again.

The current implementation uses two separate index-finger Z strokes for ZZ. Other linguistic double-Z variants are not implemented. A pose held still never produces J/Z. A deliberate stationary I or pointing pose can resume static recognition after the reservation expires. Consequently, begin a J/Z movement promptly: holding its starting pose beyond the reservation interval can commit the static letter before any future motion is observable. There is no retrospective deletion of already committed text.

The existing recognition panel now shows **STATIC / MOTION**, progress and committed text. During a trajectory, the camera overlay draws a subtle fingertip trail. Developer metrics include motion processing time/state. Static top-five diagnostics remain available. For motion, static top-five scores are not presented as probabilities for J/Z.

## Record and evaluate your motions

Use **Record J motion** or **Record Z motion** in the existing Dataset section. The static label dropdown remains the original 24 letters.

1. Keep the relevant handshape visible at the starting position, click the motion button, begin promptly, perform one complete trajectory, then hold its endpoint for about 0.3 seconds.
2. Click **Stop and save**. Check the saved label, valid frame count and path. Each file is one performance; do not combine many repetitions into one clip.
3. Record 5–10 performances of each needed motion with modest variation in speed, size, position and distance. Retain the same session ID within one sitting.
4. Take a real break, return with slightly changed framing, click **New session**, and record another sitting. A new ID alone is not independence. Keep at least one sitting for evaluation rather than parameter tuning.

Motion clips retain the existing Dataset v1 `features`, `sequence`, timestamps, label and session fields. New J/Z clips additionally store `raw_landmarks` as `(frames, 2, 21, 3)` float32 with fixed Right/Left slots, `frame_sizes`, and `motion_schema=raw-unmirrored-two-hand-v1`. Presence is retained in the existing sequence. No RGB video is stored. Static archives are neither rewritten nor migrated, and J/Z clips remain excluded from static training.

The geometric recognizer runs immediately; recording does not silently retrain or alter it. **Train** and **Evaluate** retain their static-model meanings. To measure J/Z clip completion, run from the project directory:

```powershell
.\.venv\Scripts\python.exe -m training.evaluate_motion --data-dir data/samples --output work/motion-evaluation.json
```

Add `--sessions SESSION_ID` to select an independent sitting. The report lists every clip, completed letters, timestamps, duplicate/missed detections and per-class completion accuracy. Legacy clips without raw trajectories are reported as unevaluable and preserved. No untouched-session performance is claimed until real clips exist.

## Exact IC800 acceptance test

Close older SignBridge/probe windows. Launch the normal command from README, select **Live camera → Fingerspelling**, and confirm yourself and the hand landmarks in the existing IC800 preview. Do not change the camera configuration.

Leave automatic spacing off for the first tests. Use **Clear** with your hand out of view before each word. Optional **Hide dataset** gives the video and text more room. Keep Developer metrics visible.

| Test | Action | Required text |
|---|---|---|
| IAN | Sign I, A, N with distinct stable shapes; hold I about one second because it shares J's starting shape. | IAN |
| HELLO | H, E, L; after L commits, briefly steady then shift the same L to your right and settle; finish O. | HELLO |
| ANNA | A, N; repeat N by the same rightward movement; finish A. | ANNA |
| MY NAME | M, Y, press Space or Ctrl+Space, then N, A, M, E. | MY NAME |
| J | Form the pinky handshape and promptly make one complete J trajectory; hold the end briefly. | J exactly once |
| Z | Form the pointing handshape and promptly trace a complete Z; hold the end briefly. | Z exactly once |
| JAZZ | J, A, complete Z, return to the next starting position after cooldown, steady briefly, then complete another Z. | JAZZ |
| MY NAME IS IAN | Enter each word and insert spaces with the button/shortcut. Then enable automatic spacing and repeat using about 1.5 seconds of hand withdrawal between words. | MY NAME IS IAN |

Also check:

- Hold A and L for five seconds: one letter each. Tiny shakes and a return movement after a repeat must not append a duplicate.
- A quick landmark flicker must not add a space or a letter. With automatic spacing off, even prolonged withdrawal must not insert a space.
- An ordinary index swipe or pinky movement must not commit Z/J. If an incomplete attempt enters Retry, relax the shape before retrying.
- Try Backspace, Clear, Speak, and a changed Space shortcut. Reopen the app and check the saved movement/spacing preferences and existing dataset counts.
- Retest T, U, V, S and Q. Their static model was not retrained or changed.

Record failures with expected/actual text, STATIC/MOTION state, progress, confidence and physical signing hand. Confirm each of the eight rows above before this milestone is considered live-validated.

## Automated evidence

The suite covers the requested words, movement repeats, held A/L, jitter, cooldown and return, confidence/shape/hand changes, brief gaps, no leading/duplicate spaces, buffer editing and speech routing, J/Z at different speeds/sizes/positions and both handednesses, ordinary/partial/reversed movement rejection, incomplete-motion static suppression, JAZZ rearming, raw clip round trips, static-training exclusion, settings and actual controller/UI integration. Fixtures are explicitly synthetic; they are not ASL recognition accuracy measurements.

The full non-hardware suite passed **132 tests in 17.56 seconds**.

An additional check feeds one correctly classified saved pose from each of the 24 classes through the unchanged CPU model and the new coordinator for 2.6 seconds. Each commits exactly its letter once. This preserves static hold behavior, not proof of natural live transitions. The numeric coordinator benchmark measured about 0.12 ms median and 0.23 ms p95; these are not webcam capture or landmark timings.

Evidence lives in `work/continuous-fingerspelling/`: `tests.xml`, `baseline.json`, `static-preservation-check.json`, app/UI screenshots and the before/after preservation hashes. The actual app entry point was smoke-tested offscreen with explicitly synthetic replay and no camera opened. **Real webcam acceptance remains pending.**
