# Incremental static alphabet collection

The user confirmed live A/B recognition. The expanded collector uses the same camera, normalized features, CPU classifier, model format, confidence threshold, temporal filter and live prediction display. J/Z and FPGA work remain outside this change.

## Collecting the 24 static letters

The label selector contains exactly `A B C D E F G H I K L M N O P Q R S T U V W X Y`. Previous label and Next label wrap through that sequence, skipping J and Z. Navigation is disabled while recording, saving or retrying a failed save.

1. Start Live camera with the saved IC800 configuration. Select a letter and press Record.
2. Aim for 100–150 valid samples per letter in a sitting. Stop and save; check the saved count and path.
3. Use Next label to keep collecting within this sitting. Optionally enable **After saving, go to next incomplete letter**. It advances only after a new successful receipt is matched to refreshed persisted counts, skips READY letters and never starts recording automatically. Failed or empty saves and replay do not advance.
4. Take a break and vary framing slightly, then click New session once. Collect each desired letter again. Multiple files in one sitting count as one session; do not create session IDs for adjacent frames merely to satisfy readiness.
5. The compact row uses ✓ for READY and ○ for NEED DATA. **Alphabet progress** lists every static letter's valid sample count, distinct session count and status. Select a letter or hover a status for the missing requirements.
6. Click Train once at least two letters are READY. All READY letters are included; other recordings are retained for later. You do not need all 24 letters. The trainer still requires a session-disjoint split containing each selected class in both partitions, and reports an actionable error if the session grouping cannot provide one.
7. Training results shows the actual supported classes, counts, split sizes, accuracy and confusion matrix. The saved model automatically loads into Fingerspelling. Confirm its class list, then test every newly trained letter on the webcam.

## Readiness policy

A letter is READY with **at least 100 valid samples total and at least two distinct sessions containing at least 50 valid samples each**. A 99-sample session and a 1-sample session do not qualify. Counts combine multiple files from the same session. Malformed archives are reported; no-hand and zero static vectors do not count toward readiness or incremental training.

READY describes collection sufficiency, not recognition accuracy. Keep collecting different positions/lighting and validate newly added classes. The minimum does not guarantee that visually similar letters will be distinguished reliably.

## Existing data and model

At implementation verification, A has 260 samples / 2 sessions, B has 473 / 2, and C has 36 / 1. A/B are READY. The active A/B model was left unchanged, as were the seven active archives, twelve existing archived recordings, camera settings and current model reports. Archived recordings remain in `data/removed_samples` and are not silently restored into training.

The canonical A..Z IDs and 63→32→26 model format remain compatible with the proven A/B model. The explicit `STATIC_CLASS_TO_ID` / `static_class_mapping` lists all 24 static letters using their original IDs; J=9 and Z=25 remain masked. Models recognize only their trained `supported_labels`.

On successful retraining, the previous model, mapping, reports and confusion CSVs are copied and verified under `models/history/<version>/` before any replacement is published. A manifest records original paths and hashes. A failed transaction restores the old artifacts. Existing history is never removed. No production model was retrained or replaced during implementation.

## Verification

The complete non-hardware suite passed: **79 tests in 20.10 seconds**, zero failures/errors/skips. The absolute app entry point also passed an offscreen replay startup smoke check without opening a webcam.

Regression tests cover unchanged A/B live-adapter behavior, READY aggregation and validity checks, partial A/B→A/B/C/D training, full 24-class canonical coverage, J/Z exclusion, model-history preservation and rollback, navigation, and successful/failed/empty-save advancement. Generated fixtures stay under `work/pytest-*` and provide software evidence only.

`work/static-alphabet-preservation-check.json` verifies 27 existing data/model/settings artifacts remain byte-for-byte unchanged. `work/static-alphabet-current-ab-check.json` records offline predictions from the existing A/B model: 733/733 active A/B samples correctly classified, including the model's 484 validation samples. Validation was used for model selection; there are no untouched new-session samples in this check. This is separate from the user's confirmed live A/B acceptance and provides no recognition claim for uncollected letters.

The exact local launch command remains in README. All non-hardware test results are in `work/static-alphabet-verification.xml` and `docs/test-results.md`.
