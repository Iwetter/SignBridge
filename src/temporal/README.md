# Temporal integration

`from src.temporal import TemporalDecoder, MotionRecognizer`

Call `MotionRecognizer.update(observation)` once for each new observation. A
returned prediction takes priority over static classification. If it returns
`None` and `motion_active` is true, suppress static I/D (or all static commits)
by passing a named low-confidence prediction to the decoder. Passing `None`
means an observed release; it must not represent a skipped or deferred inference.

A match emits a 500 ms confirmation plateau, with the current observation's
timestamp each time, so the default 350 ms decoder dwell can commit it. Motion
is false during confirmation. A missing hand or a substantial pose change
cancels confirmation. Change `confirmation_ms` if configuring a longer decoder
dwell. After a match the history is cleared and a cooldown excludes old motion.
Static sign templates additionally require release or a sustained pose change
before another identical sign. J/Z always require a nontrivial new trajectory.

The decoder commits a held label once. A sustained different confident label
unlocks it after `change_ms`; the new label still needs its own full dwell.
Explicit unknown/no-hand observations unlock after `release_ms`. A named
low-confidence prediction breaks pending timers without unlocking. A gap over
250 ms resets pending timers but preserves the held-label lock. Backwards time
raises `ValueError`; call `reset()` before switching streams. Deliberately
release between the two L letters in HELLO.

The recognizer loads schema-v1 NPZ clips from `data/samples`, including two-hand
presence and image-coordinate wrists. J/Z wrist matching subtracts each hand's
clip-start wrist; signs retain calibrated image position. Frozen J/Z templates
are rejected, and frozen live frames never match J/Z. Real movement must settle
before a motion match starts confirmation: finish a clip with about 300 ms of
steady endpoint pose, then hold that pose for confirmation during use. Template
collection and live signing must use consistent framing and hand assignments.

The default method resamples actual timestamps to 32 points and compares three
duration windows (0.75x, 1x, 1.3x) with weighted pose, wrist, and presence error.
Matches require distance at most `threshold` (0.20) and at least `margin` (0.025)
separation from every other class or negative template. These values require
real held-out calibration; match confidence is a score, not accuracy. More
templates can increase processing time; the sequence history is bounded to
512 frames and clips longer than six seconds are rejected.

An optional scalar Unicode JSON `metadata` field containing `{"reject":true}`
marks a clip as a negative example. Noncanonical labels, including UNKNOWN,
REST, BACKGROUND, and NONE, are also reject-only and can never emit tokens.
Static A–Y clips other than J are ignored. `supported_labels`,
`calibration_info`, `diagnostics`, and `last_match` make coverage and rejection
visible. An empty/missing dataset has no recognition coverage and does not
invent signs or activate motion suppression.
