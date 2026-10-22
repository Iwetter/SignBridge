"""Guided real-landmark calibration. Run: python -m training.collect --help."""
from __future__ import annotations

import argparse
from collections import Counter
import sys

from src.contracts import CLASS_TO_ID, LABELS, STATIC_LABELS
from src.vision import CameraError, CameraTracker, DatasetRecorder, draw_landmarks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=",".join(LABELS),
                        help="Comma-separated labels, e.g. A,B,C,J,Z,MY,NAME,THANK YOU")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--data-dir", default="data/samples")
    parser.add_argument("--session-id", help="Same ID for one sitting; use a new ID for held-out performances")
    parser.add_argument("--primary", choices=("Right", "Left"), default="Right")
    parser.add_argument("--samples", type=int, default=60, help="Auto-stop each static capture after this many samples")
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--clip-seconds", type=float, default=4, help="Maximum length of one motion/sign performance")
    parser.add_argument("--no-mirror", action="store_true", help="Display without selfie mirroring; features never mirror")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    labels = [" ".join(label.upper().split()) for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in labels if label not in CLASS_TO_ID]
    if not labels or unknown:
        parser.error(f"Choose canonical labels; unknown labels: {unknown}")
    if args.samples < 1 or args.sample_interval_ms < 1 or args.clip_seconds < 0.3:
        parser.error("Samples/interval must be positive and --clip-seconds must be at least 0.3")
    recorder = DatasetRecorder(args.data_dir, args.session_id, args.primary, args.sample_interval_ms)
    print(f"Session: {recorder.session_id}\nSaving landmark-only data in {recorder.data_dir}")
    print("SPACE starts/stops; N/P next/previous label; R discards; Q/Esc quits.")
    print("Static letters: vary the pose slightly while holding the correct letter.")
    print("J/Z and signs: perform ONE complete motion per clip; lower both hands to finish.")
    print("Use SPACE to stop signs that end with hands still visible. Repeat 5-10 clips per sign.")
    print("For evaluation, repeat the labels in a separate session after a break.")
    print("Check the displayed Right/Left labels against your anatomical hands before collecting.")
    counts = Counter()
    index = 0
    status = "SPACE to start; N/P changes label"

    def finish_capture():
        nonlocal status
        saved_label = recorder.label
        try:
            path = recorder.finish()
            if path:
                counts[saved_label] += 1
                print(f"Saved: {path}")
            status = recorder.last_message
        except ValueError as exc:
            status = str(exc)
            print(status, file=sys.stderr)

    try:
        import cv2
        with CameraTracker(args.camera, args.model) as tracker:
            while True:
                frame, observation = tracker.read()
                label = labels[index]
                if recorder.active:
                    recorder.add(observation)
                    if ((not recorder.is_temporal and recorder.count >= args.samples)
                            or recorder.release_ready
                            or (recorder.is_temporal and recorder.duration_ms >= args.clip_seconds * 1000)):
                        finish_capture()
                display = draw_landmarks(frame, observation, mirror=not args.no_mirror)
                mode = "STATIC" if label in STATIC_LABELS else "MOTION CLIP"
                recording = "RECORDING" if recorder.active else "READY"
                text_lines = [
                    f"{label}  |  {mode}  |  {recording}",
                    f"Frames: {recorder.count}   Saved captures for label: {counts[label]}",
                    "SPACE start/stop   N/P label   R discard   Q quit",
                    ("Perform once; lower both hands or SPACE to finish" if label not in STATIC_LABELS
                     else f"Hold correct letter; automatic {args.samples} samples"),
                    status,
                ]
                cv2.rectangle(display, (0, 0), (display.shape[1], 135), (24, 24, 24), -1)
                for row, line in enumerate(text_lines):
                    color = (100, 235, 165) if row == 0 else (235, 235, 235)
                    cv2.putText(display, line[:95], (10, 23 + row * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                cv2.imshow("SignBridge - Real calibration", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    if recorder.active:
                        finish_capture()
                    break
                if cv2.getWindowProperty("SignBridge - Real calibration", cv2.WND_PROP_VISIBLE) < 1:
                    if recorder.active:
                        finish_capture()
                    break
                if key == ord(" "):
                    if recorder.active:
                        finish_capture()
                    else:
                        recorder.start(label, observation.timestamp_ms)
                        status = f"Recording {label}; perform the prompted sign only"
                elif key == ord("r"):
                    recorder.cancel()
                    status = "Discarded. SPACE to retry"
                elif key in (ord("n"), ord("p")):
                    if recorder.active:
                        finish_capture()
                    index = (index + (1 if key == ord("n") else -1)) % len(labels)
                    status = "SPACE to start the next capture"
    except (CameraError, ImportError, OSError) as exc:
        print(f"Collection cannot continue: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if recorder.active:
            finish_capture()
    finally:
        if "cv2" in locals():
            cv2.destroyAllWindows()
    print(f"Captured {sum(counts.values())} files in session {recorder.session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
