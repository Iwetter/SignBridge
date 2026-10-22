"""Launch SignBridge: local live tutor or explicitly synthetic integration replay."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--camera-backend', choices=['DSHOW'], default='DSHOW',help='Camera capture backend (DSHOW)')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/alphabet.npz')
    parser.add_argument('--landmarker', type=Path, default=ROOT / 'models/hand_landmarker.task')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/samples')
    parser.add_argument('--quantized-model', type=Path, default=ROOT / 'models/alphabet_quantized.npz')
    parser.add_argument('--port', help='Verified FPGA inference serial endpoint, never auto-selected')
    parser.add_argument('--dwell-ms', type=int, default=350)
    parser.add_argument('--replay', action='store_true', help='Clearly labeled synthetic token-flow rehearsal')
    parser.add_argument('--headless', action='store_true', help='Run synthetic integration replay, print JSON, exit')
    parser.add_argument('--mute', action='store_true')
    parser.add_argument('--input-mirrored', action=argparse.BooleanOptionalAction, default=None,
                        help='Camera driver already mirrors raw pixels; undo before inference')
    parser.add_argument('--smoke-seconds', type=float, help='Close UI automatically after smoke test')
    parser.add_argument('--screenshot', type=Path, help='Save this app window during smoke test')
    args = parser.parse_args()
    settings_path = ROOT / 'data/camera_settings.json'
    from src.camera_settings import read_settings
    args.camera_settings_path = settings_path
    args.camera_settings_error = ''
    try:
        settings = read_settings(settings_path)
    except (OSError,ValueError) as exc:
        settings = {}
        args.camera_settings_error = f'Cannot read camera settings: {exc}'
    if args.camera is not None:
        parser.error('Numeric camera selection is no longer supported. Select the camera by name in Settings / Diagnostics.')
    args.camera_device=settings.get('selected_device')
    if not isinstance(args.camera_device,dict):
        args.camera_device=None
    # Legacy index-only settings are intentionally not used to identify a device.
    if args.input_mirrored is None:
        args.input_mirrored = bool(settings.get('input_mirrored',False))
    from src.application import ApplicationController, run_headless_replay
    if args.headless:
        print(json.dumps(run_headless_replay(), indent=2))
        return
    from src.ui import SignBridgeUI
    controller = None
    def on_action(action, value=None):
        if controller:
            controller.action(action, value)
    ui = SignBridgeUI(on_action)
    controller = ApplicationController(args, ui.update)
    controller.start()
    if args.smoke_seconds:
        from PySide6.QtCore import QTimer
        def finish():
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                ui.window.grab().save(str(args.screenshot))
            ui.window.close()
        QTimer.singleShot(int(args.smoke_seconds*1000), finish)
    try:
        ui.run()
    finally:
        controller.close()


if __name__ == '__main__':
    main()
