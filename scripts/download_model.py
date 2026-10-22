"""Download the official MediaPipe landmark model, not an ASL classifier."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import io
import zipfile

URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'models/hand_landmarker.task')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        print(f'Already present: {args.output}')
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        content = response.read(20000000)
    if len(content) < 1000000 or not zipfile.is_zipfile(io.BytesIO(content)):
        raise ValueError('Download is not the expected MediaPipe task model archive')
    temporary = args.output.with_suffix('.download')
    temporary.write_bytes(content)
    temporary.replace(args.output)
    args.output.with_suffix('.provenance.json').write_text(json.dumps({
        'url': URL, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
        'purpose': 'Hand landmark extraction only; not an ASL recognition model',
        'documentation': 'https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker'
    }, indent=2), encoding='utf-8')
    print(f'Downloaded {args.output} ({len(content)} bytes)')

if __name__ == '__main__':
    main()
