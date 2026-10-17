#!/usr/bin/env python3
"""Send three actual image files using the frozen blind protocol, save native receipt."""
import argparse
import base64
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from skill3d.segmentation.intern_decision_client import InternDecisionClient
from skill3d.infra.m5_identity import file_identity

def main():
    p = argparse.ArgumentParser()
    p.add_argument('images', type=Path, nargs=3, help='context, original crop, mask overlay')
    p.add_argument('--endpoint', default='http://127.0.0.1:18445')
    p.add_argument('--identity', default='')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    client = InternDecisionClient(args.endpoint, expected_identity_sha256=args.identity)
    images = [{'url': 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode(),
               'identity': file_identity(path)} for path in args.images]
    observation, raw, request_sha = client.observe(images)
    with args.output.open('x') as stream:
        json.dump({'observation': observation, 'raw_response': raw,
                   'request_sha256': request_sha, 'images': [r['identity'] for r in images]},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(observation))

if __name__ == '__main__':
    main()
