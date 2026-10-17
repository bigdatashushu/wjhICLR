#!/usr/bin/env python3
"""Stop only the verified service PID from this deployment's startup receipt."""
import argparse
import json
import os
from pathlib import Path
import signal

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
receipt = json.loads((a.root / 'runtime/startup.json').read_text())
pid = receipt['pid']
cmdline = Path(f'/proc/{pid}/cmdline')
if not cmdline.exists():
    print('Service already exited')
else:
    args = cmdline.read_bytes().split(b'\0')
    expected = str(a.root.resolve() / 'model').encode()
    if b'--model-dir' not in args or expected not in args or not any(
            value.endswith(b'/scripts/intern_decision/serve.py') for value in args):
        raise SystemExit('PID command identity differs; refusing to stop')
    os.kill(pid, signal.SIGTERM)
    print(f'Termination requested for verified PID {pid}')
