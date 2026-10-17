#!/usr/bin/env python3
"""Download and hash the pinned official model package; standard library only."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL = 'internlm/Intern-Decision-4B'
REVISION = '1e65a988dacbcf00b84aee7366f9e47563407000'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}', timeout=45) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise ValueError('Resolved model revision changed')
    names = [row['rfilename'] for row in metadata['siblings']]
    def download(name):
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unexpected artifact path')
        path = args.destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}', timeout=60) as response:
                with path.with_suffix(path.suffix + '.part').open('wb') as output:
                    while block := response.read(8 * 1024 * 1024):
                        output.write(block)
            path.with_suffix(path.suffix + '.part').replace(path)
        h = hashlib.sha256()
        with path.open('rb') as source:
            while block := source.read(8 * 1024 * 1024):
                h.update(block)
        return name, {'sha256': h.hexdigest(), 'size': path.stat().st_size}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = dict(pool.map(download, names))
    manifest = {'model': MODEL, 'revision': REVISION, 'files': files}
    (args.destination / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'revision': REVISION, 'files': len(files),
                      'bytes': sum(row['size'] for row in files.values())}))

if __name__ == '__main__':
    main()
