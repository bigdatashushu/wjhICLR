#!/usr/bin/env python3
"""Loopback HTTP adapter around the pinned model package's native DecisionEngine.

Run in the isolated service environment; this file has no project imports.
Client paths/IDs never enter compiler evidence. No generate() is used.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--port', type=int, default=18445)
    parser.add_argument('--startup', type=Path, required=True)
    parser.add_argument('--temp-root', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.temp_root.mkdir(parents=True, exist_ok=True)
    # Freeze the shipped implementation and every model/processor/tokenizer artifact.
    expected = json.loads((args.model_dir / 'download_manifest.json').read_text())
    if expected['revision'] != args.revision:
        raise ValueError('Download revision mismatch')
    for name, row in expected['files'].items():
        if file_sha(args.model_dir / name) != row['sha256']:
            raise ValueError(f'Model artifact hash mismatch: {name}')
    spec = importlib.util.spec_from_file_location('intern_native_inference', args.model_dir / 'inference.py')
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import torch
    import transformers
    from PIL import Image
    torch.set_num_threads(4)
    temperature = module.DEFAULT_TEMPERATURE
    engine = module.DecisionEngine(checkpoint=str(args.model_dir), temperature=1,
        device='cuda', dtype='bfloat16', max_length=8192, attn_implementation='sdpa')
    identity = {'model': 'internlm/Intern-Decision-4B', 'model_revision': args.revision,
        'backend': 'native-hf-model-package', 'dtype': 'bfloat16',
        'implementation_sha256': file_sha(args.model_dir / 'inference.py'),
        'service_sha256': file_sha(Path(__file__)), 'files': expected['files'],
        'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'transformers': transformers.__version__, 'cuda': torch.version.cuda},
        'processor': json.loads((args.model_dir / 'preprocessor_config.json').read_text()),
        'limits': {'max_length': 8192, 'max_images': 3, 'max_body_bytes': 32*1024*1024,
                   'max_image_bytes': 12*1024*1024, 'max_pixels': 16000000,
                   'concurrency': 1, 'hard_inference_timeout_s': 120},
        'calibration': {'method': 'temperature-scaling', 'temperature': temperature},
        'calibration_source': 'DEFAULT_TEMPERATURE in same pinned model-package inference.py; no task fitting',
        'readout': 'official compile_row; logits at marker minus 1; legal single-token symbols',
        'device': {'name': torch.cuda.get_device_name(0), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}}
    identity_sha = digest(identity)
    trace = {}
    original_encode = engine.backend.encode

    def encode(row):
        compiled, batch, positions = original_encode(row)
        image_id = engine.backend.model.config.image_token_id
        trace.clear()
        trace.update(image_count=len(row.get('images', [])),
            image_token_count=int((batch['input_ids'] == image_id).sum()),
            pixel_values_shape=list(batch['pixel_values'].shape) if 'pixel_values' in batch else [],
            image_grid_thw=batch['image_grid_thw'].tolist() if 'image_grid_thw' in batch else [],
            input_ids_sha256=hashlib.sha256(batch['input_ids'].numpy().tobytes()).hexdigest(),
            input_tokens=int(batch['input_ids'].shape[-1]),
            decision_logit_positions=positions.tolist(),
            legal_symbol_token_ids={field:[engine.tokenizer.encode(s, add_special_tokens=False)[0]
                for s in symbols] for field,symbols in compiled.symbols.items()})
        return compiled, batch, positions

    engine.backend.encode = encode
    lock = threading.Lock()
    counters = {'requests': 0, 'failures': 0}
    startup = {'identity': identity, 'identity_sha256': identity_sha,
        'pid': os.getpid(), 'port': args.port, 'host': '127.0.0.1',
        'cold_start_s': time.perf_counter() - start,
        'gpu_allocated_bytes': torch.cuda.memory_allocated(),
        'gpu_reserved_bytes': torch.cuda.memory_reserved()}
    args.startup.parent.mkdir(parents=True, exist_ok=True)
    args.startup.write_text(json.dumps(startup, indent=2))

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(20)

        def log_message(self, *args):
            pass  # Never log request images, prompt or credentials.

        def send(self, status, body):
            payload = json.dumps(body, allow_nan=False).encode()
            try:
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != '/health':
                return self.send(404, {'error': 'unknown route'})
            self.send(200, {**startup, 'counters': counters,
                'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()})

        def do_POST(self):
            if self.path != '/decisions':
                return self.send(404, {'error': 'unknown route'})
            if not lock.acquire(blocking=False):
                return self.send(429, {'error': 'single inference slot busy'})
            began = time.perf_counter()
            watchdog = None
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 32*1024*1024:
                    raise ValueError('body size outside limit')
                body = json.loads(self.rfile.read(size))
                if set(body) != {'request_sha256', 'request'}:
                    raise ValueError('unexpected envelope fields')
                request = body['request']
                if set(request) - {'state', 'questions', 'images'}:
                    raise ValueError('unexpected model evidence fields')
                request_sha = digest(request)
                if body['request_sha256'] != request_sha:
                    raise ValueError('request digest mismatch')
                images = request.get('images', [])
                # Empty images only for explicitly measured no-image diagnostics.
                if len(images) not in {0, 3}:
                    raise ValueError('supply zero diagnostic images or exactly three audit images')
                with tempfile.TemporaryDirectory(dir=args.temp_root, prefix='request-') as temp:
                    paths = []
                    for index, data_url in enumerate(images):
                        if not isinstance(data_url, str) or not data_url.startswith('data:image/'):
                            raise ValueError('only inline image bytes accepted')
                        head, encoded = data_url.split(',', 1)
                        if head not in {'data:image/jpeg;base64', 'data:image/png;base64'}:
                            raise ValueError('unsupported image format')
                        raw = base64.b64decode(encoded, validate=True)
                        if len(raw) > 12*1024*1024:
                            raise ValueError('image byte limit')
                        with Image.open(io.BytesIO(raw)) as img:
                            if img.width*img.height > 16000000 or getattr(img, 'n_frames', 1) != 1:
                                raise ValueError('static image pixel limit')
                            img.verify()
                        path = Path(temp) / f'{index}.img'
                        path.write_bytes(raw)
                        paths.append(str(path))
                    counters['requests'] += 1
                    # A wedged CUDA operation terminates this service after 120s.
                    watchdog = threading.Timer(120, lambda: os.kill(os.getpid(), signal.SIGTERM))
                    watchdog.start()
                    raw_result = engine.predict({**request, 'images': paths})
                    result = module.scale_result(raw_result, temperature)
                    for answer in result['answers'].values():
                        answer['decision'] = module.argmax(answer['probabilities'])
                    self.send(200, {'request_sha256': request_sha, 'identity_sha256': identity_sha,
                        'result': result, 'uncalibrated_result': raw_result,
                        'vision_evidence': dict(trace), 'elapsed_s': time.perf_counter()-began,
                        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            except Exception as exc:
                counters['failures'] += 1
                self.send(422, {'error': f'{type(exc).__name__}: {exc}'})
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                lock.release()

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 4

    Server(('127.0.0.1', args.port), Handler).serve_forever()

if __name__ == '__main__':
    main()
