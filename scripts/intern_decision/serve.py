#!/usr/bin/env python3
"""Loopback HTTP adapter around the pinned model package's native DecisionEngine.

Run in the isolated service environment; this file has no project imports.
Client paths/IDs never enter compiler evidence. No generate() is used.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path, PurePosixPath
import signal
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


KERNEL_PACKAGES = ('fla-core', 'flash-linear-attention', 'causal-conv1d',
                   'triton', 'einops', 'ninja')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def distribution_identity(name, *, distribution=importlib.metadata.distribution):
    """Hash installed payload bytes, not RECORD's claimed hashes or install metadata."""
    try:
        dist = distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {'present': False, 'version': None, 'files': {}, 'sha256': None}
    if not dist.files:
        raise RuntimeError(f'No file inventory for kernel distribution: {name}')
    files = {}
    for entry in dist.files:
        relative = PurePosixPath(str(entry))
        # Entry-point shebangs, install metadata and bytecode are location/time dependent.
        if (relative.is_absolute() or '..' in relative.parts
                or '__pycache__' in relative.parts
                or any(part.endswith(('.dist-info', '.egg-info')) for part in relative.parts)
                or relative.suffix in {'.pyc', '.pyo', '.pth'}):
            continue
        key = relative.as_posix()
        if key in files:
            raise RuntimeError(f'Duplicate kernel distribution file: {name}/{key}')
        try:
            files[key] = file_sha(Path(dist.locate_file(entry)))
        except OSError as exc:
            raise RuntimeError(f'Cannot hash kernel distribution file: {name}/{key}') from exc
    if not files:
        raise RuntimeError(f'No runtime payload for kernel distribution: {name}')
    files = dict(sorted(files.items()))
    return {'present': True, 'version': dist.version, 'files': files,
            'sha256': digest(files)}


def operator_identity(operator, *, hash_file=file_sha):
    """Identify the bound implementation and every decorator's actual source file."""
    if operator is None:
        return {'present': False}
    if not callable(operator):
        raise RuntimeError('Kernel binding is not callable')
    chain = []
    seen = set()
    operator = getattr(operator, '__func__', operator)
    while operator is not None:
        if id(operator) in seen:
            raise RuntimeError('Cyclic kernel wrapper chain')
        seen.add(id(operator))
        module_name = getattr(operator, '__module__', None)
        qualname = getattr(operator, '__qualname__', None)
        if not module_name or not qualname:
            raise RuntimeError('Kernel binding has no source identity')
        try:
            source = inspect.getfile(operator)
        except TypeError:
            source = getattr(sys.modules.get(module_name), '__file__', None)
        if not source:
            raise RuntimeError(f'No source file for kernel binding: {module_name}.{qualname}')
        if source.endswith(('.pyc', '.pyo')):
            source = importlib.util.source_from_cache(source)
        try:
            sha = hash_file(Path(source))
        except OSError as exc:
            raise RuntimeError(f'Cannot hash kernel binding: {module_name}.{qualname}') from exc
        chain.append({'module': module_name, 'qualname': qualname, 'source_sha256': sha})
        operator = getattr(operator, '__wrapped__', None)
    return {'present': True, 'sources': chain}


def kernel_runtime_identity(model, qwen, *, kernel_mode=None,
                            distribution=importlib.metadata.distribution):
    """Observe loaded Qwen3.5 layer bindings without importing or executing kernels.

    An omitted contract preserves upstream selection. Explicit torch/fla/full modes
    require matching prefill AND cached-update paths; they never patch the model.
    """
    if kernel_mode not in {None, 'torch', 'fla', 'full'}:
        raise ValueError(f'Unknown kernel mode: {kernel_mode}')
    hashes = {}

    def hash_file(path):
        path = path.resolve()
        if path not in hashes:
            hashes[path] = file_sha(path)
        return hashes[path]

    def operator_row(value, fast, fallback, fast_backend):
        if fallback is not None and value is fallback:
            backend = 'torch'
        elif fast is not None and value is fast:
            backend = fast_backend
        else:
            backend = 'unknown'
        return {'backend': backend, **operator_identity(value, hash_file=hash_file)}

    def same_forward(instance, cls):
        actual = inspect.unwrap(instance.forward)
        expected = inspect.unwrap(cls.forward)
        return getattr(actual, '__func__', actual) is expected

    bindings = {name: getattr(qwen, name) for name in (
        'chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule',
        'causal_conv1d_fn', 'causal_conv1d_update', 'FusedRMSNormGated')}
    implementations = {}
    modes = set()
    for name, layer in sorted(model.named_modules(), key=lambda row: row[0]):
        if not isinstance(layer, qwen.Qwen3_5GatedDeltaNet):
            continue
        operators = {
            'attention_prefill': operator_row(layer.chunk_gated_delta_rule,
                bindings['chunk_gated_delta_rule'], qwen.torch_chunk_gated_delta_rule, 'fla'),
            'attention_update': operator_row(layer.recurrent_gated_delta_rule,
                bindings['fused_recurrent_gated_delta_rule'], qwen.torch_recurrent_gated_delta_rule, 'fla'),
            'convolution_update': operator_row(layer.causal_conv1d_update,
                bindings['causal_conv1d_update'], qwen.torch_causal_conv1d_update, 'causal-conv1d')}
        if layer.causal_conv1d_fn is None:
            operators['convolution_prefill'] = {
                'backend': 'torch' if same_forward(layer.conv1d, qwen.nn.Conv1d) else 'unknown',
                **operator_identity(layer.conv1d.forward, hash_file=hash_file)}
        else:
            operators['convolution_prefill'] = operator_row(layer.causal_conv1d_fn,
                bindings['causal_conv1d_fn'], None, 'causal-conv1d')
        norm_type = type(layer.norm)
        norm_backend = ('torch' if norm_type is qwen.Qwen3_5RMSNormGated else
                        'fla' if norm_type is bindings['FusedRMSNormGated'] else 'unknown')
        if not same_forward(layer.norm, norm_type):
            norm_backend = 'unknown'
        attention = {operators[key]['backend'] for key in ('attention_prefill', 'attention_update')}
        convolution = {operators[key]['backend'] for key in ('convolution_prefill', 'convolution_update')}
        mode = 'mixed'
        if attention == {'torch'} and convolution == {'torch'} and norm_backend == 'torch':
            mode = 'torch'
        elif attention == {'fla'} and norm_backend in {'torch', 'fla'}:
            if convolution == {'torch'}:
                mode = 'fla'
            elif convolution == {'causal-conv1d'}:
                mode = 'full'
        if not same_forward(layer, qwen.Qwen3_5GatedDeltaNet):
            mode = 'mixed'
        if kernel_mode is not None and mode != kernel_mode:
            raise RuntimeError(f'Kernel contract mismatch at {name}: requested {kernel_mode}, observed {mode}')
        modes.add(mode)
        row = {'mode': mode, 'operators': operators,
               'normalization': {'backend': norm_backend,
                   **operator_identity(layer.norm.forward, hash_file=hash_file)},
               'forward': operator_identity(layer.forward, hash_file=hash_file)}
        key = digest(row)
        if key not in implementations:
            implementations[key] = {'layers': [], **row}
        implementations[key]['layers'].append(name)
    if not implementations:
        raise RuntimeError('Cannot establish kernel identity: no Qwen3.5 GatedDeltaNet layers')
    return {'schema': 'intern-kernel-runtime/1', 'requested_mode': kernel_mode or 'auto',
            'mode': next(iter(modes)) if len(modes) == 1 else 'mixed',
            'distributions': {name: distribution_identity(name, distribution=distribution)
                              for name in KERNEL_PACKAGES},
            'qwen3_5': {'module': qwen.__name__, 'source_sha256': hash_file(Path(qwen.__file__)),
                'bindings': {name: operator_identity(value, hash_file=hash_file)
                             for name, value in bindings.items()},
                'implementations': [implementations[key] for key in sorted(implementations)]}}


def configure_patch_projection(model, qwen, projection='native'):
    """Keep native startup import-free; load the standalone adapter only on opt-in."""
    if projection == 'native':
        return None
    if projection != 'linear':
        raise ValueError(f'Unknown patch projection: {projection}')
    cls = getattr(qwen, 'Qwen3_5VisionPatchEmbed', None)
    layers = [(name, layer) for name, layer in model.named_modules()
              if cls is not None and isinstance(layer, cls)]
    if len(layers) != 1:
        raise RuntimeError('Linear patch projection requires exactly one official vision patch embed')
    path = Path(__file__).resolve().with_name('patch_projection.py')
    spec = importlib.util.spec_from_file_location('intern_patch_projection', path)
    adapter_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter_module
    spec.loader.exec_module(adapter_module)
    name, layer = layers[0]
    adapter = adapter_module.apply_patch_projection(layer, qwen, projection)
    return name, adapter


def patch_projection_identity(configured):
    if configured is None:
        return {'schema': 'intern-patch-projection/1',
                'requested_projection': 'native', 'observed_projection': 'native'}
    name, adapter = configured
    return {'layer': name, **adapter.identity()}


def verify_patch_projection(model, qwen, configured):
    if configured is None:
        return
    name, adapter = configured
    layers = [(key, layer) for key, layer in model.named_modules()
              if isinstance(layer, qwen.Qwen3_5VisionPatchEmbed)]
    if (len(layers) != 1 or layers[0][0] != name
            or layers[0][1] is not adapter.original_forward.__self__):
        raise RuntimeError('Model vision patch embed changed after applying adapter')
    adapter.verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--port', type=int, default=18445)
    parser.add_argument('--startup', type=Path, required=True)
    parser.add_argument('--temp-root', type=Path, required=True)
    parser.add_argument('--kernel-mode', choices=('torch', 'fla', 'full'),
        help='Require torch only, FLA attention with torch convolution, or FLA with causal-conv1d; '
             'omitting this flag preserves automatic upstream selection')
    parser.add_argument('--patch-projection', choices=('native', 'linear'), default='native',
        help='Preserve native Conv3d (default), or opt into the r03 linear vision patch adapter; '
             'BF16 genuine-request equivalence is not established by this flag')
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import torch
    import transformers
    from PIL import Image
    torch.set_num_threads(4)
    temperature = module.DEFAULT_TEMPERATURE
    engine = module.DecisionEngine(checkpoint=str(args.model_dir), temperature=1,
        device='cuda', dtype='bfloat16', max_length=8192, attn_implementation='sdpa')
    from transformers.models.qwen3_5 import modeling_qwen3_5
    kernels = kernel_runtime_identity(engine.backend.model, modeling_qwen3_5,
        kernel_mode=args.kernel_mode)
    projection = configure_patch_projection(engine.backend.model, modeling_qwen3_5,
        args.patch_projection)
    projection_identity = patch_projection_identity(projection)
    identity = {'schema': ('intern-decision-service/3' if projection is not None
                           else 'intern-decision-service/2'),
        'model': 'internlm/Intern-Decision-4B', 'model_revision': args.revision,
        'backend': 'native-hf-model-package', 'dtype': 'bfloat16',
        'implementation_sha256': file_sha(args.model_dir / 'inference.py'),
        'service_sha256': file_sha(Path(__file__)), 'files': expected['files'],
        'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__,
                    'transformers': transformers.__version__, 'cuda': torch.version.cuda,
                    'kernels': kernels, 'patch_projection': projection_identity},
        'processor': json.loads((args.model_dir / 'preprocessor_config.json').read_text()),
        'limits': {'max_length': 8192, 'max_images': 3, 'max_body_bytes': 32*1024*1024,
                   'max_image_bytes': 12*1024*1024, 'max_pixels': 16000000,
                   'concurrency': 1, 'hard_inference_timeout_s': 120},
        'calibration': {'method': 'temperature-scaling', 'temperature': temperature},
        'calibration_source': 'DEFAULT_TEMPERATURE in same pinned model-package inference.py; no task fitting',
        'readout': 'official compile_row; logits at marker minus 1; legal single-token symbols',
        'device': {'name': torch.cuda.get_device_name(0), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}}
    identity_sha = digest(identity)

    def verify_projection():
        verify_patch_projection(engine.backend.model, modeling_qwen3_5, projection)
        if projection is not None and file_sha(Path(__file__)) != identity['service_sha256']:
            raise RuntimeError('Service source changed after applying patch adapter')

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
            try:
                verify_projection()
            except Exception as exc:
                return self.send(503, {'error': f'{type(exc).__name__}: {exc}'})
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
                verify_projection()
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
                    verify_projection()
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
