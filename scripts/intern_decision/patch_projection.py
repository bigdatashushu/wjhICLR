"""Opt-in, instance-only projection for official Qwen3.5 flattened vision patches.

No project imports or model downloads. Torch is imported only when applying the
adapter. BF16 request-level equivalence must be established separately on GPU.
"""
from __future__ import annotations

import hashlib
import inspect
import marshal
from pathlib import Path
import sys
from types import MethodType


QWEN_MODULE = 'transformers.models.qwen3_5.modeling_qwen3_5'
PATCH_KERNEL = (2, 16, 16)
PATCH_WIDTH = 3 * 2 * 16 * 16


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_sha(code):
    constants = tuple(
        value.replace(co_filename='') if inspect.iscode(value) else value
        for value in code.co_consts)
    return hashlib.sha256(marshal.dumps(
        code.replace(co_filename='', co_consts=constants))).hexdigest()


def _callable_identity(operator):
    function = getattr(operator, '__func__', operator)
    if not inspect.isfunction(function) or hasattr(function, '__wrapped__'):
        raise RuntimeError('Patch projection requires an unwrapped Python forward')
    try:
        source = inspect.getsource(function)
        path = Path(inspect.getfile(function)).resolve()
        return {'module': function.__module__, 'qualname': function.__qualname__,
                'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'file_sha256': _file_sha(path),
                'code_sha256': _code_sha(function.__code__)}
    except (OSError, TypeError) as exc:
        raise RuntimeError('Cannot establish patch projection source identity') from exc


def _tensor_digest(tensor, torch):
    if tensor is None:
        return None
    # A byte view also supports BF16, which NumPy cannot represent directly.
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _tensor_signature(tensor):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError:
        version = None  # Inference tensors need the outer byte-digest check.
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype),
            str(tensor.device), version)


def _geometry(layer, torch):
    conv = layer.proj
    if type(conv) is not torch.nn.Conv3d:
        raise RuntimeError('Patch projection requires an exact Conv3d')
    if (layer.in_channels != 3 or layer.temporal_patch_size != 2
            or layer.patch_size != 16 or conv.in_channels != 3
            or conv.kernel_size != PATCH_KERNEL or conv.stride != PATCH_KERNEL
            or conv.padding != (0, 0, 0) or conv.dilation != (1, 1, 1)
            or conv.groups != 1 or conv.padding_mode != 'zeros'):
        raise RuntimeError('Unsupported patch projection geometry')
    if (type(conv.out_channels) is not int or conv.out_channels <= 0
            or layer.embed_dim != conv.out_channels):
        raise RuntimeError('Patch projection output dimension mismatch')
    if (getattr(conv.forward, '__func__', None) is not torch.nn.Conv3d.forward
            or getattr(conv._conv_forward, '__func__', None) is not torch.nn.Conv3d._conv_forward
            or torch.nn.Conv3d.forward.__module__ != 'torch.nn.modules.conv'
            or torch.nn.Conv3d.forward.__qualname__ != 'Conv3d.forward'
            or torch.nn.Conv3d._conv_forward.__module__ != 'torch.nn.modules.conv'
            or torch.nn.Conv3d._conv_forward.__qualname__ != 'Conv3d._conv_forward'
            or any(getattr(instance, name) for instance in (layer, conv)
                   for name in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks'))):
        raise RuntimeError('Modified Conv3d implementation or hooks')
    weight, bias = conv.weight, conv.bias
    if (type(weight) is not torch.nn.Parameter
            or tuple(weight.shape) != (conv.out_channels, 3, *PATCH_KERNEL)
            or weight.layout != torch.strided or weight.device.type == 'meta'
            or weight.dtype not in {torch.bfloat16, torch.float32, torch.float64}):
        raise RuntimeError('Unsupported patch projection weight')
    if bias is not None and (
            type(bias) is not torch.nn.Parameter or tuple(bias.shape) != (conv.out_channels,)
            or bias.dtype != weight.dtype or bias.device != weight.device
            or bias.layout != torch.strided):
        raise RuntimeError('Unsupported patch projection bias')
    return {'in_channels': 3, 'out_channels': conv.out_channels,
            'kernel_size': list(PATCH_KERNEL), 'stride': list(PATCH_KERNEL),
            'padding': [0, 0, 0], 'dilation': [1, 1, 1], 'groups': 1,
            'padding_mode': 'zeros', 'input_shape': ['n', PATCH_WIDTH],
            'output_shape': ['n', conv.out_channels], 'patches_per_sample': 1,
            'weight_shape': list(weight.shape), 'bias_shape': list(bias.shape) if bias is not None else None,
            'dtype': str(weight.dtype), 'input_cast': 'weight dtype, as in native forward'}


def _linear_projection(layer, hidden_states, torch):
    if (not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2
            or hidden_states.shape[0] <= 0 or hidden_states.shape[1] != PATCH_WIDTH
            or not hidden_states.is_floating_point()
            or hidden_states.device != layer.proj.weight.device):
        raise RuntimeError('Expected floating one-patch-per-row input [n, 1536] on weight device')
    return torch.nn.functional.linear(
        hidden_states.to(dtype=layer.proj.weight.dtype),
        layer.proj.weight.flatten(1), layer.proj.bias)


class PatchProjection:
    """Keep the original bound callable and guard the single runtime replacement."""

    def __init__(self, layer, qwen, projection='native'):
        if projection not in {'native', 'linear'}:
            raise ValueError(f'Unknown patch projection: {projection}')
        cls = getattr(qwen, 'Qwen3_5VisionPatchEmbed', None)
        if (qwen.__name__ != QWEN_MODULE or sys.modules.get(QWEN_MODULE) is not qwen
                or cls is None or cls.__module__ != QWEN_MODULE
                or cls.__name__ != 'Qwen3_5VisionPatchEmbed' or type(layer) is not cls):
            raise RuntimeError('Patch projection requires the exact official Qwen3_5VisionPatchEmbed')
        if Path(inspect.getfile(cls)).resolve() != Path(qwen.__file__).resolve():
            raise RuntimeError('Patch embed class is not from the official Qwen3.5 source file')
        original = layer.forward
        if (getattr(original, '__self__', None) is not layer
                or getattr(original, '__func__', None) is not cls.forward
                or not inspect.isfunction(cls.forward)
                or cls.forward.__module__ != QWEN_MODULE
                or cls.forward.__qualname__ != 'Qwen3_5VisionPatchEmbed.forward'
                or Path(inspect.getfile(cls.forward)).resolve() != Path(qwen.__file__).resolve()
                or 'forward' in vars(layer)):
            raise RuntimeError('Patch embed forward was modified before applying adapter')
        import torch

        self.original_forward = original
        self.original_identity = _callable_identity(original)
        self.projection = projection
        self._layer, self._qwen, self._cls, self._torch = layer, qwen, cls, torch
        self._conv = layer.proj
        self._constraints = _geometry(layer, torch)
        self._weight, self._bias = self._conv.weight, self._conv.bias
        self._signatures = (_tensor_signature(self._weight), _tensor_signature(self._bias))
        self._digests = (_tensor_digest(self._weight, torch), _tensor_digest(self._bias, torch))
        self._original_function = original.__func__
        self._conv_functions = (torch.nn.Conv3d.forward, torch.nn.Conv3d._conv_forward)
        self._linear = torch.nn.functional.linear
        self._constants = (QWEN_MODULE, PATCH_KERNEL, PATCH_WIDTH)
        self._helpers = {name: (globals()[name], globals()[name].__code__) for name in (
            '_file_sha', '_code_sha', '_callable_identity', '_tensor_digest',
            '_tensor_signature', '_geometry', '_linear_projection')}
        self._methods = {name: (getattr(type(self), name), getattr(type(self), name).__code__)
                         for name in ('verify', 'identity')}
        self._native_codes = tuple(function.__code__ for function in (
            self._original_function, *self._conv_functions))
        paths = {Path(__file__).resolve(), Path(qwen.__file__).resolve(),
                 Path(inspect.getfile(torch.nn.Conv3d.forward)).resolve(),
                 Path(torch.nn.functional.__file__).resolve()}
        self._files = {path: _file_sha(path) for path in paths}
        self._installed_forward = None
        self._installed_code = None
        self._installed_closure = ()
        if projection == 'linear':
            verify = self.verify

            def forward(bound_layer, hidden_states):
                verify(check_files=False)
                return _linear_projection(bound_layer, hidden_states, torch)

            self._installed_forward = MethodType(forward, layer)
            self._installed_code = forward.__code__
            self._installed_closure = tuple(cell.cell_contents for cell in forward.__closure__)
            # Resolve all identity sources and validate before the only mutation.
            _callable_identity(forward)
            layer.forward = self._installed_forward

    def verify(self, *, check_files=True):
        layer, torch = self._layer, self._torch
        if ((QWEN_MODULE, PATCH_KERNEL, PATCH_WIDTH) != self._constants
                or sys.modules.get(QWEN_MODULE) is not self._qwen
                or type(layer) is not self._cls or self._qwen.Qwen3_5VisionPatchEmbed is not self._cls
                or self._cls.forward is not self._original_function
                or layer.proj is not self._conv
                or torch.nn.Conv3d.forward is not self._conv_functions[0]
                or torch.nn.Conv3d._conv_forward is not self._conv_functions[1]
                or torch.nn.functional.linear is not self._linear):
            raise RuntimeError('Patch projection implementation changed')
        for name, (function, code) in self._helpers.items():
            if globals()[name] is not function or function.__code__ is not code:
                raise RuntimeError(f'Patch projection helper changed: {name}')
        for name, (function, code) in self._methods.items():
            if getattr(type(self), name) is not function or function.__code__ is not code:
                raise RuntimeError(f'Patch projection guard changed: {name}')
        if any(function.__code__ is not code for function, code in zip(
                (self._original_function, *self._conv_functions), self._native_codes)):
            raise RuntimeError('Native patch projection code changed')
        actual = layer.forward
        if self.projection == 'linear':
            if (actual is not self._installed_forward
                    or actual.__self__ is not layer
                    or actual.__func__.__code__ is not self._installed_code
                    or any(cell.cell_contents is not value for cell, value in zip(
                        actual.__func__.__closure__, self._installed_closure))):
                raise RuntimeError('Bound patch projection forward changed')
        elif (getattr(actual, '__self__', None) is not layer
              or getattr(actual, '__func__', None) is not self._original_function):
            raise RuntimeError('Native patch projection forward changed')
        if _geometry(layer, torch) != self._constraints:
            raise RuntimeError('Patch projection constraints changed')
        if (self._conv.weight is not self._weight or self._conv.bias is not self._bias
                or (_tensor_signature(self._weight), _tensor_signature(self._bias)) != self._signatures):
            raise RuntimeError('Patch projection weights changed')
        if check_files:
            if any(_file_sha(path) != sha for path, sha in self._files.items()):
                raise RuntimeError('Patch projection source file changed')
            if (_tensor_digest(self._weight, torch), _tensor_digest(self._bias, torch)) != self._digests:
                raise RuntimeError('Patch projection weight bytes changed')

    def identity(self):
        self.verify()
        return {'schema': 'intern-patch-projection/1',
                'requested_projection': self.projection, 'observed_projection': self.projection,
                'adapter_revision': 'r03' if self.projection == 'linear' else None,
                'original_forward': self.original_identity,
                'observed_forward': _callable_identity(self._layer.forward),
                'projected_forward': (_callable_identity(self._installed_forward)
                                      if self._installed_forward is not None else None),
                'helpers': {name: _callable_identity(function)
                            for name, (function, _) in self._helpers.items()},
                'guards': {name: _callable_identity(function)
                           for name, (function, _) in self._methods.items()},
                'native_conv_forward': _callable_identity(self._conv_functions[0]),
                'linear_operator': {'module': self._linear.__module__,
                                    'qualname': self._linear.__qualname__},
                'adapter_file_sha256': self._files[Path(__file__).resolve()],
                'constraints': self._constraints,
                'weights': {'weight_sha256': self._digests[0], 'bias_sha256': self._digests[1]},
                'bf16_request_equivalence': 'unproven; requires genuine GPU request replay'}


def apply_patch_projection(layer, qwen, projection='native'):
    """Validate before mutation; native records identity without changing the layer."""
    return PatchProjection(layer, qwen, projection)
