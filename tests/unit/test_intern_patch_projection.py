"""Offline adapter contracts, with real CPU Conv3d math when Torch is available.

The Qwen interface fixture is model-free; numerical tests use actual Torch
operators and parameters. No GPU/BF16 request-equivalence claim is made here.
"""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType, ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / 'scripts/intern_decision/patch_projection.py'
SERVICE_PATH = ROOT / 'scripts/intern_decision/serve.py'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = load_module('intern_patch_projection_tests', ADAPTER_PATH)
serve = load_module('intern_patch_service_tests', SERVICE_PATH)


QWEN_FIXTURE = '''
import torch
from torch import nn

class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(self, out_channels=4, bias=True, dtype=torch.float32):
        super().__init__()
        self.in_channels = 3
        self.temporal_patch_size = 2
        self.patch_size = 16
        self.embed_dim = out_channels
        self.proj = nn.Conv3d(3, out_channels, kernel_size=(2, 16, 16),
                             stride=(2, 16, 16), bias=bias, dtype=dtype)

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
'''


@pytest.fixture
def cpu_runtime(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch', reason='Real CPU Torch required for numerical/parameter tests')
    source = tmp_path / 'modeling_qwen3_5.py'
    source.write_text(QWEN_FIXTURE)
    qwen = ModuleType(patch.QWEN_MODULE)
    qwen.__file__ = str(source)
    monkeypatch.setitem(sys.modules, qwen.__name__, qwen)
    exec(compile(QWEN_FIXTURE, str(source), 'exec'), qwen.__dict__)
    return torch, qwen


def test_standalone_import_help_and_native_configuration_need_no_torch():
    code = '''
import importlib.util
import sys
from types import SimpleNamespace
for name, path in zip(("patch", "serve"), sys.argv[1:]):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    if name == "serve":
        assert module.configure_patch_projection(None, None) is None
        identity = module.patch_projection_identity(None)
        assert identity["requested_projection"] == identity["observed_projection"] == "native"
        sys.argv = ["serve.py", "--help"]
        try:
            module.main()
        except SystemExit as exc:
            assert exc.code == 0
        assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, str(ADAPTER_PATH), str(SERVICE_PATH)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--patch-projection {native,linear}' in result.stdout
    assert 'r03 linear vision patch adapter' in ' '.join(result.stdout.split())


def test_unknown_projection_and_unrecognized_class_reject_without_torch():
    with pytest.raises(ValueError, match='Unknown patch projection'):
        patch.apply_patch_projection(None, None, 'conv-fast')
    with pytest.raises(RuntimeError, match='exact official'):
        patch.apply_patch_projection(object(), SimpleNamespace(__name__='custom'), 'linear')
    with pytest.raises(ValueError, match='Unknown patch projection'):
        serve.configure_patch_projection(None, None, 'conv-fast')
    empty = SimpleNamespace(named_modules=lambda: iter([]))
    with pytest.raises(RuntimeError, match='exactly one official'):
        serve.configure_patch_projection(empty, SimpleNamespace(), 'linear')


def test_callable_identity_hashes_executable_code_as_well_as_source(monkeypatch):
    before = patch._callable_identity(patch._linear_projection)

    def modified(layer, hidden_states, torch):
        raise AssertionError('modified helper must never execute')

    original_code = patch._linear_projection.__code__
    changed_code = modified.__code__.replace(
        co_filename=original_code.co_filename, co_firstlineno=original_code.co_firstlineno)
    monkeypatch.setattr(patch._linear_projection, '__code__', changed_code)
    after = patch._callable_identity(patch._linear_projection)
    assert before['file_sha256'] == after['file_sha256']
    assert before['source_sha256'] == after['source_sha256']
    assert before['code_sha256'] != after['code_sha256']


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('rows', [1, 3])
@pytest.mark.parametrize('cast_input', [False, True])
def test_real_cpu_conv3d_equivalence_and_weights_unchanged(
        cpu_runtime, dtype_name, bias, rows, cast_input):
    torch, qwen = cpu_runtime
    dtype = getattr(torch, dtype_name)
    layer = qwen.Qwen3_5VisionPatchEmbed(dtype=dtype, bias=bias).eval()
    data = torch.randn(rows, 1536, dtype=torch.float64 if cast_input else dtype,
                       generator=torch.Generator().manual_seed(31))
    saved = {name: value.clone() for name, value in layer.state_dict().items()}
    original_function = qwen.Qwen3_5VisionPatchEmbed.forward
    conv_function = layer.proj.forward.__func__
    with torch.inference_mode():
        expected = layer(data)
        adapter = patch.apply_patch_projection(layer, qwen, 'linear')
        actual = layer(data)
        original_again = adapter.original_forward(data)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(original_again, expected, rtol=0, atol=0)
    assert actual.shape == (rows, 4)
    assert actual.dtype == dtype
    assert adapter.original_forward.__self__ is layer
    assert adapter.original_forward.__func__ is original_function
    assert qwen.Qwen3_5VisionPatchEmbed.forward is original_function
    assert layer.proj.forward.__func__ is conv_function
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
    identity = adapter.identity()
    assert identity['requested_projection'] == identity['observed_projection'] == 'linear'
    assert identity['adapter_revision'] == 'r03'
    assert identity['original_forward'] == adapter.original_identity
    assert identity['observed_forward'] == identity['projected_forward']
    assert identity['original_forward'] != identity['projected_forward']
    assert identity['weights']['weight_sha256']
    assert bool(identity['weights']['bias_sha256']) == bias
    assert identity['constraints']['weight_shape'] == [4, 3, 2, 16, 16]
    assert identity['constraints']['input_shape'] == ['n', 1536]
    assert identity['constraints']['patches_per_sample'] == 1
    assert '_linear_projection' in identity['helpers']
    assert 'unproven' in identity['bf16_request_equivalence']
    assert adapter.identity() == identity


def test_noncontiguous_rows_preserve_native_flatten_order(cpu_runtime):
    torch, qwen = cpu_runtime
    layer = qwen.Qwen3_5VisionPatchEmbed(dtype=torch.float64).eval()
    data = torch.randn(3, 1536, 2, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(73))[:, :, 0]
    assert not data.is_contiguous()
    with torch.inference_mode():
        expected = layer(data)
        adapter = patch.apply_patch_projection(layer, qwen, 'linear')
        torch.testing.assert_close(layer(data), expected, rtol=1e-12, atol=1e-12)
    adapter.verify()


def test_native_is_a_true_instance_no_op(cpu_runtime):
    torch, qwen = cpu_runtime
    layer = qwen.Qwen3_5VisionPatchEmbed().eval()
    keys = set(vars(layer))
    weight, bias = layer.proj.weight, layer.proj.bias
    function = layer.forward.__func__
    adapter = patch.apply_patch_projection(layer, qwen)
    assert set(vars(layer)) == keys
    assert 'forward' not in vars(layer)
    assert layer.forward.__func__ is function
    assert layer.proj.weight is weight and layer.proj.bias is bias
    identity = adapter.identity()
    assert identity['requested_projection'] == identity['observed_projection'] == 'native'
    assert identity['projected_forward'] is None
    assert identity['observed_forward'] == identity['original_forward']


@pytest.mark.parametrize('invalid', [
    'layer_channels', 'temporal_size', 'patch_size', 'embed_dim', 'conv_channels',
    'kernel', 'stride', 'padding', 'dilation', 'groups', 'padding_mode',
    'output_channels', 'weight_shape', 'weight_dtype', 'bias_shape', 'bias_dtype',
    'conv_forward', 'conv_hook', 'conv_subclass', 'layer_forward', 'layer_subclass',
    'layer_hook', 'class_forward',
])
def test_invalid_geometry_or_implementation_rejects_before_mutation(cpu_runtime, invalid):
    torch, qwen = cpu_runtime
    layer = qwen.Qwen3_5VisionPatchEmbed().eval()
    changes = {
        'layer_channels': (layer, 'in_channels', 1),
        'temporal_size': (layer, 'temporal_patch_size', 1),
        'patch_size': (layer, 'patch_size', 8),
        'embed_dim': (layer, 'embed_dim', 7),
        'conv_channels': (layer.proj, 'in_channels', 1),
        'kernel': (layer.proj, 'kernel_size', (1, 16, 16)),
        'stride': (layer.proj, 'stride', (1, 16, 16)),
        'padding': (layer.proj, 'padding', (1, 0, 0)),
        'dilation': (layer.proj, 'dilation', (2, 1, 1)),
        'groups': (layer.proj, 'groups', 3),
        'padding_mode': (layer.proj, 'padding_mode', 'reflect'),
        'output_channels': (layer.proj, 'out_channels', 7),
    }
    if invalid in changes:
        owner, name, value = changes[invalid]
        setattr(owner, name, value)
    elif invalid == 'weight_shape':
        layer.proj.weight = torch.nn.Parameter(torch.zeros(4, 3, 1, 16, 16))
    elif invalid == 'weight_dtype':
        layer.proj.weight = torch.nn.Parameter(layer.proj.weight.to(torch.float16))
    elif invalid == 'bias_shape':
        layer.proj.bias = torch.nn.Parameter(torch.zeros(3))
    elif invalid == 'bias_dtype':
        layer.proj.bias = torch.nn.Parameter(layer.proj.bias.to(torch.float64))
    elif invalid == 'conv_forward':
        layer.proj.forward = lambda _: None
    elif invalid == 'conv_hook':
        layer.proj.register_forward_hook(lambda *_: None)
    elif invalid == 'layer_hook':
        layer.register_forward_pre_hook(lambda *_: None)
    elif invalid == 'conv_subclass':
        class CustomConv(torch.nn.Conv3d):
            pass
        layer.proj = CustomConv(3, 4, (2, 16, 16), stride=(2, 16, 16))
    elif invalid == 'layer_forward':
        layer.forward = MethodType(qwen.Qwen3_5VisionPatchEmbed.forward, layer)
    elif invalid == 'layer_subclass':
        class CustomPatch(qwen.Qwen3_5VisionPatchEmbed):
            pass
        layer = CustomPatch()
    elif invalid == 'class_forward':
        qwen.Qwen3_5VisionPatchEmbed.forward = lambda *_: None
    before = dict(vars(layer))
    weight, bias = layer.proj.weight, layer.proj.bias
    with pytest.raises(RuntimeError):
        patch.apply_patch_projection(layer, qwen, 'linear')
    assert vars(layer).keys() == before.keys()
    assert vars(layer).get('forward') is before.get('forward')
    assert layer.proj.weight is weight and layer.proj.bias is bias


@pytest.mark.parametrize('shape', [(1536,), (1, 3072), (1, 3, 2, 16, 16), (0, 1536)])
def test_runtime_input_must_be_one_flattened_patch_per_row(cpu_runtime, shape):
    torch, qwen = cpu_runtime
    layer = qwen.Qwen3_5VisionPatchEmbed().eval()
    adapter = patch.apply_patch_projection(layer, qwen, 'linear')
    with pytest.raises(RuntimeError, match='one-patch-per-row'):
        layer(torch.zeros(shape))
    with pytest.raises(RuntimeError, match='one-patch-per-row'):
        layer(torch.zeros((1, 1536), dtype=torch.int64))
    adapter.verify()


@pytest.mark.parametrize('mutation', [
    'bound_forward', 'wrong_self', 'forward_code', 'native_class_forward',
    'conv_forward', 'helper_binding', 'helper_code', 'linear_operator',
    'stride', 'weight_replacement', 'weight_inplace', 'bias_inplace', 'layer_hook',
    'native_code_copy', 'helper_constant', 'closure',
])
def test_later_mutations_fail_closed_before_projection(cpu_runtime, monkeypatch, mutation):
    torch, qwen = cpu_runtime
    layer = qwen.Qwen3_5VisionPatchEmbed().eval()
    adapter = patch.apply_patch_projection(layer, qwen, 'linear')

    def unexpected(*args):
        raise AssertionError('unexpected implementation executed')

    if mutation == 'bound_forward':
        layer.forward = MethodType(qwen.Qwen3_5VisionPatchEmbed.forward, layer)
    elif mutation == 'wrong_self':
        layer.forward = MethodType(layer.forward.__func__, qwen.Qwen3_5VisionPatchEmbed())
    elif mutation == 'forward_code':
        code = layer.forward.__func__.__code__
        monkeypatch.setattr(layer.forward.__func__, '__code__', code.replace(
            co_consts=(*code.co_consts, 'unexpected code mutation')))
    elif mutation == 'native_class_forward':
        monkeypatch.setattr(qwen.Qwen3_5VisionPatchEmbed, 'forward', unexpected)
    elif mutation == 'conv_forward':
        layer.proj.forward = unexpected
    elif mutation == 'layer_hook':
        layer.register_forward_hook(lambda *_: None)
    elif mutation == 'native_code_copy':
        function = qwen.Qwen3_5VisionPatchEmbed.forward
        monkeypatch.setattr(function, '__code__', function.__code__.replace())
    elif mutation == 'helper_constant':
        monkeypatch.setattr(patch, 'PATCH_WIDTH', 3072)
    elif mutation == 'closure':
        function = layer.forward.__func__
        index = function.__code__.co_freevars.index('torch')
        function.__closure__[index].cell_contents = SimpleNamespace()
    elif mutation == 'helper_binding':
        monkeypatch.setattr(patch, '_linear_projection', unexpected)
    elif mutation == 'helper_code':
        monkeypatch.setattr(patch._linear_projection, '__code__', unexpected.__code__)
    elif mutation == 'linear_operator':
        monkeypatch.setattr(torch.nn.functional, 'linear', unexpected)
    elif mutation == 'stride':
        layer.proj.stride = (1, 16, 16)
    elif mutation == 'weight_replacement':
        layer.proj.weight = torch.nn.Parameter(layer.proj.weight.detach().clone())
    elif mutation in {'weight_inplace', 'bias_inplace'}:
        with torch.no_grad():
            getattr(layer.proj, mutation.split('_')[0]).add_(1)
    with pytest.raises(RuntimeError):
        adapter.verify()
    if mutation not in {'bound_forward', 'wrong_self', 'forward_code'}:
        with pytest.raises(RuntimeError):
            layer(torch.zeros(1, 1536))


def test_raw_weight_data_change_is_caught_by_outer_byte_guard(cpu_runtime):
    torch, qwen = cpu_runtime
    layer = qwen.Qwen3_5VisionPatchEmbed().eval()
    adapter = patch.apply_patch_projection(layer, qwen, 'linear')
    version = layer.proj.weight._version
    layer.proj.weight.data.add_(1)
    assert layer.proj.weight._version == version
    with pytest.raises(RuntimeError, match='weight bytes changed'):
        adapter.verify()


def test_native_library_source_change_fails_closed(cpu_runtime):
    _, qwen = cpu_runtime
    adapter = patch.apply_patch_projection(qwen.Qwen3_5VisionPatchEmbed(), qwen, 'linear')
    path = Path(qwen.__file__)
    path.write_text(path.read_text() + '\n# unexpected model-library modification\n')
    with pytest.raises(RuntimeError, match='source file changed'):
        adapter.identity()


def test_helper_source_change_changes_identity_and_invalidates_existing_adapter(
        cpu_runtime, tmp_path):
    _, qwen = cpu_runtime
    path = tmp_path / 'patch_projection.py'
    source = ADAPTER_PATH.read_text()
    path.write_text(source)
    first_module = load_module('projection_copy', path)
    first_layer = qwen.Qwen3_5VisionPatchEmbed()
    first = first_module.apply_patch_projection(first_layer, qwen, 'linear')
    before = first.identity()
    changed = source.replace('layer.proj.weight.flatten(1), layer.proj.bias)',
                             'layer.proj.weight.reshape(layer.embed_dim, -1), layer.proj.bias)')
    assert changed != source
    path.write_text(changed)
    with pytest.raises(RuntimeError, match='source file changed'):
        first.identity()
    second_module = load_module('projection_copy', path)
    second_layer = qwen.Qwen3_5VisionPatchEmbed()
    second_layer.load_state_dict(first_layer.state_dict())
    after = second_module.apply_patch_projection(second_layer, qwen, 'linear').identity()
    assert before['original_forward'] == after['original_forward']
    assert before['weights'] == after['weights']
    assert before['adapter_file_sha256'] != after['adapter_file_sha256']
    assert (before['helpers']['_linear_projection']['code_sha256']
            != after['helpers']['_linear_projection']['code_sha256'])
    assert serve.digest(before) != serve.digest(after)
    assert str(tmp_path) not in json.dumps(after)


def test_service_checks_unique_patch_layer_before_mutating_and_detects_replacement(cpu_runtime):
    _, qwen = cpu_runtime
    first, second = qwen.Qwen3_5VisionPatchEmbed(), qwen.Qwen3_5VisionPatchEmbed()
    rows = [('model.visual.patch_embed', first), ('other.visual.patch_embed', second)]
    model = SimpleNamespace(named_modules=lambda: iter(rows))
    with pytest.raises(RuntimeError, match='exactly one official'):
        serve.configure_patch_projection(model, qwen, 'linear')
    assert 'forward' not in vars(first) and 'forward' not in vars(second)
    rows.pop()
    configured = serve.configure_patch_projection(model, qwen, 'linear')
    identity = serve.patch_projection_identity(configured)
    assert identity['layer'] == 'model.visual.patch_embed'
    serve.verify_patch_projection(model, qwen, configured)
    rows[0] = (rows[0][0], second)
    with pytest.raises(RuntimeError, match='Model vision patch embed changed'):
        serve.verify_patch_projection(model, qwen, configured)


@pytest.mark.parametrize('invalid_startup', [False, True])
def test_service_linear_startup_schema_and_request_health_guards(
        cpu_runtime, tmp_path, monkeypatch, invalid_startup):
    torch, qwen = cpu_runtime
    # BF16 here tests metadata/byte hashing only, never numerical equivalence.
    layer = qwen.Qwen3_5VisionPatchEmbed(out_channels=1024, dtype=torch.bfloat16).eval()
    if invalid_startup:
        layer.proj.stride = (1, 16, 16)
    model = SimpleNamespace(named_modules=lambda: iter([('model.visual.patch_embed', layer)]))
    received = []

    def predict(request):
        received.append(request)
        return {'answers': {'field': {'probabilities': {'yes': 0.8, 'no': 0.2}}}}

    engine = SimpleNamespace(backend=SimpleNamespace(model=model, encode=lambda _: None),
                             predict=predict)
    fixture = ModuleType('_intern_patch_test_engine')
    fixture.engine = engine
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    inference = checkpoint / 'inference.py'
    inference.write_text('from _intern_patch_test_engine import engine\n'
        'DEFAULT_TEMPERATURE = 1.99241824\n'
        'def DecisionEngine(**kwargs): return engine\n'
        'def scale_result(result, temperature): return result\n'
        'def argmax(probabilities): return max(probabilities, key=probabilities.get)\n')
    (checkpoint / 'download_manifest.json').write_text(json.dumps({
        'revision': 'test-revision',
        'files': {'inference.py': {'sha256': serve.file_sha(inference)}}}))
    (checkpoint / 'preprocessor_config.json').write_text('{}')
    for name in ('transformers', 'transformers.models', 'transformers.models.qwen3_5'):
        module = ModuleType(name)
        module.__path__ = []
        module.__version__ = 'test-transformers'
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['transformers.models.qwen3_5'].modeling_qwen3_5 = qwen
    monkeypatch.setattr(torch, 'set_num_threads', lambda _: None)
    for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'CPU metadata fixture')
    monkeypatch.setattr(serve, 'kernel_runtime_identity', lambda *a, **k: {'mode': 'fixture'})
    handlers = []

    class NoSocketServer:
        def __init__(self, address, handler):
            handlers.append(handler)

        def serve_forever(self):
            pass

    monkeypatch.setattr(serve, 'ThreadingHTTPServer', NoSocketServer)
    startup_path = tmp_path / 'startup.json'
    monkeypatch.setattr(sys, 'argv', ['serve.py', '--model-dir', str(checkpoint),
        '--revision', 'test-revision', '--startup', str(startup_path),
        '--temp-root', str(tmp_path / 'requests'), '--patch-projection', 'linear'])
    if invalid_startup:
        with pytest.raises(RuntimeError, match='geometry'):
            serve.main()
        assert not startup_path.exists() and handlers == []
        assert 'forward' not in vars(layer)
        return
    serve.main()
    startup = json.loads(startup_path.read_text())
    assert startup['identity']['schema'] == 'intern-decision-service/3'
    identity = startup['identity']['runtime']['patch_projection']
    assert identity['requested_projection'] == identity['observed_projection'] == 'linear'
    assert identity['constraints']['weight_shape'] == [1024, 3, 2, 16, 16]
    assert identity['constraints']['dtype'] == 'torch.bfloat16'
    assert startup['identity_sha256'] == serve.digest(startup['identity'])
    responses = []
    handler = object.__new__(handlers[0])
    handler.send = lambda status, body: responses.append((status, body))
    handler.path = '/health'
    handler.do_GET()
    assert responses[-1][0] == 200
    request = {'state': 'unchanged', 'questions': {}, 'images': []}
    body = json.dumps({'request': request, 'request_sha256': serve.digest(request)}).encode()
    handler.path = '/decisions'
    handler.headers = {'Content-Length': str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.do_POST()
    assert responses[-1][0] == 200
    assert received == [request]
    assert responses[-1][1]['identity_sha256'] == startup['identity_sha256']
    layer.forward = MethodType(qwen.Qwen3_5VisionPatchEmbed.forward, layer)
    handler.path = '/health'
    handler.do_GET()
    assert responses[-1][0] == 503
    handler.path = '/decisions'
    handler.rfile = io.BytesIO(body)
    handler.do_POST()
    assert responses[-1][0] == 422
    assert received == [request]
