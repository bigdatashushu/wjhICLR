"""Model-free checks for the standalone Intern service's kernel identity."""
import csv
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SERVICE_PATH = Path(__file__).resolve().parents[2] / 'scripts/intern_decision/serve.py'
SPEC = importlib.util.spec_from_file_location('intern_service_identity_tests', SERVICE_PATH)
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)


def fake_distribution(root, name, payloads):
    """Use real importlib metadata, including deliberately stale RECORD hashes."""
    info = root / f'{name.replace("-", "_")}-0.5.2.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(f'Name: {name}\nVersion: 0.5.2\n')
    inventory = io.StringIO()
    writer = csv.writer(inventory)
    for relative, content in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        writer.writerow([relative, 'sha256=deliberately-stale', len(content)])
    for name in ('METADATA', 'RECORD', 'INSTALLER', 'direct_url.json'):
        writer.writerow([f'{info.name}/{name}', '', ''])
    (info / 'RECORD').write_text(inventory.getvalue())
    (info / 'INSTALLER').write_text('pip\n')
    (info / 'direct_url.json').write_text(json.dumps({'url': str(root)}))
    return importlib.metadata.Distribution.at(info)


def absent_distribution(name):
    raise importlib.metadata.PackageNotFoundError(name)


def load_fake_module(root, monkeypatch, name, source):
    path = root / (name.replace('.', '/') + '.py')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    module = ModuleType(name)
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    qwen = load_fake_module(tmp_path, monkeypatch,
        'transformers.models.qwen3_5.modeling_qwen3_5', '''
def torch_chunk_gated_delta_rule(): raise AssertionError("kernel executed")
def torch_recurrent_gated_delta_rule(): raise AssertionError("kernel executed")
def torch_causal_conv1d_update(): raise AssertionError("kernel executed")
class TorchConv:
    def forward(self): raise AssertionError("kernel executed")
class Qwen3_5RMSNormGated:
    def forward(self): raise AssertionError("kernel executed")
class Qwen3_5GatedDeltaNet:
    def forward(self): raise AssertionError("model executed")
''')
    qwen.nn = SimpleNamespace(Conv1d=qwen.TorchConv)
    fla = load_fake_module(tmp_path, monkeypatch, 'fla.ops.gated_delta_rule', '''
def chunk_gated_delta_rule(): raise AssertionError("kernel executed")
def fused_recurrent_gated_delta_rule(): raise AssertionError("kernel executed")
class FusedRMSNormGated:
    def forward(self): raise AssertionError("kernel executed")
''')
    causal = load_fake_module(tmp_path, monkeypatch, 'causal_conv1d.causal_conv1d_interface', '''
def causal_conv1d_fn(): raise AssertionError("kernel executed")
def causal_conv1d_update(): raise AssertionError("kernel executed")
''')

    def build(mode, *, layers=2):
        qwen.chunk_gated_delta_rule = fla.chunk_gated_delta_rule if mode != 'torch' else None
        qwen.fused_recurrent_gated_delta_rule = (
            fla.fused_recurrent_gated_delta_rule if mode != 'torch' else None)
        qwen.FusedRMSNormGated = fla.FusedRMSNormGated if mode != 'torch' else None
        qwen.causal_conv1d_fn = causal.causal_conv1d_fn if mode == 'full' else None
        qwen.causal_conv1d_update = causal.causal_conv1d_update if mode == 'full' else None
        qwen.is_fast_path_available = False  # Deliberately misleading aggregate warning.
        rows = []
        for index in range(layers):
            layer = qwen.Qwen3_5GatedDeltaNet()
            layer.chunk_gated_delta_rule = (
                qwen.chunk_gated_delta_rule or qwen.torch_chunk_gated_delta_rule)
            layer.recurrent_gated_delta_rule = (
                qwen.fused_recurrent_gated_delta_rule or qwen.torch_recurrent_gated_delta_rule)
            layer.causal_conv1d_fn = qwen.causal_conv1d_fn
            layer.causal_conv1d_update = (
                qwen.causal_conv1d_update or qwen.torch_causal_conv1d_update)
            layer.norm = (qwen.FusedRMSNormGated or qwen.Qwen3_5RMSNormGated)()
            layer.conv1d = qwen.TorchConv()
            rows.append((f'language.layers.{index}.linear_attn', layer))
        model = SimpleNamespace(named_modules=lambda: iter(rows))
        return model, qwen, rows

    return build


def test_distributions_are_explicitly_absent_and_no_optional_imports(fake_runtime):
    model, qwen, _ = fake_runtime('torch')
    identity = serve.kernel_runtime_identity(model, qwen, distribution=absent_distribution)
    assert identity['schema'] == 'intern-kernel-runtime/1'
    assert identity['requested_mode'] == 'auto'
    assert identity['mode'] == 'torch'
    assert set(identity['distributions']) == {
        'fla-core', 'flash-linear-attention', 'causal-conv1d', 'triton', 'einops', 'ninja'}
    for row in identity['distributions'].values():
        assert row['present'] is False
        assert row['version'] is None
        assert row['sha256'] is None
    assert all(row['present'] is False for row in identity['qwen3_5']['bindings'].values())


@pytest.mark.parametrize('relative', [
    'fla/ops/kernel.py',
    'causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so',
    'triton/_C/libtriton.so',
    'ninja/data/bin/ninja',
])
def test_actual_payload_bytes_not_versions_or_record_determine_identity(tmp_path, relative):
    dist = fake_distribution(tmp_path, 'test-kernels', {relative: b'original kernel'})
    before = serve.distribution_identity('test-kernels', distribution=lambda _: dist)
    (tmp_path / relative).write_bytes(b'modified kernel')
    after = serve.distribution_identity('test-kernels', distribution=lambda _: dist)
    assert before['version'] == after['version'] == '0.5.2'
    assert before['files'][relative] != after['files'][relative]
    assert before['sha256'] != after['sha256']


def test_relocation_inventory_order_and_install_noise_do_not_change_identity(tmp_path):
    payloads = {'fla/kernel.py': b'kernel', 'native.so': b'ELF',
                'fla/__pycache__/kernel.cpython-312.pyc': b'old cache'}
    first = fake_distribution(tmp_path / 'one', 'fla-core', payloads)
    second = fake_distribution(tmp_path / 'two', 'fla-core', dict(reversed(list(payloads.items()))))
    (tmp_path / 'two/fla/__pycache__/kernel.cpython-312.pyc').write_bytes(b'new cache')
    os.utime(tmp_path / 'two/fla/kernel.py', (42, 42))
    a = serve.distribution_identity('fla-core', distribution=lambda _: first)
    b = serve.distribution_identity('fla-core', distribution=lambda _: second)
    assert a == b
    assert set(a['files']) == {'fla/kernel.py', 'native.so'}
    assert str(tmp_path) not in json.dumps(a)


def test_installed_package_is_distinct_from_absence(tmp_path):
    dist = fake_distribution(tmp_path, 'einops', {'einops/__init__.py': b''})
    present = serve.distribution_identity('einops', distribution=lambda _: dist)
    absent = serve.distribution_identity('einops', distribution=absent_distribution)
    assert present['present'] is True
    assert serve.digest(present) != serve.digest(absent)


def test_missing_native_payload_refuses_incomplete_identity(tmp_path):
    dist = fake_distribution(tmp_path, 'causal-conv1d', {'causal_conv1d_cuda.so': b'ELF'})
    (tmp_path / 'causal_conv1d_cuda.so').unlink()
    with pytest.raises(RuntimeError, match='Cannot hash kernel distribution file'):
        serve.distribution_identity('causal-conv1d', distribution=lambda _: dist)


@pytest.mark.parametrize('inventory', [None, []])
def test_unverifiable_distribution_inventory_fails_closed(inventory):
    dist = SimpleNamespace(files=inventory, version='0.5.2')
    with pytest.raises(RuntimeError, match='No file inventory'):
        serve.distribution_identity('fla-core', distribution=lambda _: dist)


def test_metadata_only_installation_is_not_a_complete_kernel_identity(tmp_path):
    dist = fake_distribution(tmp_path, 'fla-core', {})
    with pytest.raises(RuntimeError, match='No runtime payload'):
        serve.distribution_identity('fla-core', distribution=lambda _: dist)


@pytest.mark.parametrize('actual', ['torch', 'fla', 'full'])
@pytest.mark.parametrize('requested', ['torch', 'fla', 'full'])
def test_explicit_contract_requires_exact_active_attention_and_convolution(
        fake_runtime, actual, requested):
    model, qwen, _ = fake_runtime(actual)
    if actual != requested:
        with pytest.raises(RuntimeError, match='Kernel contract mismatch'):
            serve.kernel_runtime_identity(model, qwen, kernel_mode=requested,
                distribution=absent_distribution)
        return
    identity = serve.kernel_runtime_identity(model, qwen, kernel_mode=requested,
        distribution=absent_distribution)
    assert identity['mode'] == actual
    assert identity['requested_mode'] == requested
    assert len(identity['qwen3_5']['implementations']) == 1
    row = identity['qwen3_5']['implementations'][0]
    assert len(row['layers']) == 2
    assert row['operators']['attention_prefill']['backend'] == ('torch' if actual == 'torch' else 'fla')
    assert row['operators']['convolution_prefill']['backend'] == (
        'causal-conv1d' if actual == 'full' else 'torch')


def test_loaded_bindings_not_package_presence_or_warning_define_fla_only(fake_runtime, tmp_path):
    model, qwen, _ = fake_runtime('fla')
    dist = fake_distribution(tmp_path / 'installed', 'causal-conv1d',
        {'causal_conv1d_cuda.so': b'not-loadable-on-this-host'})

    def lookup(name):
        return dist if name == 'causal-conv1d' else absent_distribution(name)

    identity = serve.kernel_runtime_identity(model, qwen, kernel_mode='fla', distribution=lookup)
    assert qwen.is_fast_path_available is False
    assert identity['distributions']['causal-conv1d']['present'] is True
    assert identity['qwen3_5']['bindings']['causal_conv1d_fn']['present'] is False
    assert identity['mode'] == 'fla'


def test_every_layer_and_both_prefill_and_update_are_checked(fake_runtime):
    model, qwen, rows = fake_runtime('fla')
    rows[-1][1].recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
    with pytest.raises(RuntimeError, match='language.layers.1.*requested fla, observed mixed'):
        serve.kernel_runtime_identity(model, qwen, kernel_mode='fla',
            distribution=absent_distribution)
    identity = serve.kernel_runtime_identity(model, qwen, distribution=absent_distribution)
    assert identity['mode'] == 'mixed'
    assert len(identity['qwen3_5']['implementations']) == 2


def test_actual_operator_override_is_not_mistaken_for_available_fla(fake_runtime):
    model, qwen, rows = fake_runtime('fla')
    for _, layer in rows:
        layer.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
        layer.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
        layer.norm = qwen.Qwen3_5RMSNormGated()
    identity = serve.kernel_runtime_identity(model, qwen, kernel_mode='torch',
        distribution=absent_distribution)
    assert identity['qwen3_5']['bindings']['chunk_gated_delta_rule']['present'] is True
    assert identity['mode'] == 'torch'


@pytest.mark.parametrize('target', ['layer', 'convolution', 'normalization'])
def test_overridden_forward_cannot_satisfy_a_kernel_contract(fake_runtime, target):
    model, qwen, rows = fake_runtime('fla')
    layer = rows[-1][1]
    instance = {'layer': layer, 'convolution': layer.conv1d, 'normalization': layer.norm}[target]
    instance.forward = qwen.torch_chunk_gated_delta_rule
    with pytest.raises(RuntimeError, match='Kernel contract mismatch'):
        serve.kernel_runtime_identity(model, qwen, kernel_mode='fla',
            distribution=absent_distribution)


def test_source_changes_and_wrappers_affect_operator_identity(tmp_path, monkeypatch):
    original = load_fake_module(tmp_path, monkeypatch, 'fake_kernel', '''
def kernel(): raise AssertionError("kernel executed")
''')
    wrapper = load_fake_module(tmp_path, monkeypatch, 'fake_wrapper', '''
def decorated(): raise AssertionError("wrapper executed")
''')
    wrapper.decorated.__wrapped__ = original.kernel
    before = serve.operator_identity(wrapper.decorated)
    assert len(before['sources']) == 2
    assert before != serve.operator_identity(original.kernel)
    Path(original.__file__).write_text('# changed underlying kernel\n')
    after = serve.operator_identity(wrapper.decorated)
    assert before['sources'][0] == after['sources'][0]
    assert before['sources'][1] != after['sources'][1]
    assert str(tmp_path) not in json.dumps(after)


def test_active_source_change_affects_runtime_digest_and_order_does_not(fake_runtime):
    model, qwen, rows = fake_runtime('fla')
    a = serve.kernel_runtime_identity(model, qwen, distribution=absent_distribution)
    rows.reverse()
    b = serve.kernel_runtime_identity(model, qwen, distribution=absent_distribution)
    assert a == b
    source = Path(qwen.__file__)
    source.write_text(source.read_text() + '\n# modified fallback source\n')
    c = serve.kernel_runtime_identity(model, qwen, distribution=absent_distribution)
    assert serve.digest(a) != serve.digest(c)


def test_unrecognized_model_and_contract_refuse_identity(fake_runtime):
    model, qwen, _ = fake_runtime('torch', layers=0)
    with pytest.raises(RuntimeError, match='no Qwen3.5'):
        serve.kernel_runtime_identity(model, qwen, distribution=absent_distribution)
    with pytest.raises(ValueError, match='Unknown kernel mode'):
        serve.kernel_runtime_identity(model, qwen, kernel_mode='all-fast',
            distribution=absent_distribution)


def test_cli_help_is_model_free_and_explains_legacy_default(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['serve.py', '--help'])
    with pytest.raises(SystemExit) as exc:
        serve.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert '--kernel-mode {torch,fla,full}' in help_text
    assert 'automatic upstream selection' in help_text


@pytest.mark.parametrize('requested', [None, 'fla', 'full'])
def test_startup_publishes_new_schema_only_after_contract_passes(
        tmp_path, monkeypatch, fake_runtime, requested):
    model, qwen, _ = fake_runtime('fla')
    engine = SimpleNamespace(backend=SimpleNamespace(model=model, encode=lambda _: None))
    fixture = ModuleType('_intern_test_engine')
    fixture.engine = engine
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    monkeypatch.setitem(sys.modules, 'intern_native_inference', ModuleType('intern_native_inference'))
    model_dir = tmp_path / 'checkpoint'
    model_dir.mkdir()
    inference = model_dir / 'inference.py'
    inference.write_text('from _intern_test_engine import engine\n'
        'DEFAULT_TEMPERATURE = 1.99241824\n'
        'def DecisionEngine(**kwargs): return engine\n')
    (model_dir / 'download_manifest.json').write_text(json.dumps({
        'revision': 'test-revision',
        'files': {'inference.py': {'sha256': serve.file_sha(inference)}}}))
    (model_dir / 'preprocessor_config.json').write_text('{}')
    torch = ModuleType('torch')
    torch.__version__ = 'test-torch'
    torch.version = SimpleNamespace(cuda='test-cuda')
    torch.set_num_threads = lambda _: None
    torch.cuda = SimpleNamespace(get_device_name=lambda _: 'test-device',
        memory_allocated=lambda: 0, memory_reserved=lambda: 0)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    for name in ('transformers', 'transformers.models', 'transformers.models.qwen3_5'):
        module = ModuleType(name)
        module.__path__ = []
        module.__version__ = '5.14.1'
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules['transformers.models.qwen3_5'].modeling_qwen3_5 = qwen
    real_identity = serve.kernel_runtime_identity
    monkeypatch.setattr(serve, 'kernel_runtime_identity',
        lambda *args, **kwargs: real_identity(*args, **kwargs, distribution=absent_distribution))
    servers = []

    class NoSocketServer:
        def __init__(self, address, handler):
            servers.append(address)

        def serve_forever(self):
            pass

    monkeypatch.setattr(serve, 'ThreadingHTTPServer', NoSocketServer)
    startup_path = tmp_path / 'startup.json'
    argv = ['serve.py', '--model-dir', str(model_dir), '--revision', 'test-revision',
            '--startup', str(startup_path), '--temp-root', str(tmp_path / 'requests')]
    if requested is not None:
        argv += ['--kernel-mode', requested]
    monkeypatch.setattr(sys, 'argv', argv)
    if requested == 'full':
        with pytest.raises(RuntimeError, match='Kernel contract mismatch'):
            serve.main()
        assert not startup_path.exists()
        assert servers == []
        return
    serve.main()
    startup = json.loads(startup_path.read_text())
    assert startup['identity']['schema'] == 'intern-decision-service/2'
    assert startup['identity']['runtime']['kernels']['mode'] == 'fla'
    assert startup['identity']['runtime']['kernels']['requested_mode'] == (requested or 'auto')
    assert startup['identity_sha256'] == serve.digest(startup['identity'])
    assert servers == [('127.0.0.1', 18445)]
