import json
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from experiments.server_readiness import child_path, check_files, code_hash, digest


def test_model_manifest_rejects_escape_and_empty_relative(tmp_path):
    for name in ['../outside', str(tmp_path.parent / 'outside'), '.', 'sub/../../outside']:
        with pytest.raises(ValueError):
            child_path(tmp_path, name)
    assert child_path(tmp_path, 'weights/model.safetensors') == tmp_path / 'weights/model.safetensors'


def test_model_manifest_requires_exact_size_and_hash(tmp_path):
    target = tmp_path / 'weights'
    target.write_bytes(b'abc')
    manifest = {'weights': {'size': 3, 'sha256': digest(target)}}
    check_files(tmp_path, manifest)
    target.write_bytes(b'abd')
    with pytest.raises(ValueError, match='Checksum'):
        check_files(tmp_path, manifest)
    target.write_bytes(b'ab')
    with pytest.raises(ValueError, match='truncated'):
        check_files(tmp_path, manifest)
    with pytest.raises(ValueError, match='No recorded'):
        check_files(tmp_path, {})


def test_server_source_hash_includes_launcher_and_config(tmp_path):
    (tmp_path / 'qihc').mkdir()
    (tmp_path / 'qihc/model.py').write_text('value=1', encoding='utf-8')
    (tmp_path / 'scripts/server').mkdir(parents=True)
    script = tmp_path / 'scripts/server/start.sh'
    script.write_text('echo first', encoding='utf-8')
    before = code_hash(tmp_path)
    script.write_text('echo second', encoding='utf-8')
    assert before != code_hash(tmp_path)


def test_safetensors_header_and_index_agree(tmp_path):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    from experiments.server_readiness import check_model_headers
    save_file({'weight': torch.ones(2, 3)}, str(tmp_path / 'shard.safetensors'))
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {'weight': 'shard.safetensors'}}), encoding='utf-8')
    assert check_model_headers(tmp_path) == {'shards': 1, 'tensors': 1}
    index.write_text(json.dumps({'weight_map': {'missing': 'shard.safetensors'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='Index/header'):
        check_model_headers(tmp_path)


def test_offline_launcher_passes_local_data_and_does_not_install():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'scripts/server/02_run_offline_gpu.sh').read_text(encoding='utf-8')
    assert '--data "$MEDIUM" "$HARD"' in source
    assert 'HF_HUB_OFFLINE=1' in source
    assert 'pip install' not in source
    assert 'CUDA_VISIBLE_DEVICES=' not in source
    assert '--deepspeed-config' in source
    assert 'distributed-smoke' in source


def test_launcher_source_uses_lf_and_shared_user_directory():
    root = Path(__file__).resolve().parents[1]
    for name in ['01_prepare_online.sh', '02_run_offline_gpu.sh']:
        raw = (root / 'scripts/server' / name).read_bytes()
        assert b'\r\n' not in raw
        assert b'/inspire/hdd/global_user/czxs25210022/why' in raw


def test_two_h100_defaults_and_grpo_batch_forwarding():
    root = Path(__file__).resolve().parents[1]
    online = (root / 'scripts/server/01_prepare_online.sh').read_text(encoding='utf-8')
    offline = (root / 'scripts/server/02_run_offline_gpu.sh').read_text(encoding='utf-8')
    assert '${QIHC_MODEL_ID:-Qwen/Qwen3.5-9B}' in online
    assert '${QIHC_NPROC:-2}' in offline
    assert '${QIHC_PROFILE:-formal}' in offline
    assert '${QIHC_GRPO_GENERATION_BATCH_SIZE:-4}' in offline
    assert '--grpo-generation-batch-size "$GRPO_BATCH"' in offline


def test_grpo_generation_memory_budget_independent_of_accumulation():
    from experiments.train_s2e_lora import resolve_generation_batch
    assert resolve_generation_batch(4, 2, 4, 8) == 4
    assert resolve_generation_batch(0, 2, 4, 8) == 16
    assert resolve_generation_batch(0, 3, 4, 1) == 12
    with pytest.raises(ValueError, match='divisible'):
        resolve_generation_batch(4, 8, 4, 8)


@pytest.mark.parametrize('name', ['01_prepare_online.sh', '02_run_offline_gpu.sh'])
def test_conda_hooks_allow_unset_variables_and_restore_strict_mode(name):
    root = Path(__file__).resolve().parents[1]
    bash = shutil.which('bash')
    if bash is None and Path('C:/Program Files/Git/bin/bash.exe').is_file():
        bash = 'C:/Program Files/Git/bin/bash.exe'
    if bash is None:
        pytest.skip('Bash is required to execute activation-hook regression')
    source = (root / 'scripts/server' / name).read_text(encoding='utf-8')
    blocks = re.findall(r'^set \+u\n.*?^set -u$', source, flags=re.MULTILINE | re.DOTALL)
    assert blocks and 'source ' in '\n'.join(blocks) and 'conda activate' in '\n'.join(blocks)
    prelude = '''set -Eeuo pipefail
unset SYS_SYSROOT
CONDA_ROOT=/unused
BASE=/unused
ENV_PREFIX=/unused
CONDA_ACTION=install
TOOLKIT=12.8
source() { printf '%s' "$SYS_SYSROOT" >/dev/null; }
conda() { printf '%s' "$SYS_SYSROOT" >/dev/null; }
'''
    check = '\ncase "$-" in *u*) ;; *) exit 44;; esac\nprintf "strict_restored"\n'
    passed = subprocess.run([bash, '-c', prelude + '\n'.join(blocks) + check], capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert 'strict_restored' in passed.stdout
    # Errexit must still stop a genuinely failing activation hook.
    failed = subprocess.run([bash, '-c', prelude + '\nconda() { return 23; }\n' + '\n'.join(blocks) + check], capture_output=True, text=True)
    assert failed.returncode == 23


def test_hf_cli_download_explicitly_lists_all_weight_shards(tmp_path):
    from experiments.server_readiness import hf_download_command
    files = ['config.json', 'model.safetensors.index.json',
             *[f'model.safetensors-{n:05d}-of-00004.safetensors' for n in range(1, 5)]]
    command = hf_download_command('hf', 'Qwen/Qwen3.5-9B', 'fixed-sha', tmp_path, files)
    assert '--include' not in command
    assert command[command.index('--') + 1:] == sorted(files)
    assert command[command.index('--revision') + 1] == 'fixed-sha'
    assert not any('*' in value for value in command)


def test_download_model_cli_then_verifies_complete_manifest(tmp_path, monkeypatch):
    import experiments.server_readiness as readiness
    payloads = {'config.json': b'{}', 'model.safetensors-00001-of-00002.safetensors': b'abc',
                'model.safetensors-00002-of-00002.safetensors': b'def'}
    siblings = [SimpleNamespace(rfilename=name, size=len(payload), lfs=None) for name, payload in payloads.items()]
    info = SimpleNamespace(sha='immutable-revision', siblings=siblings)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: info)))
    monkeypatch.setattr(readiness.shutil, 'which', lambda name: 'hf')
    monkeypatch.setattr(readiness.shutil, 'disk_usage', lambda directory: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(readiness, 'check_model_headers', lambda directory: {'shards': 2, 'tensors': 2})
    calls = []

    def downloaded(command, check):
        calls.append(command)
        directory = Path(command[command.index('--local-dir') + 1])
        for name in command[command.index('--') + 1:]:
            (directory / name).write_bytes(payloads[name])

    monkeypatch.setattr(readiness.subprocess, 'run', downloaded)
    args = SimpleNamespace(repo='Qwen/test', revision='main', model_path=tmp_path / 'model', output=tmp_path / 'manifest.json')
    readiness.download_model(args)
    saved = json.loads(args.output.read_text(encoding='utf-8'))
    assert saved['revision'] == 'immutable-revision'
    assert set(saved['files']) == set(payloads)
    assert len(calls) == 1
    readiness.check_files(args.model_path, saved['files'])


def test_server_requirements_include_hf_cli_dependency():
    root = Path(__file__).resolve().parents[1]
    assert 'click==8.3.0' in (root / 'configs/server-requirements.lock.txt').read_text(encoding='utf-8')
