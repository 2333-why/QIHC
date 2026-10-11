#!/usr/bin/env python3
"""Prepare/verify shared offline inputs and exercise real single-node ZeRO-3."""
from __future__ import annotations

import argparse
from datetime import timedelta
import fnmatch
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PATTERNS = ['*.safetensors', '*.json', '*.txt', '*.model', '*.jinja', '*.tiktoken']


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def child_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if path == directory.resolve() or not path.is_relative_to(directory.resolve()):
        raise ValueError(f'Unsafe file name in model manifest: {relative}')
    return path


def code_hash(directory: Path) -> str:
    paths = set(directory.glob('pyproject.toml'))
    for folder, pattern in [('qihc', '*.py'), ('experiments', '*.py'), ('tests', '*.py'), ('configs', '*'), ('scripts/server', '*.sh')]:
        paths.update(p for p in (directory / folder).rglob(pattern) if p.is_file())
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.relative_to(directory).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def package_versions() -> dict[str, str]:
    return {str(d.metadata['Name']).lower().replace('_', '-'): d.version for d in distributions() if d.metadata['Name']}


def check_model_headers(directory: Path) -> dict:
    """Read safetensors headers only, never materialize a 35B CPU model."""
    from safetensors import safe_open
    index = directory / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text(encoding='utf-8'))['weight_map']
        if not mapping:
            raise ValueError('Empty model weight index')
        by_shard = {}
        for key, name in mapping.items():
            by_shard.setdefault(name, set()).add(key)
    else:
        if not (directory / 'model.safetensors').is_file():
            raise ValueError('Require safetensors weights or complete safetensors shard index')
        by_shard = {'model.safetensors': None}
    count = 0
    for name, expected in sorted(by_shard.items()):
        path = child_path(directory, name)
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            keys = set(handle.keys())
            if expected is not None and keys != expected:
                raise ValueError(f'Index/header mismatch in {name}')
            count += len(keys)
    return {'shards': len(by_shard), 'tensors': count}


def check_files(directory: Path, files: dict) -> None:
    if not files:
        raise ValueError('No recorded model files')
    for name, metadata in sorted(files.items()):
        path = child_path(directory, name)
        if not path.is_file() or path.stat().st_size != metadata['size']:
            raise ValueError(f'Missing/truncated file: {path}')
        print(f'VERIFY {name}', flush=True)
        if digest(path) != metadata['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')


def hf_download_command(hf: str, repo: str, revision: str, directory: Path, filenames: list[str]) -> list[str]:
    """Use explicit files; Typer's list options consume one value per occurrence."""
    if not filenames:
        raise ValueError('No selected model files to download')
    for name in filenames:
        child_path(directory, name)
    return [hf, 'download', repo, '--revision', revision, '--local-dir', str(directory),
            '--max-workers', '4', '--', *sorted(set(filenames))]


def download_model(args) -> None:
    from huggingface_hub import HfApi
    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    if not info.sha:
        raise ValueError('Hub returned no immutable revision')
    directory = args.model_path.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    selected = [f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in PATTERNS)]
    if not any(f.rfilename.endswith('.safetensors') for f in selected):
        raise ValueError('Model repository contains no selected safetensors')
    missing_bytes = sum(int(f.size or 0) for f in selected
                        if not child_path(directory, f.rfilename).is_file()
                        or (f.size is not None and child_path(directory, f.rfilename).stat().st_size != f.size))
    if shutil.disk_usage(directory).free < missing_bytes + 10 * 1024**3:
        raise ValueError('Not enough filesystem free space for model plus 10 GiB reserve; check quota too')
    hf = shutil.which('hf')
    if not hf:
        raise RuntimeError('hf CLI missing from active environment')
    print(json.dumps({'repo': args.repo, 'revision': info.sha, 'files': len(selected),
                      'weight_shards': sum(f.rfilename.endswith('.safetensors') for f in selected),
                      'download_bytes_missing_or_truncated': missing_bytes}), flush=True)
    subprocess.run(hf_download_command(hf, args.repo, info.sha, directory,
                                      [f.rfilename for f in selected]), check=True)
    files = {}
    for item in selected:
        path = child_path(directory, item.rfilename)
        if not path.is_file() or (item.size is not None and path.stat().st_size != item.size):
            raise ValueError(f'Hub file missing/truncated: {item.rfilename}')
        sha = digest(path)
        lfs = item.lfs
        remote_sha = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
        if remote_sha and sha != remote_sha:
            raise ValueError(f'Hub SHA256 mismatch: {item.rfilename}')
        files[item.rfilename] = {'size': path.stat().st_size, 'sha256': sha, 'remote_lfs_sha256': remote_sha}
    headers = check_model_headers(directory)
    atomic_json(args.output, {'repo': args.repo, 'requested_revision': args.revision, 'revision': info.sha,
                             'model_path': str(directory), 'files': files, 'headers': headers})
    print(json.dumps({'model_ready': str(directory), 'revision': info.sha, **headers}))


def seal(args) -> None:
    manifest = json.loads(args.model_manifest.read_text(encoding='utf-8'))
    cpu = json.loads(args.cpu_report.read_text(encoding='utf-8'))
    training = json.loads(args.training_smoke.read_text(encoding='utf-8'))
    suite = json.loads(args.suite_smoke.read_text(encoding='utf-8'))
    if cpu.get('scope') != 'cpu_metadata_and_apis' or cpu.get('status') != 'passed':
        raise ValueError('CPU environment checks did not pass')
    if training.get('status') != 'complete' or not training.get('adapter_inference_exercised'):
        raise ValueError('Actual tiny training/inference did not complete')
    if not suite.get('methods') or any(row['failed_jobs'] for row in suite['methods'].values()):
        raise ValueError('CPU research suite has failed/missing arms')
    inputs = {}
    for path in args.data:
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'Empty benchmark input: {path}')
        if any('known_feasible_routes' in row.get('metadata', {}) for row in rows):
            raise ValueError('Benchmark exports witness routes')
        inputs[str(path.resolve())] = {'sha256': digest(path), 'rows': len(rows)}
    record = {'status': 'cpu_prepared_gpu_checks_pending', 'base': str(args.base.resolve()),
              'env_prefix': str(args.env_prefix.resolve()), 'code_dir': str(args.code_dir.resolve()),
              'source_sha256': code_hash(args.code_dir.resolve()), 'cuda_flavor': args.cuda_flavor,
              'model_path': manifest['model_path'], 'model': manifest, 'data': list(inputs), 'inputs': inputs,
              'packages': package_versions(), 'cpu_report': str(args.cpu_report.resolve()),
              'training_smoke': str(args.training_smoke.resolve()), 'suite_smoke': str(args.suite_smoke.resolve()),
              'deferred': ['CUDA/driver', 'NCCL', 'ZeRO-3', 'full_model_generation', 'GPU_memory']}
    atomic_json(args.output, record)
    print(f'Prepared manifest: {args.output}')


def verify(args) -> None:
    record = json.loads(args.manifest.read_text(encoding='utf-8'))
    if record.get('status') != 'cpu_prepared_gpu_checks_pending':
        raise ValueError('Preparation did not complete')
    if str(args.env_prefix.resolve()) != record['env_prefix'] or str(Path(sys.prefix).resolve()) != record['env_prefix']:
        raise ValueError('Wrong conda environment / CUDA wheel selection')
    if code_hash(args.code_dir.resolve()) != record['source_sha256']:
        raise ValueError('Code/config changed after online verification; rerun script 01')
    if package_versions() != record['packages']:
        raise ValueError('Environment packages changed after preparation; rerun script 01')
    for name, metadata in record['inputs'].items():
        if digest(Path(name)) != metadata['sha256']:
            raise ValueError(f'Benchmark input changed: {name}')
    directory = Path(record['model_path'])
    check_files(directory, record['model']['files'])
    check_model_headers(directory)
    print('Offline model/data/source/environment verification passed')


def distributed_smoke(args) -> None:
    import torch
    import torch.distributed as dist
    rank, local_rank, world = (int(os.environ[x]) for x in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    try:
        value = torch.tensor([rank + 1.0], device=f'cuda:{local_rank}')
        dist.all_reduce(value)
        if value.item() != world * (world + 1) / 2:
            raise ValueError('NCCL all_reduce returned incorrect value')
        import deepspeed
        config = json.loads(args.deepspeed_config.read_text(encoding='utf-8'))
        config.update(train_micro_batch_size_per_gpu=1, gradient_accumulation_steps=1,
                      train_batch_size=world, gradient_clipping=1.0)
        torch.manual_seed(123)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config, dist_init_required=False)
        if engine.zero_optimization_stage() != 3:
            raise ValueError('Distributed smoke did not activate ZeRO-3')
        losses = []
        for _ in range(2):
            prediction = engine(torch.randn(2, 8, device=f'cuda:{local_rank}', dtype=torch.bfloat16))
            loss = prediction.float().square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Non-finite distributed training loss')
            engine.backward(loss)
            engine.step()
            losses.append(float(loss.detach()))
        if engine.global_steps != 2:
            raise ValueError(f'Expected two optimizer steps, got {engine.global_steps}')
        dist.barrier()
        if rank == 0:
            atomic_json(args.output, {'status': 'passed', 'world_size': world, 'backend': 'nccl',
                                     'zero_stage': 3, 'actual_optimizer_steps': 2, 'losses_rank0': losses,
                                     'scope': 'tiny real distributed training, not full 35B memory validation'})
    finally:
        dist.destroy_process_group()


def model_smoke(args) -> None:
    from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector
    import torch
    selector = LocalLLMNeighborhoodSelector(str(args.model_path), device='cuda:0', max_new_tokens=64)
    answer = selector._generate('Return exactly this JSON: {"constraints": []}')
    if not answer.strip():
        raise ValueError('Full model returned empty generation')
    torch.cuda.synchronize(0)
    atomic_json(args.output, {'status': 'passed', 'generated_chars': len(answer),
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                             'scope': 'actual full-model generation on one GPU, not semantic quality'})


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download-model')
    download.add_argument('--repo', required=True)
    download.add_argument('--revision', default='main')
    download.add_argument('--model-path', type=Path, required=True)
    download.add_argument('--output', type=Path, required=True)
    prepare = commands.add_parser('seal')
    for name in ['base', 'env-prefix', 'code-dir', 'model-manifest', 'cpu-report', 'training-smoke', 'suite-smoke', 'output']:
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--cuda-flavor', required=True)
    prepare.add_argument('--data', type=Path, nargs='+', required=True)
    check = commands.add_parser('verify')
    for name in ['manifest', 'code-dir', 'env-prefix']:
        check.add_argument('--' + name, type=Path, required=True)
    gpu = commands.add_parser('distributed-smoke')
    gpu.add_argument('--deepspeed-config', type=Path, required=True)
    gpu.add_argument('--output', type=Path, required=True)
    generation = commands.add_parser('model-smoke')
    generation.add_argument('--model-path', type=Path, required=True)
    generation.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'download-model': download_model, 'seal': seal, 'verify': verify,
     'distributed-smoke': distributed_smoke, 'model-smoke': model_smoke}[args.command](args)


if __name__ == '__main__':
    main()
