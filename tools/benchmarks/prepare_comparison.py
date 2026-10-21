#!/usr/bin/env python3
"""CPU-only data/model preparation; reuses existing weights, no training/environment reinstall."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmarks.download_public import atomic_write, json_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--per-dataset', type=int, default=40)
    parser.add_argument('--cases-file', type=Path)
    args = parser.parse_args()
    # Defaults are local/offline after preparation; remove inherited offline flags only here.
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'PIP_NO_INDEX'):
        os.environ.pop(name, None)
    if not args.cases_file:
        subprocess.run([sys.executable, str(ROOT/'tools/benchmarks/download_public.py'), '--output',
                        str(args.base/'data/public_benchmarks_v1')], check=True)
    from experiments.server_readiness import download_model, check_model_headers
    from transformers import AutoTokenizer, AutoConfig
    from qihc.generic.bounded_reference import solve
    import numpy, ortools, torch
    print(json.dumps({'numpy': numpy.__version__, 'ortools': ortools.__version__, 'torch': torch.__version__}), flush=True)
    model = args.base/'models/Qwen3-32B'
    lock = json.loads(Path(__file__).with_name('related_methods.lock.json').read_text())
    chosen = lock['models'][0]
    marker = model/'.comparison-download.json'
    reused = model.exists() and not marker.exists()
    if not model.exists() or (marker.exists() and json.loads(marker.read_text()).get('status') != 'complete'):
        atomic_write(marker, json_bytes({'status': 'downloading', 'revision': chosen['revision']}))
        errors = []
        for endpoint in ('https://hf-mirror.com', 'https://huggingface.co'):
            os.environ['HF_ENDPOINT'] = endpoint
            try:
                download_model(SimpleNamespace(repo=chosen['repo'], revision=chosen['revision'], model_path=model,
                                                output=args.base/'artifacts/comparison-model.json'))
                break
            except Exception as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError('Model download failed: '+str(errors))
        atomic_write(marker, json_bytes({'status': 'complete', 'revision': chosen['revision']}))
    # Existing incomplete models are NOT silently overwritten. Repair explicitly if this fails.
    headers = check_model_headers(model)
    AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    assert tokenizer.encode('Optimization smoke test')
    result = solve({'schema': 'bounded-integer-v2', 'variables': [{'name': 'x', 'lower': 0, 'upper': 1}],
                    'objective': {'sense': 'max', 'linear': [1], 'quadratic': []}, 'constraints': [], 'logic': []}, seconds=2)
    if result['objective'] != 1 or not result['optimality_proven']:
        raise RuntimeError('CP-SAT environment smoke failed')
    atomic_write(args.base/'artifacts/comparison-cpu-check.json', json_bytes({'model': str(model), 'headers': headers,
                  'tokenizer': 'passed', 'cp_sat': 'passed', 'gpu_full_model': 'deferred',
                  'existing_weights_reused': reused, 'paper_reproduction': 'not_enabled_no_solver_license'}))
    revision = json.loads((ROOT/'tools/benchmarks/public_benchmarks.lock.json').read_text())['revision']
    subprocess.run([sys.executable, str(ROOT/'tools/benchmarks/comparison.py'), 'prepare',
                    '--run', str(args.run), '--data', str(args.base/'data/public_benchmarks_v1'/revision),
                    '--model', str(model), '--per-dataset', str(args.per_dataset)] +
                   (['--cases-file', str(args.cases_file)] if args.cases_file else []), check=True)


if __name__ == '__main__': main()
