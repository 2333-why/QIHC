"""One-command Linux launcher; owns and cleans up only its own processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from qihc.experiment_runtime.storage import load, save, value_digest, worker_lock
from qihc.experiment_runtime.identity import model_identity

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def stop(proc):
    if proc.poll() is not None: return
    try:
        if os.name != 'nt': os.killpg(proc.pid, signal.SIGTERM)
        else: proc.terminate()
    except ProcessLookupError: return
    try: proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name != 'nt': os.killpg(proc.pid, signal.SIGKILL)
        else: proc.kill()
        proc.wait(timeout=10)


def visible_device(logical_index):
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    return mask.split(',')[logical_index].strip() if mask else str(logical_index)


def server_failure(run, servers, gpu_ids, reason):
    """Surface child errors in the platform job log without loading huge logs."""
    details=[]
    for gpu, proc in zip(gpu_ids, servers):
        path=run/'logs'/f'local_gpu{gpu}.log'
        tail='(log unavailable)'
        if path.exists():
            with path.open('rb') as f:
                f.seek(0,2); f.seek(max(0,f.tell()-24000))
                tail='\n'.join(f.read().decode('utf-8',errors='replace').splitlines()[-100:])
        details.append(dict(gpu=gpu,exit_code=proc.poll(),log=str(path),tail=tail))
        print(f'\n--- GPU {gpu}, exit={proc.poll()}, {path} ---\n{tail}',file=sys.stderr,flush=True)
    save(run/'startup_failure.json',dict(time=time.time(),reason=reason,servers=details))
    return RuntimeError(reason+'; child log tails printed above; see startup_failure.json')


def gpu_command(model, port, cfg):
    return [sys.executable, '-u', '-m', 'experiments.feedback_active_v3.local_server',
            '--model', str(model), '--port', str(port), '--max-model-len', str(cfg['max_model_len'])]



def ready(url):
    try:
        with urllib.request.urlopen(url + '/health', timeout=2) as r: return r.status == 200
    except Exception: return False


def telemetry(root, endpoints, stopped, interval):
    path = root / 'telemetry.jsonl'
    while not stopped.is_set():
        row = {'time': time.time(), 'endpoints_healthy': [ready(url) for url in endpoints]}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            row['gpus'] = result.stdout.strip().splitlines()
        except Exception as exc: row['gpu_error'] = str(exc)
        with path.open('a', encoding='utf-8') as f: f.write(json.dumps(row) + '\n')
        stopped.wait(interval)


def preflight(args, cfg):
    if os.name != 'posix': raise ValueError('GPU launcher targets Linux; local CPU tests work on Windows')
    if not (args.model / 'config.json').is_file(): raise ValueError('Missing local model config: ' + str(args.model))
    for package in ('numpy', 'httpx', 'jsonschema', 'pyscipopt', 'highspy', 'ortools', 'transformers', 'accelerate'):
        importlib.metadata.version(package)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4: raise RuntimeError('Four visible CUDA devices required')
    devices = []
    for i in cfg['gpu_ids']:
        p = torch.cuda.get_device_properties(i)
        devices.append(dict(index=i, name=p.name, memory_gib=p.total_memory / 2**30))
        if p.total_memory < 75 * 2**30: raise RuntimeError('BF16 Qwen3-32B per replica requires an 80GB-class GPU; configure TP separately')
    for i in range(4):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port + i))
    # Fresh child per native backend prevents binary-loader collisions.
    from .data import make_case
    case = make_case('KP', 8, 3)
    for backend in ('scip', 'highs', 'cp_sat'):
        job = dict(stage='solve', model=case['gold'], backend=backend, seconds=.5, seed=17, chains=8)
        ip = args.run / 'preflight' / (backend + '.input.json'); op = args.run / 'preflight' / (backend + '.output.json')
        save(ip, job)
        subprocess.run([sys.executable, '-m', 'experiments.feedback_active_v3.cpu', str(ip), str(op)], check=True, timeout=40)
        result = load(op)
        if not result.get('feasible'): raise RuntimeError('Solver preflight failed: ' + backend + ' ' + str(result))
    packages = {p: importlib.metadata.version(p) for p in ('numpy', 'httpx', 'jsonschema', 'pyscipopt', 'highspy', 'ortools', 'transformers', 'accelerate', 'torch')}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT)
    files=model_identity(args.model); cache_path=args.run/'weight_hashes.json'
    cache=load(cache_path) if cache_path.exists() else {}
    def hash_weight(row):
        if row['sha256'] is not None: return row
        previous=cache.get(row['path'])
        if previous and all(previous[k]==row[k] for k in ('bytes','mtime_ns')):
            return dict(row,sha256=previous['sha256'])
        h=hashlib.sha256()
        with (args.model/row['path']).open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
        print('Hashed model shard:',row['path'],flush=True)
        return dict(row,sha256=h.hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool: files=list(pool.map(hash_weight,files))
    save(cache_path,{row['path']:row for row in files})
    return dict(devices=devices, packages=packages, python=sys.version, model_files=files,
                git_head=revision.stdout.strip(), model_path=str(args.model.resolve()),
                inference_cpu_workers=cfg['cpu_workers'], evaluation_workers=args.evaluation_workers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--data', type=Path, default=HERE / 'assets/public_dataset'); p.add_argument('--port', type=int, default=8100)
    p.add_argument('--stages', default='development,main,ablation,scale,replay,public'); p.add_argument('--cpu-workers', type=int, default=12)
    p.add_argument('--evaluation-workers', type=int, default=4); p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--mechanism-workers', type=int, default=2)
    p.add_argument('--skip-mechanism', action='store_true')
    p.add_argument('--max-num-seqs', type=int, default=16); p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args(); a.run = a.run.resolve(); a.data = a.data.resolve(); a.model = a.model.resolve()
    cfg = load(HERE / 'config.json'); cfg.update(cpu_workers=a.cpu_workers, requests_per_gpu=1, inference_backend='historical_transformers_localmodel')
    reserved_mechanism = 0 if a.skip_mechanism else a.mechanism_workers
    if a.cpu_workers + a.evaluation_workers + reserved_mechanism > max(1, (os.cpu_count() or 24) - 4):
        raise ValueError('Reserve at least four CPU cores for GPU services; reduce worker counts')
    a.run.mkdir(parents=True, exist_ok=True)
    children, handles = [], []; stopped = threading.Event(); monitor = None
    def interrupted(signum, frame): raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    with worker_lock(a.run / 'launcher.lock'):
        try:
            from .assets import restore
            restore(a.data)
            data_manifest=load(a.data/'manifest.json')
            if not data_manifest.get('formal_public_only'):
                raise ValueError('Formal launcher accepts only the locked public dataset')
            stages=a.stages.split(',')
            if len(set(stages))!=len(stages) or any(s not in ('development','main','ablation','scale','replay','public') for s in stages):
                raise ValueError('Invalid/duplicate stages')
            if 'replay' in stages and ('main' not in stages or stages.index('main')>stages.index('replay')):
                if not (a.run/'stages/main.json').exists(): raise ValueError('Replay requires completed main evidence first')
            environment = preflight(a, cfg); save(a.run / 'environment.json', environment)
            if a.preflight_only:
                print('Preflight passed; no experiment launched.', flush=True); return
            identity = value_digest(environment['model_files'])
            config_path = a.run / 'config.requested.json'; save(config_path, cfg)
            endpoints = [f'http://127.0.0.1:{a.port+i}' for i in range(4)]
            def spawn(command, log_name, env=None):
                logfile = a.run / 'logs' / log_name; logfile.parent.mkdir(parents=True, exist_ok=True)
                handle = logfile.open('ab'); handles.append(handle)
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(proc); return proc
            servers = []
            for i, gpu in enumerate(cfg['gpu_ids']):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible_device(gpu), TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn')
                servers.append(spawn(gpu_command(a.model, a.port+i, cfg), f'local_gpu{gpu}.log', env))
            timeout = time.monotonic() + 1200
            while not all(ready(url) for url in endpoints):
                if any(proc.poll() is not None for proc in servers): raise server_failure(a.run,servers,cfg['gpu_ids'],'LocalModel exited during startup')
                if time.monotonic() > timeout: raise server_failure(a.run,servers,cfg['gpu_ids'],'Model loading exceeded 20 minutes')
                print('Waiting for four model replicas...', flush=True); time.sleep(10)
            monitor = threading.Thread(target=telemetry, args=(a.run, endpoints, stopped, cfg['telemetry_seconds']), daemon=True); monitor.start()
            command = [sys.executable, '-u', '-m', 'experiments.feedback_active_v3.runtime', '--run', str(a.run),
                '--data', str(a.data), '--config', str(config_path), '--model', str(a.model), '--model-identity', identity,
                '--endpoints', ','.join(endpoints), '--stages', a.stages]
            runner = spawn(command, 'pipeline.log')
            while not (a.run / 'manifest.json').exists():
                if runner.poll() is not None: raise RuntimeError('Pipeline failed before manifest; inspect logs/pipeline.log')
                time.sleep(1)
            evaluator = spawn([sys.executable, '-u', '-m', 'experiments.feedback_active_v3.evaluation', '--run', str(a.run),
                '--data', str(a.data), '--workers', str(a.evaluation_workers), '--watch'], 'evaluation.log')
            mechanism = None
            if not a.skip_mechanism:
                mechanism = spawn([sys.executable, '-u', '-m', 'experiments.feedback_active_v3.mechanism', '--run', str(a.run / 'mechanism'),
                    '--data', str(a.data / 'evaluation/mechanism.json'), '--workers', str(a.mechanism_workers)], 'mechanism.log')
            print('RUNNING:', a.run, '\nFollow logs/pipeline.log and report.md; resume with the same command.', flush=True)
            while runner.poll() is None:
                if any(proc.poll() is not None for proc in servers): raise RuntimeError('A GPU service failed; results retained for resume')
                if evaluator.poll() not in (None, 0): raise RuntimeError('Evaluator failed; inspect evaluation.log')
                if mechanism and mechanism.poll() not in (None,0): raise RuntimeError('Mechanism experiment failed; inspect mechanism.log')
                time.sleep(5)
            if runner.returncode: raise RuntimeError('Pipeline failed; inspect logs/pipeline.log')
            for proc in servers: stop(proc)
            stopped.set()
            stop(evaluator)
            final = subprocess.run([sys.executable, '-m', 'experiments.feedback_active_v3.evaluation', '--run', str(a.run),
                '--data', str(a.data), '--workers', str(a.cpu_workers + a.evaluation_workers)], cwd=ROOT)
            if final.returncode: raise RuntimeError('Final CPU evaluation failed')
            if mechanism:
                if mechanism.poll() is None:
                    stop(mechanism)
                    # Free GPU/evaluation cores now accelerate the remaining
                    # cached CPU-only mechanism jobs without rerunning results.
                    mechanism=spawn([sys.executable,'-u','-m','experiments.feedback_active_v3.mechanism',
                        '--run',str(a.run/'mechanism'),'--data',str(a.data/'evaluation/mechanism.json'),
                        '--workers',str(a.cpu_workers+a.evaluation_workers)],'mechanism.log')
                if mechanism.wait()!=0: raise RuntimeError('Mechanism experiment failed')
            from .report import report
            result = report(a.run, statistics=True); print('FINISHED:', result['status'], a.run / 'report.md', flush=True)
            save(a.run/'completion.json',dict(status=result['status'],stages=a.stages.split(','),second_model_enabled=False,
                public_scope=data_manifest['scope'],human_audit='not_completed_by_launcher',
                mechanism='disabled' if a.skip_mechanism else 'complete'))
        finally:
            stopped.set()
            for proc in reversed(children): stop(proc)
            if monitor: monitor.join(timeout=15)
            for handle in handles: handle.close()


if __name__ == '__main__': main()
