"""Batched GPU inference for the NEW queue only; never loads CPU-only cases."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.benchmarks import comparison as c
from tools.runtime.modeling_v2.generation import execute
from tools.runtime.modeling_v2.experiment import validate

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def pending(run, rank, replicas, limit):
    rows = []
    for path in (run/'requests').glob('*.json'):
        if (run/'responses'/path.name).exists() or int(path.stem, 16) % replicas != rank:
            continue
        request = c.load(path)
        rows.append((0 if request['phase'].startswith('review_') else 1,
                     path.stat().st_mtime_ns, path.name, path, request))
    rows.sort(key=lambda row: row[:3])
    return [(r[3], r[4]) for r in rows[:limit]]


def worker(args):
    cfg = validate(args.run, cpu=False, model=True)['settings']
    from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator
    generator, idle = None, time.monotonic()
    with c.worker_lock(args.run/f'gpu-replica-{args.rank}.lock'):
        while not STOP:
            items = pending(args.run, args.rank, args.replicas, args.batch_size)
            if not items:
                summary = args.run/'summary.json'
                if summary.exists() and c.load(summary)['status'] == 'complete':
                    break
                # Do not strand a temporarily empty shard while other replicas
                # still consume real work that may generate follow-up requests.
                if pending(args.run, 0, 1, 1):
                    idle = time.monotonic()
                if time.monotonic()-idle > args.idle_exit:
                    break
                time.sleep(1)
                continue
            if generator is None:
                generator = LocalDirectLLMGenerator(cfg['model'], device='balanced', temperature=0,
                                                     max_input_tokens=16384, max_new_tokens=4096)
                placement = getattr(generator.model, 'hf_device_map', {})
                if any(str(device) in ('cpu', 'disk') for device in placement.values()):
                    raise RuntimeError('Model offloaded to CPU/disk; refusing unexpectedly slow GPU run')
                c.save(args.run/f'gpu_replica_{args.rank}_placement.json',
                       {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'device_map': {k: str(v) for k, v in getattr(generator.model, 'hf_device_map', {}).items()}})
            batch_id = f'{args.rank}-{time.time_ns()}'
            c.save(args.run/f'gpu_replica_{args.rank}.json', {'status': 'generating', 'time': time.time(),
                   'batch_id': batch_id, 'requests': [r['case_id']+':'+r['phase'] for _, r in items]})
            execute(generator, items, args.run, args.rank, batch_id)
            telemetry = [{'device': i, 'allocated_bytes': generator.torch.cuda.memory_allocated(i),
                          'reserved_bytes': generator.torch.cuda.memory_reserved(i)}
                         for i in range(generator.torch.cuda.device_count())]
            c.save(args.run/f'gpu_replica_{args.rank}.json', {'status': 'batch_saved', 'time': time.time(),
                    'batch_id': batch_id, 'memory': telemetry})
            idle = time.monotonic()
        c.save(args.run/f'gpu_replica_{args.rank}.json', {'status': 'stopped_checkpointed', 'time': time.time()})


def launch(args):
    validate(args.run, cpu=False, model=True)
    import torch
    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count < args.replicas or count % args.replicas:
        raise ValueError('Visible CUDA GPUs must divide evenly across model replicas; default 2 replicas / 4 GPUs')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(count)]
    if len(devices) != count:
        raise ValueError('CUDA visible-device inventory mismatch')
    processes = []
    with c.worker_lock(args.run/'gpu.lock'):
        try:
            width = count//args.replicas
            for rank in range(args.replicas):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices[rank*width:(rank+1)*width]))
                cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--run', str(args.run),
                       '--rank', str(rank), '--replicas', str(args.replicas), '--batch-size', str(args.batch_size),
                       '--idle-exit', str(args.idle_exit)]
                processes.append(subprocess.Popen(cmd, env=env))
            while any(p.poll() is None for p in processes) and not STOP:
                if any(p.poll() not in (None, 0) for p in processes):
                    raise RuntimeError('GPU replica failed; completed response files remain resumable')
                time.sleep(1)
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('GPU replica failed')
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.terminate()
            for proc in processes:
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
            c.save(args.run/'gpu_status.json', {'status': 'stopped_checkpointed', 'time': time.time(),
                   'note': 'If summary is partial, rerun after CPU has queued requests. No fake GPU keep-alive work.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['launch', 'worker'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--replicas', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--idle-exit', type=int, default=120)
    args = p.parse_args()
    if not 1 <= args.replicas <= 8 or not 1 <= args.batch_size <= 32 or not 0 <= args.rank < args.replicas or not 1 <= args.idle_exit <= 600:
        p.error('Invalid inference execution settings')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    {'launch': launch, 'worker': worker}[args.stage](args)


if __name__ == '__main__':
    main()
