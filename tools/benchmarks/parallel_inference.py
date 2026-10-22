"""Real batched inference, isolated replica queues; no synthetic keep-alive work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmarks import comparison as c
from tools.benchmarks.public_simulation import model_identity

LEGACY_SOURCE = '12b0c0928907d9f2de0533de48c0dc35d1b3a68f37f71dcdefb623dd218fbbcf'


def migrate(run):
    """Only the audited cebfb1b run may migrate. Never discard answers or gold."""
    if not (run/'manifest.json').exists(): return
    with c.worker_lock(run/'cpu.lock'), c.worker_lock(run/'gpu.lock'):
        m = c.load(run/'manifest.json')
        if m['settings']['source'] == c.code_hash():
            c.validate_run(run); return
        if m['settings']['source'] != LEGACY_SOURCE:
            raise ValueError('Unknown source revision; refusing automatic migration')
        if hashlib.sha256((run/'cases.CPU_ONLY.json').read_bytes()).hexdigest() != m['cases_sha256']:
            raise ValueError('Case file changed')
        if model_identity(Path(m['settings']['model'])) != m['settings']['model_files']:
            raise ValueError('Model identity changed')
        for p in (run/'responses').glob('*.json'):
            request = run/'requests'/p.name
            if c.load(p)['request_sha256'] != hashlib.sha256(request.read_bytes()).hexdigest():
                raise ValueError('Response integrity failed: '+p.name)
        backup = run/'manifest.before_parallel.json'
        if not backup.exists(): c.save(backup, m)
        m['execution_migration'] = {'from_source': LEGACY_SOURCE, 'to_source': c.code_hash(),
            'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'note': 'Same model/prompts/search budgets; batch execution may change numerical outputs. Legacy response costs retained.'}
        m['settings']['source'] = c.code_hash()
        c.save(run/'manifest.json', m)
        print('Migrated execution only; all existing answers and search checkpoints retained.', flush=True)


def pending(run, rank, replicas, limit):
    rows = []
    for p in (run/'requests').glob('*.json'):
        if (run/'responses'/p.name).exists() or int(p.stem, 16) % replicas != rank: continue
        r = c.load(p)
        priority = 0 if r['phase'] in ('qihc', 'solver_feedback') else 1
        rows.append((priority, p.stat().st_mtime_ns, p.name, p, r))
    rows.sort(key=lambda x: x[:3])
    return [(x[3], x[4]) for x in rows[:limit]]


def batch_generate(generator, requests):
    """Left-pad pre-tokenized prompts; preserve original chat template and limits."""
    torch = generator.torch
    inputs = [generator._inputs(r['prompt']) for r in requests]
    lengths = [int(v['input_ids'].shape[1]) for v in inputs]
    if max(lengths) > generator.max_input_tokens:
        raise ValueError('Prompt exceeds explicit input limit; no silent truncation')
    if generator.processor is not None:
        raise ValueError('Parallel runner currently supports text-only causal models')
    tokenizer = generator.tokenizer
    pad = tokenizer.pad_token_id
    if pad is None: pad = tokenizer.eos_token_id
    if pad is None: raise ValueError('Missing padding/EOS token')
    width = max(lengths)
    ids = torch.full((len(inputs), width), pad, dtype=torch.long, device=generator.device)
    mask = torch.zeros_like(ids)
    for i, v in enumerate(inputs):
        ids[i, -lengths[i]:] = v['input_ids'][0].to(generator.device)
        mask[i, -lengths[i]:] = 1
    with torch.inference_mode():
        output = generator.model.generate(input_ids=ids, attention_mask=mask,
                    max_new_tokens=generator.max_new_tokens, do_sample=False, pad_token_id=pad)
    eos = generator.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    records = []
    for i in range(len(inputs)):
        tokens = output[i, width:].tolist()
        for j, t in enumerate(tokens):
            if t in eos:
                tokens = tokens[:j+1]; break
        records.append({'response': tokenizer.decode(tokens, skip_special_tokens=True),
                        'status': 'generated', 'input_tokens': lengths[i], 'output_tokens': len(tokens)})
    return records


def execute(generator, items, run, rank, batch_id):
    started = time.perf_counter()
    try:
        records = batch_generate(generator, [r for _, r in items])
    except (generator.torch.cuda.OutOfMemoryError, ValueError) as exc:
        # Release failed generation frame/tensors before a smaller retry.
        exc.__traceback__ = None
        if isinstance(exc, generator.torch.cuda.OutOfMemoryError):
            generator.torch.cuda.empty_cache()
        if len(items) > 1:
            elapsed = time.perf_counter()-started
            c.save(run/'inference_events'/f'{batch_id}.json', {'event':'batch_retry', 'seconds':elapsed,
                   'error':str(exc), 'batch_size':len(items), 'rank':rank})
            middle = len(items)//2
            execute(generator, items[:middle], run, rank, batch_id+'a')
            execute(generator, items[middle:], run, rank, batch_id+'b')
            return
        if isinstance(exc, generator.torch.cuda.OutOfMemoryError):
            raise  # Leave request pending, not a model quality failure.
        records = [{'response':'', 'status':'generation_error', 'error':str(exc)}]
    elapsed = time.perf_counter()-started
    for (path, request), record in zip(items, records):
        record.update(seconds=elapsed/len(items), batch_seconds=elapsed, batch_size=len(items),
                      batch_id=batch_id, replica=rank, execution='transformers_batched_v1',
                      request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        c.save(run/'responses'/path.name, record)
    print(f'REPLICA {rank} saved {len(items)} responses in {elapsed:.1f}s', flush=True)


def worker(args):
    from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator
    cfg = c.load(args.run/'manifest.json')['settings']
    if cfg['source'] != c.code_hash() or model_identity(Path(cfg['model'])) != cfg['model_files']:
        raise ValueError('Source/model changed')
    generator = None
    idle = time.monotonic()
    with c.worker_lock(args.run/f'gpu-replica-{args.rank}.lock'):
        while not c.STOP:
            items = pending(args.run, args.rank, args.replicas, args.batch_size)
            if not items:
                if (args.run/'summary.json').exists() and c.load(args.run/'summary.json')['status'] == 'complete': break
                if time.monotonic()-idle > args.idle_exit: break
                time.sleep(1); continue
            if generator is None:
                generator = LocalDirectLLMGenerator(cfg['model'], device='balanced', temperature=0,
                                                     max_input_tokens=16384, max_new_tokens=8192)
                c.save(args.run/f'gpu_replica_{args.rank}_placement.json',
                       {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'device_map': {k:str(v) for k,v in getattr(generator.model,'hf_device_map',{}).items()}})
            batch_id = f'{args.rank}-{time.time_ns()}'
            c.save(args.run/f'gpu_replica_{args.rank}.json', {'status':'generating',
                   'batch_id':batch_id, 'requests':[r['case_id']+':'+r['phase'] for _,r in items]})
            execute(generator, items, args.run, args.rank, batch_id)
            telemetry = [{'device':i, 'allocated_bytes':generator.torch.cuda.memory_allocated(i),
                          'reserved_bytes':generator.torch.cuda.memory_reserved(i)}
                         for i in range(generator.torch.cuda.device_count())]
            c.save(args.run/f'gpu_replica_{args.rank}.json', {'status':'batch_saved',
                    'batch_id':batch_id, 'memory':telemetry})
            idle = time.monotonic()
        c.save(args.run/f'gpu_replica_{args.rank}.json', {'status':'stopped_checkpointed'})


def launch(args):
    import torch
    count = torch.cuda.device_count()
    cfg = c.load(args.run/'manifest.json')['settings']
    if cfg['source'] != c.code_hash():
        raise ValueError('Run CPU migration/preparation first')
    if count < args.replicas or count % args.replicas:
        raise ValueError('Visible GPU count must divide evenly across replicas')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else [str(i) for i in range(count)]
    if len(devices) != count: raise ValueError('Visible-device inventory mismatch')
    processes = []
    with c.worker_lock(args.run/'gpu.lock'):
        try:
            width = count//args.replicas
            for rank in range(args.replicas):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(devices[rank*width:(rank+1)*width]))
                cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--run',str(args.run),
                       '--rank',str(rank),'--replicas',str(args.replicas),'--batch-size',str(args.batch_size),
                       '--idle-exit',str(args.idle_exit)]
                processes.append(subprocess.Popen(cmd, env=env))
            while any(p.poll() is None for p in processes) and not c.STOP:
                if any(p.poll() not in (None,0) for p in processes):
                    raise RuntimeError('Replica failed; retained requests can resume after fixing the error')
                time.sleep(1)
            if any(p.poll() not in (None,0) for p in processes): raise RuntimeError('Replica failed')
        finally:
            for p in processes:
                if p.poll() is None: p.terminate()
            for p in processes:
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
        c.save(args.run/'gpu_status.json', {'status':'stopped_checkpointed', 'replicas':args.replicas,
               'batch_size':args.batch_size, 'note':'Check summary; if partial, resume after CPU queues work.'})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['migrate','launch','worker'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--replicas', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--idle-exit', type=int, default=45)
    args = p.parse_args()
    if not 1 <= args.replicas <= 8 or not 1 <= args.batch_size <= 32 or not 0 <= args.rank < args.replicas or not 1 <= args.idle_exit <= 600:
        p.error('Invalid execution settings')
    signal.signal(signal.SIGTERM,c.stop); signal.signal(signal.SIGINT,c.stop)
    {'migrate':lambda a:migrate(a.run),'launch':launch,'worker':worker}[args.stage](args)


if __name__ == '__main__': main()
