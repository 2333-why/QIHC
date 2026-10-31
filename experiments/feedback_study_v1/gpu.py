"""Independent GPU replicas serving a durable dynamic request queue."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from .common import s, validate, default_root

STOP = False


def stop(*_):
    global STOP
    STOP = True


def response_path(root, identity):
    return root/'responses'/(identity+'.json')


def terminal_error(root, path, error, attempted=False):
    out = response_path(root, path.stem)
    if not out.exists():
        s.save(out, {'status':'generation_error', 'error':error, 'response':'',
                    'request_sha256':s.digest(path), 'generation_attempts':int(attempted),
                    'input_tokens':0, 'output_tokens':0,
                    'cost_incomplete':attempted, 'time':time.time()})


def pending(root, rank, count):
    candidates=[]
    for path in (root/'pending'/str(rank)).glob('*.json'):
        try:
            candidates.append((path.stat().st_mtime_ns,path.name,path))
        except FileNotFoundError:
            # Coordinator can remove completed markers concurrently.
            continue
    markers=[row[2] for row in sorted(candidates)]
    result = []; stage = None
    for marker in markers:
        if response_path(root, marker.stem).exists():
            marker.unlink(missing_ok=True); continue
        path = root/'requests'/marker.name; req = s.load(path)
        try:
            marker_row=s.load(marker)
        except FileNotFoundError:
            continue
        if marker_row['request_sha256'] != s.digest(path):
            raise ValueError('Pending request identity mismatch')
        if stage is None:
            stage = req['stage']
        if req['stage'] == stage:
            result.append((path, req))
        if len(result) >= count:
            break
    return result


def worker(root, rank):
    from qihc.modeling.backend import LocalModel
    from qihc.modeling.generation import generate
    from qihc.experiment_runtime.identity import model_identity
    from .protocol import bounded_evidence
    m = validate(root); cfg = m['config']
    if model_identity(Path(m['model'])) != m['model_files']:
        raise ValueError('Model metadata changed')
    hb = root/f'gpu-{rank}.json'
    with s.worker_lock(root/f'gpu-{rank}.lock'):
        s.save(hb, {'status':'loading', 'time':time.time()})
        generator = LocalModel(m['model'], device='cuda:0' if cfg['gpu_replicas']==4 else 'balanced',
                               temperature=0., max_input_tokens=cfg['max_input_tokens'])
        placement = getattr(generator.model, 'hf_device_map', {})
        if not placement or any(str(v) in ('cpu', 'disk') for v in placement.values()):
            raise RuntimeError('Model does not fit on assigned GPUs; CPU/disk offload forbidden')
        print('MODEL LOADED', rank, placement, flush=True)
        while not STOP:
            control = s.load(root/'control.json')
            if control['status'] != 'running':
                break
            items = pending(root, rank, cfg['batch_size'])
            if not items:
                s.save(hb, {'status':'idle', 'time':time.time()}); time.sleep(1); continue
            prepared = []
            for path, req in items:
                task = dict(req)
                if 'evidence_payload' in req:
                    packet = bounded_evidence(req['evidence_payload'], generator.tokenizer, cfg['feedback_tokens'])
                    task['prompt'] = task['prompt'].replace('__E004_EVIDENCE__', json.dumps(packet, ensure_ascii=False, separators=(',', ':')))
                try:
                    n = int(generator._inputs(task['prompt'])['input_ids'].shape[1])
                    if n > min(cfg['max_input_tokens'], req.get('input_remaining', cfg['max_input_tokens'])):
                        terminal_error(root, path, f'Input token budget exceeded: {n}; source not truncated')
                        continue
                except (ValueError, TypeError) as exc:
                    terminal_error(root, path, repr(exc)); continue
                prepared.append((path, task))
            if not prepared:
                continue
            ids = [p.stem for p,_ in prepared]
            s.save(hb, {'status':'generating', 'time':time.time(), 'requests':ids})
            batch = str(time.time_ns()); started = time.monotonic()
            # Persist attempted calls BEFORE generation. On a killed process, the
            # coordinator records unknown token cost, never pretends it was free.
            s.save(root/'inference_events'/(batch+'.json'), {'status':'started', 'requests':ids,
                    'time':time.time(), 'rank':rank, 'request_hashes':[s.digest(p) for p,_ in prepared]})
            try:
                rows = generate(generator, prepared)
            except Exception as exc:
                for path, _ in prepared:
                    terminal_error(root, path, repr(exc), attempted=True)
                generator.torch.cuda.empty_cache()
                # Device failures require a new process/context; no silent CPU fallback.
                if isinstance(exc, RuntimeError):
                    raise
                continue
            elapsed = time.monotonic()-started
            for (path, task), row in zip(prepared, rows):
                row.update(request_sha256=s.digest(path), generation_attempts=1,
                           attributed_seconds=elapsed/len(rows), batch_seconds=elapsed,
                           time=time.time(), cost_incomplete=False, gpu_replica=rank,
                           gpu_memory_allocated_bytes=[generator.torch.cuda.memory_allocated(i) for i in range(generator.torch.cuda.device_count())])
                s.save(response_path(root, path.stem), row)
            s.save(root/'inference_events'/(batch+'.json'), {'status':'complete', 'requests':ids,
                    'seconds':elapsed, 'rank':rank, 'time':time.time()})
            s.save(hb, {'status':'batch_complete', 'time':time.time()})
            print('BATCH', rank, ids, round(elapsed, 2), flush=True)
        s.save(hb, {'status':'stopped', 'time':time.time()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, default=default_root()); p.add_argument('--rank', type=int, required=True)
    a = p.parse_args(); signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    worker(a.run, a.rank)
