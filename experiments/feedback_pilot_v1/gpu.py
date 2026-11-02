import argparse
import signal
import time
import gc
from pathlib import Path
from .adapter import install, default_root
from qihc.experiment_runtime import resilient_storage as s

STOP = False


def stop(*_):
    global STOP
    STOP = True


def requires_process_restart(exc):
    # RecursionError is a RuntimeError subclass but not a broken CUDA context.
    return isinstance(exc, RuntimeError) and not isinstance(exc, RecursionError)


def worker(root, rank):
    from experiments.feedback_study_v1.common import validate
    from experiments.feedback_study_v1.gpu import pending, terminal_error, response_path
    from experiments.feedback_study_v1.protocol import bounded_evidence
    from experiments.feedback_study_v1.placement import verify_placement, DevicePlacementError
    from qihc.experiment_runtime.identity import model_identity
    from qihc.modeling.backend import LocalModel
    from qihc.modeling.generation import generate
    import json
    m=validate(root);cfg=m['config'];hb=root/f'gpu-{rank}.json'
    if model_identity(Path(m['model'])) != m['model_files']:
        raise ValueError('Model metadata changed')
    with s.worker_lock(root/f'gpu-{rank}.lock'):
        s.save(hb,{'status':'loading','time':time.time()})
        generator=LocalModel(m['model'],device='cuda:0',temperature=0.,max_input_tokens=cfg['max_input_tokens'])
        try:placement=verify_placement(generator.model,generator.torch.cuda.device_count())
        except DevicePlacementError as exc:
            s.save(root/f'gpu-{rank}-placement.json',dict(exc.report,time=time.time(),rank=rank))
            s.save(hb,{'status':'placement_failed','time':time.time(),'error':str(exc)})
            raise
        s.save(root/f'gpu-{rank}-placement.json',dict(placement,time=time.time(),rank=rank))
        print('MODEL LOADED',rank,flush=True)
        cleanup=False
        while not STOP:
            if cleanup:
                gc.collect();generator.torch.cuda.empty_cache();cleanup=False
            if s.load(root/'control.json')['status']!='running':break
            items=pending(root,rank,1)
            if not items:
                s.save(hb,{'status':'idle','time':time.time()});time.sleep(1);continue
            path,req=items[0];task=dict(req)
            try:
                if 'evidence_payload' in req:
                    packet=bounded_evidence(req['evidence_payload'],generator.tokenizer,cfg['feedback_tokens'])
                    task['prompt']=task['prompt'].replace('__E004_EVIDENCE__',json.dumps(packet,ensure_ascii=False,separators=(',',':')))
                n=int(generator._inputs(task['prompt'])['input_ids'].shape[1])
                if n>min(cfg['max_input_tokens'],req.get('input_remaining',cfg['max_input_tokens'])):
                    raise ValueError(f'Input token budget exceeded: {n}; source not truncated')
            except (ValueError,TypeError,RecursionError) as exc:
                terminal_error(root,path,repr(exc));continue
            ids=[path.stem];batch=str(time.time_ns());started=time.monotonic()
            eventpath=root/'inference_events'/(batch+'.json')
            s.save(hb,{'status':'generating','time':time.time(),'requests':ids})
            s.save(eventpath,{'status':'started','requests':ids,'time':time.time(),'rank':rank,'request_hashes':[s.digest(path)]})
            try:rows=generate(generator,[(path,task)])
            except Exception as exc:
                terminal_error(root,path,repr(exc),attempted=True)
                s.save(eventpath,{'status':'failed','requests':ids,'time':time.time(),'rank':rank,
                                 'error':repr(exc),'seconds':time.monotonic()-started,'cost_incomplete':True})
                if isinstance(exc,generator.torch.cuda.OutOfMemoryError):cleanup=True
                elif requires_process_restart(exc):raise
                else:cleanup=True
                continue
            elapsed=time.monotonic()-started
            row=rows[0]
            row.update(request_sha256=s.digest(path),generation_attempts=1,attributed_seconds=elapsed,
                       batch_seconds=elapsed,time=time.time(),cost_incomplete=False,gpu_replica=rank,
                       gpu_memory_allocated_bytes=[generator.torch.cuda.memory_allocated(0)])
            s.save(response_path(root,path.stem),row)
            s.save(eventpath,{'status':'complete','requests':ids,'seconds':elapsed,'rank':rank,'time':time.time()})
            s.save(hb,{'status':'batch_complete','time':time.time()})
            print('BATCH',rank,ids,round(elapsed,2),flush=True)
        s.save(hb,{'status':'stopped','time':time.time()})

if __name__ == '__main__':
    install()
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=default_root())
    p.add_argument('--rank', type=int, required=True)
    a = p.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker(a.run, a.rank)
