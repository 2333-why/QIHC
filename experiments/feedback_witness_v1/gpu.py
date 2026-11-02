"""Persistent single-GPU worker; no automatic repeat or shared-module mutation."""
import argparse
import json
import time
from pathlib import Path
from .common import s
from .protocol import compact_evidence


class TimedModel:
    def __init__(self,model,seconds):self.inner,self.seconds=model,seconds
    def __getattr__(self,key):return getattr(self.inner,key)
    def generate(self,**kwargs):
        kwargs['max_time']=self.seconds
        return self.inner.generate(**kwargs)


class TimedGenerator:
    def __init__(self,generator,seconds):
        self.inner=generator;self.model=TimedModel(generator.model,seconds)
    def __getattr__(self,key):return getattr(self.inner,key)


def infer(generator,req,cfg):
    from qihc.modeling.generation import generate
    task=dict(req)
    packet=compact_evidence(req['evidence_payload'],generator.tokenizer,cfg['feedback_tokens'])
    task['prompt']=task['prompt'].replace('"__WITNESS_EVIDENCE__"',json.dumps(packet,separators=(',',':')))
    n=int(generator._inputs(task['prompt'])['input_ids'].shape[1])
    if n>min(cfg['request_input_tokens'],req['input_remaining']):
        return {'status':'input_budget_exceeded','input_tokens':n,'output_tokens':0,'generation_attempts':0}
    t=time.monotonic()
    timed=TimedGenerator(generator,req['time_limit_seconds'])
    row=generate(timed,[(Path('request.json'),task)])[0]
    if '_v6_tokenizer_data' in timed.__dict__:
        generator._v6_tokenizer_data=timed._v6_tokenizer_data
    elapsed=time.monotonic()-t
    row.update(attributed_seconds=elapsed,generation_attempts=1,time_limit_seconds=req['time_limit_seconds'],
               delivered_evidence=packet)
    if row.get('status')!='generated' and elapsed>=req['time_limit_seconds']:row['finish_reason']='time_limit'
    return row


def worker(root,rank):
    m=s.load(root/'manifest.json');cfg=m['config'];folder=root/'gpu'/str(rank)
    folder.mkdir(parents=True,exist_ok=True);hb=folder/'heartbeat.json'
    s.save(hb,{'status':'loading','time':time.time()})
    from qihc.modeling.backend import LocalModel
    from experiments.feedback_study_v1.placement import verify_placement
    from qihc.experiment_runtime.identity import model_identity
    if model_identity(Path(m['model']))!=m['model_files']:raise ValueError('Model files changed')
    generator=LocalModel(m['model'],device='cuda:0',temperature=0.,max_input_tokens=cfg['request_input_tokens'])
    placement=verify_placement(generator.model,generator.torch.cuda.device_count())
    s.save(folder/'placement.json',placement)
    s.save(hb,{'status':'ready','time':time.time()})
    print('MODEL LOADED',rank,flush=True)
    while not (root/'stop.json').exists():
        paths=sorted((folder/'pending').glob('*.json')) if (folder/'pending').exists() else []
        paths=[p for p in paths if not (root/'responses'/p.name).exists()]
        if not paths:time.sleep(.2);continue
        path=paths[0];req=s.load(path);started=root/'attempts'/path.name
        if started.exists():
            row={'status':'interrupted_no_retry','generation_attempts':1,'cost_incomplete':True}
        else:
            s.save(started,{'time':time.time(),'rank':rank,'request_sha256':s.digest(path)})
            s.save(hb,{'status':'generating','request':path.stem,'time':time.time()})
            try:row=infer(generator,req,cfg)
            except Exception as exc:
                row={'status':'generation_error','error':repr(exc),'generation_attempts':1,'cost_incomplete':True}
                s.save(root/'responses'/path.name,dict(row,request_sha256=s.digest(path),gpu_replica=rank))
                if isinstance(exc,RuntimeError) and not isinstance(exc,RecursionError):raise
        row.update(request_sha256=s.digest(path),gpu_replica=rank)
        s.save(root/'responses'/path.name,row)
        s.save(hb,{'status':'ready','time':time.time()})
        print('REQUEST',rank,path.stem,row['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    a=p.parse_args()
    with s.worker_lock(a.run/'gpu'/str(a.rank)/'worker.lock'):worker(a.run,a.rank)
