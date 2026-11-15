"""Frozen E005-1.14 evidence, four independent adapters and paired 780-case evaluation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib
import os
from pathlib import Path
import time
import sys
import subprocess

import numpy as np
from qihc.experiment_runtime import posttrain_screen as runtime
from qihc.experiment_runtime.storage import load,save,value_digest,digest,worker_lock
from qihc.experiment_runtime.pbit_control import seed
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.pbit_joint_v1 import run as common, search
from experiments.pbit_joint_v2 import run as previous
from experiments.pbit_joint_v2.signals import control_prompt
from . import audit, methods

HERE = Path(__file__).resolve().parent
CFG = load(HERE/'config.json')
RUN = Path(os.getenv('QIHC_SCALE_RUN',previous.BASE/'results/E005-1.15_four'))
OLD = Path(os.getenv('QIHC_114_RUN',previous.BASE/'results/E005-1.14_four'))
PREVIOUS = previous.PREVIOUS
MODELS = ('weighted','anchor','slow')
POLICIES = ('schedule','rank','balanced')
ARMS = ('anchor','slow','rank','balanced')


def job():
    return importlib.import_module('experiments.pbit_joint_v3.run')


def cases():
    return [c for c in load(RUN/'inputs/cohort.json') if c['stage'] in ('main','scale')]


def sources():
    return {str(p.relative_to(HERE.parent)):digest(p) for d in (common.HERE,previous.HERE)
            for p in d.iterdir() if p.suffix in ('.py','.json')}


def put_snapshot(path,data):
    if path.exists():
        if load(path)!=data:
            raise ValueError(f'Frozen snapshot differs: {path}')
    else:
        save(path,data)


def audit_job(task):
    cid,gold,base_path = task
    out = RUN/'scale_audit'/f'{cid}.json'
    if not out.exists():
        row = audit.examine(cid,load(OLD/'scores'/f'{cid}.json'),load(Path(base_path)),
                            load(RUN/'predictions/weighted'/f'{cid}.json'),gold,load(RUN/'references'/f'{cid}.json'))
        save(out,row)
    return load(out)


def prepare():
    if OLD.resolve()==RUN.resolve() or OLD.resolve() in RUN.resolve().parents or RUN.resolve() in OLD.resolve().parents:
        raise ValueError('Use independent old/new result directories')
    runtime.prepare(job())
    manifest = runtime.validate(job())
    old_manifest = load(OLD/'manifest.json')
    if old_manifest['experiment_id']!='E005-1.14' or any(manifest[k]!=old_manifest[k] for k in ('inputs','dependencies','model_files')):
        raise ValueError('E005-1.14 input or parent mismatch')
    old_report = load(OLD/'report.json')
    if old_report['evaluations']!=9360:
        raise ValueError('Expected completed E005-1.14 report')
    old_ready = load(OLD/'ready.json')
    gold = {r['id']:r['gold'] for p in manifest['gold_paths'] for r in load(Path(p))}
    with worker_lock(RUN/'prepare15.lock'):
        evidence = {}
        for row in load(RUN/'inputs/train.json'):
            cid=row['case']['id']
            path=OLD/'evidence'/f'{cid}.json'
            if digest(path)!=old_ready['files'][cid]:
                raise ValueError('Changed training evidence: '+cid)
            evidence[cid]=load(path)
        put_snapshot(RUN/'training.json',evidence)
        jobs=[]
        for case in cases():
            cid=case['id']
            for name in ('predictions/weighted','references'):
                put_snapshot(RUN/name/f'{cid}.json',load(OLD/name/f'{cid}.json'))
            if case['stage']=='scale':
                jobs.append((cid,gold[cid],manifest['predictions'][cid]))
        with ProcessPoolExecutor(max_workers=8) as pool:
            rows=list(pool.map(audit_job,jobs))
        observed=[sum(row['hits'][i] for row in rows) for i in (0,1)]
        if len(rows)!=300 or observed!=[46,40]:
            raise ValueError(f'Scale snapshot mismatch: n={len(rows)}, hits={observed}')
        save(RUN/'scale_diagnostic.json',audit.summarize(rows))
        diagnostics={}
        for arm in POLICIES[1:]:
            samples,_=methods.control_data(evidence,arm)
            diagnostics[arm]=dict(examples=len(samples),winners=dict(Counter(search.ACTIONS[w] for _,_,w,_ in samples)),
                                  intervention_cases=len({cid for cid,_,w,_ in samples if w!=0}))
        ready=dict(training=digest(RUN/'training.json'),sources=sources(),diagnostics=diagnostics,
                   manifest=digest(RUN/'manifest.json'))
        put_snapshot(RUN/'ready.json',ready)
        print('Scale diagnosis:',audit.summarize(rows),'Training supervision:',diagnostics,flush=True)


def subtasks():
    for i in range(CFG['shards']):
        for m in MODELS[1:]:
            yield ('predict',m,m,i)
        for m in MODELS:
            for p in POLICIES[1:]:
                yield ('search',m,p,i)


def marker(t):
    return RUN/'shards'/('_'.join(map(str,t))+'.json')


def eligible(t):
    op,m,a,i=t
    return (RUN/'trained'/a/'adapter.pt').exists() and (
        op=='predict' or m=='weighted' or marker(('predict',m,m,i)).exists())


def process_shard(task,engine,ready,subset):
    from experiments.posttraining_v1.learn import parsed
    import torch
    op,model,arm,idx=task
    engine.net.eval()
    if op=='search':
        ids=common.action_ids(engine)
        supported=ready['diagnostics'][arm]['winners']
        choices=[i for i,a in enumerate(search.ACTIONS) if a=='cool' or a in supported]
        def choose(spec,state,remaining):
            with torch.inference_mode():
                logits=common.logits(engine,control_prompt(spec,state,remaining),ids)
                return search.ACTIONS[choices[int(logits[choices].argmax())]]
    for case in subset[idx::CFG['shards']]:
        cid=case['id']
        path=RUN/'predictions'/model/f'{cid}.json'
        if op=='predict':
            if path.exists():
                continue
            start=time.monotonic()
            res=engine.evaluate(model_prompt(case),seed(cid,'model') & 0x7fffffff)
            choice=res['choices'][0]
            spec,error=parsed(choice['message']['content'],case) if choice['finish_reason']=='stop' else (None,'length')
            save(path,dict(model=spec,error=error,generation_seconds=time.monotonic()-start))
        else:
            dest=RUN/'episodes'/f'{model}__{arm}'/f'{cid}.json'
            if not dest.exists():
                spec=load(path)['model']
                ep=search.episode(spec,cid,CFG,choose) if spec is not None else previous.trajectory(None,cid,'schedule')
                save(dest,ep)


def worker(arm):
    import torch
    from experiments.posttraining_v1.engine import Engine
    manifest=runtime.validate(job())
    ready=load(RUN/'ready.json')
    if ready['sources']!=sources() or ready['training']!=digest(RUN/'training.json') or ready['manifest']!=digest(RUN/'manifest.json'):
        raise ValueError('Run input changed')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_properties(0).total_memory<75*1024**3 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Need one visible BF16 80GB GPU per worker')
    torch.manual_seed(CFG['seed'])
    engine=Engine(Path(manifest['model']))
    with worker_lock(RUN/'trained'/arm/'train.lock'):
        methods.train(engine,arm,RUN/'trained'/arm,manifest,ready,load(RUN/'inputs/train.json'),load(RUN/'training.json'),CFG)
    current=arm
    subset=cases()
    while pending:=(sorted((t for t in subtasks() if not marker(t).exists()),key=lambda t:t[2]!=current)):
        for t in pending:
            if not eligible(t):
                continue
            try:
                with worker_lock(marker(t).with_suffix('.lock')):
                    if marker(t).exists():
                        continue
                    if current!=t[2]:
                        current=t[2]
                        engine.restore(torch.load(RUN/'trained'/current/'adapter.pt',map_location='cpu',weights_only=True))
                    print('running',t,flush=True)
                    process_shard(t,engine,ready,subset)
                    save(marker(t),dict(task=t))
                break
            except BlockingIOError:
                continue
        else:
            time.sleep(2)


def gpu():
    import torch
    from experiments.pvcd_cycle_v1.launch import stop_process
    devices=os.getenv('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    if len(set(devices))!=4 or len(devices)!=4 or torch.cuda.device_count()!=4:
        raise ValueError('Allocate four visible GPUs')
    live=[]
    with worker_lock(RUN/'gpu.lock'):
        try:
            for arm,device in zip(ARMS,devices):
                path=RUN/'logs'/f'{arm}.log'
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('a') as log:
                    proc=subprocess.Popen([sys.executable,'-m',job().__name__,'worker',arm],
                        stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),start_new_session=True)
                live.append((proc,path))
            next_sample=time.monotonic()
            while True:
                for proc,path in live:
                    if proc.poll() not in (None,0):
                        with path.open('rb') as tail:
                            tail.seek(max(0,path.stat().st_size-4096))
                            raise RuntimeError(f'{path} exit {proc.returncode}: '+tail.read().decode(errors='replace'))
                if all(proc.poll()==0 for proc,_ in live):
                    return
                if time.monotonic()>=next_sample:
                    try:
                        usage=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used',
                                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                        if usage.returncode==0:
                            with (RUN/'logs/gpu_usage.csv').open('a') as log:
                                log.write(usage.stdout)
                    except (OSError,subprocess.TimeoutExpired):
                        pass
                    next_sample=time.monotonic()+60
                time.sleep(2)
        finally:
            for proc,_ in live:
                stop_process(proc)


def main():
    from .report import score_all,report
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','gpu','worker','score','report'])
    parser.add_argument('arm',nargs='?',choices=ARMS)
    a=parser.parse_args()
    if a.stage=='worker':
        if not a.arm: parser.error('worker requires arm')
        worker(a.arm)
    else:
        dict(prepare=prepare,gpu=gpu,score=score_all,report=report)[a.stage]()


if __name__=='__main__':
    main()
