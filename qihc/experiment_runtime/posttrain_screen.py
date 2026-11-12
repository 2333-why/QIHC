"""Reusable single-seed posttraining screen with immutable inputs and atomic restart."""
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from .storage import load,save,digest,worker_lock,value_digest
from .identity import model_identity

_GOLD_CACHE={}

def module_job(name):return importlib.import_module(name)

def source_hashes(job):
    repo=job.HERE.parents[1]
    files=list((repo/'qihc').rglob('*.py'))
    for name in ('posttraining_v1','feedback_active_v3','feedback_active_v4','pvcd_cycle_v1',job.HERE.name):
        folder=repo/'experiments'/name
        files += [p for p in folder.iterdir() if p.suffix in ('.py','.sh','.json','.txt')]
    return {str(p.relative_to(repo).as_posix()):digest(p) for p in sorted(set(files))}

def case_list(job,mode):
    if mode=='pilot':return [dict(r['case'],stage='pilot') for r in load(job.RUN/'inputs/validation.json')]
    return load(job.RUN/'inputs/cohort.json')

def train_list(job,mode):
    from experiments.feedback_active_v3.runtime import seed_for
    rows=load(job.RUN/'inputs/train.json')
    if mode=='pilot':rows=sorted(rows,key=lambda r:seed_for(r['case']['id'],'pilot'))[:job.CFG['pilot_train_cases']]
    return rows

def validate(job):
    from experiments.pvcd_cycle_v1.common import environment_identity
    m=load(job.RUN/'manifest.json')
    if m['config']!=job.CFG or m['source']!=source_hashes(job):raise ValueError('Run code/config changed')
    if m['environment']!=environment_identity():raise ValueError('Shared environment changed')
    for path,h in m['dependencies'].items():
        if digest(Path(path))!=h:raise ValueError('Frozen dependency changed '+path)
    attrs=lambda fs:[(r['path'],r['bytes'],r['mtime_ns']) for r in fs]
    if attrs(model_identity(Path(m['model'])))!=attrs(m['model_files']):raise ValueError('Base model changed')
    for name,h in m['inputs'].items():
        if digest(job.RUN/'inputs'/name)!=h:raise ValueError('Input snapshot changed '+name)
    return m

def prepare(job):
    from experiments.pvcd_cycle_v1.prepare import stream_hash
    from experiments.pvcd_cycle_v1.common import environment_identity
    from experiments.feedback_active_v3.check_local_model import check
    from qihc.generic.solver_baselines.registry import solve
    from qihc.generic.feedback.search import sample
    a,b=job.RUN.resolve(),job.PREVIOUS.resolve()
    if a==b or a in b.parents or b in a.parents:raise ValueError('Use a separate experiment result directory')
    if not (job.RUN/'manifest.json').exists() and any((job.RUN/s).exists() for s in ('pilot','screen')):raise ValueError('Existing unbound outputs')
    previous=load(job.PREVIOUS/'manifest.json')
    if previous['experiment_id']!='E005-1.9':raise ValueError('Expected frozen E005-1.9 inputs')
    inputs={name:load(job.PREVIOUS/'inputs'/name) for name in ('train.json','validation.json','cohort.json')}
    for name in inputs:
        path=job.PREVIOUS/'inputs'/name
        if digest(path)!=previous['input_files']['inputs/'+name]:raise ValueError('Previous input changed')
    if [len(inputs[n]) for n in inputs]!=[90,24,910]:raise ValueError('Wrong cohort sizes')
    from collections import Counter
    cohort=inputs['cohort.json']
    if Counter(c['stage'] for c in cohort)!=dict(development=120,main=480,scale=300,public=10):raise ValueError('Unexpected stage counts')
    development={c['id'] for c in cohort if c['stage']=='development'}
    train_ids={r['case']['id'] for r in inputs['train.json']};val_ids={r['case']['id'] for r in inputs['validation.json']}
    if train_ids & val_ids or not (train_ids|val_ids)<=development:raise ValueError('Invalid train/validation split')
    if len({c['id'] for c in cohort})!=910:raise ValueError('Duplicate cohort identifiers')
    adapter=Path(previous['history'])/'formal/models'/str(job.CFG['seed'])/'sft/adapter.pt'
    frozen={str(adapter):previous['donor_files'][str(adapter)]}
    predictions={}
    for case in inputs['cohort.json']:
        path=adapter.parent/'predictions'/(case['id']+'.json')
        if str(path) not in previous['donor_predictions']:raise ValueError('Missing seed17 SFT baseline '+str(path))
        predictions[case['id']]=str(path);frozen[str(path)]=previous['donor_files'][str(path)]
    gold_paths=[p for p in previous['data_files'] if p.replace('\\','/').endswith(('/evaluation/cases.gold.json','/evaluation/cases.public_external.gold.json'))]
    if len(gold_paths)!=2:raise ValueError('Missing frozen evaluation data')
    for p in gold_paths:frozen[p]=previous['data_files'][p]
    for path,h in frozen.items():
        if stream_hash(Path(path))!=h:raise ValueError('Historical dependency changed '+path)
    model=Path(previous['model']);check(model);files=model_identity(model)
    for row in files:
        if row['sha256'] is None:row['sha256']=stream_hash(model/row['path'])
    if files!=previous['fingerprints']:raise ValueError('Base model changed')
    gold=inputs['train.json'][0]['gold']
    if solve(gold,'scip',2,17).get('x') is None:raise RuntimeError('SCIP preflight failed')
    sample(gold,.1,17,chains=4,read_q=False)
    repo=job.HERE.parents[1];source=source_hashes(job)
    changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=repo,text=True).splitlines()
    changed+=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=repo,text=True).splitlines()
    if any(p in source for p in changed):raise ValueError('Commit code before preparing')
    manifest=dict(experiment_id=job.CFG['experiment_id'],config=job.CFG,source=source,environment=environment_identity(),
        code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        previous_manifest_sha256=digest(job.PREVIOUS/'manifest.json'),dependencies=frozen,
        model=str(model),model_files=files,adapter=str(adapter),predictions=predictions,gold_paths=gold_paths,
        inputs={name:digest(job.PREVIOUS/'inputs'/name) for name in inputs})
    # Input hashes use the exact frozen bytes, not a second JSON serialization.
    with worker_lock(job.RUN/'prepare.lock'):
        path=job.RUN/'manifest.json'
        if path.exists() and load(path)!=manifest:raise ValueError('Run identity changed')
        for name in inputs:
            dest=job.RUN/'inputs'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            raw=(job.PREVIOUS/'inputs'/name).read_bytes()
            if dest.exists() and dest.read_bytes()!=raw:raise ValueError('Snapshot changed')
            if not dest.exists():dest.write_bytes(raw)
        save(path,manifest)
    print('VERIFIED_SCREEN_READY',job.RUN,manifest['code_sha'],flush=True)

def worker(job,mode,arm):
    if mode not in ('pilot','screen') or arm not in job.CFG['arms']:raise ValueError('Unsupported worker task')
    import torch
    from experiments.posttraining_v1.engine import Engine
    from experiments.posttraining_v1.learn import put_checkpoint,parsed
    from experiments.feedback_active_v3.runtime import seed_for
    from qihc.modeling.active_protocol_v2 import model_prompt
    cfg=job.CFG;seed=cfg['seed'];m=load(job.RUN/'manifest.json');out=job.RUN/mode/'models'/str(seed)/arm
    with worker_lock(out/'lock'):
        engine=None
        if arm!='sft':
            torch.manual_seed(seed);engine=Engine(Path(m['model']))
            reference=torch.load(m['adapter'],map_location='cpu',weights_only=True);engine.restore(reference)
            final=out/'adapter.pt';ck=out/'checkpoint.pt'
            if not final.exists():
                optimizer=torch.optim.AdamW(engine.params.values(),lr=cfg['learning_rate']);start=0
                if ck.exists():
                    state=torch.load(ck,map_location='cpu',weights_only=True)
                    engine.restore(state['weights']);optimizer.load_state_dict(state['optimizer']);start=state['step']
                rows=train_list(job,mode);random.Random(seed).shuffle(rows)
                for i in range(start,len(rows)):
                    began=time.monotonic();optimizer.zero_grad(set_to_none=True)
                    info=job.step(engine,arm,rows[i],seed_for(seed,i),reference,cfg)
                    norm=torch.nn.utils.clip_grad_norm_(list(engine.params.values()),1.,error_if_nonfinite=True);optimizer.step()
                    save(out/'trace'/f'{i:04}.json',info)
                    save(out/'steps'/f'{i:04}.json',dict(step=i+1,seconds=time.monotonic()-began,gradient_norm=float(norm),gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0))
                    if (i+1)%cfg['checkpoint_steps']==0 or i+1==len(rows):put_checkpoint(ck,dict(weights=engine.snapshot(),optimizer=optimizer.state_dict(),step=i+1))
                    print(arm,i+1,len(rows),'positives',info['positive_used'],'repairs',info['verified_repairs'],flush=True)
                weights=engine.snapshot()
                if all(torch.equal(weights[k],reference[k]) for k in weights):raise RuntimeError('No adapter update')
                put_checkpoint(final,weights)
            engine.restore(torch.load(final,map_location='cpu',weights_only=True))
        for case in case_list(job,mode):
            path=out/'predictions'/(case['id']+'.json')
            if path.exists():continue
            if arm=='sft':result=load(Path(m['predictions'][case['id']]))
            else:
                response=engine.evaluate(model_prompt(case),seed_for(case['id'],0,'initial'));c=response['choices'][0]
                text=c['message']['content'];model,error=parsed(text,case) if c['finish_reason']=='stop' else (None,'length')
                result=dict(model=model,error=error,raw=text,usage=response.get('usage'))
            save(path,dict(result,case_id=case['id'],stage=case['stage']))
        save(out/'done.json',dict(count=len(case_list(job,mode))))

def score_one(args):
    run,cfg,path,gold_paths=args;run=Path(run);path=Path(path)
    from experiments.posttraining_v1.evaluate import worker_lock_wait
    from experiments.feedback_active_v3.cpu import execute
    from experiments.feedback_active_v3.runtime import seed_for
    from qihc.generic.feedback.verification import equivalent,score
    cache_key=tuple(gold_paths)
    if cache_key not in _GOLD_CACHE:_GOLD_CACHE[cache_key]={r['id']:r for p in gold_paths for r in load(Path(p))}
    record=_GOLD_CACHE[cache_key][path.stem]
    gold=record['gold'];p=load(path);cid=p['case_id'];model=p['model']
    cache=run/'score_cache';ref_path=cache/'references'/(cid+'.json')
    with worker_lock_wait(ref_path):
        if not ref_path.exists():save(ref_path,execute(dict(stage='reference',gold=gold,seconds=120,seed=seed_for(cid,'reference'))))
        ref=load(ref_path)
        if ref.get('status')=='execution_error':raise RuntimeError('Reference solver failed: '+str(ref))
    key=cache/'models'/(value_digest([cid,model,cfg['solver_seconds']])+'.json')
    with worker_lock_wait(key):
        if not key.exists():
            cross={}
            for backend in ('scip','pbit'):
                row=execute(dict(stage='solve',model=model,backend=backend,seconds=cfg['solver_seconds'],chains=cfg['chains'],seed=seed_for(cid,'terminal')))
                if row.get('status')=='execution_error':raise RuntimeError(str(row))
                cross[backend]=dict(row,score=score(gold,row,ref))
            save(key,dict(semantic=equivalent(model,gold),cross=cross))
        result=load(key)
    save(path.parent.parent/'evaluations'/path.name,dict(result,case_id=cid,stage=p['stage'],family=record.get('family','public'),n_variables=len(gold['variables'])))

def score_run(job,mode):
    m=load(job.RUN/'manifest.json');folder=job.RUN/mode
    paths=[p for p in folder.glob('models/*/*/predictions/*.json') if not (p.parent.parent/'evaluations'/p.name).exists()]
    with worker_lock(folder/'score.lock'),ProcessPoolExecutor(max_workers=job.CFG['cpu_workers']) as pool:
        list(pool.map(score_one,[(str(job.RUN),job.CFG,str(p),m['gold_paths']) for p in paths]))
    result=job.report(mode)
    if not result['complete']:raise RuntimeError('Evaluation coverage incomplete; inspect report before resuming')

def launch(job,mode):
    import torch
    from experiments.pvcd_cycle_v1.launch import stop_process
    if mode not in ('pilot','screen'):raise ValueError('Only pilot or single-seed screen supported')
    validate(job)
    if mode=='screen':
        job.report('pilot')
        if not load(job.RUN/'pilot/gate.json')['passed']:raise RuntimeError('Pilot execution/positive-signal gate failed')
    if torch.cuda.device_count()!=4:raise RuntimeError('Allocate four GPUs')
    devices=[]
    for i in range(4):
        with torch.cuda.device(i):
            p=torch.cuda.get_device_properties(i)
            if p.total_memory<75*1024**3 or not torch.cuda.is_bf16_supported():raise RuntimeError('Requires BF16 80GB class GPUs')
            x=torch.ones((64,64),device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True)
            (x@x).float().mean().backward();torch.cuda.synchronize(i)
            devices.append(dict(name=p.name,bytes=p.total_memory))
    try:driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=15).strip()
    except (OSError,subprocess.SubprocessError):driver='unavailable; CUDA BF16 runtime check passed'
    save(job.RUN/'gpu_preflight.json',dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda,driver=driver))
    ids=os.getenv('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    if len(ids)!=4 or len(set(ids))!=4:raise ValueError('Ambiguous GPU assignment')
    live=[]
    with worker_lock(job.RUN/'launcher.lock'):
        try:
            for gpu,arm in zip(ids,job.CFG['arms']):
                if (job.RUN/mode/'models'/str(job.CFG['seed'])/arm/'done.json').exists():continue
                path=job.RUN/mode/'logs'/(arm+'.log');path.parent.mkdir(parents=True,exist_ok=True);log=path.open('a')
                proc=subprocess.Popen([sys.executable,'-m',job.__name__,'worker',mode,arm],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                live.append((proc,log))
            while any(p.poll() is None for p,l in live):
                if any(p.poll() not in (None,0) for p,l in live):raise RuntimeError('GPU worker failed; inspect logs')
                time.sleep(2)
            if any(p.returncode for p,l in live):raise RuntimeError('GPU worker failed')
        finally:
            for p,l in live:stop_process(p);l.close()
        score_run(job,mode)
