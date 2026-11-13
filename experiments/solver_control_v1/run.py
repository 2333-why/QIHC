"""E005-1.11: offline branching data, categorical LLM control, sequential evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from qihc.experiment_runtime.storage import load, save, digest, worker_lock, value_digest
from qihc.experiment_runtime.pbit_control import ACTIONS, initial, advance, preference, prompt, rule, seed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = Path('/inspire/hdd/global_user/czxs25210022/why')
RUN = Path(os.getenv('QIHC_CONTROL_RUN', BASE/'results/E005-1.11_controller'))
PREVIOUS = Path(os.getenv('QIHC_CYCLE_RUN', BASE/'results/E005-1.9_pvcd_cycle'))
CFG = load(Path(os.getenv('QIHC_CONTROL_CONFIG', HERE/'config.json')))


def sources():
    paths = list((REPO/'qihc').rglob('*.py'))
    for folder in ('posttraining_v1', 'feedback_active_v3', 'pvcd_cycle_v1', 'solver_control_v1'):
        paths += [p for p in (REPO/'experiments'/folder).iterdir() if p.suffix in ('.py', '.json', '.sh', '.txt')]
    return {p.relative_to(REPO).as_posix():digest(p) for p in sorted(paths)}


def validate(manifest=None):
    from experiments.pvcd_cycle_v1.common import environment_identity
    from qihc.experiment_runtime.identity import model_identity
    m = load(RUN/'manifest.json') if manifest is None else manifest
    if m['config'] != CFG or m['sources'] != sources(): raise ValueError('Code/config changed')
    if m['environment'] != environment_identity(): raise ValueError('Shared environment changed')
    attrs = lambda rows:[(r['path'],r['bytes'],r['mtime_ns']) for r in rows]
    if attrs(model_identity(Path(m['model']))) != attrs(m['model_files']): raise ValueError('Local model changed')
    for name, h in m['inputs'].items():
        if digest(RUN/'inputs'/name) != h: raise ValueError('Frozen input changed '+name)
    return m


def migrate_reference_seed_fix():
    """Accept only the published 1.12 launcher change; preserve all run artifacts."""
    import hashlib
    from contextlib import ExitStack
    original='7fb82f7f552f8918ca7a77ea844f4e697efc6fa0'
    key='experiments/solver_control_v1/run.py'
    with ExitStack() as locks:
        for path in [RUN/'prepare.lock',RUN/'launch.lock',RUN/'full/labels.lock']+[
                RUN/'full/models'/arm/'worker.lock' for arm in CFG['arms']]:
            locks.enter_context(worker_lock(path))
        m=load(RUN/'manifest.json'); current=sources()
        if m['sources']==current:
            validate();print('Already current; existing run verified');return
        if m['experiment_id']!='E005-1.12' or m['config']!=CFG:
            raise ValueError('Migration supports only unchanged E005-1.12 configuration')
        old=hashlib.sha256(subprocess.check_output(['git','show',original+':'+key],cwd=REPO)).hexdigest()
        if m['sources'].get(key)!=old or set(m['sources'])!=set(current):
            raise ValueError('Not the published E005-1.12 source layout')
        if any(m['sources'][p]!=h for p,h in current.items() if p!=key):
            raise ValueError('Unrelated source changed; refusing migration')
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        committed=hashlib.sha256(subprocess.check_output(['git','show',head+':'+key],cwd=REPO)).hexdigest()
        if current[key]!=committed:raise ValueError('Commit launcher fix first')
        updated=dict(m,sources=current,code_sha=head,
            reference_seed_fix=dict(previous_code_sha=m['code_sha'],mapping='uint32 & 0x7fffffff'))
        validate(updated)  # Check environment/model/inputs before mutating the manifest.
        backup=RUN/'manifest.before_reference_seed_fix.json'
        if backup.exists() and load(backup)!=m:raise ValueError('Conflicting migration backup')
        if not backup.exists():save(backup,m)
        save(RUN/'manifest.json',updated)
        print('Reference seed fix bound; labels, checkpoints and evaluations preserved',flush=True)


def prepare():
    from experiments.pvcd_cycle_v1.common import environment_identity
    from experiments.pvcd_cycle_v1.prepare import stream_hash
    from experiments.feedback_active_v3.check_local_model import check
    from qihc.experiment_runtime.identity import model_identity
    a, b = RUN.resolve(), PREVIOUS.resolve()
    if a == b or a in b.parents or b in a.parents: raise ValueError('Use separate result directories')
    if (RUN/'manifest.json').exists():
        validate(); print('Existing frozen run verified', RUN); return
    if RUN.exists() and any(p.name not in ('inputs','prepare.lock') for p in RUN.iterdir()):
        raise ValueError('Unbound nonempty run directory')
    previous = load(PREVIOUS/'manifest.json')
    if previous['experiment_id'] != 'E005-1.9': raise ValueError('Expected E005-1.9 frozen split')
    inputs = {}
    for name in ('train.json', 'validation.json', 'cohort.json'):
        p = PREVIOUS/'inputs'/name
        if digest(p) != previous['input_files']['inputs/'+name]: raise ValueError('Previous input changed')
        inputs[name] = load(p)
    train, val, cohort = (inputs[n] for n in ('train.json','validation.json','cohort.json'))
    if (len(train),len(val),len(cohort)) != (90,24,910): raise ValueError('Unexpected split sizes')
    tids = {r['case']['id'] for r in train}; vids = {r['case']['id'] for r in val}
    test = [c for c in cohort if c['stage'] in ('main','scale')]
    eids = {c['id'] for c in test}
    if len(tids)!=90 or len(vids)!=24 or len(eids)!=780 or tids & vids or (tids|vids) & eids:
        raise ValueError('Invalid or overlapping split')
    gold = {}
    for p, h in previous['data_files'].items():
        if p.replace('\\','/').endswith('/evaluation/cases.gold.json'):
            if digest(Path(p)) != h: raise ValueError('Evaluation source changed')
            gold.update({r['id']:r['gold'] for r in load(Path(p))})
    # GOLD is the actual solver input in this track, not hidden online feedback.
    inputs = {'train.json':train, 'validation.json':val,
              'test.json':[dict(case=c,gold=gold[c['id']]) for c in test]}
    if sum(r['case']['stage']=='main' for r in inputs['test.json']) != 480: raise ValueError('Wrong test stages')
    model = Path(previous['model']); check(model); files = model_identity(model)
    for f in files:
        if f['sha256'] is None: f['sha256'] = stream_hash(model/f['path'])
    if files != previous['fingerprints']: raise ValueError('Model fingerprint changed')
    code = sources()
    dirty = subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=REPO,text=True).splitlines()
    dirty += subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=REPO,text=True).splitlines()
    if any(p in code for p in dirty): raise ValueError('Commit experiment code first')
    with worker_lock(RUN/'prepare.lock'):
        for name, rows in inputs.items():
            path=RUN/'inputs'/name
            if path.exists() and load(path)!=rows:raise ValueError('Interrupted preparation inputs changed')
            save(path, rows)
        save(RUN/'manifest.json',dict(experiment_id=CFG['experiment_id'], config=CFG,
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            sources=code, environment=environment_identity(), model=str(model), model_files=files,
            previous_manifest_sha256=digest(PREVIOUS/'manifest.json'),
            inputs={n:digest(RUN/'inputs'/n) for n in inputs}))
    print('CONTROL_READY', RUN, flush=True)


def rows(mode, training=False):
    data = load(RUN/'inputs'/('train.json' if training else 'validation.json' if mode=='pilot' else 'test.json'))
    if mode=='pilot' and training:
        data = sorted(data, key=lambda r:seed(r['case']['id'],'controller-pilot'))[:CFG['pilot_train_cases']]
    return data


def label_one(job):
    row, mode = job; cid = row['case']['id']; spec = row['gold']
    path = RUN/mode/'labels'/(cid+'.json')
    with worker_lock(path.with_suffix('.lock')):
        if path.exists(): return load(path)
        began = time.monotonic()
        examples = []
        temperatures = CFG.get('training_temperatures',[.5])
        for trajectory,temperature in enumerate(temperatures):
            # Preserve the original trajectory's RNG; extra trajectories use independent seeds.
            def draw_seed(*parts):
                return seed(*parts) if trajectory==0 else seed(*parts,'trajectory',trajectory)
            state = initial(spec, draw_seed(cid,'train-init'), CFG['chains'])
            state['temperature'] = temperature
            state = advance(spec,state,'hold',draw_seed(cid,'train-warmup'),CFG['window_steps'])
            for decision in range(CFG['decisions']):
                label = preference(spec,state,draw_seed(cid,decision,'label'),CFG['window_steps'],
                                   CFG['selection_repeats'],CFG['confirmation_repeats'])
                examples.append(dict(trajectory=trajectory,decision=decision,state=copy.deepcopy(state),pair=label))
                # Predetermined behavior; neither test data nor selected labels steer it.
                action = ACTIONS[draw_seed(cid,decision,'behavior') % len(ACTIONS)]
                state = advance(spec,state,action,draw_seed(cid,decision,'behavior-draw'),CFG['window_steps'])
        out = dict(case_id=cid,examples=examples,seconds=time.monotonic()-began)
        save(path,out); print('labels',mode,cid,sum(e['pair']['accepted'] for e in examples),flush=True)
        return out


def label_gate(mode):
    accepted = []; complete = True; states=0; labeled_cases=0
    for row in rows(mode,True):
        p = RUN/mode/'labels'/(row['case']['id']+'.json')
        if not p.exists(): complete=False; continue
        examples=load(p)['examples'];states+=len(examples);labeled_cases+=1
        accepted += [(row['case']['id'], e['pair']['chosen']) for e in examples if e['pair']['accepted']]
    result = dict(complete=complete,pairs=len(accepted),cases=len({c for c,a in accepted}),
                  nonhold=sum(a!='hold' for c,a in accepted),actions=sorted({a for c,a in accepted}))
    result.update(labeled_cases=labeled_cases,expected_cases=len(rows(mode,True)),candidate_states=states,
                  accepted_fraction=len(accepted)/states if states else 0.,
                  action_counts={a:sum(chosen==a for _,chosen in accepted) for a in ACTIONS})
    result['passed'] = complete and result['cases']>=3 and result['nonhold']>=1 and len(result['actions'])>=2
    save(RUN/mode/'label_gate.json',result)
    return result


def make_labels(mode):
    validate()
    if mode=='screen' and not report('pilot')['gate']['passed']: raise RuntimeError('Pilot gate failed')
    with worker_lock(RUN/mode/'labels.lock'), ProcessPoolExecutor(max_workers=CFG['cpu_workers']) as pool:
        list(pool.map(label_one,[(r,mode) for r in rows(mode,True)]))
    gate=label_gate(mode); print(gate,flush=True)
    if not gate['passed']: raise RuntimeError('Insufficient diverse confirmed training signal; inspect labels, do not bypass gate')


def action_ids(engine):
    ids = [engine.encode(c) for c in 'ABCDE']
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=5:
        raise ValueError('Tokenizer must encode A..E as five distinct single tokens')
    return [x[0] for x in ids]


def policy_logits(engine, text, ids):
    import torch
    from torch.nn import functional as F
    # Hidden state before the single output token. Exact categorical action policy;
    # no unconstrained JSON generation or silent parser fallback.
    h = engine.hidden(text,[ids[0]])[0]
    index = torch.tensor(ids,device=h.device)
    head = engine.net.lm_head
    bias = head.bias[index] if getattr(head,'bias',None) is not None else None
    return F.linear(h.to(head.weight.dtype),head.weight[index],bias).float()


def training_examples(mode):
    result=[]
    for row in rows(mode,True):
        cid=row['case']['id']
        for i,e in enumerate(load(RUN/mode/'labels'/(cid+'.json'))['examples']):
            if e['pair']['accepted']: result.append((cid,i,row['gold'],e))
    return result


def train(engine,mode,arm):
    import torch
    from torch.nn import functional as F
    from experiments.posttraining_v1.learn import put_checkpoint
    out=RUN/mode/'models'/arm
    final=out/'adapter.pt'; ck=out/'checkpoint.pt'
    ids=action_ids(engine); examples=training_examples(mode)
    label_hash=value_digest(examples)
    parent=RUN/mode/'models/trajectory_sft/adapter.pt'
    parent_hash=digest(parent) if arm=='trajectory_dpo' else None
    binding=dict(labels=label_hash,parent=parent_hash)
    if (out/'binding.json').exists() and load(out/'binding.json')!=binding: raise ValueError('Training inputs changed')
    save(out/'binding.json',binding)
    if final.exists():
        engine.restore(torch.load(final,map_location='cpu',weights_only=True)); return
    if arm=='trajectory_dpo': engine.restore(torch.load(parent,map_location='cpu',weights_only=True))
    ref=None
    if arm=='trajectory_dpo':
        ref={};engine.net.eval()
        with torch.no_grad():
            for cid,i,spec,e in examples:
                lp=policy_logits(engine,prompt(spec,e['state'],True),ids).log_softmax(-1)
                c,r=(ACTIONS.index(e['pair'][k]) for k in ('chosen','rejected'))
                ref[cid,i]=float(lp[c]-lp[r])
    optimizer=torch.optim.AdamW(engine.params.values(),lr=CFG['dpo_learning_rate'] if ref is not None else CFG['learning_rate'])
    order=[]
    for epoch in range(CFG['dpo_epochs'] if ref is not None else CFG['sft_epochs']):
        local=list(range(len(examples)));random.Random(CFG['seed']+epoch).shuffle(local);order+=local
    start=0
    if ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=True)
        engine.restore(state['weights']);optimizer.load_state_dict(state['optimizer']);start=state['step']
    for step in range(start,len(order)):
        began=time.monotonic();cid,i,spec,e=examples[order[step]]
        optimizer.zero_grad(set_to_none=True);engine.net.train()
        logits=policy_logits(engine,prompt(spec,e['state'],arm!='best_sft'),ids)
        c,r=(ACTIONS.index(e['pair'][k]) for k in ('chosen','rejected'))
        if ref is None: loss=F.cross_entropy(logits[None],torch.tensor([c],device=logits.device))
        else:
            lp=logits.log_softmax(-1);loss=-F.logsigmoid(CFG['beta']*(lp[c]-lp[r]-ref[cid,i]))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite controller loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(engine.params.values()),1.,error_if_nonfinite=True)
        optimizer.step()
        save(out/'steps'/f'{step:05}.json',dict(case_id=cid,decision=i,loss=float(loss.detach()),gradient_norm=float(norm),seconds=time.monotonic()-began))
        if (step+1)%CFG['checkpoint_steps']==0 or step+1==len(order):
            put_checkpoint(ck,dict(weights=engine.snapshot(),optimizer=optimizer.state_dict(),step=step+1))
        print(arm,step+1,len(order),float(loss.detach()),flush=True)
    put_checkpoint(final,engine.snapshot())


def episode(spec,cid,arm,engine=None):
    began=time.monotonic();search_seconds=0.;decision_seconds=0.;decisions=[]
    tick=time.monotonic();state=initial(spec,seed(cid,CFG['seed'],'eval-init'),CFG['chains'])
    state=advance(spec,state,'hold',seed(cid,CFG['seed'],'eval-warmup'),CFG['window_steps'])
    search_seconds+=time.monotonic()-tick
    ids=action_ids(engine) if engine is not None else None
    for i in range(CFG['decisions']):
        tick=time.monotonic();error=None
        if arm=='fixed': action='hold'
        elif arm=='schedule': action='cool'
        elif arm=='rule': action=rule(state)
        elif arm=='random': action=ACTIONS[seed(cid,CFG['seed'],i,'random-action')%len(ACTIONS)]
        else:
            import torch
            engine.net.eval()
            with torch.no_grad(): action=ACTIONS[int(policy_logits(engine,prompt(spec,state,arm!='best_sft'),ids).argmax())]
        decision_seconds+=time.monotonic()-tick
        decisions.append(action);tick=time.monotonic()
        state=advance(spec,state,action,seed(cid,CFG['seed'],i,'eval-draw'),CFG['window_steps'])
        search_seconds+=time.monotonic()-tick
    return dict(best=state['best'],history=state['history'],actions=decisions,
                bit_updates=state['steps']*CFG['chains'],search_seconds=search_seconds,
                decision_seconds=decision_seconds,episode_seconds=time.monotonic()-began)


def evaluate(mode,arm,engine=None):
    from experiments.feedback_active_v3.cpu import execute
    from qihc.generic.feedback.verification import score
    for row in rows(mode):
        cid=row['case']['id'];spec=row['gold'];path=RUN/mode/'evaluations'/arm/(cid+'.json')
        if path.exists(): continue
        result=episode(spec,cid,arm,engine)
        # Reference is read only AFTER all decisions, never placed in the prompt.
        rp=RUN/'references'/(cid+'.json')
        from experiments.posttraining_v1.evaluate import worker_lock_wait
        with worker_lock_wait(rp):
            # SCIP accepts a nonnegative signed 32-bit int. Keep all other RNG streams unchanged.
            if not rp.exists(): save(rp,execute(dict(stage='reference',gold=spec,seconds=120,seed=seed(cid,'reference') & 0x7fffffff)))
            reference=load(rp)
            if reference.get('status')=='execution_error':raise RuntimeError(str(reference))
        result.update(case_id=cid,stage='pilot' if mode=='pilot' else row['case']['stage'],
            score=score(spec,dict(x=result['best']['x']),reference),model_sha256=value_digest(spec))
        save(path,result);print('evaluation',mode,arm,cid,flush=True)


def worker(mode,arm):
    validate()
    if not label_gate(mode)['passed']:raise RuntimeError('Training signal gate failed')
    with worker_lock(RUN/mode/'models'/arm/'worker.lock'):
        engine=None
        if arm in CFG['trained_arms']:
            import torch
            from experiments.posttraining_v1.engine import Engine
            torch.manual_seed(CFG['seed']);engine=Engine(Path(load(RUN/'manifest.json')['model']))
            train(engine,mode,arm)
            if not training_complete(mode,arm):
                raise RuntimeError('Incomplete training or missing finite nonzero gradients; evaluation not started')
        evaluate(mode,arm,engine)
        save(RUN/mode/'models'/arm/'done.json',dict(cases=len(rows(mode))))


def training_complete(mode,arm):
    out=RUN/mode/'models'/arm
    count=len(training_examples(mode))*(CFG['dpo_epochs'] if arm=='trajectory_dpo' else CFG['sft_epochs'])
    paths=list((out/'steps').glob('*.json'))
    if not count or {p.stem for p in paths}!={f'{i:05}' for i in range(count)}:
        return False
    logs=[load(p) for p in paths]
    return bool((out/'adapter.pt').exists() and any(r['gradient_norm']>0 for r in logs)
                and all(np.isfinite(r['gradient_norm']) and np.isfinite(r['loss']) for r in logs))


def report(mode):
    summary={};records={};expected={r['case']['id'] for r in rows(mode)}
    lg=label_gate(mode)
    expected_stages={stage:sum(stage=='all' or r['case'].get('stage')==stage for r in rows(mode))
                     for stage in (('all','main','scale') if mode!='pilot' else ('all',))}
    for arm in CFG['arms']:
        records[arm]={p.stem:load(p) for p in (RUN/mode/'evaluations'/arm).glob('*.json') if p.stem in expected}
        summary[arm]={}
        for stage in expected_stages:
            rs=[r for r in records[arm].values() if stage=='all' or r['stage']==stage]
            if not rs:continue
            summary[arm][stage]=dict(n=len(rs),expected=expected_stages[stage],complete=len(rs)==expected_stages[stage],
                feasible=float(np.mean([r['score']['original_feasible'] for r in rs])),
                optimal=float(np.mean([r['score']['optimal_hit'] for r in rs])),reference_proven=sum(r['score']['reference_proven'] for r in rs),
                episode_seconds=float(np.mean([r['episode_seconds'] for r in rs])),
                search_seconds=float(np.mean([r['search_seconds'] for r in rs])),
                decision_seconds=float(np.mean([r['decision_seconds'] for r in rs])),
                nonhold=sum(a!='hold' for r in rs for a in r['actions']),
                actions={a:sum(r['actions'].count(a) for r in rs) for a in ACTIONS})
    complete=all(set(v)==expected for v in records.values())
    gradients={}
    for arm in CFG['trained_arms']:
        gradients[arm]=bool(lg.get('complete') and training_complete(mode,arm))
    gate=dict(passed=bool(complete and lg['passed'] and all(gradients.values())),complete=complete,labels=lg,gradients=gradients)
    comparisons={}
    if gate['passed']:
        pairs=[('trajectory_sft','best_sft'),('trajectory_dpo','trajectory_sft'),('trajectory_sft','rule')]
        for a,b in pairs:
            d=np.array([float(records[a][cid]['score']['optimal_hit'])-float(records[b][cid]['score']['optimal_hit']) for cid in sorted(expected)])
            rng=np.random.default_rng(20261112);boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(CFG['bootstrap_repeats'])])
            comparisons[a+'-'+b]=dict(delta=float(d.mean()),bonferroni_98_33_ci=np.quantile(boot,[.05/6,1-.05/6]).tolist(),n=len(d))
    result=dict(experiment_id=CFG['experiment_id'],mode=mode,track='fixed_correct_model',seed=CFG['seed'],
        expected_cases=len(expected),expected_evaluations=len(expected)*len(CFG['arms']),
        completed_evaluations=sum(len(v) for v in records.values()),summary=summary,gate=gate,comparisons=comparisons,
        caveat='Single-seed exploratory screen; equal bit updates, NOT equal wall time. 780-case total is case-weighted. Not comparable to LLM modeling accuracy.')
    save(RUN/mode/'report.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def launch(mode):
    import torch
    validate()
    if mode=='screen' and not report('pilot')['gate']['passed']:raise RuntimeError('Pilot gate failed')
    if not label_gate(mode)['passed']:raise RuntimeError('Prepare labels on CPU first')
    if torch.cuda.device_count()<2:raise RuntimeError('Allocate at least two 80GB GPUs')
    devices=[]
    for i in range(2):
        with torch.cuda.device(i):
            p=torch.cuda.get_device_properties(i)
            if p.total_memory<75*1024**3 or not torch.cuda.is_bf16_supported():raise RuntimeError('Requires BF16 80GB GPUs')
            x=torch.ones((64,64),device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True)
            (x@x).float().mean().backward();torch.cuda.synchronize(i)
            devices.append(dict(name=p.name,bytes=p.total_memory))
    save(RUN/mode/'gpu_preflight.json',dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda))
    ids=os.getenv('CUDA_VISIBLE_DEVICES',','.join(map(str,range(torch.cuda.device_count())))).split(',')
    from experiments.pvcd_cycle_v1.launch import stop_process
    def batch(assignments):
        live=[]
        def check_failed():
            failed=[(p,l,a,path) for p,l,a,path in live if p.poll() not in (None,0)]
            if failed:
                details=[]
                for p,log,arm,path in failed:
                    log.flush()
                    with path.open('rb') as f:
                        f.seek(0,2);f.seek(max(0,f.tell()-8192))
                        tail=f.read().decode('utf-8',errors='replace')
                    details.append(f'{arm}: exit={p.returncode}; log={path}\n{tail}')
                raise RuntimeError('Worker failed:\n'+'\n'.join(details))
        try:
            for arm,gpu in assignments:
                if (RUN/mode/'models'/arm/'done.json').exists():continue
                path=RUN/mode/'logs'/(arm+'.log');path.parent.mkdir(parents=True,exist_ok=True)
                log=path.open('a')
                proc=subprocess.Popen([sys.executable,'-m','experiments.solver_control_v1.run','worker',mode,arm],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                live.append((proc,log,arm,path))
            while any(p.poll() is None for p,l,a,path in live):
                check_failed()
                time.sleep(2)
            check_failed()
        finally:
            for p,l,a,path in live:stop_process(p);l.close()
    with worker_lock(RUN/'launch.lock'):
        batch([('best_sft',ids[0]),('trajectory_sft',ids[1])]+[(a,'') for a in CFG['arms'] if a not in CFG['trained_arms']])
        batch([('trajectory_dpo',ids[0])])
    if not report(mode)['gate']['passed']:raise RuntimeError('Incomplete run or missing gradients')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','labels','gpu','worker','report','migrate-reference-seed'])
    p.add_argument('mode',nargs='?',choices=['pilot','screen','full'],default='pilot');p.add_argument('arm',nargs='?',choices=CFG['arms'])
    a=p.parse_args()
    if a.command=='migrate-reference-seed':migrate_reference_seed_fix()
    elif a.command=='prepare':prepare()
    elif a.command=='labels':make_labels(a.mode)
    elif a.command=='gpu':launch(a.mode)
    elif a.command=='worker':
        if a.arm is None:p.error('worker requires arm')
        worker(a.mode,a.arm)
    else:validate();report(a.mode)


if __name__=='__main__':main()
