"""E005-1.13: isolated modeling/control training and their factorial evaluation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from qihc.experiment_runtime import posttrain_screen as runtime
from qihc.experiment_runtime.storage import load,save,digest,value_digest,worker_lock
from qihc.experiment_runtime.pbit_control import seed
from . import search,modeling

HERE=Path(__file__).resolve().parent
BASE=Path('/inspire/hdd/global_user/czxs25210022/why')
RUN=Path(os.getenv('QIHC_JOINT_RUN',BASE/'results/E005-1.13_joint'))
PREVIOUS=Path(os.getenv('QIHC_CYCLE_RUN',BASE/'results/E005-1.9_pvcd_cycle'))
CFG=load(HERE/'config.json')


def job():return importlib.import_module('experiments.pbit_joint_v1.run')
def validate():return runtime.validate(job())
def train_rows():return load(RUN/'inputs/train.json')
def test_cases():return [c for c in load(RUN/'inputs/cohort.json') if c['stage'] in ('main','scale')]
def prediction(arm,cid):return RUN/'models'/arm/'predictions'/(cid+'.json')


def prepare():
    runtime.prepare(job())
    m=validate()
    if len(test_cases())!=780:raise ValueError('Expected 780 test cases')
    with worker_lock(RUN/'cards.lock'):
        for row in train_rows():
            cid=row['case']['id'];path=RUN/'cards'/(cid+'.json')
            if path.exists():continue
            candidate=load(Path(m['predictions'][cid]))['model']
            save(path,dict(model=candidate,**modeling.collect(candidate,row['gold'],cid,CFG)))
    print('JOINT_READY: CPU evidence prepared; GPU all trains and evaluates all nine combinations',flush=True)


def optimize(engine,out,order,lr,backward,binding):
    import torch
    from experiments.posttraining_v1.learn import put_checkpoint
    if (out/'binding.json').exists() and load(out/'binding.json')!=binding:raise ValueError('Training binding changed')
    save(out/'binding.json',binding)
    final=out/'adapter.pt';checkpoint=out/'checkpoint.pt'
    if final.exists():
        if not training_status(out,len(order))['passed']:raise RuntimeError('Final adapter lacks complete training evidence')
        engine.restore(torch.load(final,map_location='cpu',weights_only=True));return
    if not order:raise ValueError('No training examples')
    optimizer=torch.optim.AdamW(engine.params.values(),lr=lr);start=0
    if checkpoint.exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        engine.restore(ck['weights']);optimizer.load_state_dict(ck['optimizer']);start=ck['step']
    for i in range(start,len(order)):
        began=time.monotonic();optimizer.zero_grad(set_to_none=True);engine.net.train()
        info=backward(order[i])
        norm=torch.nn.utils.clip_grad_norm_(list(engine.params.values()),1.,error_if_nonfinite=True)
        optimizer.step()
        save(out/'steps'/f'{i:05}.json',dict(info,gradient_norm=float(norm),seconds=time.monotonic()-began))
        if (i+1)%CFG['checkpoint_steps']==0 or i+1==len(order):
            put_checkpoint(checkpoint,dict(weights=engine.snapshot(),optimizer=optimizer.state_dict(),step=i+1))
        print(out.name,i+1,len(order),flush=True)
    if not any(load(p)['gradient_norm']>0 for p in (out/'steps').glob('*.json')):raise RuntimeError('No training gradients')
    put_checkpoint(final,engine.snapshot())


def training_status(out,expected):
    paths=list((out/'steps').glob('*.json'));rows=[load(p) for p in paths]
    return dict(expected_steps=expected,steps=len(rows),
        passed=bool(expected and {p.stem for p in paths}=={f'{i:05}' for i in range(expected)}
                    and (out/'adapter.pt').exists() and any(x['gradient_norm']>0 for x in rows)
                    and all(np.isfinite(x['gradient_norm']) and np.isfinite(x['loss']) for x in rows)),
        weighted_tokens=sum(x.get('weighted_tokens',0) for x in rows),
        weighted_cases=len({x['case_id'] for x in rows if x.get('weighted_tokens',0)>0}),
        seconds=sum(x.get('seconds',0) for x in rows))


def order_for(n,epochs):
    order=[]
    for ep in range(epochs):
        ids=list(range(n));random.Random(CFG['seed']+ep).shuffle(ids);order.extend(ids)
    return order


def model_worker(arm):
    m=validate();out=RUN/'models'/arm
    from qihc.modeling.active_protocol_v2 import model_prompt
    from experiments.posttraining_v1.learn import parsed
    from experiments.posttraining_v1.data import target
    with worker_lock(out/'worker.lock'):
        engine=None
        if arm!='base':
            import torch
            from experiments.posttraining_v1.engine import Engine
            torch.manual_seed(CFG['seed']);engine=Engine(Path(m['model']))
            engine.restore(torch.load(m['adapter'],map_location='cpu',weights_only=True))
            rows=train_rows();cards={r['case']['id']:load(RUN/'cards'/(r['case']['id']+'.json')) for r in rows}
            def backward(i):
                row=rows[i];cid=row['case']['id'];answer=target(row['gold']);e=cards[cid]
                weights=modeling.token_weights(engine.tok,answer,e['model'],row['gold'],e['cards']) if arm=='pbit' else [1.]*len(engine.encode(answer))
                ids=engine.encode(answer)+[engine.tok.eos_token_id];weights=weights+[1.]
                loss=engine.backward(model_prompt(row['case']),ids,'nll',weights=weights)
                return dict(case_id=cid,loss=loss,weighted_tokens=sum(w>1 for w in weights),tokens=len(ids))
            optimize(engine,out,order_for(len(rows),CFG['model_epochs']),CFG['model_lr'],backward,
                dict(parent=digest(Path(m['adapter'])),cards=value_digest(cards),arm=arm,config=CFG))
        cases={c['id']:c for c in test_cases()}
        cases.update({r['case']['id']:r['case'] for r in train_rows()})
        for cid,case in cases.items():
            path=prediction(arm,cid)
            if path.exists():continue
            if arm=='base':p=load(Path(m['predictions'][cid]))
            else:
                began=time.monotonic()
                response=engine.evaluate(model_prompt(case),seed(cid,'model') & 0x7fffffff)
                choice=response['choices'][0];text=choice['message']['content']
                model,error=parsed(text,case) if choice['finish_reason']=='stop' else (None,'length')
                p=dict(model=model,error=error,raw=text,usage=response.get('usage'),generation_seconds=time.monotonic()-began)
            save(path,dict(p,case_id=cid))
        save(out/'done.json',dict(predictions=len(cases)))


def control_labels_one(task):
    cid,spec=task;path=RUN/'control_labels'/(value_digest([cid,spec])+'.json')
    with worker_lock(path.with_suffix('.lock')):
        if path.exists():return load(path)
        examples=[]
        for t,temp in enumerate(CFG['training_temperatures']):
            state=search.initial(spec,seed(cid,t,'train'),CFG['chains']);state['temperature']=temp
            state=search.advance(spec,state,'cool',seed(cid,t,'warm'),CFG['window_steps'])
            if t in CFG.get('collapsed_training_trajectories',[]):state=search.collapse_training_state(spec,state)
            for i in range(CFG['decisions']):
                remaining=CFG['decisions']-i
                label=search.label(spec,state,seed(cid,t,i,'label'),remaining,CFG)
                examples.append(dict(state=state,remaining=remaining,label=label))
                action=search.ACTIONS[seed(cid,t,i,'behavior')%len(search.ACTIONS)]
                state=search.advance(spec,state,action,seed(cid,t,i,'behavior-draw'),CFG['window_steps'])
        result=dict(case_id=cid,model=spec,examples=examples)
        save(path,result);return result


def make_control_labels():
    validate();tasks={};invalid=0
    # Only training IDs. Both old and improved model distributions feed one shared controller.
    for row in train_rows():
        cid=row['case']['id']
        for arm in CFG['arms']:
            spec=load(prediction(arm,cid))['model']
            if spec is None:invalid+=1;continue
            tasks[value_digest([cid,spec])]=(cid,spec)
    with worker_lock(RUN/'control_labels.lock'),ProcessPoolExecutor(max_workers=CFG['cpu_workers']) as pool:
        list(pool.map(control_labels_one,tasks.values()))
    save(RUN/'control_label_index.json',dict(files=sorted(tasks),invalid_training_models=invalid))
    gate=control_gate();print(json.dumps(gate),flush=True)
    return gate


def control_examples():
    index=load(RUN/'control_label_index.json');result=[]
    for key in index['files']:
        row=load(RUN/'control_labels'/(key+'.json'))
        result.extend((row['case_id'],row['model'],e) for e in row['examples'])
    return result


def control_gate():
    examples=control_examples();counts=Counter(e['label']['action'] for _,_,e in examples)
    gain_cases={cid for cid,_,e in examples if e['label']['confirmed_gain']}
    result=dict(examples=len(examples),action_counts=dict(counts),gain_cases=len(gain_cases),
        passed=len(gain_cases)>=CFG['min_control_gain_cases'] and len(counts)>=2)
    save(RUN/'control_gate.json',result);return result


def action_ids(engine):
    ids=[engine.encode(c) for c in 'ABC']
    if any(len(x)!=1 for x in ids) or len({x[0] for x in ids})!=3:raise ValueError('ABC must be distinct single tokens')
    return [x[0] for x in ids]


def logits(engine,text,ids):
    import torch
    from torch.nn import functional as F
    h=engine.hidden(text,[ids[0]])[0];index=torch.tensor(ids,device=h.device);head=engine.net.lm_head
    return F.linear(h.to(head.weight.dtype),head.weight[index],head.bias[index] if head.bias is not None else None).float()


def controller_worker():
    import torch
    from experiments.posttraining_v1.engine import Engine
    m=validate();out=RUN/'controller'
    with worker_lock(out/'worker.lock'):
        if not control_gate()['passed']:raise RuntimeError('Control label gate failed')
        torch.manual_seed(CFG['seed']);engine=Engine(Path(m['model']));ids=action_ids(engine)
        rows=control_examples();counts=Counter(e['label']['action'] for _,_,e in rows);per_case=Counter(cid for cid,_,_ in rows)
        weights=[1/(per_case[cid]*np.sqrt(counts[e['label']['action']])) for cid,_,e in rows]
        weights=np.asarray(weights)/np.mean(weights)
        def backward(i):
            cid,spec,e=rows[i];target=search.ACTIONS.index(e['label']['action'])
            logit=logits(engine,search.prompt(spec,e['state'],e['remaining']),ids)
            loss=torch.nn.functional.cross_entropy(logit[None],torch.tensor([target],device=logit.device))*float(weights[i])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite control loss')
            loss.backward();return dict(case_id=cid,loss=float(loss.detach()),action=e['label']['action'])
        optimize(engine,out,order_for(len(rows),CFG['control_epochs']),CFG['control_lr'],backward,
                 dict(labels=value_digest(rows),config=CFG,initialization='fresh_base_seed17'))


def evaluate(arm,control):
    m=validate();engine=None;ids=None
    if all((RUN/'evaluations'/(arm+'__'+control)/(c['id']+'.json')).exists() for c in test_cases()):return
    if control=='learned':
        import torch
        from experiments.posttraining_v1.engine import Engine
        engine=Engine(Path(m['model']));engine.restore(torch.load(RUN/'controller/adapter.pt',map_location='cpu',weights_only=True))
        engine.net.eval();ids=action_ids(engine)
    def choose(spec,state,remaining):
        if control=='schedule':return 'cool'
        if control=='rule':return search.rule(spec,state)
        import torch
        with torch.no_grad():return search.ACTIONS[int(logits(engine,search.prompt(spec,state,remaining),ids).argmax())]
    gold={r['id']:r['gold'] for p in m['gold_paths'] for r in load(Path(p))}
    from experiments.feedback_active_v3.cpu import execute
    from experiments.posttraining_v1.evaluate import worker_lock_wait
    from qihc.generic.feedback.verification import score,equivalent
    out=RUN/'evaluations'/(arm+'__'+control)
    with worker_lock(out/'worker.lock'):
        for case in test_cases():
            cid=case['id'];path=out/(cid+'.json')
            if path.exists():continue
            pred=load(prediction(arm,cid));spec=pred['model']
            # No gold model, optimum or test outcome is passed to the policy/search.
            result=search.episode(spec,cid,CFG,choose) if spec is not None else dict(x=None,actions=[],episode_seconds=0,decision_seconds=0,search_seconds=0,heat_bath_attempts=0,intervention_writes=0)
            refpath=RUN/'references'/(cid+'.json')
            with worker_lock_wait(refpath):
                if not refpath.exists():save(refpath,execute(dict(stage='reference',gold=gold[cid],seconds=120,seed=seed(cid,'reference') & 0x7fffffff)))
                ref=load(refpath)
            if ref.get('status')=='execution_error':raise RuntimeError(str(ref))
            semantic_path=RUN/'semantic'/arm/(cid+'.json')
            with worker_lock_wait(semantic_path):
                if not semantic_path.exists():save(semantic_path,equivalent(spec,gold[cid]))
                semantic=load(semantic_path)
            save(path,dict(result,case_id=cid,stage=case['stage'],score=score(gold[cid],result,ref),
                semantic=semantic,invalid_model=spec is None,model_hash=value_digest(spec),generation_seconds=pred.get('generation_seconds')))
            print('evaluated',arm,control,cid,flush=True)


def report():
    expected={c['id'] for c in test_cases()};records={};summary={}
    for arm in CFG['arms']:
        for control in CFG['controls']:
            name=arm+'__'+control
            rs={p.stem:load(p) for p in (RUN/'evaluations'/name).glob('*.json') if p.stem in expected};records[name]=rs
            summary[name]={}
            for stage in ['all','main','scale']:
                xs=[r for r in rs.values() if stage=='all' or r['stage']==stage]
                if not xs:continue
                generated=[r['generation_seconds'] for r in xs if r.get('generation_seconds') is not None]
                gaps=[r['score']['gap'] for r in xs if r['score']['reference_proven'] and r['score'].get('gap') is not None]
                summary[name][stage]=dict(n=len(xs),expected=780 if stage=='all' else 480 if stage=='main' else 300,
                    feasible=float(np.mean([r['score']['original_feasible'] for r in xs])),
                    optimal=float(np.mean([r['score']['optimal_hit'] for r in xs])),
                    model_correct=float(np.mean([r['semantic']['status']=='verified_correct' for r in xs])),
                    model_unknown=float(np.mean([r['semantic']['status']=='unknown' for r in xs])),
                    invalid_models=sum(r['invalid_model'] for r in xs),reference_proven=sum(r['score']['reference_proven'] for r in xs),
                    proven_feasible_gap=float(np.mean(gaps)) if gaps else None,proven_feasible_gap_cases=len(gaps),
                    episode_seconds=float(np.mean([r['episode_seconds'] for r in xs])),
                    decision_seconds=float(np.mean([r['decision_seconds'] for r in xs])),
                    search_seconds=float(np.mean([r['search_seconds'] for r in xs])),
                    generation_seconds=float(np.mean(generated)) if generated else None,generation_timed_cases=len(generated),
                    heat_bath_attempts=sum(r['heat_bath_attempts'] for r in xs),intervention_writes=sum(r['intervention_writes'] for r in xs),
                    actions=dict(Counter(a for r in xs for a in r['actions'])))
    complete=all(set(v)==expected for v in records.values());comparisons={}
    training={a:training_status(RUN/'models'/a,len(train_rows())*CFG['model_epochs']) for a in ('plain','pbit')}
    gate=load(RUN/'control_gate.json') if (RUN/'control_gate.json').exists() else None
    training['controller']=training_status(RUN/'controller',gate['examples']*CFG['control_epochs'] if gate else 0)
    passed=bool(complete and gate and gate['passed'] and all(x['passed'] for x in training.values()))
    if passed:
        # Three principal effects and factorial interaction; familywise four comparisons.
        ids=sorted(expected)
        vectors={k:np.array([v[i]['score']['optimal_hit'] for i in ids],dtype=float) for k,v in records.items()}
        b=vectors['base__schedule'];m=vectors['pbit__schedule'];c=vectors['base__learned'];j=vectors['pbit__learned']
        for name,d in dict(model=m-b,control=c-b,joint=j-b,interaction=j-m-c+b).items():
            rng=np.random.default_rng(20261114)
            boot=[rng.choice(d,len(d),replace=True).mean() for _ in range(CFG['bootstrap_repeats'])]
            comparisons[name]=dict(delta=float(d.mean()),bonferroni_98_75_ci=np.quantile(boot,[.05/8,1-.05/8]).tolist())
    result=dict(experiment_id=CFG['experiment_id'],complete=complete,passed=passed,expected_evaluations=7020,
        completed_evaluations=sum(len(v) for v in records.values()),summary=summary,comparisons=comparisons,
        control_gate=gate,training=training,
        caveat='Single seed; equal heat-bath attempts, not equal wall time or intervention writes; test historically exposed.')
    save(RUN/'report.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return result


def batch(tasks,gpu_ids):
    from experiments.pvcd_cycle_v1.launch import stop_process
    for offset in range(0,len(tasks),len(gpu_ids)):
        live=[]
        try:
            for task,gpu in zip(tasks[offset:offset+len(gpu_ids)],gpu_ids):
                path=RUN/'logs'/('_'.join(task)+'.log');path.parent.mkdir(parents=True,exist_ok=True);log=path.open('a')
                proc=subprocess.Popen([sys.executable,'-m','experiments.pbit_joint_v1.run',*task],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                live.append((proc,log,path))
            while True:
                failed=[(p,path) for p,l,path in live if p.poll() not in (None,0)]
                if failed:
                    for p,path in failed:
                        with path.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-8192));print(f.read().decode(errors='replace'),flush=True)
                    raise RuntimeError('Workers failed: '+str([(p.returncode,str(path)) for p,path in failed]))
                if all(p.poll()==0 for p,l,path in live):break
                time.sleep(2)
        finally:
            for p,l,path in live:stop_process(p);l.close()


def launch():
    import torch
    validate()
    if torch.cuda.device_count()<2:raise RuntimeError('Allocate at least two BF16 80GB GPUs')
    devices=[]
    for i in range(2):
        with torch.cuda.device(i):
            p=torch.cuda.get_device_properties(i)
            if p.total_memory<75*1024**3 or not torch.cuda.is_bf16_supported():raise RuntimeError('Requires BF16 80GB GPUs')
            x=torch.ones((64,64),device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True)
            (x@x).float().mean().backward();torch.cuda.synchronize(i);devices.append(dict(name=p.name,bytes=p.total_memory))
    ids=os.getenv('CUDA_VISIBLE_DEVICES',','.join(map(str,range(torch.cuda.device_count())))).split(',')[:2]
    save(RUN/'gpu_preflight.json',dict(devices=devices,torch=torch.__version__,cuda=torch.version.cuda))
    with worker_lock(RUN/'launch.lock'):
        model_worker('base')
        batch([['model',a] for a in ('plain','pbit')],ids)
        gate=make_control_labels()
        # Reuse exactly the same generated model for each control, never regenerate per cell.
        batch([['evaluate',a,c] for a in CFG['arms'] for c in CFG['controls'] if c!='learned'],['',''])
        if not gate['passed']:
            report()
            raise RuntimeError('Non-cooling control signal insufficient; modeling comparisons saved; inspect control_gate.json')
        batch([['controller']],ids)
        batch([['evaluate',a,'learned'] for a in CFG['arms']],ids)
        if not report()['passed']:raise RuntimeError('Incomplete evaluation or training evidence')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','all','model','labels','controller','evaluate','report'])
    p.add_argument('arm',nargs='?',choices=CFG['arms']);p.add_argument('control',nargs='?',choices=CFG['controls']);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='all':launch()
    elif a.command=='model':
        if a.arm is None:p.error('model requires arm')
        model_worker(a.arm)
    elif a.command=='labels':
        if not make_control_labels()['passed']:raise RuntimeError('Control signal gate failed')
    elif a.command=='controller':controller_worker()
    elif a.command=='evaluate':
        if a.arm is None or a.control is None:p.error('evaluate requires arm and control')
        evaluate(a.arm,a.control)
    else:validate();report()


if __name__=='__main__':main()
