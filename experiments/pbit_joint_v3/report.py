"""Paired comparisons with original-feasibility scoring and intervention accounting."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from experiments.pbit_joint_v2.report import summarize
from qihc.generic.feedback.verification import equivalent,score
from qihc.experiment_runtime.storage import load,save,worker_lock
from . import run as r


def score_case(task):
    case,gold=task
    cid=case['id']
    path=r.RUN/'scores'/f'{cid}.json'
    if path.exists():
        return
    ref=load(r.RUN/'references'/f'{cid}.json')
    cells={}
    for model in r.MODELS:
        pred=load(r.RUN/'predictions'/model/f'{cid}.json')
        spec=pred['model']
        semantic=equivalent(spec,gold)['status']
        for policy in r.POLICIES:
            name=f'{model}__{policy}'
            if policy=='schedule':
                ep=r.previous.trajectory(spec,cid,'schedule')
            else:
                ep=load(r.RUN/'episodes'/name/f'{cid}.json')
            cells[name]=dict(score=score(gold,ep,ref),semantic=semantic,invalid=spec is None,
                actions=ep['actions'],episode_seconds=ep['episode_seconds'],decision_seconds=ep['decision_seconds'],
                attempts=ep['heat_bath_attempts'],writes=ep['intervention_writes'])
    save(path,dict(id=cid,stage=case['stage'],cells=cells))


def contrasts(rows):
    def hit(cell):
        return np.array([int(row['cells'][cell]['score']['optimal_hit']) for row in rows])
    b,m,c,j=map(hit,('weighted__schedule','anchor__schedule','weighted__rank','anchor__rank'))
    out={}
    for name,d in dict(model=m-b,control=c-b,joint=j-b,interaction=j-m-c+b).items():
        draws=np.random.default_rng(20261115).choice(d,(5000,len(d))).mean(axis=1)
        out[name]=dict(delta=float(d.mean()),ci=np.quantile(draws,[.025/4,1-.025/4]).tolist())
    return out


def intervention_results(rows):
    out={}
    for model in r.MODELS:
        for policy in r.POLICIES[1:]:
            counts=Counter(dict(valid_cases=0,intervened_cases=0,improved=0,worse=0,tied=0,
                                gained_optimum=0,lost_optimum=0,actual_writes=0))
            for row in rows:
                cells=row['cells']
                ep=cells[f'{model}__{policy}']
                controls=ep['actions']
                counts['valid_cases']+=int(bool(controls))
                if all(a=='cool' for a in controls):
                    continue
                counts['intervened_cases']+=1
                counts['actual_writes']+=ep['writes']
                base=cells[f'{model}__schedule']['score']
                new=ep['score']
                def quality(s):
                    # Both scores share reference and direction-normalized gap.
                    return (s['original_feasible'],-s['gap'] if s['gap'] is not None else 0.)
                diff=int(new['optimal_hit'])-int(base['optimal_hit'])
                counts['gained_optimum']+=int(diff>0)
                counts['lost_optimum']+=int(diff<0)
                a,b=quality(new),quality(base)
                counts['improved' if a>b else 'worse' if a<b else 'tied']+=1
            out[f'{model}__{policy}']=dict(counts)
    return out


def report():
    rows=[load(p) for c in r.cases() if (p:=r.RUN/'scores'/(c['id']+'.json')).exists()]
    result=dict(experiment=r.CFG['experiment_id'],cases=len(rows),evaluations=len(rows)*9,
                expected=7020,shards=sum(r.marker(t).exists() for t in r.subtasks()),summary=summarize(rows))
    if (r.RUN/'ready.json').exists():
        ready=load(r.RUN/'ready.json')
        result['supervision']=ready['diagnostics']
        result['training']={}
        for a in r.ARMS:
            n=ready['diagnostics'][a]['examples'] if a in r.POLICIES else len(load(r.RUN/'inputs/train.json'))
            result['training'][a]=r.common.training_status(r.RUN/'trained'/a,n*r.CFG['epochs'])
    if len(rows)==780:
        result['effects']=contrasts(rows)
        result['interventions']=intervention_results(rows)
        result['baseline_reproduction']=dict(expected_hits=286,observed_hits=sum(
            bool(x['cells']['weighted__schedule']['score']['optimal_hit']) for x in rows))
    save(r.RUN/'report.json',result)
    print(result,flush=True)


def score_all():
    manifest=r.runtime.validate(r.job())
    if not all(r.marker(t).exists() for t in r.subtasks()):
        raise RuntimeError('GPU stage incomplete')
    gold={row['id']:row['gold'] for p in manifest['gold_paths'] for row in load(Path(p))}
    with worker_lock(r.RUN/'score.lock'),ProcessPoolExecutor(max_workers=8) as pool:
        for i,_ in enumerate(pool.map(score_case,[(c,gold[c['id']]) for c in r.cases()]),1):
            if i%20==0: print('scored',i,flush=True)
        report()
