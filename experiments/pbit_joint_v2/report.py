"""Score all policies against the original problem using one reference per case."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import Counter
from pathlib import Path
import numpy as np
from experiments.feedback_active_v3.cpu import execute
from qihc.generic.feedback.verification import equivalent, score
from qihc.experiment_runtime.pbit_control import seed
from qihc.experiment_runtime.storage import load, save, worker_lock
from . import run as r


def score_case(task, predictions):
    case, gold = task
    cid = case['id']
    path = r.RUN/'scores'/f'{cid}.json'
    if path.exists():
        return
    ref_path = r.RUN/'references'/f'{cid}.json'
    if not ref_path.exists():
        ref = execute(dict(stage='reference',gold=gold,seconds=120,seed=seed(cid,'reference') & 0x7fffffff))
        if ref.get('status') == 'execution_error':
            raise RuntimeError(str(ref))
        save(ref_path,ref)
    ref = load(ref_path)
    cells = {}
    for model in r.MODELS:
        pred = load(r.prediction(model,cid,{'predictions':predictions}))
        spec = pred['model']
        semantic = equivalent(spec,gold)['status']
        for policy in r.CONTROLS:
            name = f'{model}__{policy}'
            if policy in ('schedule','rule'):
                ep = r.trajectory(spec,cid,policy)
            else:
                ep = load(r.RUN/'episodes'/name/f'{cid}.json')
            cells[name] = dict(score=score(gold,ep,ref), semantic=semantic, invalid=spec is None,
                               actions=ep['actions'],episode_seconds=ep['episode_seconds'],
                               decision_seconds=ep['decision_seconds'],writes=ep['intervention_writes'],
                               attempts=ep['heat_bath_attempts'],generation_seconds=pred.get('generation_seconds'))
    save(path,dict(stage=case['stage'],cells=cells))


def summarize(rows):
    summary = {}
    for stage in ('all','main','scale'):
        subset = [row for row in rows if stage=='all' or row['stage']==stage]
        if not subset:
            continue
        stats = {}
        for name in subset[0]['cells']:
            cells = [row['cells'][name] for row in subset]
            gaps = [c['score']['gap'] for c in cells if c['score']['reference_proven'] and c['score']['gap'] is not None]
            stats[name] = dict(n=len(cells),optimal=np.mean([c['score']['optimal_hit'] for c in cells]),
                feasible=np.mean([c['score']['original_feasible'] for c in cells]),
                correct=np.mean([c['semantic']=='verified_correct' for c in cells]),
                unknown=np.mean([c['semantic']=='unknown' for c in cells]),invalid=sum(c['invalid'] for c in cells),
                reference_proven=sum(c['score']['reference_proven'] for c in cells),
                mean_gap=np.mean(gaps) if gaps else None,gap_cases=len(gaps),
                episode_seconds=np.mean([c['episode_seconds'] for c in cells]),
                decision_seconds=np.mean([c['decision_seconds'] for c in cells]),
                attempts=sum(c['attempts'] for c in cells),writes=sum(c['writes'] for c in cells),
                actions=Counter(a for c in cells for a in c['actions']))
        summary[stage]=stats
    return summary


def report():
    rows = [load(p) for c in r.cases() if (p:=r.RUN/'scores'/(c['id']+'.json')).exists()]
    summary = summarize(rows)
    effects = {}
    if len(rows)==780:
        def hits(name):
            return np.array([row['cells'][name]['score']['optimal_hit'] for row in rows],dtype=float)
        b,m,c,j = map(hits,['base__schedule','paired__schedule','base__trajectory','paired__trajectory'])
        diffs = dict(model=m-b,control=c-b,joint=j-b,interaction=j-m-c+b,
                     paired_vs_weighted=m-hits('weighted__schedule'),
                     trajectory_vs_terminal=c-hits('base__terminal'))
        for name,values in diffs.items():
            rng = np.random.default_rng(20261115)
            boot = rng.choice(values,(5000,len(values))).mean(axis=1)
            effects[name] = dict(delta=values.mean(),ci=np.quantile(boot,[.025/6,1-.025/6]).tolist())
    result = dict(experiment='E005-1.14',cases=len(rows),evaluations=12*len(rows),expected=9360,
                  completed_shards=sum(r.marker(t).exists() for t in r.tasks()),summary=summary,effects=effects)
    ready_path = r.RUN/'ready.json'
    if ready_path.exists():
        ready = load(ready_path)
        result['labels'] = ready['action_counts']
        result['training'] = {}
        for arm in r.ARMS:
            expected = len(ready['files']) if arm in r.MODELS else sum(ready['action_counts'][arm].values())
            folder = r.RUN/'trained'/arm
            status = r.v1.training_status(folder,expected*r.CFG['epochs'])
            status['pair_steps'] = sum(bool(load(p).get('pair')) for p in (folder/'steps').glob('*.json'))
            result['training'][arm]=status
    save(r.RUN/'report.json',result)
    print(result,flush=True)


def score_all():
    manifest = r.runtime.validate(r.job())
    if any(not r.marker(t).exists() for t in r.tasks()):
        raise RuntimeError('Finish or resume the four-GPU stage first')
    gold = {row['id']:row['gold'] for path in manifest['gold_paths'] for row in load(Path(path))}
    cases = r.cases()
    with worker_lock(r.RUN/'score.lock'), ProcessPoolExecutor(max_workers=8) as pool:
        scorer = partial(score_case,predictions=manifest['predictions'])
        for i,_ in enumerate(pool.map(scorer,[(c,gold[c['id']]) for c in cases]),1):
            if i%20==0:
                print('scored',i,flush=True)
        report()
