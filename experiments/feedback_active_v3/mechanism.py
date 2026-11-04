"""Verified controlled mutations; oracle-local checks are explicitly labeled."""
import argparse
import asyncio
import copy
from collections import defaultdict
from pathlib import Path
import signal

import numpy as np

from qihc.experiment_runtime.storage import load, save, key
from qihc.experiment_runtime.isolated_jobs import JobPool
from qihc.generic.feedback.verification import equivalent
from .data import digest

ERRORS = ('omission', 'direction', 'boundary', 'index', 'coefficient', 'domain', 'objective_sense', 'objective_term')
ARMS = ('A', 'X', 'M', 'T', 'R', 'D', 'P0', 'P1', 'P2')


def mutation(gold, kind, rng):
    for _ in range(200):
        m = copy.deepcopy(gold); n = len(m['variables'])
        if kind == 'domain': m['variables'][int(rng.integers(n))]['upper'] = 0
        elif kind == 'objective_sense': m['objective']['sense'] = 'max' if m['objective']['sense'] == 'min' else 'min'
        elif kind == 'objective_term':
            ids = [i for i, c in enumerate(m['objective']['linear']) if c]
            m['objective']['linear'][int(rng.choice(ids))] = 0
        else:
            k = int(rng.integers(len(m['constraints']))); c = m['constraints'][k]
            if kind == 'omission': m['constraints'].pop(k)
            elif kind == 'direction': c['op'] = 'ge' if c['op'] == 'le' else 'le'
            elif kind == 'boundary': c['rhs'] += int(rng.choice([-1, 1]))
            elif kind == 'coefficient':
                i = int(rng.choice(np.flatnonzero(c['a']))); c['a'][i] += int(rng.choice([-1, 1]))
            elif kind == 'index':
                i = int(rng.choice(np.flatnonzero(c['a']))); j = int(rng.integers(n))
                if i == j: continue
                c['a'][j] += c['a'][i]; c['a'][i] = 0
        proof = equivalent(m, gold)
        if proof['status'] == 'disproved': return m, proof
    raise ValueError('Cannot construct an effective mutation: ' + kind)


def prepare(output):
    # Mutate models of PUBLIC development instances, never generate problems.
    cases = load(output.parent / 'cases.gold.json')
    bases = sorted((c for c in cases if c['split']=='development' and c['tier']=='S'
                    and c['family']!='MAXCUT'), key=lambda c:c['id'])
    if len(bases)!=60: raise ValueError('Expected 60 public development bases')
    rows = []; excluded = []; rng = np.random.default_rng(20261109)
    for case in bases:
        checks = []
        for c in case['gold']['constraints'][:15]: checks.append(dict(c, kind='constraint', quote=case['question'], confidence=1., trusted=False))
        checks.append(dict(case['gold']['objective'], id='objective', kind='objective', quote=case['question'], confidence=1., trusted=False))
        for kind in ('correct',) + ERRORS:
            try:
                model, proof = (case['gold'], {'status': 'verified_correct'}) if kind == 'correct' else mutation(case['gold'], kind, rng)
            except ValueError as exc:
                excluded.append(dict(base_id=case['id'], error_type=kind, reason=str(exc)))
                continue  # An ineffective mutation is not a labeled modeling error.
            rows.append(dict(id=case['id'] + '_' + kind, base_id=case['id'], error_type=kind, model=model,
                gold=case['gold'], checks={'checks': checks}, verification=proof,
                provenance=case['provenance'],
                scope='controlled_model_errors_on_public_development_instances; oracle_local_checks_not_end_to_end'))
    audit=dict(models=len(rows), incorrect=sum(r['error_type']!='correct' for r in rows), correct=60,
               bases=[c['id'] for c in bases], excluded=excluded, sha256=digest(rows))
    save(output, rows); save(output.with_suffix('.audit.json'),audit); return audit


async def run(args):
    cfg = load(Path(__file__).parent / 'config.json'); cfg['probe_seconds'] = args.seconds
    rows = load(args.data); pool = JobPool(args.run / 'jobs', args.workers, 'experiments.feedback_active_v3.cpu')
    gate = asyncio.Semaphore(args.workers * 2)
    async def one(case, arm, seed):
        path = args.run / 'results' / (key(case['id'], arm, str(seed)) + '.json')
        if path.exists(): return
        async with gate:
            result = await pool.run(dict(stage='probe', model=case['model'], checks=case['checks'], arm=arm,
                                         seed=seed, config=cfg), args.seconds + 30)
            save(path, dict(case_id=case['id'], base_id=case['base_id'], error_type=case['error_type'], arm=arm, seed=seed,
                           result=result, scope=case['scope']))
    await asyncio.gather(*(one(c, a, s) for c in rows for a in ARMS for s in (17, 29, 43)))
    from qihc.generic.bounded import violations
    byid={c['id']:c for c in rows}; groups=defaultdict(lambda:dict(evaluated=0,cards_emitted=0,certified_feasibility_conflicts=0,worker_failures=0))
    for path in (args.run/'results').glob('*.json'):
        row=load(path); c=byid[row['case_id']]; r=row['result']; g=groups[(row['arm'],row['error_type'])]
        g['evaluated']+=1; cards=r.get('cards',[]); g['cards_emitted']+=bool(cards)
        g['worker_failures']+=r.get('status') in ('worker_failed','worker_timeout','execution_error')
        certified=False
        for card in cards:
            x=card.get('x')
            if x is not None:
                try: certified |= bool(violations(c['model'],x).sum()==0 and violations(c['gold'],x).sum()>0)
                except (ValueError,TypeError): pass
        g['certified_feasibility_conflicts']+=certified
    save(args.run/'summary.json',dict(groups=[dict(arm=a,error_type=e,**g) for (a,e),g in sorted(groups.items())],
        scope='public_development_controlled_mutations; card emission is not semantic detection; objective errors need separate analysis',
        models=len(rows),bases=len({c['base_id'] for c in rows})))
    save(args.run / 'status.json', dict(status='complete', expected=len(rows) * len(ARMS) * 3,
         scope='20s_final_only; selection/confirmation trace retained; 1/5s card checkpoints not certified'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prepare', action='store_true')
    p.add_argument('--data', type=Path, default=Path(__file__).parent / 'assets/public_dataset/evaluation/mechanism.json')
    p.add_argument('--run', type=Path); p.add_argument('--workers', type=int, default=16); p.add_argument('--seconds', type=float, default=20)
    a = p.parse_args()
    def interrupted(*_): raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    if a.prepare: print(prepare(a.data))
    else:
        if a.run is None: p.error('--run required')
        asyncio.run(run(a))
