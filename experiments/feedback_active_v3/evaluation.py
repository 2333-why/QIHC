"""Separate, gold-owning evaluator; can watch frozen trajectories during inference."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import signal
import time

from qihc.experiment_runtime.storage import load, save, key, worker_lock
from qihc.experiment_runtime.isolated_jobs import JobPool
from qihc.generic.feedback.verification import score
from .runtime import seed_for

BACKENDS = ('scip', 'pbit', 'cp_sat', 'highs', 'sa', 'pt', 'penalty_pt')


class Evaluator:
    def __init__(self, run, data, workers):
        self.run = run; self.cfg = load(run / 'manifest.json')['config']
        self.cases = {c['id']: c for c in load(data / 'evaluation/cases.gold.json')}
        public = data / 'evaluation/cases.public_external.gold.json'
        if public.exists(): self.cases.update({c['id']: c for c in load(public)})
        self.subsets = load(data / 'subsets.json')
        reference_path = data / 'evaluation/references.json'
        self.references = load(reference_path) if reference_path.exists() else {}
        save(run / 'evaluation_design.json', dict(audit_case_ids=self.subsets['audit'], backends=list(BACKENDS)))
        self.pool = JobPool(run / 'evaluation/jobs', workers, 'experiments.feedback_active_v3.cpu')
        self.limit = asyncio.Semaphore(workers * 2); self.initials = {}

    async def reference(self, case):
        cached = self.references.get(case['id'])
        if cached and cached.get('optimality_proven'): return cached
        return await self.pool.run(dict(stage='reference', gold=case['gold'], seconds=120, seed=seed_for(case['id'], 'reference')), 180)

    async def solve(self, case, model, backend, seconds, ref):
        row = await self.pool.run(dict(stage='solve', model=model, backend=backend, seconds=seconds,
            chains=self.cfg['chains'], seed=seed_for(case['id'], 'terminal')), seconds + 45)
        return dict(row, score=score(case['gold'], row, ref))

    async def initial(self, case, initial_key=None):
        cid = initial_key or key(case['id'])
        if cid not in self.initials: self.initials[cid] = asyncio.create_task(self._initial(case, cid))
        return await self.initials[cid]

    async def _initial(self, case, initial_key):
        path = self.run / 'evaluation/initial' / (initial_key + '.json')
        if path.exists(): return load(path)
        initial = load(self.run / 'initial' / (initial_key + '.json')); ref = await self.reference(case)
        row = dict(case_id=case['id'], reference=ref,
            semantic=await self.pool.run(dict(stage='semantic', gold=case['gold'], model=initial['model']), 120),
            check_audit=await self.pool.run(dict(stage='check_audit', gold=case['gold'], checks=initial['checks']), 120),
            cross={}, gold_cross={}, compensated={})
        backends = BACKENDS if case['split'] == 'main' else BACKENDS[:2]
        async def one(backend):
            row['cross'][backend] = await self.solve(case, initial['model'], backend, self.cfg['solver_seconds'], ref)
            row['gold_cross'][backend] = await self.solve(case, case['gold'], backend, self.cfg['solver_seconds'], ref)
            # A fresh 40s run is explicitly labeled; do not claim it is a 20s
            # continuation. Exact continuation requires backend-specific state.
            if case['split'] == 'main':
                row['compensated'][backend] = await self.solve(case, initial['model'], backend, self.cfg['solver_seconds'] + 2 * self.cfg['probe_seconds'], ref)
                row['compensated'][backend]['comparison_scope'] = 'independent_equal_seed_40s_run_not_checkpoint_continuation'
        await asyncio.gather(*(one(b) for b in backends))
        save(path, row); return row

    async def trajectory(self, state):
        path = self.run / 'evaluation/trajectories' / (state['id'] + '.json')
        if path.exists(): return
        async with self.limit:
            case = self.cases[state['case_id']]; initial = await self.initial(case, state.get('initial_key'))
            if state['arm'] == 'C0':
                row = dict(semantic=initial['semantic'], cross=initial['cross'])
            else:
                row = dict(semantic=await self.pool.run(dict(stage='semantic', gold=case['gold'], model=state['model']), 120), cross={})
                backends = BACKENDS if state['stage'] == 'main' and case['id'] in self.subsets['audit'] else BACKENDS[:2]
                async def one(backend): row['cross'][backend] = await self.solve(case, state['model'], backend, self.cfg['solver_seconds'], initial['reference'])
                await asyncio.gather(*(one(b) for b in backends))
            row.update(trajectory_id=state['id'], stage=state['stage'], case_id=case['id'], arm=state['arm'], seed=state['seed'],
                       initial_semantic=initial['semantic']['status'], final_hash=state['final_hash'], evaluated_at=time.time())
            save(path, row)

    async def run_all(self, watch=False):
        while True:
            pending = []
            for p in sorted((self.run / 'trajectories').glob('*.json')):
                if (self.run / 'evaluation/trajectories' / p.name).exists(): continue
                st = load(p)
                if st.get('status') == 'complete': pending.append(st)
            if pending: await asyncio.gather(*(self.trajectory(st) for st in pending))
            from .report import report
            report(self.run)
            if not watch or (self.run / 'inference_complete.json').exists(): break
            await asyncio.sleep(10)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--data', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    def interrupted(*_): raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    with worker_lock(a.run / 'evaluator.lock'): asyncio.run(Evaluator(a.run, a.data, a.workers).run_all(a.watch))


if __name__ == '__main__': main()
