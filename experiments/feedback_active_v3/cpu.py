"""Isolated CPU operations; gold is supplied to offline evaluation only."""
import argparse
import time
from pathlib import Path

from qihc.experiment_runtime.storage import load, save, value_digest


def execute(job):
    stage = job['stage']
    if stage == 'probe':
        if any(k in job for k in ('gold', 'reference', 'answer', 'family')): raise ValueError('Gold/label in feedback payload')
        from qihc.generic.feedback.active import probe
        return probe(job['model'], job['checks'], job['arm'], job['config'], job['seed'])
    if stage == 'semantic':
        from qihc.generic.feedback.verification import equivalent
        return equivalent(job['model'], job['gold'])
    if stage == 'reference':
        from qihc.generic.feedback.verification import reference
        result = reference(job['gold'])
        if result['status'] != 'needs_solver_reference': return result
        from qihc.generic.solver_baselines.registry import solve
        return solve(job['gold'], 'scip', job['seconds'], job.get('seed', 0))
    if stage == 'solve':
        if job['model'] is None: return dict(status='input_invalid', x=None, feasible=False)
        from qihc.generic.solver_baselines.registry import solve
        row = solve(job['model'], job['backend'], job['seconds'], job['seed'], job['chains'], collect_trace=True)
        checkpoints = {}
        for limit in (1, 5, 20, 40):
            if limit > job['seconds']: continue
            seen = [r for r in row.get('trace', []) if r['seconds'] <= limit]
            if row.get('seconds', float('inf')) <= limit:
                checkpoints[str(limit)] = {k: row.get(k) for k in ('x', 'objective', 'feasible', 'optimality_proven', 'status')}
            elif seen:
                checkpoints[str(limit)] = dict(seen[-1], optimality_proven=False, status='observed_incumbent')
            else: checkpoints[str(limit)] = dict(x=None, status='no_incumbent_observed', optimality_proven=False)
        row['checkpoints'] = checkpoints
        return row
    if stage == 'check_audit':
        import numpy as np
        from qihc.generic.feedback.verification import enumerate_states, ordering_equal
        from qihc.generic.feedback.active import check_residual
        from qihc.generic.bounded import violations, objective
        xs = enumerate_states(job['gold']['variables']); rows = []
        if xs is None: return dict(status='unknown', checks=[])
        feasible = xs[violations(job['gold'], xs).sum(axis=1) == 0]
        for c in job['checks'].get('checks', []):
            if c['kind'] == 'constraint': valid = bool(np.all(check_residual(c, feasible) == 0))
            else:
                candidate = dict(job['gold'], objective={k: c[k] for k in ('sense', 'linear', 'quadratic')})
                a = objective(job['gold'], feasible) * (1 if job['gold']['objective']['sense'] == 'min' else -1)
                b = objective(candidate, feasible) * (1 if c['sense'] == 'min' else -1)
                valid = ordering_equal(a, b)
            rows.append(dict(id=c['id'], valid_on_gold=valid))
        return dict(status='complete', checks=rows)
    raise ValueError('Unknown CPU stage: ' + stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('input', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); job = load(args.input); start = time.monotonic()
    try: row = execute(job)
    except Exception as exc: row = dict(status='execution_error', error=f'{type(exc).__name__}: {exc}', x=None)
    row.update(job_hash=value_digest(job), worker_seconds=time.monotonic() - start)
    save(args.output, row)
