"""One bounded CPU job per process. Probe workers never load evaluator files."""
import argparse
import time
from pathlib import Path
import numpy as np
from qihc.generic import feedback
from qihc.generic.bounded import objective, violations, validate
from .common import s, backend
from .protocol import alternatives


def execute(job):
    cfg, spec = job['config'], job.get('model')
    stage = job['stage']; seed = int(job.get('seed', 0)); arm = job.get('arm', 'C0')
    if stage in ('reference', 'evaluate'):
        return evaluate(job)
    if spec is None:
        return {'status':'invalid_model', 'x':None, 'seconds':0.}
    start = time.monotonic(); validate(spec)
    if stage in ('base', 'final'):
        seconds = (cfg['base_seconds'] if stage == 'base' else cfg['final_seconds'])-job.get('prior_compile_seconds',0.)
        if seconds<=0:
            return {'status':'compile_budget_exhausted','x':None,'seconds':0.}
        archive = [x for x in job.get('archive', []) if violations(spec,x)[0]==0]
        initial=job.get('initial')
        if initial is not None and np.any(violations(spec,initial)[:,0]):
            initial=None
        chosen = None
        if archive:
            e = feedback.Energy(spec)
            f, r, _ = e.values(np.array(archive))
            i = min(range(len(archive)), key=lambda j:(r[j].sum()>0, r[j].sum(), e.sign*f[j]))
            chosen = archive[i]
        row = feedback.run_window(spec, max(.001, seconds-(time.monotonic()-start)), seed,
                                  backend(arm), cfg['chains'], read_q=arm != 'no_q', initial=initial)
        if chosen is not None:
            candidates = [chosen]+([row['x']] if row.get('x') is not None else [])
            sign = 1 if spec['objective']['sense']=='min' else -1
            row['x'] = min(candidates, key=lambda x:(float(violations(spec,x).sum())>0,
                                      float(violations(spec,x).sum()),sign*float(objective(spec,x))))
        row['archive_revalidated_states'] = len(archive)
    else:
        alts, errors = alternatives(spec, job.get('review', {}), job['question'])
        remaining=cfg['probe_seconds']-job.get('prior_compile_seconds',0.)-(time.monotonic()-start)
        if remaining<=0:
            return {'status':'compile_budget_exhausted','x':None,'seconds':time.monotonic()-start}
        row = feedback.evidence(spec, arm, remaining,
             seed, backend(arm), cfg['chains'], job['base'], alts, cfg)
        row['hypothesis_errors'] = errors
    row['status'] = 'complete'; row['wall_seconds'] = time.monotonic()-start
    return row


def evaluate(job):
    # Evaluation is a separate subprocess and its records are never copied to
    # requests or trajectory state. A coordinator may schedule it but not read
    # metrics when choosing feedback actions.
    from qihc.generic.solver_baselines.registry import solve
    case, cfg = job['case'], job['config']
    gold = case['gold']
    if job['stage'] == 'reference':
        result = solve(gold, 'cp_sat', cfg['reference_seconds'], seed=0)
        return {'case_id':case['id'], 'gold_sha256':s.value_digest(gold), 'result':result}
    spec, x = job.get('model'), job.get('x')
    result = {'case_id':case['id'], 'arm':job['arm'], 'seed':job['seed'], 'split':case['split'],
              'dataset':case['dataset'], 'size':case.get('size'), 'compiled':spec is not None,
              'original_feasible':False, 'objective':None, 'predicted_feasible':False,
              'gap_percent':None, 'optimal_hit':False, 'reference_proven':False}
    if x is not None:
        try:
            result['original_feasible'] = bool(violations(gold, x).sum() == 0)
            result['objective'] = float(objective(gold, x))
            result['predicted_feasible'] = spec is not None and bool(violations(spec, x).sum() == 0)
        except (ValueError, TypeError, IndexError):
            result['candidate_error'] = 'domain/dimension mismatch'
    def check(candidate):
        try:
            return candidate is not None and bool(violations(gold,candidate).sum()==0)
        except (ValueError,TypeError,IndexError):
            return False
    result['initial_candidate_feasible']=check(job.get('initial_x'))
    result['round_results']=[{'round':r['round'],'original_feasible':check(r['x']),
        'objective':float(objective(gold,r['x'])) if r['x'] is not None else None} for r in job.get('round_candidates',[])]
    result['repaired_feasibility']=not result['initial_candidate_feasible'] and result['original_feasible']
    result['regressed_feasibility']=result['initial_candidate_feasible'] and not result['original_feasible']
    ref = job.get('reference', {}).get('result', {})
    if ref.get('feasible') and ref.get('objective') is not None:
        val = ref['objective']; result['reference_proven'] = bool(ref.get('optimality_proven'))
        if result['original_feasible']:
            sign = 1 if gold['objective']['sense'] == 'min' else -1
            result['gap_percent'] = 100*sign*(result['objective']-val)/max(1., abs(val))
            result['optimal_hit'] = result['reference_proven'] and abs(result['objective']-val) < 1e-7
    if spec is not None:
        try:
            diagnostic = solve(spec, 'scip', cfg['diagnostic_seconds'], seed=job['seed'])
            dx = diagnostic.get('x')
            result['common_solver_diagnostic'] = diagnostic
            result['common_solver_original_feasible'] = dx is not None and bool(violations(gold, dx).sum() == 0)
            rng = np.random.default_rng(20261101+job['seed'])
            xs = np.column_stack([rng.integers(v['lower'], v['upper']+1, 128) for v in gold['variables']])
            a = violations(gold, xs).sum(axis=1) == 0; b = violations(spec, xs).sum(axis=1) == 0
            result['hidden_feasibility_agreement'] = float(np.mean(a == b))
            gf = objective(gold, xs)*(1 if gold['objective']['sense']=='min' else -1)
            pf = objective(spec, xs)*(1 if spec['objective']['sense']=='min' else -1)
            result['hidden_objective_order_agreement'] = float(np.mean(np.sign(np.diff(gf)) == np.sign(np.diff(pf))))
            result['hidden_probe_scope'] = '128 independent domain samples, not equivalence proof; rare feasible regions may be missed'
        except Exception as exc:
            result['diagnostic_error'] = repr(exc)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('job', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); job = s.load(a.job)
    try:
        row = execute(job)
    except Exception as exc:
        row = {'status':'execution_error', 'error':repr(exc), 'x':None}
    row['job_sha256'] = s.digest(a.job)
    s.save(a.output, row)


if __name__ == '__main__':
    main()
