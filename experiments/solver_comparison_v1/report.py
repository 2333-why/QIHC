"""Case-level paired statistics; predicted-model and original quality separated."""
from statistics import mean
import numpy as np
from qihc.experiment_runtime import storage as s


def average(values):
    return mean(values) if values else None


def report(root):
    manifest = s.load(root/'manifest.json'); cfg = manifest['config']
    cases = s.load(root/'cases.CPU_ONLY.json'); data = {}; models = {}; refs = {}
    for c in cases:
        rp = root/'references'/(s.key(c['id'])+'.json')
        if rp.exists():
            ref = s.load(rp)
            if ref['gold_sha256'] != s.value_digest(c['gold']):
                raise ValueError('Reference identity mismatch')
            refs[c['id']] = ref
        for arm in cfg['source_arms']:
            identity = s.key(c['id'], arm); mp = root/'models'/(identity+'.json')
            if not mp.exists():
                continue
            model = s.load(mp); models[c['id'], arm] = model
            for backend in cfg['backends']:
                for seconds in cfg['seconds']:
                    rows = []
                    for seed in cfg['seeds']:
                        p = root/'scores'/identity/f'{backend}-{seconds}-{seed}.json'
                        if p.exists():
                            r = s.load(p)
                            if (r['response_sha256'] != model['response_sha256'] or
                                    r['model_sha256'] != model.get('model_sha256') or
                                    (r['case_id'], r['source_arm'], r['backend'], r['budget_seconds'], r['seed']) !=
                                    (c['id'], arm, backend, seconds, seed)):
                                raise ValueError('Score/model/config identity mismatch')
                            rows.append(r)
                    if rows:
                        data[c['id'], arm, backend, seconds] = rows
    def gap(case, row):
        ref = refs.get(case['id'])
        if not ref or not row.get('verified_feasible'):
            return None, None
        if ref['result'].get('optimality_proven') and ref['result'].get('feasible'):
            value, label = ref['result']['objective'], 'proven_original_optimum'
        elif ref.get('published_reference') is not None:
            value, label = ref['published_reference'], 'published_reference'
        else:
            return None, None
        sign = 1 if case['gold']['objective']['sense'] == 'min' else -1
        return sign*(row['verified_objective']-value)/max(1., abs(value)), label

    def groups(selected):
        out = []
        for arm in cfg['source_arms']:
            for seconds in cfg['seconds']:
                for backend in cfg['backends']:
                    bundles = [(c, data[c['id'], arm, backend, seconds]) for c in selected
                               if len(data.get((c['id'], arm, backend, seconds), [])) == len(cfg['seeds'])]
                    gaps = [average([g for r in rs if (g := gap(c, r)[0]) is not None]) for c, rs in bundles]
                    raw = [r for _, rs in bundles for r in rs if r['status'] == 'searched']
                    compiled = [(c, rs) for c, rs in bundles if models[c['id'], arm]['status'] == 'compiled']
                    out.append({'source_arm': arm, 'seconds': seconds, 'backend': backend,
                        'completed_cases': len(bundles), 'expected_cases': len(selected),
                        'compiled_cases': len(compiled),
                        'unsupported_runs': sum(r['status'] == 'unsupported' for _, rs in bundles for r in rs),
                        'original_feasible': average([mean(bool(r.get('verified_feasible')) for r in rs) for _, rs in bundles]),
                        'predicted_feasible': average([mean(bool(r.get('predicted_model_feasible')) for r in rs) for _, rs in bundles]),
                        'predicted_feasible_given_compiled': average([mean(bool(r.get('predicted_model_feasible')) for r in rs) for _, rs in compiled]),
                        'original_reference_gap': average([g for g in gaps if g is not None]),
                        'gap_cases': sum(g is not None for g in gaps),
                        'gap_scope': 'Feasible subset only; per-row provenance in references. Published values not assumed optimal.',
                        'mean_solver_wall_seconds_searched_only': average([r['raw_metrics']['wall_seconds'] for r in raw]),
                        'mean_limit_overrun_seconds': average([r['raw_metrics']['time_limit_overrun_seconds'] for r in raw]),
                        'predicted_optimal_runs': sum(r['raw_metrics'].get('optimality_proven', False) for r in raw)})
        return out
    paired = []
    for arm in cfg['source_arms']:
        for seconds in cfg['seconds']:
            for backend in cfg['backends']:
                if backend == 'pbit':
                    continue
                differences, predicted_differences, quality, win, tie, loss = [], [], [], 0, 0, 0
                predicted_quality, predicted_wtl = [], [0, 0, 0]
                for c in cases:
                    a = data.get((c['id'], arm, backend, seconds), [])
                    b = data.get((c['id'], arm, 'pbit', seconds), [])
                    if len(a) != len(cfg['seeds']) or len(b) != len(cfg['seeds']):
                        continue
                    differences.append(mean(bool(r.get('verified_feasible')) for r in b)-mean(bool(r.get('verified_feasible')) for r in a))
                    predicted_differences.append(mean(bool(r.get('predicted_model_feasible')) for r in b)-mean(bool(r.get('predicted_model_feasible')) for r in a))
                    ds, pds = [], []
                    for aa, bb in zip(a, b):
                        assert aa['seed'] == bb['seed']
                        if aa.get('predicted_model_feasible') and bb.get('predicted_model_feasible'):
                            sign = 1 if models[c['id'], arm]['model']['objective']['sense'] == 'min' else -1
                            av, bv = aa['raw_metrics']['objective'], bb['raw_metrics']['objective']
                            pd = sign*(bv-av)/max(1., abs(av), abs(bv))
                            pds.append(pd)
                            predicted_wtl[0 if pd < -1e-9 else 2 if pd > 1e-9 else 1] += 1
                        if aa.get('verified_feasible') and bb.get('verified_feasible'):
                            sign = 1 if c['gold']['objective']['sense'] == 'min' else -1
                            d = sign*(bb['verified_objective']-aa['verified_objective'])/max(1., abs(aa['verified_objective']), abs(bb['verified_objective']))
                            ds.append(d); win += d < -1e-9; loss += d > 1e-9; tie += abs(d) <= 1e-9
                    if ds:
                        quality.append(mean(ds))
                    if pds:
                        predicted_quality.append(mean(pds))
                ci = None
                if len(differences) > 1:
                    rng = np.random.default_rng(20261028)
                    ci = np.quantile(rng.choice(differences, (2000, len(differences)), replace=True).mean(axis=1), [.025, .975]).tolist()
                paired.append({'source_arm': arm, 'seconds': seconds, 'before': backend, 'after': 'pbit',
                    'paired_cases': len(differences), 'original_feasible_delta': average(differences),
                    'predicted_feasible_delta': average(predicted_differences), 'descriptive_bootstrap95': ci,
                    'predicted_paired_feasible_objective_delta': average(predicted_quality),
                    'predicted_objective_cases': len(predicted_quality),
                    'predicted_paired_seed_wins_ties_losses': predicted_wtl,
                    'paired_feasible_normalized_objective_delta': average(quality), 'objective_cases': len(quality),
                    'paired_feasible_seed_wins_ties_losses': [int(win), int(tie), int(loss)]})
    completed = sum(map(len, data.values()))
    expected = len(cases)*len(cfg['source_arms'])*len(cfg['backends'])*len(cfg['seconds'])*len(cfg['seeds'])
    summary = {'status': 'complete' if completed == expected and len(refs) == len(cases) else 'partial',
        'completed_scores': completed, 'expected_scores': expected, 'references': len(refs),
        'imported_models': len(models), 'compiled_models': sum(m['status'] == 'compiled' for m in models.values()),
        'overall': groups(cases), 'datasets': {d: groups([c for c in cases if c['dataset'] == d]) for d in sorted({c['dataset'] for c in cases})},
        'paired': paired, 'scope': manifest['scope'],
        'generation_costs_once_per_model': [{'case_id': c, 'source_arm': arm, **m['generation_cost']} for (c, arm), m in models.items()],
        'statistics': 'Case means over seeds. Partial nonrandom completion. Bootstrap descriptive, no multiplicity adjustment. Equal wall budgets, not equal FLOPs.'}
    s.save(root/'summary.json', summary)
    def fmt(v):
        return '-' if v is None else f'{v:.4f}'
    lines = ['# E002-1.1 shared-model solver comparison', '', f"Status: {summary['status']}; scores {completed}/{expected}", '',
        '| Modeling arm | Budget s | Solver | Cases | Compiled | Predicted feasible | Original feasible | Reference gap | Gap cases | Actual solver s |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['overall']:
        lines.append(f"| {r['source_arm']} | {r['seconds']} | {r['backend']} | {r['completed_cases']}/{r['expected_cases']} | {r['compiled_cases']} | {fmt(r['predicted_feasible'])} | {fmt(r['original_feasible'])} | {fmt(r['original_reference_gap'])} | {r['gap_cases']} | {fmt(r['mean_solver_wall_seconds_searched_only'])} |")
    lines += ['', summary['scope'], '', summary['statistics'], '',
              'Original feasibility is not semantic equivalence. Predicted optimality is not original optimality.',
              'Only compiled common models identify solver capability; invalid models count in end-to-end totals.',
              'PT/penalty_PT are reference kernels, not an original-paper reproduction. All backends here run on CPU.',
              'Generation cost charged once per model, never summed seven times. No measured hardware energy advantage.']
    (root/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return summary
