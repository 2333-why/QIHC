"""Case-paired quality and cost summaries, including failed generations."""
import time
from collections import Counter
import numpy as np
from qihc.experiment_runtime import storage as s
from .study import PAIRS


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(root, phase):
    manifest = s.load(root/'manifest.json'); cfg = manifest['config']; directory = root/phase
    cases = s.load(directory/'cases.CPU_ONLY.json')
    data = {}
    for case in cases:
        for arm in cfg['arms']:
            identity = s.key(case['id'], arm)
            path = directory/'responses'/(identity+'.json')
            if not path.exists():
                continue
            response = s.load(path)
            if response['request_sha256'] != s.digest(directory/'requests'/(identity+'.json')):
                raise ValueError('Report response/request identity mismatch')
            rh = s.digest(path)
            scores = []
            for backend, seed in [('cp_sat', 0)]+[('pbit', seed) for seed in cfg['seeds']]:
                p = directory/'scores'/identity/f'{backend}-{seed}.json'
                if p.exists():
                    value = s.load(p)
                    if value['response_sha256'] != rh:
                        raise ValueError('Score belongs to a different generation')
                    scores.append(value)
            data[case['id'], arm] = {'generation': response, 'scores': scores, 'dataset': case['dataset']}
    expected = len(cases)*len(cfg['arms'])
    complete = sum(len(row['scores']) == len(cfg['seeds'])+1 for row in data.values())
    def groups(selected):
        out = {}
        for arm in cfg['arms']:
            rows = [data[c['id'], arm] for c in selected if (c['id'], arm) in data]
            g = [r['generation'] for r in rows]
            entry = {'expected_cases': len(selected), 'generated_cases': len(g),
                     'compiled_cases': sum(v.get('mathematical_validity') == 'valid' for v in g),
                     'failure_reasons': dict(Counter(v.get('validation_error', v.get('mathematical_validity', 'unknown'))
                                                    for v in g if v.get('mathematical_validity') != 'valid')),
                     'compilable_fraction_of_generated': mean([v.get('mathematical_validity') == 'valid' for v in g]),
                     'mean_input_tokens': mean([v['input_tokens'] for v in g]),
                     'mean_output_tokens': mean([v['output_tokens'] for v in g]),
                     'mean_generation_calls': mean([v['generation_attempts'] for v in g]),
                     'mean_attributed_generation_seconds': mean([v['seconds'] for v in g]), 'backends': {}}
            for backend in ('pbit', 'cp_sat'):
                count = len(cfg['seeds']) if backend == 'pbit' else 1
                bundles = [[v for v in r['scores'] if v['backend'] == backend] for r in rows]
                bundles = [b for b in bundles if len(b) == count]
                # Mean seeds within each case, then average cases; invalid outputs count as failures.
                quality = [mean([bool(v.get('verified_feasible')) for v in b]) for b in bundles]
                gaps = [mean([v['reference_gap'] for v in b if v.get('verified_feasible') and 'reference_gap' in v]) for b in bundles]
                entry['backends'][backend] = {'completed_cases': len(bundles),
                    'verified_feasible_rate': mean(quality),
                    'mean_search_seconds_per_seed': mean([mean([v.get('seconds', 0) for v in b]) for b in bundles]),
                    'mean_reference_gap_feasible_only': mean([v for v in gaps if v is not None]),
                    'reference_gap_cases': sum(v is not None for v in gaps),
                    'searched_runs': sum(v['status'] == 'searched' for b in bundles for v in b)}
            out[arm] = entry
        return out
    paired = []
    for before, after in PAIRS:
        for backend in ('pbit', 'cp_sat'):
            count = len(cfg['seeds']) if backend == 'pbit' else 1
            deltas, cost_deltas, gaps = [], [], []
            for case in cases:
                a, b = data.get((case['id'], before)), data.get((case['id'], after))
                if a is None or b is None:
                    continue
                aa = [v for v in a['scores'] if v['backend'] == backend]
                bb = [v for v in b['scores'] if v['backend'] == backend]
                if len(aa) != count or len(bb) != count:
                    continue
                deltas.append(mean([bool(v.get('verified_feasible')) for v in bb])-mean([bool(v.get('verified_feasible')) for v in aa]))
                cost_deltas.append(b['generation']['input_tokens']+b['generation']['output_tokens']-a['generation']['input_tokens']-a['generation']['output_tokens'])
                matched = [(x, y) for x in aa for y in bb if x['seed'] == y['seed']
                           and x.get('verified_feasible') and y.get('verified_feasible')
                           and 'reference_gap' in x and 'reference_gap' in y]
                if matched:
                    gaps.append(mean([y['reference_gap']-x['reference_gap'] for x, y in matched]))
            ci = None
            if len(deltas) >= 2:
                rng = np.random.default_rng(20261026)
                boot = np.mean(rng.choice(deltas, size=(2000, len(deltas)), replace=True), axis=1)
                ci = np.quantile(boot, [.025, .975]).tolist()
            paired.append({'before': before, 'after': after, 'backend': backend, 'paired_cases': len(deltas),
                           'feasible_rate_delta': mean(deltas), 'descriptive_bootstrap95': ci,
                           'mean_total_token_delta': mean(cost_deltas),
                           'paired_feasible_reference_gap_delta': mean(gaps), 'gap_paired_cases': len(gaps)})
    result = {'phase': phase, 'status': 'complete' if complete == expected else 'partial',
              'completed_case_arms': complete, 'expected_case_arms': expected,
              'responses': len(data), 'overall': groups(cases),
              'datasets': {d: groups([c for c in cases if c['dataset'] == d]) for d in sorted({c['dataset'] for c in cases})},
              'paired': paired, 'updated': time.time(),
              'scope': 'Initial modeling only; no post-training or solver feedback. Feasibility is not semantic equivalence.',
              'cost_note': 'Attributed generation time includes failed attempts. Not equal FLOPs or total token costs; CP-SAT has a wall-clock cap.',
              'gap_note': 'Published reference gap, not proven optimality. Only paired feasible same-seed results for gap deltas.',
              'ci_note': 'Case bootstrap is descriptive, unadjusted for multiple comparisons; not a significance claim.'}
    s.save(directory/'summary.json', result)
    lines = ['# Typed modeling contract study v2', '', f'Status: {result["status"]}; {complete}/{expected} case-arms', '',
             '| Arm | Backend | Compiled/generated | Scored cases | Original feasible rate | Feasible reference gap | Gap cases | Input tokens | Output tokens | Calls | Generation s | Search s/seed |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: '-' if x is None else f'{x:.4f}'
    for arm, row in result['overall'].items():
        for backend, b in row['backends'].items():
            lines.append(f'| {arm} | {backend} | {row["compiled_cases"]}/{row["generated_cases"]} | {b["completed_cases"]}/{len(cases)} | '+
                         ' | '.join(fmt(x) for x in (b['verified_feasible_rate'], b['mean_reference_gap_feasible_only'], b['reference_gap_cases'],row['mean_input_tokens'],row['mean_output_tokens'],row['mean_generation_calls'],row['mean_attributed_generation_seconds'],b['mean_search_seconds_per_seed']))+' |')
    lines += ['', '## Generation failures', '', '| Arm | Reason | Cases |', '|---|---|---:|']
    for arm, row in result['overall'].items():
        for reason, count in row['failure_reasons'].items():
            safe_reason = str(reason).replace('|', '/').replace('\n', ' ')
            lines.append(f'| {arm} | {safe_reason} | {count} |')
    lines += ['', result['scope'], '', result['cost_note'], '', result['gap_note'], '',
              'Pairwise deltas and descriptive confidence intervals: summary.json / paired.', result['ci_note']]
    (directory/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result
