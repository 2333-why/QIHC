"""Read-only E001-1.2 snapshot analysis; no model inference or solver calls.

Run from repository root with optional research dependencies on PYTHONPATH:
python docs/experiments/E001_prompt/E001-1.2_analyze_archive.py --run PATH --output NEW_JSON
"""
import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean

from experiments.prompt_efficiency_v2.report import summarize
from experiments.prompt_efficiency_v3.study import failure_attribution
from qihc.experiment_runtime.storage import key
from qihc.modeling import typed_sources


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(run):
    # The historical report helper writes report.md/summary.json. Run it only
    # against a disposable copy, keeping all archived evidence byte-identical.
    with tempfile.TemporaryDirectory(prefix='qihc-e001-analysis-') as tmp:
        report_run = Path(tmp)/'run'
        shutil.copytree(run, report_run)
        return _analyze(run, report_run)


def _analyze(run, report_run):
    manifest = load(run/'manifest.json')
    result = {'source_run': str(run.resolve()), 'code_commit': manifest['code_commit'],
              'source_hash': manifest['source_hash'], 'config': manifest['config'],
              'status_snapshot': load(run/'status.json'), 'phases': {}}
    for phase in ('gate', 'formal'):
        directory = run/phase
        assert digest(directory/'cases.CPU_ONLY.json') == manifest['phase_cases_sha256'][phase]
        cases = load(directory/'cases.CPU_ONLY.json')
        by_id = {c['id']: c for c in cases}
        summary = summarize(report_run, phase)
        old = load(directory/'summary.json')
        for field in ('overall', 'datasets', 'paired', 'completed_case_arms', 'responses'):
            assert summary[field] == old[field], (phase, field)
        rows = []
        requests = list((directory/'requests').glob('*.json'))
        for p in requests:
            assert digest(p) == manifest['request_hashes'][p.stem]
        for arm in manifest['config']['arms']:
            for case in cases:
                identity = key(case['id'], arm)
                p = directory/'responses'/(identity+'.json')
                if not p.exists():
                    continue
                response = load(p)
                scores = [load(s) for s in sorted((directory/'scores'/identity).glob('*.json'))]
                assert len(scores) == 4
                rows.append({'case_id': case['id'], 'arm': arm, 'response': response, 'scores': scores})
        typed_findings = []
        for row in rows:
            if row['arm'] != 'typed_flat':
                continue
            case = by_id[row['case_id']]
            model, reason, _ = typed_sources.inspect(row['response']['response'], case['question'])
            assert model is not None, reason
            scores = [{'backend': s['backend'], 'seed': s['seed'],
                       **failure_attribution(case, model, s)} for s in row['scores']]
            typed_findings.append({'case_id': case['id'], 'dataset': case['dataset'],
                                   'gold_constraints': len(case['gold']['constraints']),
                                   'predicted_constraints': len(model['constraints']),
                                   'gold_objective': case['gold']['objective'],
                                   'predicted_objective': model['objective'],
                                   'predicted_constraint_examples': model['constraints'][:2],
                                   'scores': scores})
        backend_pairs = {}
        for arm in manifest['config']['arms']:
            outcomes = Counter()
            searched_times = {'pbit': [], 'cp_sat': []}
            for row in rows:
                if row['arm'] != arm:
                    continue
                for s in row['scores']:
                    if s['status'] == 'searched':
                        searched_times[s['backend']].append(s['seconds'])
                cp = next(s for s in row['scores'] if s['backend'] == 'cp_sat')
                case = by_id[row['case_id']]
                for pb in (s for s in row['scores'] if s['backend'] == 'pbit'):
                    if cp.get('verified_feasible') and pb.get('verified_feasible'):
                        delta = pb['verified_objective']-cp['verified_objective']
                        if case['gold']['objective']['sense'] == 'max':
                            delta = -delta
                        outcomes['both_feasible'] += 1
                        outcomes['pbit_better' if delta < -1e-8 else 'cp_sat_better' if delta > 1e-8 else 'objective_tie'] += 1
                    elif pb.get('verified_feasible'):
                        outcomes['only_pbit_feasible'] += 1
                    elif cp.get('verified_feasible'):
                        outcomes['only_cp_sat_feasible'] += 1
                    else:
                        outcomes['neither_feasible'] += 1
            backend_pairs[arm] = {'seed_comparisons': dict(outcomes),
                'mean_seconds_searched_only': {k: mean(v) if v else None for k, v in searched_times.items()},
                'scope': 'Three pbit seeds reuse one deterministic CP-SAT result per case; comparisons are not independent cases.'}
        result['phases'][phase] = {
            'summary_reproduced_exactly': True, 'verified_requests': len(requests),
            'verified_responses': len(rows), 'verified_scores': sum(len(r['scores']) for r in rows),
            'observed_distinct_cases': len({r['case_id'] for r in rows}),
            'observed_datasets': dict(Counter(by_id[c]['dataset'] for c in {r['case_id'] for r in rows})),
            'reference_present_in_observed_cases': sum(by_id[c].get('reference_answer') is not None for c in {r['case_id'] for r in rows}),
            'overall': summary['overall'], 'paired': summary['paired'],
            'datasets': {k: v for k, v in summary['datasets'].items() if any(a['generated_cases'] for a in v.values())},
            'typed_findings': typed_findings,
            'typed_categories_seed_counts': dict(Counter(s['category'] for r in typed_findings for s in r['scores'])),
            'typed_cases_with_counterexample': sum(any('witness' in s['category'] for s in r['scores']) for r in typed_findings),
            'backend_comparisons': backend_pairs}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.run.resolve() in args.output.resolve().parents:
        raise ValueError('Use a new output file outside the original result directory')
    result = analyze(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({phase: {k: v for k, v in d.items() if k in
        ('summary_reproduced_exactly', 'verified_requests', 'verified_responses', 'verified_scores',
         'observed_datasets', 'typed_cases_with_counterexample', 'backend_comparisons')}
        for phase, d in result['phases'].items()}, ensure_ascii=False, indent=2))
