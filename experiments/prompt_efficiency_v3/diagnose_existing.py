"""Read-only analysis of saved v2/v3 typed_flat scores; no new solver/model calls."""
import argparse
from collections import Counter
from pathlib import Path
from qihc.experiment_runtime import storage as s
from qihc.modeling import typed_sources
from .study import failure_attribution


def diagnose(run, output):
    run, output = run.resolve(), output.resolve()
    if output == run or run in output.parents:
        raise ValueError('Diagnostic output must be outside the old experiment directory')
    if output.exists():
        raise ValueError('Choose a new snapshot output; never overwrite diagnostics')
    manifest = s.load(run/'manifest.json')
    findings = []; pending = 0
    for phase in ('gate', 'formal'):
        directory = run/phase
        cases_path = directory/'cases.CPU_ONLY.json'
        if not cases_path.exists():
            continue
        if s.digest(cases_path) != manifest['phase_cases_sha256'][phase]:
            raise ValueError('Historical case identity mismatch')
        for case in s.load(cases_path):
            key = s.key(case['id'], 'typed_flat')
            path = directory/'responses'/(key+'.json')
            if not path.exists():
                pending += 1
                continue
            response_hash = s.digest(path)
            response = s.load(path)
            req = directory/'requests'/(key+'.json')
            if response['request_sha256'] != s.digest(req):
                raise ValueError('Historical response/request identity mismatch')
            row = {'phase': phase, 'case_id': case['id'], 'response_sha256': response_hash,
                   'response_path': str(path), 'scores': []}
            try:
                if response.get('status') != 'generated':
                    raise ValueError(response.get('validation_error', response.get('status')))
                model, reason, audits = typed_sources.inspect(response['response'], case['question'])
                if model is None:
                    raise ValueError(reason)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                row.update(category='not_compilable', error=str(exc))
            else:
                row.update(category='compiled', audits=audits)
                for score in sorted((directory/'scores'/key).glob('*.json')):
                    result = s.load(score)
                    if result['response_sha256'] != response_hash:
                        raise ValueError('Historical score/response identity mismatch')
                    row['scores'].append({'backend': result['backend'], 'seed': result['seed'],
                                          **failure_attribution(case, model, result)})
            findings.append(row)
    result = {'source_run': str(run), 'source_code_commit': manifest['code_commit'],
              'pending_case_arms': pending, 'observed_typed_flat_case_arms': len(findings),
              'categories_per_scored_seed': dict(Counter(v['category'] for r in findings for v in r['scores'])),
              'scope': 'Read-only snapshot, no regeneration/search. Gate and formal may overlap; counts are not independent instances. Unresolved search failure is not labeled a semantic error.',
              'findings': findings}
    s.save(output/'diagnosis.json', result)
    lines = ['# typed_flat historical diagnostics', '', result['scope'], '',
             '| Phase | Case | Backend | Seed | Attribution | Violated original constraints |',
             '|---|---|---|---:|---|---|']
    for row in findings:
        for score in row['scores']:
            violations = ', '.join(v['id'] for v in score.get('original_violations', []))
            lines.append(f'| {row["phase"]} | {row["case_id"]} | {score["backend"]} | {score["seed"]} | {score["category"]} | {violations} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.run, args.output)
    print({k: v for k, v in result.items() if k != 'findings'})
