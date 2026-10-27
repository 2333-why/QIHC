"""Read-only archive replay. Run from repository root with PYTHONPATH=.; no inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

from qihc.modeling.protocol import compile_math
from qihc.modeling.workflow import Job


def analyze(archive):
    with tarfile.open(archive) as tar:
        raw = {}
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.json'):
                if member.name in raw:
                    raise ValueError('Duplicate archive member: '+member.name)
                raw[member.name] = tar.extractfile(member).read()
    data = {name: json.loads(value) for name, value in raw.items()}
    digest = lambda name: hashlib.sha256(raw[name]).hexdigest()
    manifest = data['manifest.json']
    assert digest('gate/cases.CPU_ONLY.json') == manifest['phase_cases_sha256']['gate']
    cases = {c['id']: c for c in data['gate/cases.CPU_ONLY.json']}
    rows = []
    finishes = Counter()
    for path, req in sorted(data.items()):
        if not path.startswith('gate/requests/'):
            continue
        identity = Path(path).stem
        assert digest(path) == manifest['request_hashes'][identity]
        response_path = f'gate/responses/{identity}.json'
        response = data[response_path]
        assert response['request_sha256'] == digest(path)
        job = Job(Path(path), req)
        steps = []
        for name, step in sorted(data.items()):
            if not name.startswith(f'gate/builder_steps/{identity}/'):
                continue
            task = job.task()
            assert task and task['prompt'] == step['prompt']
            job.accept(step['generation'])
            gen = step['generation']
            finishes[gen.get('finish_reason', 'unknown')] += 1
            steps.append({'path': name, 'kind': step['kind'], 'status': gen['status'],
                          'finish_reason': gen.get('finish_reason'),
                          'input_tokens': gen.get('input_tokens'), 'output_tokens': gen.get('output_tokens')})
        if not job.done:
            job.task()
        assert job.done and job.result['status'] == response['status']
        assert job.result.get('validation_error') == response.get('validation_error')
        scores = []
        for name, score in sorted(data.items()):
            if name.startswith(f'gate/scores/{identity}/'):
                assert score['response_sha256'] == digest(response_path)
                scores.append({k: score.get(k) for k in ('backend', 'seed', 'status', 'verified_feasible',
                    'verified_objective', 'objective', 'reference_gap', 'seconds', 'solver_status')})
        assert len(scores) == len(manifest['config']['seeds'])+1
        row = {'case': req['case_id'], 'arm': req['phase'], 'response_path': response_path,
               'validity': response['mathematical_validity'], 'failed_step': response.get('failed_step'),
               'error': response.get('validation_error'), 'reference_answer': cases[req['case_id']].get('reference_answer'),
               'scores': scores, 'steps': steps}
        if row['validity'] == 'valid':
            spec, _ = compile_math(response['response'], req['question'])
            gold = cases[req['case_id']]['gold']
            row.update(exact_model_equal=spec == gold, exact_objective_equal=spec['objective'] == gold['objective'],
                       model_constraint_count=len(spec['constraints']), gold_constraint_count=len(gold['constraints']))
        rows.append(row)
    return {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'code_commit': manifest['code_commit'], 'replayed_requests': len(rows),
            'step_finish_counts': dict(finishes),
            'validity_counts': dict(Counter(r['validity'] for r in rows)),
            'error_counts': dict(Counter(r['error'] for r in rows if r['error'])),
            'scope': 'Replay saved outputs through original validators; no new model inference or solver run. Exact inequality is not proof of semantic inequality.',
            'cases': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.archive)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, ensure_ascii=False))
