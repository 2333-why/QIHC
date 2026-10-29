"""Read-only, CPU-only audit of the returned E001-1.3 gate (no generation/search).

Usage: python THIS_FILE EXTRACTED_ROOT OUTPUT_JSON
Gold is used only for retrospective diagnostics and never for model selection.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys

from qihc.generic.bounded import objective, violations
from qihc.modeling import semantic_review, typed_sources


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root):
    gate = root / 'gate'
    cases = {c['id']: c for c in read(gate / 'cases.CPU_ONLY.json')}
    rows = []
    for path in sorted((gate / 'responses').glob('*.json')):
        response = read(path)
        review = response.get('semantic_review')
        if not review:
            continue
        request_path = gate / 'requests' / path.name
        request = read(request_path)
        assert response['request_sha256'] == hashlib.sha256(request_path.read_bytes()).hexdigest()
        step = read(sorted((gate / 'builder_steps' / path.stem).glob('*.json'))[-1])
        assert step['kind'] == 'semantic_review'
        generation = step['generation']
        error = review.get('error', '')
        category = ('accepted' if review['status'] == 'accepted_self_consistent' else
                    'token_limit' if error == 'Review unfinished: token_limit' else
                    'duplicate_assignment' if error == 'Duplicate semantic check assignment' else
                    'contradictory_checks' if error.startswith('Review candidate contradicts') else error)
        row = {'case_id': request['case_id'], 'arm': request['phase'],
               'response_file': str(path.relative_to(root)), 'category': category,
               'status': review['status'], 'output_tokens': generation.get('output_tokens'),
               'generation_seconds': generation.get('seconds'),
               'mathematics_changed': review.get('mathematics_changed')}
        text = generation['response']
        # Repeated 160-character windows are evidence of repetition, not a decoding fix.
        windows = collections.Counter(text[i:i+160] for i in range(max(0, len(text)-159)))
        row['most_repeated_160_char_window_count'] = max(windows.values(), default=0)
        if category == 'token_limit':
            row['output_tail'] = text[-240:]
        else:
            obj = json.loads(text)
            try:
                _, replay = semantic_review.validate_review(obj, response['initial_response'],
                                                           request['question'], request['variable_names'])
                assert replay['status'] == review['status']
                row['validator_reproduced'] = True
            except ValueError as exc:
                assert str(exc) == error, (request['case_id'], str(exc), error)
                row['validator_reproduced'] = True
            model, _ = typed_sources.compile_math(json.dumps(obj['candidate']), request['question'])
            gold = cases[request['case_id']]['gold']
            assert [v['name'] for v in model['variables']] == [v['name'] for v in gold['variables']]
            checks, seen = [], set()
            for c in obj['checks']:
                x = [v['lower'] for v in model['variables']]
                for key, value in c['assignment'].items():
                    x[int(key)] = value
                checks.append({'assignment': x, 'duplicate': tuple(x) in seen,
                    'expected_objective': c['expected_objective'], 'expected_feasible': c['expected_feasible'],
                    'candidate_objective': float(objective(model, x)),
                    'candidate_feasible': bool(violations(model, x).sum() == 0),
                    'gold_objective': float(objective(gold, x)),
                    'gold_feasible': bool(violations(gold, x).sum() == 0),
                    'reason_excerpt': c['reason'][:500]})
                seen.add(tuple(x))
            row['checks_cpu_only'] = checks
        rows.append(row)
    counts = {}
    for arm in ('semantic_flat', 'semantic_staged'):
        counts[arm] = dict(collections.Counter(r['category'] for r in rows if r['arm'] == arm))
    scores = [read(p) for p in (gate / 'scores').rglob('*.json')]
    rates = {}
    for arm in sorted({s['arm'] for s in scores}):
        rates[arm] = {}
        for backend in ('pbit', 'cp_sat'):
            subset = [s for s in scores if s['arm'] == arm and s['backend'] == backend]
            groups = collections.defaultdict(list)
            for s in subset:
                groups[s['case_id']].append(int(s['verified_feasible']))
            rates[arm][backend] = sum(sum(v)/len(v) for v in groups.values())/len(groups)
    return {'scope': 'Retrospective CPU-only archive audit; no inference, search or gate modification.',
            'gate_acceptance': read(root/'gate_acceptance.json'), 'score_files': len(scores),
            'review_categories': counts, 'feasible_rates_recomputed': rates, 'reviews': rows}


if __name__ == '__main__':
    result = audit(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('score_files', 'review_categories', 'feasible_rates_recomputed')}))
