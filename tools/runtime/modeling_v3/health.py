"""Distinct-response health counts; failures are not multiplied by seeds/arms."""
import hashlib
from tools.benchmarks import comparison as c


def health(run):
    seen = {}
    audits = {}
    for path in (run/'progress').glob('*/*.json'):
        row = c.load(path)
        identity = (path.parent.name, row.get('input_response_sha256'))
        searched = row.get('status') == 'searched' and bool(row.get('runs'))
        prev = seen.get(identity)
        if prev is None or searched:
            seen[identity] = (searched, path.stat().st_mtime)
            audits[identity] = {'citation': row.get('citation_audit', {}).get('status'),
                                'coverage': row.get('coverage_audit', {}).get('status'),
                                'verified_feasible': any(r.get('verified_feasible') is True for r in row.get('runs', []))}
    latest = sorted(seen.values(), key=lambda r: r[1])[-24:]
    failures = sum(not ok for ok, _ in latest)
    result = {'distinct_models': len(seen), 'searched_models': sum(ok for ok, _ in seen.values()),
              'recent_models': len(latest), 'recent_failures': failures}
    result['trip'] = len(latest) >= 12 and failures/len(latest) >= .8
    result['citation_issues'] = sum(a['citation'] == 'issues' for a in audits.values())
    result['coverage_issues'] = sum(a['coverage'] != 'consistent' for a in audits.values())
    result['independently_feasible_models'] = sum(a['verified_feasible'] for a in audits.values())
    recent_ids = sorted(seen, key=lambda k: seen[k][1])[-24:]
    coverage_failures = sum(audits[k]['coverage'] != 'consistent' for k in recent_ids)
    result['coverage_trip'] = len(recent_ids) >= 12 and coverage_failures/len(recent_ids) >= .8
    result['trip'] = result['trip'] or result['coverage_trip']
    return result


def gate_check(run):
    manifest = c.load(run/'manifest.json')
    cases = c.load(run/'cases.CPU_ONLY.json')
    if not manifest['settings'].get('gate'):
        raise ValueError('Not a gate run')
    details = []
    for case in cases:
        for phase, stage in [('model', 'initial'), ('review_pbit_selected', 'revised')]:
            req = run/'requests'/(c.key(case['id'], phase)+'.json')
            response = c.load(run/'responses'/req.name)
            record = c.load(run/'progress'/c.key(case['id'], 'case')/f'pbit_selected.{stage}.0.json')
            real = response.get('execution') == 'audited_modeling_v3' and response.get('status') == 'generated'
            integrity = response['request_sha256'] == hashlib.sha256(req.read_bytes()).hexdigest()
            integrity &= record['input_response_sha256'] == hashlib.sha256(response['response'].encode()).hexdigest()
            searched = record['status'] == 'searched' and bool(record['runs']) and record['runs'][0].get('candidate_evaluations', 0) > 0
            coverage = record.get('coverage_audit', {}).get('status') == 'consistent'
            independently_feasible = bool(record.get('runs')) and record['runs'][0].get('verified_feasible') is True
            details.append({'case': case['id'], 'stage': stage,
                            'passed': bool(real and integrity and searched and coverage and independently_feasible),
                            'mathematical_validity': record.get('mathematical_validity'),
                            'diagnostic_error': record.get('diagnostic', {}).get('error'),
                            'citation_audit': record.get('citation_audit'),
                            'coverage_audit': record.get('coverage_audit'),
                            'verified_feasible': record['runs'][0].get('verified_feasible') if record.get('runs') else None})
    return {'passed': bool(details) and all(r['passed'] for r in details), 'checks': details,
            'scope': 'Real search, self-ledger consistency and independently checked candidate feasibility; NOT proof of complete semantic equivalence or performance superiority.'}
