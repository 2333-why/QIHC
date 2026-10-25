"""Distinct-response health counts; failures are not multiplied by seeds/arms."""
import hashlib
from tools.benchmarks import comparison as c
from .experiment import initial_phase


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
    result['execution_trip'] = len(latest) >= 12 and failures/len(latest) >= .8
    result['trip'] = result['execution_trip']
    result['citation_issues'] = sum(a['citation'] == 'issues' for a in audits.values())
    result['coverage_issues'] = sum(a['coverage'] != 'consistent' for a in audits.values())
    result['independently_feasible_models'] = sum(a['verified_feasible'] for a in audits.values())
    recent_ids = sorted(seen, key=lambda k: seen[k][1])[-24:]
    coverage_failures = sum(audits[k]['coverage'] != 'consistent' for k in recent_ids)
    result['coverage_warning_high'] = len(recent_ids) >= 12 and coverage_failures/len(recent_ids) >= .8
    result['semantic_correctness'] = 'unverified'
    result['scope'] = 'Only inability to compile/search trips. Citation, coverage and solution quality are nonblocking measurements.'
    return result


def gate_check(run):
    manifest = c.load(run/'manifest.json')
    cases = c.load(run/'cases.CPU_ONLY.json')
    if not manifest['settings'].get('gate'):
        raise ValueError('Not a gate run')
    details = []
    for case,arm in ((case,arm) for case in cases for arm in manifest['settings']['arms']):
        for phase, stage in [(initial_phase(arm), 'initial'), ('review_'+arm, 'revised')]:
            req = run/'requests'/(c.key(case['id'], phase)+'.json')
            response = c.load(run/'responses'/req.name)
            record = c.load(run/'progress'/c.key(case['id'], 'case')/f'{arm}.{stage}.0.json')
            real = (response.get('execution') == 'audited_modeling_v6' and response.get('status') == 'generated'
                    and response.get('format_enforcement') == 'lm-format-enforcer-0.11.3-json-schema')
            integrity = response['request_sha256'] == hashlib.sha256(req.read_bytes()).hexdigest()
            integrity &= record['input_response_sha256'] == hashlib.sha256(response['response'].encode()).hexdigest()
            searched = record['status'] == 'searched' and bool(record['runs']) and record['runs'][0].get('candidate_evaluations', 0) > 0
            math_valid = record.get('mathematical_validity') == 'valid'
            details.append({'case': case['id'], 'arm':arm, 'stage': stage,
                            'passed': bool(real and integrity and math_valid and searched),
                            'real_generation': real, 'integrity': integrity, 'search_executed': searched,
                            'mathematical_validity': record.get('mathematical_validity'),
                            'diagnostic_error': record.get('diagnostic', {}).get('error'),
                            'citation_audit': record.get('citation_audit'),
                            'coverage_audit': record.get('coverage_audit'),
                            'verified_feasible': record['runs'][0].get('verified_feasible') if record.get('runs') else None})
    return {'passed': bool(details) and all(r['passed'] for r in details), 'checks': details,
            'purpose': 'execution_readiness_only',
            'quality_summary': {
                'stage_models': len(details),
                'citation_issues': sum((r.get('citation_audit') or {}).get('status') != 'pass' for r in details),
                'coverage_issues': sum((r.get('coverage_audit') or {}).get('status') != 'consistent' for r in details),
                'independently_feasible': sum(r['verified_feasible'] is True for r in details)},
            'semantic_correctness': 'unverified',
            'scope': 'Real generation, mathematical validation and search execution only. Semantic/solution quality is nonblocking and NOT proven by passing this gate.'}
