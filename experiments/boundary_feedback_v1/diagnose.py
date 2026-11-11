"""Read frozen pilot artifacts without importing a model, solver or experiment config."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def diagnose(run):
    run = Path(run).resolve()
    pilot = run / 'pilot'
    if not pilot.is_dir():
        raise ValueError(f'Missing pilot directory: {pilot}')
    read_errors = []
    fingerprints = {}

    def read(path):
        try:
            raw = path.read_bytes()
            fingerprints[path.relative_to(run).as_posix()] = hashlib.sha256(raw).hexdigest()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('Expected artifact object')
            return value
        except (OSError, ValueError) as exc:
            read_errors.append(dict(path=path.relative_to(run).as_posix(), error=str(exc)))
            return {}

    summary = read(pilot / 'summary.json')
    shared = {}
    failures = []
    actions = {arm: Counter() for arm in ('A', 'B', 'C', 'S', 'P', 'Q')}
    calls = Counter()

    def call_records(row):
        records = []
        for call in row.get('usage', []):
            # Usage is counted once per saved call, not once per consuming arm.
            reason = call.get('finish_reason')
            calls['total'] += 1
            if reason == 'length':
                calls['length'] += 1
            if call.get('error') == 'context_limit':
                calls['context_limit'] += 1
            records.append({key: call[key] for key in
                            ('error', 'detail', 'finish_reason', 'usage', 'text') if key in call})
        return records

    for path in sorted((pilot / 'shared').glob('*/*.json')):
        row = read(path)
        key = (path.parent.name, path.stem)
        relative = path.relative_to(run).as_posix()
        shared[key] = relative
        usage = call_records(row)
        if row.get('action') == 'candidate_parse_error':
            failures.append(dict(kind='candidate', path=relative, seed=key[0], case_id=key[1],
                                 error=row.get('error'), text=row.get('text'), calls=usage))

    for path in sorted((pilot / 'models').glob('*/*/predictions/*.json')):
        row = read(path)
        arm = path.parents[1].name
        seed = path.parents[2].name
        if arm not in actions:
            read_errors.append(dict(path=path.relative_to(run).as_posix(), error='Unexpected arm'))
            continue
        actions[arm][row.get('action', 'missing_action')] += 1
        usage = call_records(row)
        if row.get('action') == 'candidate_parse_error':
            # Its raw output belongs to the shared candidate, not this arm.
            if (seed, path.stem) not in shared:
                read_errors.append(dict(path=path.relative_to(run).as_posix(), error='Missing shared candidate'))
        if row.get('action') == 'judge_parse_error':
            bad = [e for e in row.get('evidence', [])
                   if e.get('verdict', {}).get('status') == 'judge_parse_error']
            failures.append(dict(kind='judge', path=path.relative_to(run).as_posix(), arm=arm,
                                 seed=seed, case_id=row.get('case_id', path.stem),
                                 shared_candidate=shared.get((seed, path.stem)),
                                 evidence=bad, calls=usage))
            if not bad:
                read_errors.append(dict(path=path.relative_to(run).as_posix(), error='Missing failed judge evidence'))

    mismatches = []
    reasons = Counter()
    for failure in failures:
        if failure['kind'] == 'candidate':
            reasons['candidate: '+str(failure.get('error'))] += 1
        else:
            for entry in failure['evidence']:
                reasons['judge: '+str(entry['verdict'].get('error'))] += 1
    reported = summary.get('stages', {}).get('development', {}).get('arms', {})
    for arm, counts in actions.items():
        expected = reported.get(arm, {}).get('actions')
        if expected is not None and dict(counts) != expected:
            mismatches.append(dict(arm=arm, reported=expected, observed=dict(counts)))
    return dict(schema_version=1, experiment='E004_2.6', run=str(run),
                scope='Read-only evidence collection; no parser relaxation, model calls or gate bypass',
                reported_complete=summary.get('complete'), reported_expected=summary.get('expected'),
                observed_predictions=sum(sum(c.values()) for c in actions.values()),
                actions={a: dict(c) for a, c in actions.items()}, unique_saved_calls=dict(calls),
                failure_reasons=dict(reasons), failures=failures, read_errors=read_errors, summary_mismatches=mismatches,
                input_sha256=fingerprints)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = diagnose(args.run)
    # Preserve both old evidence and earlier diagnostic snapshots.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(dict(output=str(args.output), failures=len(report['failures']),
                          observed_predictions=report['observed_predictions'],
                          unique_saved_calls=report['unique_saved_calls'],
                          failure_reasons=report['failure_reasons'],
                          read_errors=len(report['read_errors']),
                          summary_mismatches=len(report['summary_mismatches'])), ensure_ascii=False))
    if report['read_errors'] or report['summary_mismatches']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
