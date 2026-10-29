"""Reuse paired statistics; expose failures instead of silently dropping them."""
from collections import Counter
import errno
import time
from qihc.experiment_runtime import resilient_storage as s
from experiments.solver_comparison_v1.report import report as paired_report


def report(root):
    # Serialize manual and automatic snapshots without stopping the solver queue.
    while True:
        lock = s.worker_lock(root/'report.lock')
        try:
            lock.__enter__()
            break
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            time.sleep(.2)
    try:
        return _report(root)
    finally:
        lock.__exit__(None, None, None)


def _report(root):
    result = paired_report(root)
    m = s.load(root/'manifest.json')
    counts = Counter()
    for p in (root/'scores').rglob('*.json'):
        r = s.load(p)
        counts[r['source_arm'], r['backend'], r['status']] += 1
    result.update(experiment_id='E003-1.1', selected_before_validation=m['selected_before_validation'],
                  adapter_exclusions=m['excluded'],
                  failures=[{'track': t, 'backend': b, 'status': status, 'runs': n}
                            for (t, b, status), n in sorted(counts.items())])
    # Complete describes terminal accounting, never automatic scientific success.
    result['execution_errors'] = sum(n for (t, b, status), n in counts.items() if status in ('execution_error', 'hard_timeout', 'worker_crash'))
    result['all_terminal'] = result['status'] == 'complete'
    result['quality_status'] = 'inspect_coverage_and_failures' if result['execution_errors'] else 'inspect_model_validity_and_quality'
    per_track = result['expected_scores']//len(m['config']['source_arms'])
    result['tracks'] = {}
    for track in m['config']['source_arms']:
        done = sum(n for (t,b,status),n in counts.items() if t == track)
        result['tracks'][track] = {'completed_scores': done, 'expected_scores': per_track,
            'status': 'complete' if done == per_track else 'not_started' if done == 0 else 'partial'}
    s.save(root/'summary.json', result)
    path = root/'report.md'
    text = path.read_text(encoding='utf-8').replace('# E002-1.1 shared-model solver comparison', '# E003-1.1 public solver benchmark')
    text += '\n\n## Independent track progress\n\n| Track | Status | Scores |\n|---|---|---:|\n'
    text += '\n'.join(f'| {t} | {r["status"]} | {r["completed_scores"]}/{r["expected_scores"]} |' for t,r in result['tracks'].items())+'\n'
    text += '\n\n## Terminal status accounting\n\n| Track | Backend | Status | Runs |\n|---|---|---|---:|\n'
    text += '\n'.join(f'| {t} | {b} | {status} | {n} |' for (t,b,status),n in sorted(counts.items()))+'\n'
    text += f'\nSelected before validation: 900; adapter exclusions: {len(m["excluded"])}. No replacement by easier cases.\n'
    text += '\nA complete run can contain failures. Canonical is the solver-only track; LLM is end-to-end and must not be pooled with canonical. Published references are not automatically optima.\n'
    path.write_text(text, encoding='utf-8')
    return result
