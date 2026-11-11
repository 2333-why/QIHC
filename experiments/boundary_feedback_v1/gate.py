"""Pilot interface gate, not a significance or performance selection gate."""
from qihc.experiment_runtime.storage import load, save
from .data import RUN, sources

def check():
    summary = load(RUN/'pilot/summary.json')
    if not summary['complete'] or summary['expected'] != 144:
        raise RuntimeError('Pilot must complete 24 cases x 6 arms before formal')
    calls = []
    for pattern in ('shared/*/*.json', 'models/*/*/predictions/*.json'):
        for path in (RUN/'pilot').glob(pattern):
            calls.extend(load(path).get('usage', []))
    failures = sum(r.get('error') == 'context_limit' or r.get('finish_reason') == 'length' for r in calls)
    if calls and failures/len(calls) > .2:
        raise RuntimeError(f'Pilot context/truncation failures {failures}/{len(calls)}; inspect saved raw outputs')
    for arm in ('S', 'P', 'Q'):
        counts = summary['stages']['development']['arms'][arm]['actions']
        errors = sum(counts.get(k, 0) for k in ('candidate_parse_error', 'judge_parse_error'))
        if errors > 6:
            raise RuntimeError(f'Pilot {arm}: {errors}/24 parse failures; inspect outputs before formal')
    save(RUN/'pilot/gate.json', dict(passed=True, source=sources(), observed_calls=len(calls), transport_failures=failures))
    print('Pilot interface gate passed; no efficacy claim')

if __name__ == '__main__':
    check()
