"""Gate on actual attempted interfaces, never on observed performance."""
from qihc.experiment_runtime.storage import load,save
from .data import RUN,sources

def check():
    summary=load(RUN/'pilot/summary.json')
    if not summary['complete'] or summary['expected']!=168:
        raise RuntimeError('Pilot must complete 24 cases x 7 arms')
    attempts=failures=0
    for path in (RUN/'pilot/shared').glob('*/*.json'):
        row=load(path)
        if row.get('usage'):
            attempts+=1
            failures+=int(row['action']=='candidate_parse_error')
    if attempts and failures/attempts>.2:
        raise RuntimeError(f'Candidate parse errors {failures}/{attempts} actual attempts')
    counts={}
    for arm in ('R','S','P','Q'):
        calls=errors=0
        for path in (RUN/f'pilot/models').glob(f'*/{arm}/predictions/*.json'):
            row=load(path)
            for entry in row.get('evidence',[]):
                calls+=1
                errors+=int(entry['verdict']['status']=='review_parse_error')
        counts[arm]=dict(attempts=calls,parse_errors=errors)
        if calls and errors/calls>.2:
            raise RuntimeError(f'{arm}: local review parse failures {errors}/{calls}; inspect prediction evidence')
    save(RUN/'pilot/gate.json',dict(passed=True,source=sources(),candidate_attempts=attempts,
                                  candidate_errors=failures,review_interfaces=counts))
    print('Pilot interface passed:',counts,'; no efficacy or semantic certification')

if __name__=='__main__':
    check()
