"""Read-only progress, including partial results not yet counted by final summary."""
import argparse
from collections import Counter
import json
from pathlib import Path


def status(run):
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    requests = {p.name:read(p) for p in (run/'requests').glob('*.json')}
    responses = {p.name:read(p) for p in (run/'responses').glob('*.json')}
    stages = Counter(r['phase'] for k,r in requests.items() if k not in responses)
    completed = Counter(requests[k]['phase'] for k in responses if k in requests)
    errors = Counter(r.get('status','unknown') for r in responses.values())
    return {'requests':len(requests),'responses':len(responses),
            'final_cases':len(list((run/'final').glob('*.json'))),
            'pending_by_phase':dict(stages),'saved_by_phase':dict(completed),
            'response_statuses':dict(errors),
            'saved_search_stages':dict(Counter(p.name for p in (run/'progress').glob('*/*.json'))),
            'generation_seconds_sum_allocated':sum(r.get('seconds',0) for r in responses.values()),
            'output_tokens':sum(r.get('output_tokens',0) for r in responses.values()),
            'retry_overhead_seconds':sum(read(p).get('seconds',0) for p in (run/'inference_events').glob('*.json')),
            'replicas':[read(p) for p in sorted(run.glob('gpu_replica_[0-9].json'))],
            'note':'Saved status files do not establish process liveness. Batch seconds allocated across requests; retry overhead reported separately.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(status(p.parse_args().run),indent=2,ensure_ascii=False))
