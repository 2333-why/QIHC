"""Versioned protocol; reuse solver, probe and semantic implementations unchanged."""
import argparse
from pathlib import Path
import time
from .common import s
from .protocol import parse_model, guard
from experiments.feedback_witness_v1.cpu import execute as legacy_execute


def execute(job):
    if job['stage'] == 'contract_compile':
        value, model = parse_model(job['response'], job['case'])
        return {'status': 'complete', 'model': model, 'source_refs': value['source_refs'],
                'source_ledger': value['source_ledger'],
                'quotes': value['source_refs']}
    if job['stage'] == 'guard':
        return guard(job['case'], job.get('model'), job['response'], job.get('contract'), job.get('evidence'))
    if job['stage'] == 'probe':
        row = legacy_execute(job)
        for card in row.get('cards', []):
            card['source_span_ids'] = card.pop('source_quotes', [])
        return row
    return legacy_execute(job)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('job', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); started = time.monotonic()
    try:
        row = execute(s.load(a.job))
    except Exception as exc:
        row = {'status': 'execution_error', 'error': str(exc)[:1600], 'x': None}
    row.update(job_sha256=s.digest(a.job), worker_seconds=time.monotonic()-started)
    s.save(a.output, row)
