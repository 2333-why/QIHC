"""Reuse CPU search/evaluation, rejecting stale or empty warm-start pools."""
import argparse
from pathlib import Path
import numpy as np
from qihc.generic.bounded import violations
from qihc.experiment_runtime import resilient_storage as s
from .adapter import install


def execute(job):
    from experiments.feedback_study_v1.cpu import execute as legacy_execute
    job = dict(job)
    rejected = 0
    if job['stage'] in ('base','final') and job.get('model') is not None:
        def valid(candidate):
            try:
                v = violations(job['model'], candidate)
                return v.size > 0 and not np.any(v[..., 0])
            except (ValueError, TypeError, IndexError):
                return False
        initial = job.get('initial')
        if initial is not None and not valid(initial):
            job['initial'] = None
            rejected += 1
        archive = job.get('archive', [])
        job['archive'] = [x for x in archive if valid(x)]
        rejected += len(archive)-len(job['archive'])
    row = legacy_execute(job)
    row['rejected_stale_candidate_records'] = rejected
    return row


if __name__ == '__main__':
    install()
    p = argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    try:row=execute(s.load(a.job))
    except Exception as exc:row={'status':'execution_error','error':repr(exc),'x':None}
    row['job_sha256']=s.digest(a.job);s.save(a.output,row)
