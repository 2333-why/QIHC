"""Real CPU/mock inference lifecycle tests. No claim of real H100 validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tools.runtime.distribution_job import run as job
from tools.benchmarks.distribution import experiment as e
from tools.benchmarks import comparison as c
from qihc.generic.bounded import VERSION


@pytest.fixture
def prepared(tmp_path):
    model = tmp_path/'model'
    c.save(model/'config.json', {'fixture': True})
    spec = {'schema': VERSION, 'variables': [{'name': 'x', 'lower': 0, 'upper': 1}],
            'objective': {'sense': 'max', 'linear': [1], 'quadratic': []},
            'constraints': [{'id': 'one', 'a': [1], 'op': 'eq', 'rhs': 1}], 'logic': []}
    case = {'id': 'fixture', 'dataset': 'fixture', 'question': 'Binary x equals 1. Maximize x.',
            'gold': spec, 'reference_answer': 1, 'reference_status': 'fixture_exhaustive'}
    raw = json.dumps({'model': spec, 'grounding': [
        {'target': target, 'quote': case['question'], 'explanation': 'Fixture'}
        for target in ('objective', 'variable:x', 'constraint:one')]})
    source, run = tmp_path/'source', tmp_path/'run'
    c.save(source/'cases.CPU_ONLY.json', [case])
    c.save(source/'manifest.json', {'cases_sha256': e.digest(source/'cases.CPU_ONLY.json'),
           'settings': {'model': str(model), 'model_files': e.model_identity(model)}})
    e.prepare(argparse.Namespace(run=run, from_run=source, budget=256, chains=2,
                                seeds=3, solver_seconds=.1, limit=0, arms=list(e.ARMS)))
    return run, case, raw


def deliver(run, raw):
    for path in (run/'requests').glob('*.json'):
        if not (run/'responses'/path.name).exists():
            c.save(run/'responses'/path.name, {'response': raw, 'status': 'generated',
                   'seconds': .1, 'output_tokens': 10, 'request_sha256': e.digest(path)})


def test_early_feedback_same_prompt_and_checkpoint_resume(prepared):
    run, case, raw = prepared
    cfg = e.validate(run)['settings']
    deliver(run, raw)
    assert job.process_case(run, case, cfg) == 'searched_initial'
    arm = cfg['arms'][0]
    folder = run/'progress'/c.key(case['id'], 'case')
    seed0 = folder/f'{arm}.initial.0.json'
    before = seed0.read_bytes()
    assert not (folder/f'{arm}.initial.1.json').exists()
    request = c.load(run/'requests'/(c.key(case['id'], 'review_'+arm)+'.json'))
    assert request['prompt'] == e.review_prompt(case, raw, c.load(seed0)['diagnostic'], arm)
    for _ in range(150):
        deliver(run, raw)
        if job.process_case(run, case, cfg) == 'complete':
            break
    assert seed0.read_bytes() == before
    assert e.summarize(run, [case], cfg)['status'] == 'complete'
    assert job.process_case(run, case, cfg) == 'complete'
    final = c.load(run/'final'/(c.key(case['id'], 'case')+'.json'))
    for arm, row in final['arms'].items():
        assert len(row['initial']) == (1 if arm == 'cp_sat_feedback' else 3)
        if arm != 'cp_sat_feedback':
            assert all(r['runs'][0]['candidate_evaluations'] == 512
                       for stage in ('initial', 'revised') for r in row[stage])


def test_real_spawn_cpu_complete_and_resume(prepared):
    run, _, raw = prepared
    command = [sys.executable, str(Path(job.__file__)), 'cpu', '--run', str(run), '--workers', '2']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    log = run/'test-cpu.log'
    with log.open('w') as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+90
            while proc.poll() is None and time.monotonic() < deadline:
                deliver(run, raw)
                time.sleep(.1)
            assert proc.wait(timeout=5) == 0, log.read_text()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert job.complete(run)
    checkpoints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'progress').glob('*/*.json')}
    subprocess.run(command, env=env, check=True, timeout=30, capture_output=True)
    assert checkpoints == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'progress').glob('*/*.json')}


@pytest.mark.parametrize('visible,expected', [(None, ['0,1', '2,3']), ('2,4,5,7', ['2,4', '5,7']),
                                              ('GPU-a,GPU-b,GPU-c,GPU-d', ['GPU-a,GPU-b', 'GPU-c,GPU-d'])])
def test_gpu_mapping(visible, expected):
    assert job.device_groups(visible, 4, 2) == expected


@pytest.mark.parametrize('visible,count,replicas', [('0,0,1,2', 4, 2), (None, 2, 2), (None, 4, 4), ('0,1', 4, 2)])
def test_bad_gpu_mapping(visible, count, replicas):
    with pytest.raises(ValueError):
        job.device_groups(visible, count, replicas)


@pytest.mark.parametrize('name,code,finished,fails', [('cpu', 1, False, True), ('gpu-0', 0, False, True),
                         ('report', 0, False, False), ('report', 1, True, True), ('cpu', 0, True, False),
                         ('gpu-0', None, False, False)])
def test_fail_fast(name, code, finished, fails):
    children = {name: SimpleNamespace(poll=lambda: code)}
    if fails:
        with pytest.raises(RuntimeError):
            job.check_children(children, finished)
    else:
        job.check_children(children, finished)


def test_environment_and_pending(prepared):
    run, _, raw = prepared
    assert job.pending_count(run) == 1
    deliver(run, raw)
    assert job.pending_count(run) == 0
    assert job.child_env()['CUDA_VISIBLE_DEVICES'] == ''
    assert job.child_env('2,3')['CUDA_VISIBLE_DEVICES'] == '2,3'
    assert job.child_env()['HF_HUB_OFFLINE'] == '1'


def test_scientific_hashes_not_changed():
    assert c.code_hash() == '2559c9609a22b94258d18e296c37cc6873fb0a385b52b8a75db453f507bdf787'
    assert e.source_hash() == '8f8040684e337b2794590463a5e8d8c711990874a1644c81cdf44e6e31d87c1c'


def test_active_worker_lock_is_exclusive(tmp_path):
    with c.worker_lock(tmp_path/'cpu.lock'):
        with pytest.raises(OSError):
            with c.worker_lock(tmp_path/'cpu.lock'):
                pass


def options(run, **kwargs):
    return argparse.Namespace(run=run, from_run=run.parent/'source', output=Path(str(run)+'_reports'),
                              workers=32, batch_size=8, max_idle=180, prepare_only=True, **kwargs)


def test_online_preparation_has_no_compute_and_keeps_source(prepared):
    previous, _, _ = prepared
    run = previous.parent/'prepared_only'
    source = previous.parent/'source'
    before = {p.name: p.read_bytes() for p in source.glob('*.json')}
    assert job.launch(options(run)) == 0
    assert (run/'manifest.json').exists()
    assert not (run/'progress').exists()
    assert not (run/'job_logs').exists()
    assert {p.name: p.read_bytes() for p in source.glob('*.json')} == before
    assert job.launch(options(run)) == 0


@pytest.mark.parametrize('name', ['cpu.lock', 'gpu.lock'])
def test_live_run_refused_before_preparation(prepared, monkeypatch, name):
    run, _, _ = prepared
    monkeypatch.setattr(e, 'prepare', lambda _: pytest.fail('Must not prepare a live run'))
    with c.worker_lock(run/name):
        with pytest.raises(OSError):
            job.launch(options(run))


@pytest.mark.parametrize('mode', ['idle', 'child_failure', 'complete', 'signal'])
def test_supervision_cleanup_and_final_report(tmp_path, monkeypatch, mode):
    run = tmp_path/'run'
    run.mkdir()
    args = options(run)
    procs, cleanup, reports = [], [], []
    ticks = iter(range(0, 10000, 100))
    monkeypatch.setattr(job.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(job.time, 'sleep', lambda _: None)
    monkeypatch.setattr(job, 'STOP', mode == 'signal')
    monkeypatch.setattr(job, 'complete', lambda _: mode == 'complete')
    monkeypatch.setattr(job, 'pending_count', lambda _: 0)
    def popen(command, **kwargs):
        proc = SimpleNamespace(poll=lambda: 1 if mode == 'child_failure' else None)
        procs.append((proc, command, kwargs))
        return proc
    monkeypatch.setattr(job.subprocess, 'Popen', popen)
    monkeypatch.setattr(job.subprocess, 'run', lambda *a, **kw: reports.append(a))
    monkeypatch.setattr(job, 'shutdown', lambda children: cleanup.extend(children))
    if mode in ('idle', 'child_failure'):
        with pytest.raises(RuntimeError):
            job.supervise(args, ['0,1', '2,3'])
    else:
        assert job.supervise(args, ['0,1', '2,3']) == (130 if mode == 'signal' else 0)
    assert len(procs) == len(cleanup) == 4
    assert [p[2]['env']['CUDA_VISIBLE_DEVICES'] for p in procs] == ['', '0,1', '2,3', '']
    assert len(reports) == int(mode == 'complete')
    state = c.load(run/'job_status.json')
    assert state['status'] == ('complete' if mode == 'complete' else 'stopped_checkpointed')
    assert bool(state['error']) == (mode in ('idle', 'child_failure'))
