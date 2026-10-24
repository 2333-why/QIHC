import argparse
import json
from types import SimpleNamespace

import pytest
from tools.runtime.modeling_v2 import protocol as p, experiment as e, generation as g, health as h
from tools.benchmarks import comparison as c
from tools.benchmarks.distribution import experiment as old


def fixture():
    question = 'Choose one binary item x0 or x1. Maximize 3*x0+x1.'
    obj = {'model': {'schema': 'bounded-integer-v2',
           'variables': [{'name': 'x0', 'lower': 0, 'upper': 1}, {'name': 'x1', 'lower': 0, 'upper': 1}],
           'objective': {'sense': 'max', 'linear': {'0': 3, '1': 1}, 'quadratic': []},
           'constraints': [{'id': 'one', 'a': {'0': 1, '1': 1}, 'op': 'eq', 'rhs': 1}], 'logic': []},
           'grounding': [{'targets': ['objective', 'variable:x0', 'variable:x1', 'constraint:one'],
                          'quote': question, 'explanation': 'Binary domain and selection rule'}]}
    return question, obj


def test_sparse_grouped_grounding_equivalent():
    question, obj = fixture()
    spec, reason = p.parse(json.dumps(obj), question)
    assert reason is None
    assert spec['objective']['linear'] == [3, 1]
    assert spec['constraints'][0]['a'] == [1, 1]
    assert 'model.schema' in p.modeling_prompt({'question': question, 'gold': spec})
    assert obj['model']['objective']['linear'] == {'0': 3, '1': 1}


@pytest.mark.parametrize('mutation,message', [
    ('schema', 'model.schema'), ('quote', 'quote'), ('duplicate', 'duplicate'), ('dimension', 'dimension'),
    ('float', 'integer'), ('index', 'index')])
def test_bad_models_still_rejected(mutation, message):
    question, obj = fixture()
    if mutation == 'schema':
        del obj['model']['schema']
    elif mutation == 'quote':
        obj['grounding'][0]['quote'] = 'invented quote'
    elif mutation == 'duplicate':
        obj['grounding'][0]['targets'].append('objective')
    elif mutation == 'dimension':
        obj['model']['objective']['linear'] = [1]
    elif mutation == 'float':
        obj['model']['objective']['linear'] = {'0': 3.0}
    else:
        obj['model']['objective']['linear'] = {'02': 1}
    with pytest.raises(ValueError, match=message):
        p.parse(json.dumps(obj), question)


def test_json_boundaries():
    assert g.complete_object('{"quote":"a } brace","x":[1]} trailing') == '{"quote":"a } brace","x":[1]}'
    assert g.complete_object('{"unfinished":') is None
    assert g.complete_object('reasoning first {"x":1}') is None
    assert g.complete_object('[]') is None
    with pytest.raises(ValueError, match='Invalid JSON'):
        p.parse('{"unfinished":', '')


def test_gate_selection_no_success_filter():
    cases = [{'id': str(i), 'dataset': dataset, 'question': 'q'*length}
             for i, (dataset, length) in enumerate([('nlco_KP_S', 4), ('nlco_KP_L', 9), ('synthetic_assignment', 3)])]
    assert [v['id'] for v in p.gate_cases(cases)] == ['1', '2']


def prepared(tmp_path, gate):
    question, obj = fixture()
    spec, _ = p.parse(json.dumps(obj), question)
    case = {'id': 'fixture', 'question': question, 'gold': spec, 'dataset': 'fixture',
            'reference_answer': 3, 'reference_status': 'fixture'}
    model, source, run = tmp_path/'model', tmp_path/'source', tmp_path/'run'
    c.save(model/'config.json', {'fixture': True})
    c.save(source/'cases.CPU_ONLY.json', [case])
    c.save(source/'manifest.json', {'settings': {'model': str(model), 'model_files': e.model_identity(model)},
                                   'cases_sha256': e.digest(source/'cases.CPU_ONLY.json')})
    args = argparse.Namespace(run=run, from_run=source, gate=gate, budget=256, chains=2, seeds=1,
                              solver_seconds=.1, limit=0, arms=['pbit_selected'] if gate else list(e.ARMS))
    e.prepare(args)
    return run, case, json.dumps(obj), args


@pytest.mark.parametrize('gate', [True, False])
def test_real_cpu_plumbing_and_resume(tmp_path, gate):
    run, case, raw, args = prepared(tmp_path, gate)
    cfg = e.validate(run)['settings']
    for _ in range(70):
        for path in (run/'requests').glob('*.json'):
            if not (run/'responses'/path.name).exists():
                c.save(run/'responses'/path.name, {'response': raw, 'execution': 'TEST_MOCK',
                       'status': 'generated', 'request_sha256': e.digest(path)})
        if e.process_case(run, case, cfg) == 'complete':
            break
    assert e.summarize(run, [case], cfg)['status'] == 'complete'
    assert h.health(run)['searched_models'] > 0
    if gate:
        assert not h.gate_check(run)['passed']  # Mock output cannot pass real-model gate.
        for path in (run/'responses').glob('*.json'):
            row = c.load(path)
            row['execution'] = 'validated_compact_v2'
            c.save(path, row)
        assert h.gate_check(run)['passed']
        path = next((run/'responses').glob('*.json'))
        row = c.load(path); row['request_sha256'] = 'tampered'; c.save(path, row)
        assert not h.gate_check(run)['passed']
    e.prepare(args)
    assert e.process_case(run, case, cfg) == 'complete'


def test_failure_dedup_and_trip(tmp_path):
    for i in range(11):
        for seed in range(3):
            c.save(tmp_path/'progress'/str(i)/f'a.{seed}.json', {
                'status': 'invalid_model', 'input_response_sha256': str(i), 'runs': []})
    assert h.health(tmp_path)['distinct_models'] == 11
    assert not h.health(tmp_path)['trip']
    c.save(tmp_path/'progress'/'11'/'a.json', {'status': 'invalid_model', 'input_response_sha256': '11', 'runs': []})
    assert h.health(tmp_path)['trip']


def test_old_hashes_untouched():
    assert c.code_hash() == '2559c9609a22b94258d18e296c37cc6873fb0a385b52b8a75db453f507bdf787'
    assert old.source_hash() == '8f8040684e337b2794590463a5e8d8c711990874a1644c81cdf44e6e31d87c1c'
    assert e.source_hash() != old.source_hash()


def test_generation_metadata_and_oom_split(tmp_path, monkeypatch):
    class OOM(Exception):
        pass
    generator = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None)))
    items = []
    for i in range(2):
        path = tmp_path/'requests'/f'{i}.json'
        c.save(path, {'prompt': 'test'})
        items.append((path, {'prompt': 'test'}))
    def generate(gen, batch):
        if len(batch) > 1:
            raise OOM('fixture')
        return [{'response': '{}', 'status': 'generated', 'output_tokens': 2}]
    monkeypatch.setattr(g, 'generate', generate)
    g.execute(generator, items, tmp_path, 0, 'batch')
    assert len(list((tmp_path/'responses').glob('*.json'))) == 2
    assert len(list((tmp_path/'inference_events').glob('*.json'))) == 1
    for path in (tmp_path/'responses').glob('*.json'):
        row = c.load(path)
        assert row['execution'] == 'validated_compact_v2'
        assert row['batch_size'] == 1
        assert row['request_sha256'] == e.digest(tmp_path/'requests'/path.name)


@pytest.mark.parametrize('exitcode,accepted', [(1, False), (0, False)])
def test_failed_gate_never_launches_full_job(tmp_path, monkeypatch, exitcode, accepted):
    from tools.runtime.modeling_v2 import run as job
    monkeypatch.setattr(job.sys, 'argv', ['run.py', '--run', str(tmp_path/'formal')])
    monkeypatch.setattr(job, 'STOP', False)
    monkeypatch.setattr(job.subprocess, 'Popen', lambda *a, **kw: SimpleNamespace(
        poll=lambda: exitcode, returncode=exitcode))
    monkeypatch.setattr(job, 'shutdown', lambda *a, **kw: None)
    monkeypatch.setattr(job, 'gate_check', lambda _: {'passed': accepted})
    monkeypatch.setattr(job, 'launch', lambda _: pytest.fail('Full experiment must not launch'))
    with pytest.raises(RuntimeError, match='gate'):
        job.main()
