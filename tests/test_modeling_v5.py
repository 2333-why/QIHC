import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.runtime.modeling_v5 import protocol as p, experiment as e, generation as g, health as h
from tools.benchmarks import comparison as c
from tools.benchmarks.distribution import experiment as old


@pytest.fixture(scope='module')
def samples():
    return json.loads((Path(__file__).parent/'fixtures/modeling_v3_gate_regression.json').read_text())['cases']


@pytest.mark.parametrize('index', [0, 2, 4, 6])
def test_real_failed_samples_math_survives_citation_issues(samples, index):
    row = samples[index]
    spec, reason, audits = p.inspect(row['response'], row['question'])
    assert spec is not None and reason is None
    assert audits['coverage_audit']['status'] == 'issues'  # Old responses had no ledger.
    assert audits['semantic_correctness'] == 'unverified'
    if index == 0:
        assert audits['citation_audit']['status'] == 'issues'
    if index in (2, 4, 6):
        assert audits['citation_audit']['status'] == 'pass'
    if index in (0, 2, 6):
        assert spec['objective'] == row['gold']['objective']
    if index == 4:
        assert len(spec['constraints']) == 7
        assert len(row['gold']['constraints']) == 14  # Never auto-add gold constraints.


@pytest.mark.parametrize('index', [5, 7])
def test_real_truncated_samples_remain_failures_with_correct_diagnostic(samples, index):
    row = samples[index]
    with pytest.raises(ValueError, match='Incomplete or invalid JSON'):
        p.inspect(row['response'], row['question'])
    assert g.complete_object(row['response']) is None


def test_real_maxcut_wrong_objective_not_silently_fixed(samples):
    row = samples[1]
    with pytest.raises(ValueError, match='linear length 0'):
        p.inspect(row['response'], row['question'])


def test_real_vertex_cover_duplicate_coefficient_key_rejected(samples):
    row = samples[3]
    # The previous JSON decoder silently retained only one repeated coefficient.
    with pytest.raises(ValueError, match='Duplicate JSON key: 18'):
        p.inspect(row['response'], row['question'])


def envelope():
    question = 'Choose exactly one binary x0 or x1. Maximize 3*x0+x1.'
    obj = {'model': {'schema': 'bounded-integer-v2', 'variables': [['x0',0,1],['x1',0,1]],
           'objective': {'sense':'max','linear':{'0':3,'1':1},'quadratic':[]},
           'constraints': [['one',{'0':1,'1':1},'eq',1]], 'logic':[]},
           'requirements':[
               {'id':'o','kind':'objective','expected_count':1,'targets':['objective'],'quote':question},
               {'id':'d','kind':'domain','expected_count':2,'targets':['variable:x0','variable:x1'],'quote':question},
               {'id':'c','kind':'constraint','expected_count':1,'targets':['constraint:one'],'quote':question}],
           'grounding':[]}
    return question, obj


def test_compact_expansion_and_three_layers():
    question, obj = envelope()
    spec, _, audits = p.inspect(json.dumps(obj), question)
    assert spec['constraints'][0]['a'] == [1,1]
    assert audits['coverage_audit']['status'] == 'consistent'
    assert audits['citation_audit']['status'] == 'pass'
    obj['requirements'][0]['quote'] = 'paraphrase'
    assert p.inspect(json.dumps(obj), question)[2]['citation_audit']['status'] == 'issues'
    assert p.inspect(json.dumps(obj), question)[0] == spec


@pytest.mark.parametrize('mutation', ['count','target','id','kind','malformed_kind','missing'])
def test_generic_coverage_checks(mutation):
    question, obj = envelope()
    if mutation == 'count': obj['requirements'][2]['expected_count'] = 2
    if mutation == 'target': obj['requirements'][2]['targets'] = ['constraint:absent']
    if mutation == 'id': obj['requirements'][2]['id'] = 'o'
    if mutation == 'kind': obj['requirements'][2]['kind'] = 'domain'
    if mutation == 'malformed_kind': obj['requirements'][2]['kind'] = ['constraint']
    if mutation == 'missing': obj['requirements'].pop()
    spec, _, audits = p.inspect(json.dumps(obj), question)
    assert spec is not None
    assert audits['coverage_audit']['status'] == 'issues'


def test_bounds_and_duplicate_keys():
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        p.inspect('{"model":{},"model":{}}', '')
    assert p.output_budget(1000) == 6144
    assert p.output_budget(4000) == 12288
    assert p.output_budget(20000) == 12288


def test_real_cpu_closed_loop_and_gate(tmp_path):
    question, obj = envelope()
    model, _, _ = p.inspect(json.dumps(obj), question)
    case = {'id':'fixture','dataset':'fixture','question':question,'gold':model,
            'reference_answer':3,'reference_status':'fixture'}
    source, run, weights = tmp_path/'source', tmp_path/'run', tmp_path/'weights'
    c.save(weights/'config.json', {'fixture':True})
    c.save(source/'cases.CPU_ONLY.json', [case])
    c.save(source/'manifest.json', {'cases_sha256':e.digest(source/'cases.CPU_ONLY.json'),
           'settings':{'model':str(weights),'model_files':e.model_identity(weights)}})
    args = argparse.Namespace(run=run,from_run=source,gate=True,budget=256,chains=2,seeds=1,
                              solver_seconds=.1,limit=0,arms=['pbit_selected'])
    e.prepare(args)
    raw = json.dumps(obj)
    for _ in range(8):
        for path in (run/'requests').glob('*.json'):
            if not (run/'responses'/path.name).exists():
                c.save(run/'responses'/path.name, {'response':raw,'execution':'MOCK',
                       'status':'generated','request_sha256':e.digest(path),'generation_attempts':2})
        if e.process_case(run,case,e.validate(run)['settings']) == 'complete': break
    result = e.summarize(run,[case],e.validate(run)['settings'])
    assert result['status'] == 'complete'
    assert result['datasets']['fixture']['arms']['pbit_selected']['attributed_generation_attempts_including_truncation_retries'] == 4
    assert not h.gate_check(run)['passed']
    for path in (run/'responses').glob('*.json'):
        row = c.load(path); row['execution'] = 'audited_modeling_v5'
        row['format_enforcement'] = 'lm-format-enforcer-0.11.3-json-schema'; c.save(path,row)
    assert h.gate_check(run)['passed']
    checkpoint = next((run/'progress').glob('*/pbit_selected.revised.0.json'))
    row = c.load(checkpoint); row['coverage_audit']['status'] = 'issues'; c.save(checkpoint,row)
    row['runs'][0]['verified_feasible'] = False
    c.save(checkpoint, row)
    checked = h.gate_check(run)
    assert checked['passed']
    assert checked['quality_summary']['coverage_issues'] == 1
    assert checked['quality_summary']['independently_feasible'] == 1
    assert checked['semantic_correctness'] == 'unverified'
    assert c.load(next((run/'final').glob('*.json')))['training_eligibility'] == 'unlabeled_not_a_positive'
    row['runs'][0]['candidate_evaluations'] = 0
    c.save(checkpoint, row)
    assert not h.gate_check(run)['passed']  # Still requires actual search.
    row.update(status='invalid_model', mathematical_validity='invalid', runs=[])
    row.pop('citation_audit'); row.pop('coverage_audit')
    c.save(checkpoint, row)
    assert not h.gate_check(run)['passed']  # Missing audits must not crash failure reporting.
    e.prepare(args)


def test_truncation_retry_published_once_and_costed(tmp_path, monkeypatch):
    class OOM(Exception): pass
    generator = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))
    path = tmp_path/'requests'/'one.json'
    c.save(path,{'prompt':'fixture'})
    calls = []
    valid_raw = json.dumps(envelope()[1])
    def generate(gen, items, retry=False):
        assert not (tmp_path/'responses'/path.name).exists()
        calls.append(retry)
        return [{'response':valid_raw if retry else '{','status':'generated' if retry else 'incomplete_output',
                 'output_tokens':100 if retry else 50,'input_tokens':20,'finish_reason':'json_complete' if retry else 'token_limit'}]
    monkeypatch.setattr(g,'generate',generate)
    g.execute(generator,[(path,{'prompt':'fixture'})],tmp_path,0,'batch')
    row = c.load(tmp_path/'responses'/path.name)
    assert calls == [False,True]
    assert row['output_tokens'] == 150 and row['input_tokens'] == 40
    assert row['generation_attempts'] == 2 and row['response'] == valid_raw
    assert row['mathematical_validity'] == 'valid'
    assert len(list((tmp_path/'generation_attempts').glob('*-first.json'))) == 1
    assert len(list((tmp_path/'generation_attempts').glob('*-retry-request.json'))) == 1


def test_existing_sources_preserved():
    assert c.code_hash() == '2559c9609a22b94258d18e296c37cc6873fb0a385b52b8a75db453f507bdf787'
    assert old.source_hash() == '8f8040684e337b2794590463a5e8d8c711990874a1644c81cdf44e6e31d87c1c'


def test_execution_breaker_deduplicates_and_ignores_semantic_quality(tmp_path):
    for i in range(12):
        for seed in range(3):
            c.save(tmp_path/'progress'/str(i)/f'a.{seed}.json', {
                'status':'searched','input_response_sha256':str(i),'runs':[{'verified_feasible':True}],
                'citation_audit':{'status':'issues'},'coverage_audit':{'status':'consistent'}})
    result = h.health(tmp_path)
    assert result['distinct_models'] == 12 and result['citation_issues'] == 12
    assert not result['trip']
    for path in (tmp_path/'progress').glob('*/*.json'):
        row = c.load(path); row['coverage_audit']['status'] = 'issues'
        row['runs'][0]['verified_feasible'] = False
        c.save(path,row)
    result = h.health(tmp_path)
    assert result['coverage_warning_high']
    assert result['independently_feasible_models'] == 0
    assert not result['trip']
    for path in (tmp_path/'progress').glob('*/*.json'):
        row = c.load(path); row['status'] = 'invalid_model'; row['runs'] = []; c.save(path,row)
    result = h.health(tmp_path)
    assert result['execution_trip'] and result['trip']


@pytest.mark.parametrize('repair_success', [True, False])
def test_compile_error_repaired_once_with_same_budget_policy(tmp_path, monkeypatch, repair_success):
    class OOM(Exception): pass
    generator = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)),
                                max_input_tokens=32768,
                                _inputs=lambda text: {'input_ids':SimpleNamespace(shape=(1,len(text)))})
    question, obj = envelope()
    good = json.dumps(obj)
    obj['model']['objective']['linear'] = []
    bad = json.dumps(obj)
    path = tmp_path/'requests'/'one.json'
    c.save(path,{'prompt':question})
    calls = []
    def generate(gen, items, retry=False):
        assert not (tmp_path/'responses'/path.name).exists()
        calls.append(items[0][1]['prompt'])
        if retry:
            assert 'linear length 0' in calls[-1]
            assert calls[-1].startswith(question)
        return [{'response':good if retry and repair_success else bad,'status':'generated',
                 'output_tokens':10,'input_tokens':5,'finish_reason':'json_complete'}]
    monkeypatch.setattr(g,'generate',generate)
    g.execute(generator,[(path,{'prompt':question})],tmp_path,0,'batch')
    row = c.load(tmp_path/'responses'/path.name)
    assert len(calls) == 2 and row['generation_attempts'] == 2
    assert row['output_tokens'] == 20 and row['input_tokens'] == 10
    assert row['repair_kind'] == 'format_math_repair'
    assert row['status'] == ('generated' if repair_success else 'invalid_model_output')
    assert row['training_eligibility'] == 'unlabeled_not_a_positive'
    assert row['request_sha256'] == e.digest(path)


def test_valid_model_with_missing_audits_is_not_auto_repaired(tmp_path, monkeypatch):
    class OOM(Exception): pass
    generator = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))
    question, obj = envelope()
    del obj['requirements']
    raw = json.dumps(obj)
    path = tmp_path/'requests'/'one.json'
    c.save(path,{'prompt':question})
    def generate(gen, items, retry=False):
        assert not retry
        return [{'response':raw,'status':'generated'}]
    monkeypatch.setattr(g,'generate',generate)
    g.execute(generator,[(path,{'prompt':question})],tmp_path,0,'batch')
    row = c.load(tmp_path/'responses'/path.name)
    assert row['status'] == 'generated' and row['generation_attempts'] == 1
    assert p.inspect(raw, question)[2]['coverage_audit']['status'] == 'issues'


def test_wrong_semantics_is_searched_not_silently_corrected():
    question, obj = envelope()
    gold = p.compile_math(json.dumps(obj))[0]
    obj['model']['constraints'][0][3] = 0  # Wrong meaning, valid math: forces both zeros.
    case = {'id':'semantic-error','question':question,'gold':gold,'reference_answer':3}
    row = e.solve_one(case,json.dumps(obj),'pbit_selected',{'budget':256,'chains':2},0)
    assert row['mathematical_validity'] == 'valid' and row['status'] == 'searched'
    assert row['runs'][0]['verified_feasible'] is False
    assert 'verified_feasible' not in json.dumps(row['diagnostic'])


def test_oversized_repair_preserves_original_request():
    generator = SimpleNamespace(max_input_tokens=100, _inputs=lambda text: {'input_ids':SimpleNamespace(shape=(1,len(text)))})
    original = {'prompt':'THE COMPLETE ORIGINAL REQUEST'}
    request, included = g.repair_request(generator, original, {'validation_error':'bad shape','response':'x'*1000})
    assert not included and request['prompt'].startswith(original['prompt'])
    assert 'x'*1000 not in request['prompt']
    assert original['prompt'] == 'THE COMPLETE ORIGINAL REQUEST'


def test_compiler_errors_are_shared_across_all_arms():
    messages = [e.feedback_payload({'error':'invalid coefficient dimension'}, arm) for arm in e.ARMS]
    assert all(message == messages[0] for message in messages)
    assert messages[0]['scope'] == 'model_validation_only'
