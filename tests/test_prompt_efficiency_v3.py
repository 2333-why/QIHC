"""ID collision regression and real v1 semantic failures; no invented v2 traces."""
import copy
import json
from pathlib import Path

import jsonschema
import pytest

from qihc.modeling import builder, typed_sources, semantic_review, review_workflow
from qihc.modeling.canonical_builder import Builder
from qihc.modeling.typed_builder import Builder as OldBuilder
from qihc.experiment_runtime import storage as s
from qihc.generic.bounded import objective, violations
from experiments.prompt_efficiency_v2 import study as v2
from experiments.prompt_efficiency_v3 import study, report
from test_prompt_efficiency_v1 import case, state, QUESTION, generator, fake_generation


def collision_state():
    st = state()
    st['requirements'] += [{'id': 'other', 'kind': 'constraint', 'quote': QUESTION, 'expected_count': 3}]
    st['rules'] += [{'requirement_id': 'other', 'constraints': [
        {'id': 'cap', 'a': [1, 0, 0], 'op': 'le', 'rhs': 1},
        {'id': 'cap', 'a': [0, 1, 0], 'op': 'le', 'rhs': 1}],
        'logic': [{'id': 'cap', 'expr': {'op': 'or', 'args': [{'var': 0}, {'op': 'not', 'args': [{'var': 0}]}]}}]}]
    return st


def test_ids_collision_same_group_cross_group_logic_preserves_math_and_evidence():
    st = collision_state(); raw = copy.deepcopy(st)
    with pytest.raises(ValueError, match='Unique bounded constraint IDs'):
        OldBuilder().finish(st, QUESTION)
    b = Builder(); text = b.finish(st, QUESTION)
    model, _, audit = typed_sources.inspect(text, QUESTION)
    assert st == raw
    ids = [c['id'] for c in model['constraints']+model['logic']]
    assert len(ids) == len(set(ids)) == 4
    assert all(len(i) <= 64 for i in ids)
    expanded, _ = typed_sources.expand(builder.envelope(st), QUESTION)
    for field in ('constraints', 'logic'):
        strip = lambda cs: [{k: v for k, v in c.items() if k != 'id'} for c in cs]
        assert strip(model[field]) == strip(expanded['model'][field])
    obj = json.loads(text)
    assert obj['requirements'][-1]['targets'] == ['constraint:'+c['id'] for c in model['constraints'][1:]+model['logic']]
    assert obj['requirements'][-1]['quote'] == QUESTION
    assert not any('unknown targets' in issue for issue in audit['coverage_audit']['issues'])
    canonical, mapping = b.canonical_state(st)
    assert len(mapping) == 4 and all(m['original'] == 'cap' for m in mapping)
    assert text == b.finish(st, QUESTION) and canonical != st


def test_forall_rows_namespaces_before_expansion_even_duplicate_prefix_and_literal():
    st = state(); question = QUESTION+' Rows=[{"u":0},{"u":1},{"u":2}]'
    from qihc.modeling.sources import catalog
    source = next(v['id'] for v in catalog(question) if v['kind'] == 'table')
    rule = {'forall_rows': {'source': source, 'id_prefix': 'cap', 'terms': [
        {'index': {'column': 'u'}, 'coefficient': 1}], 'op': 'le', 'rhs': 1}}
    st['rules'][0]['constraints'] += [rule, copy.deepcopy(rule)]
    model, _ = typed_sources.compile_math(Builder().finish(st, question), question)
    assert len(model['constraints']) == 7
    assert len({c['id'] for c in model['constraints']}) == 7
    bad = copy.deepcopy(st); bad['rules'][0]['constraints'][1]['forall_rows']['terms'][0]['index'] = {'column': 'absent'}
    with pytest.raises(ValueError):
        Builder().finish(bad, question)


def review(candidate=None, feasible=True, value=0):
    return {'candidate': candidate or builder.assemble(state(), QUESTION),
            'checks': [{'assignment': {}, 'expected_objective': value, 'expected_feasible': feasible,
                        'source_quote': 'select at most two', 'reason': 'empty selection is allowed'},
                       {'assignment': {'0': 1}, 'expected_objective': 2, 'expected_feasible': True,
                        'source_quote': 'values [2,3,4]', 'reason': 'one item reward'}],
            'change_reason': 'source consistency check'}


def test_review_schema_and_atomic_validation_rejects_self_contradictions_and_names():
    initial = builder.finish(state(), QUESTION); obj = review()
    jsonschema.validate(obj, semantic_review.schema(QUESTION, 3))
    text, audit = semantic_review.validate_review(obj, initial, QUESTION, ['x0', 'x1', 'x2'])
    assert audit['status'] == 'accepted_self_consistent'
    assert audit['checks'][0]['quote_verified']
    renamed = copy.deepcopy(obj)
    renamed['candidate']['model']['constraints'][0]['id'] = 'renamed'
    _, renamed_audit = semantic_review.validate_review(renamed, initial, QUESTION, ['x0', 'x1', 'x2'])
    assert not renamed_audit['mathematics_changed']
    for field, value in [('expected_objective', 1), ('expected_feasible', False)]:
        bad = copy.deepcopy(obj); bad['checks'][0][field] = value
        with pytest.raises(ValueError, match='contradicts'):
            semantic_review.validate_review(bad, initial, QUESTION, ['x0', 'x1', 'x2'])
    bad = copy.deepcopy(obj); bad['candidate']['model']['variables']['prefix'] = 'y'
    with pytest.raises(ValueError, match='names/order'):
        semantic_review.validate_review(bad, initial, QUESTION, ['x0', 'x1', 'x2'])
    bad = copy.deepcopy(obj); bad['checks'][0]['source_quote'] = 'made up quote'
    _, audit = semantic_review.validate_review(bad, initial, QUESTION, ['x0', 'x1', 'x2'])
    assert not audit['checks'][0]['quote_verified']  # citation audit never becomes math gate


def row(obj, tokens=50):
    return {'status': 'generated', 'response': builder.dump(obj), 'output_tokens': tokens,
            'input_tokens': 10, 'finish_reason': 'json_complete'}


@pytest.mark.parametrize('valid', [True, False])
def test_review_rollback_and_cap(valid):
    req = study.request(case(), 'semantic_flat')
    job = review_workflow.factory(Path('req.json'), req)
    assert job.task()['_kind'] == 'flat'
    job.accept(row(review()['candidate']))
    assert not job.done and job.task()['_kind'] == 'semantic_review'
    initial = job.initial['response']
    job.accept(row(review(feasible=valid)))
    assert job.done
    status = job.result['semantic_review']['status']
    assert status == ('accepted_self_consistent' if valid else 'rejected_preserved_initial')
    assert job.result['initial_response'] == initial
    job = review_workflow.factory(Path('req.json'), req)
    job.accept(row(review()['candidate'], tokens=16300))
    assert job.task() is None and job.done
    assert job.result['semantic_review']['status'] == 'skipped_budget'
    assert job.result['mathematical_validity'] == 'valid'


FIXTURE = json.loads((Path(__file__).parent/'fixtures/prompt_v1_gate_failures.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case_id,arm', [('nlco_MAXCUT_L_017', 'legacy_flat'), ('assignment-7-20261026', 'concise_flat')])
def test_real_compilable_but_semantically_wrong_models_have_executable_witnesses(case_id, arm):
    c = next(c for c in FIXTURE['cases'] if c['id'] == case_id)
    r = next(r for r in FIXTURE['requests'] if r['case_id'] == case_id and r['arm'] == arm)
    text = r['steps'][-1]['generation']['response']
    model, _ = typed_sources.compile_math(text, c['question'])
    diag = semantic_review.diagnostics(text, c['question'])
    assert diag['variable_count'] == len(c['gold']['variables'])
    if case_id.startswith('nlco_'):
        # True source semantic regression: observed polynomial rewards equal endpoints.
        assert model['objective'] != c['gold']['objective']
        assert diag['audits']['semantic_diagnostics']['single_variable_equalities_total'] > 0
        x = [1]*len(model['variables'])
        assert float(objective(model, x)) != float(objective(c['gold'], x))
    else:
        assert len(model['constraints']) == 2 and len(c['gold']['constraints']) == 14
        from qihc.generic.bounded_reference import solve
        from qihc.experiment_runtime.evaluation import evaluate
        raw = solve(model, 2)
        assessed = evaluate(c, model, raw)
        assert assessed['predicted_model_feasible'] and not assessed['verified_feasible']
        evidence = study.failure_attribution(c, model, assessed)
        assert evidence['category'] == 'predicted_feasible_original_infeasible_witness'
        assert evidence['original_violations']
    # A manually corrected reference is only a checker regression, NOT generated success.
    x = [v['lower'] for v in c['gold']['variables']]
    manual = {'candidate': {'model': c['gold']}, 'checks': [{'assignment': {},
        'expected_objective': float(objective(c['gold'], x)), 'expected_feasible': bool(violations(c['gold'], x).sum() == 0),
        'source_quote': '', 'reason': 'offline reference fixture, never sent to GPU'}], 'change_reason': 'test only'}
    y = list(x); y[0] += 1
    manual['checks'].append(dict(manual['checks'][0], assignment={'0': y[0]},
        expected_objective=float(objective(c['gold'], y)), expected_feasible=bool(violations(c['gold'], y).sum() == 0)))
    _, audit = semantic_review.validate_review(manual, text, c['question'], [v['name'] for v in c['gold']['variables']])
    assert audit['mathematics_changed']


def test_unchanged_baseline_prompts_no_oracle_leak_and_matching_initial_requests():
    c = case(); altered = copy.deepcopy(c)
    altered['gold']['objective']['linear'] = [987, 988, 989]
    altered['gold']['constraints'] = []; altered['reference_answer'] = 98765
    for arm in study.ARMS:
        assert study.request(c, arm) == study.request(altered, arm)
    for arm in ('legacy_flat', 'typed_flat', 'typed_staged'):
        assert study.request(c, arm)['prompt'] == v2.request(c, arm)['prompt']
    for a, b in [('typed_flat', 'semantic_flat'), ('typed_staged', 'canonical_staged'), ('canonical_staged', 'semantic_staged')]:
        aa = review_workflow.factory(Path('a'), study.request(c, a)).task()
        bb = review_workflow.factory(Path('b'), study.request(c, b)).task()
        assert aa == bb


def test_six_arms_real_search_pairing_resume_and_review_reporting(tmp_path, monkeypatch):
    root, source, model = tmp_path/'run', tmp_path/'source', tmp_path/'model'
    s.save(model/'config.json', {'fixture': True})
    s.save(source/'cases.CPU_ONLY.json', [case()])
    s.save(source/'manifest.json', {'cases_sha256': s.digest(source/'cases.CPU_ONLY.json'), 'settings': {'model': str(model)}})
    study.prepare(root, source)
    from qihc.modeling import generation
    def generate(g, tasks):
        out = []
        for p, task in tasks:
            if task['_kind'] == 'semantic_review':
                jsonschema.validate(review(), task['_schema'])
                out.append(row(review()))
            else:
                out.extend(fake_generation(g, [(p, task)]))
        return out
    monkeypatch.setattr(generation, 'generate', generate)
    for p in (root/'formal/requests').glob('*.json'):
        study.execute(generator(), [(p, s.load(p))], root/'formal', 0, p.stem)
    cfg = s.load(root/'manifest.json')['config']; cfg.update(budget=256, chains=2, seeds=[0, 1], solver_seconds=.2)
    manifest = s.load(root/'manifest.json'); manifest['config'] = cfg; s.save(root/'manifest.json', manifest)
    for _, args in study.tasks(root/'formal', [case()], cfg):
        study.evaluate_one(*args)
    result = report.summarize(root, 'formal')
    assert result['status'] == 'complete' and result['completed_case_arms'] == 6
    assert len(result['paired']) == 10
    assert study.gate_check(result)['passed']
    bad_gate = copy.deepcopy(result)
    bad_gate['overall']['semantic_flat']['semantic_review_statuses'] = {'rejected_preserved_initial': 1}
    assert not study.gate_check(bad_gate)['passed']
    assert all(v['paired_cases'] == 1 for v in result['paired'])
    for a in ['semantic_flat', 'semantic_staged']:
        assert result['overall'][a]['semantic_review_statuses'] == {'accepted_self_consistent': 1}
        assert result['overall'][a]['backends']['cp_sat']['review_transition_runs'] == {'unchanged': 1}
    monkeypatch.setattr(generation, 'generate', lambda *a, **k: pytest.fail('Regenerated on resume'))
    for p in (root/'formal/requests').glob('*.json'):
        study.execute(generator(), [(p, s.load(p))], root/'formal', 0, 'resume')
    assert not list(study.tasks(root/'formal', [case()], cfg))
    from experiments.prompt_efficiency_v3.diagnose_existing import diagnose
    before = {str(p.relative_to(root)): s.digest(p) for p in root.rglob('*') if p.is_file()}
    diagnostic = diagnose(root, tmp_path/'diagnosis')
    assert diagnostic['observed_typed_flat_case_arms'] == 1
    assert diagnostic['categories_per_scored_seed'] == {'original_feasible_semantics_not_proven': 3}
    assert before == {str(p.relative_to(root)): s.digest(p) for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='outside'):
        diagnose(root, root/'diagnostics')
