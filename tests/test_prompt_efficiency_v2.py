"""Actual v1 failures, strict typed bindings, local repair and legacy parity."""
import copy
import json
from pathlib import Path

import pytest
import jsonschema

from qihc.modeling import workflow, builder, protocol, typed_sources, typed_schema, local_repair, typed_workflow
from qihc.modeling.typed_builder import Builder
from qihc.experiment_runtime import storage
from experiments.prompt_efficiency_v1 import study as v1
from experiments.prompt_efficiency_v2 import study as v2, report

FIXTURE = json.loads((Path(__file__).parent/'fixtures/prompt_v1_gate_failures.json').read_text(encoding='utf-8'))
CASES = {c['id']: c for c in FIXTURE['cases']}


@pytest.mark.parametrize('row', FIXTURE['requests'], ids=lambda r: r['case_id']+':'+r['arm'])
def test_real_archive_replays_identically_and_original_flat_is_unchanged(row):
    case = CASES[row['case_id']]
    request = v1.request(case, row['arm'])
    job = workflow.Job(Path('fixture.json'), request)
    for step in row['steps']:
        assert job.task() is not None
        job.accept(step['generation'])
    if not job.done:
        job.task()
    assert job.done and job.result['status'] == row['expected_status']
    assert job.result.get('validation_error') == row['expected_error']
    baseline = v2.request(case, 'legacy_flat')
    assert baseline['prompt'] == v1.request(case, 'legacy_flat')['prompt']
    assert type(typed_workflow.factory(Path('fixture.json'), baseline)) is workflow.Job


DOMAIN_FAILURES = [r for r in FIXTURE['requests'] if r['expected_error'] == 'Enumerate objective and domain obligations']


@pytest.mark.parametrize('row', DOMAIN_FAILURES, ids=lambda r: r['case_id']+':'+r['arm'])
def test_actual_missing_domain_annotations_are_audited_not_executed(row):
    case = CASES[row['case_id']]
    plan = protocol.load_object(row['steps'][-1]['generation']['response'])
    state = Builder().validate_plan(plan, case['question'], [v['name'] for v in case['gold']['variables']])
    assert any('domain' in note for note in state['audit_notes'])
    assert state['requirements'] == plan['requirements']  # No guessed obligations.
    plan['variables']['upper'] = -1
    with pytest.raises(ValueError):
        Builder().validate_plan(plan, case['question'], None)


def test_explicit_index_binding_rejects_guessing_duplicates_missing_labels_and_bounds():
    q = 'Rows=[{"id":10,"value":5},{"id":30,"value":9}]'
    resolver = typed_sources.Resolver(q)
    assert resolver.index({'column': 'id', 'offset': -10}, {'id': 10}, 0, 2, 'test') == 0
    lookup = {'index_of': {'source': 'd0', 'column': 'id', 'value': {'column': 'id'}}}
    assert resolver.index(lookup, {'id': 30}, 0, 2, 'test') == 1
    with pytest.raises(ValueError, match='outside'):
        resolver.index({'column': 'id'}, {'id': 30}, 0, 2, 'test')
    with pytest.raises(ValueError, match='absent'):
        resolver.index(lookup, {'id': 20}, 0, 2, 'test')
    with pytest.raises(ValueError, match='unique'):
        typed_sources.Resolver('Rows=[{"id":10},{"id":10}]').index(lookup, {'id': 10}, 0, 2, 'test')
    with pytest.raises(ValueError, match='outside'):
        resolver.index(lookup, {'id': 30}, 0, 1, 'test')


@pytest.mark.parametrize('row', [r for r in FIXTURE['requests'] if r['expected_error'] == 'objective.linear: index 24 outside 0..23'])
def test_actual_identifier_errors_require_an_explicit_mapping(row):
    case = CASES[row['case_id']]
    failed = protocol.load_object(row['steps'][-1]['generation']['response'])
    objective = failed['model']['objective'] if row['arm'] == 'concise_flat' else failed['objective']
    ref = objective['linear']['sum_rows']
    term = ref['terms'][0]
    assert term['index']['column'] == 'prototype_id' and term['index'].get('scale', 1) == 1
    term['index'] = {'index_of': {'source': ref['source'], 'column': 'prototype_id', 'value': {'column': 'prototype_id'}}}
    vector = typed_sources.Resolver(case['question']).vector(objective['linear'], 24, 'objective.linear')
    assert vector == case['gold']['objective']['linear']


def test_nested_schema_rejects_actual_wrong_columns_and_illegal_flatten():
    row = next(r for r in FIXTURE['requests'] if r['case_id'].startswith('knapsack') and r['arm'] == 'legacy_staged')
    case = CASES[row['case_id']]
    schema = typed_schema.schema('objective', case['question'], 64)
    for step in row['steps'][1:]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(protocol.load_object(step['generation']['response']), schema)
    jsonschema.validate({'objective': {'sense': 'max', 'linear': {'source': 'd2'}, 'quadratic': []}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({'objective': {'sense': 'max', 'linear': {'source': 'd2', 'flatten': True}, 'quadratic': []}}, schema)


def test_local_repair_atomic_preserves_ids_rules_and_source_inputs():
    obj = {'objective': {'sense': 'min', 'linear': {'source': 'd0', 'column': 'bad'}, 'quadratic': []},
           'constraints': [{'id': 'keep', 'a': {'0': 1}, 'op': 'le', 'rhs': 2}]}
    before = copy.deepcopy(obj)
    paths = local_repair.allowed_paths(obj)
    fixed = local_repair.apply(obj, json.dumps({'edits': [{'path': '/objective/linear', 'value': {'source': 'd0'}}], 'reason': 'array'}), paths)
    assert obj == before and fixed['constraints'] == before['constraints']
    for path in ('/constraints', '/constraints/0/id', '/constraints/0', '/objective/new'):
        with pytest.raises(ValueError):
            local_repair.apply(obj, json.dumps({'edits': [{'path': path, 'value': []}], 'reason': 'bad'}), paths)
    with pytest.raises(ValueError, match='duplicate'):
        local_repair.apply(obj, json.dumps({'edits': [{'path': '/objective/linear', 'value': {}}]*2, 'reason': 'bad'}), paths)


def test_real_missing_column_is_locally_repaired_with_shared_budget():
    case = next(c for c in CASES.values() if c['id'].startswith('knapsack'))
    row = next(r for r in FIXTURE['requests'] if r['case_id'] == case['id'] and r['arm'] == 'legacy_staged')
    job = typed_workflow.factory(Path('fixture.json'), v2.request(case, 'typed_staged_local'))
    job.task(); job.accept(row['steps'][0]['generation'])
    job.task(); job.accept(row['steps'][1]['generation'])
    task = job.task()
    assert task['_kind'] == 'local_repair' and '/objective/linear' in job.paths
    patch = {'edits': [{'path': '/objective/linear', 'value': {'source': 'd2'}}], 'reason': 'd2 is an integer vector'}
    jsonschema.validate(patch, task['_schema'])
    job.accept({'status': 'generated', 'response': json.dumps(patch), 'output_tokens': 30, 'input_tokens': 50})
    assert job.kind == 'rules' and job.error is None and job.repairs == 1
    assert job.tokens == sum(s['generation']['output_tokens'] for s in row['steps'][:2])+30
    assert job.state['requirements'] == protocol.load_object(row['steps'][0]['generation']['response'])['requirements']


def test_duplicate_json_keys_are_not_silently_repaired():
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        protocol.load_object('{"objective":{"linear":{},"linear":[]}}')


def test_local_patch_checkpoint_replay_does_not_generate_again(tmp_path, monkeypatch):
    from test_prompt_efficiency_v1 import generator
    from qihc.modeling import generation
    case = next(c for c in CASES.values() if c['id'].startswith('knapsack'))
    old = next(r for r in FIXTURE['requests'] if r['case_id'] == case['id'] and r['arm'] == 'legacy_staged')
    request = v2.request(case, 'typed_staged_local')
    path = tmp_path/'requests'/'test.json'; storage.save(path, request)
    calls = []
    def generate(_, items, **kwargs):
        rows = []
        for _, task in items:
            calls.append(task['_kind'])
            if task['_kind'] in ('plan', 'objective'):
                rows.append(old['steps'][0 if task['_kind'] == 'plan' else 1]['generation'])
                continue
            obj = ({'edits': [{'path': '/objective/linear', 'value': {'source': 'd2'}}], 'reason': 'vector reference'}
                   if task['_kind'] == 'local_repair' else {'items': [{'requirement_id': 'r1',
                       'constraints': case['gold']['constraints'], 'logic': []}]})
            rows.append({'status': 'generated', 'response': json.dumps(obj), 'output_tokens': 40,
                         'input_tokens': 50, 'finish_reason': 'json_complete'})
        return rows
    monkeypatch.setattr(generation, 'generate', generate)
    v2.execute(generator(), [(path, request)], tmp_path, 0, 'first')
    first = storage.load(tmp_path/'responses/test.json')
    assert first['mathematical_validity'] == 'valid' and first['local_repairs'] == 1
    assert calls == ['plan', 'objective', 'local_repair', 'rules']
    v2.execute(generator(), [(path, request)], tmp_path, 0, 'resumed')
    assert len(calls) == 4 and storage.load(tmp_path/'responses/test.json') == first


def test_data_only_prompt_assistance_and_generic_shapes():
    for case in CASES.values():
        changed = copy.deepcopy(case)
        changed['gold']['objective']['linear'] = [999]*len(case['gold']['variables'])
        changed['gold']['constraints'] = []; changed['reference_answer'] = 999
        for arm in v2.ARMS:
            assert v2.request(case, arm) == v2.request(changed, arm)
    arbitrary = 'Warehouse labels [5,9]; unrelated table [{"a":5,"b":9}]'
    contract = typed_sources.describe(arbitrary)
    assert contract[0]['columns'] == [] and contract[1]['columns'] == ['a', 'b']


def test_different_remaining_budgets_do_not_truncate_each_other(tmp_path, monkeypatch):
    from test_prompt_efficiency_v1 import generator
    from qihc.modeling import generation
    class BudgetJob(workflow.Job):
        def task(self):
            task = super().task()
            task['_max_new_tokens'] = self.request['test_cap']
            return task
    batches = []
    def generate(_, items, **kwargs):
        caps = [r['_max_new_tokens'] for _, r in items]
        batches.append(caps)
        assert len(set(caps)) == 1
        return [{'status': 'generated', 'response': '{"unsupported":"fixture"}',
                 'output_tokens': 10, 'input_tokens': 10} for _ in items]
    monkeypatch.setattr(generation, 'generate', generate)
    items = []
    case = next(iter(CASES.values()))
    for cap in (128, 6144):
        req = dict(v2.request(case, 'legacy_flat'), test_cap=cap)
        p = tmp_path/'requests'/f'{cap}.json'; storage.save(p, req); items.append((p, req))
    workflow.execute(generator(), items, tmp_path, 0, 'budgets', job_factory=BudgetJob, isolate_budgets=True)
    assert batches == [[128], [6144]]


def test_all_arms_complete_actual_cpu_scoring_resume_and_report(tmp_path, monkeypatch):
    from test_prompt_efficiency_v1 import case, plan, state, generator
    from qihc.modeling import generation
    c = case(); root = tmp_path/'run'; source = tmp_path/'source'; model = tmp_path/'model'
    storage.save(model/'config.json', {'fixture': True})
    storage.save(source/'cases.CPU_ONLY.json', [c])
    storage.save(source/'manifest.json', {'cases_sha256': storage.digest(source/'cases.CPU_ONLY.json'), 'settings': {'model': str(model)}})
    v2.prepare(root, source)
    def generate(_, items, **kwargs):
        out = []
        for _, task in items:
            value = {'flat': builder.assemble(state(), c['question']), 'plan': plan(),
                     'objective': {'objective': state()['objective']}, 'rules': {'items': state()['rules']}}[task['_kind']]
            if task.get('_schema'):
                jsonschema.validate(value, task['_schema'])
            out.append({'status': 'generated', 'response': json.dumps(value), 'output_tokens': 40, 'input_tokens': 100, 'finish_reason': 'json_complete'})
        return out
    monkeypatch.setattr(generation, 'generate', generate)
    for p in (root/'formal/requests').glob('*.json'):
        req = storage.load(p); v2.execute(generator(), [(p, req)], root/'formal', 0, p.stem)
        before = storage.load(root/'formal/responses'/p.name)
        v2.execute(generator(), [(p, req)], root/'formal', 0, p.stem)
        assert before == storage.load(root/'formal/responses'/p.name)
    cfg = storage.load(root/'manifest.json')['config']
    for _, args in v2.tasks(root/'formal', [c], cfg):
        v2.evaluate_one(*args)
    result = report.summarize(root, 'formal')
    assert result['status'] == 'complete' and result['completed_case_arms'] == 5
    assert all(r['backends']['cp_sat']['verified_feasible_rate'] == 1 for r in result['overall'].values())
    assert len(result['paired']) == 8
