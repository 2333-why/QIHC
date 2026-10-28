"""Real E001-1.2 failures: grammar compatibility and non-oracle diagnostics."""
import copy
import json
from pathlib import Path
import pytest
import itertools
from qihc.modeling import formatting, typed_schema, typed_sources, semantic_review
from qihc.generic.bounded import violations

FIXTURE = json.loads((Path(__file__).parent/'fixtures/prompt_v2_archive_failures.json').read_text(encoding='utf-8'))


def test_opt_in_dataset_queue_interleaving(tmp_path):
    import hashlib
    from qihc.experiment_runtime import runner, storage
    for rank in (0, 1):
        directory = tmp_path/str(rank)
        for i, (cid, group, order) in enumerate([('a0', 'A', 0), ('a1', 'A', 1), ('b0', 'B', 0), ('b1', 'B', 1)]):
            storage.save(directory/'requests'/f'{2*i+rank:064x}.json', {
                'case_id': cid, 'phase': 'initial', 'frontend': 'typed', 'question': 'q',
                'question_sha256': hashlib.sha256(b'q').hexdigest(), 'queue_group': group, 'queue_order': order})
        rows = runner.pending(directory, rank, 4)
        assert [r['case_id'] for _, r in rows] == ['a0', 'b0', 'a1', 'b1']
        for p, r in rows:
            r.pop('queue_group'); r.pop('queue_order'); storage.save(p, r)
        assert [r['case_id'] for _, r in runner.pending(directory, rank, 4)] == ['a0', 'a1', 'b0', 'b1']


def consume(schema, text):
    p = formatting.parser(schema_override=schema)
    for ch in text:
        assert ch in p.get_allowed_characters(), (ch, text)
        p = p.add_character(ch)
    assert p.can_end()


def test_bool_numeric_const_lmfe_regression_and_execution_guard():
    for value in (True, False, 3, -2, None):
        consume({'type': 'object', 'properties': {'flag': {'const': value}}, 'required': ['flag']}, json.dumps({'flag': value}))
    question = 'Rows=[[1,2],[3,4]]'
    sid = next(v['id'] for v in typed_sources.describe(question) if v.get('flatten_reference'))
    for linear, n in [({'source': sid, 'flatten': True}, 4),
                      ({'sum_rows': {'source': sid, 'terms': [{'index': {'row_index': True}, 'coefficient': 1}]}}, 2)]:
        obj = {'objective': {'sense': 'min', 'linear': linear, 'quadratic': []}}
        consume(typed_schema.schema('objective', question, n), json.dumps(obj, separators=(',', ':')))
    bad = {'model': {'schema': 'bounded-integer-v2', 'variables': {'count': 4, 'prefix': 'x', 'lower': 0, 'upper': 1},
        'objective': {'sense': 'min', 'linear': {'source': sid, 'flatten': False}, 'quadratic': []}, 'constraints': [], 'logic': []}}
    with pytest.raises(ValueError, match='true flag'):
        typed_sources.compile_math(json.dumps(bad), question)


@pytest.mark.parametrize('index', range(4))
def test_real_semantic_failures_remain_detectable_and_not_counted_correct(index):
    row = FIXTURE['rows'][index]; case = row['case']
    text = row['response']['response']; model, _ = typed_sources.compile_math(text, case['question'])
    d = semantic_review.diagnostics(text, case['question'])
    witnesses = [s for s in row['scores'] if s.get('predicted_model_feasible') and not s.get('verified_feasible')]
    assert witnesses
    for score in witnesses:
        assert violations(model, score['x']).sum() == 0
        assert violations(case['gold'], score['x']).sum() > 0
    if index == 0:
        assert d['structural_questions']['duplicate_constraint_groups']
        assert d['structural_questions']['objective_vector_reused_as_constraint']
    if index == 1:
        assert d['structural_questions']['uniform_linear_binary_objective']
    # Regression checks validate evidence, not a claim an LLM fixed the sample.
    assert 'Warnings only' in d['structural_questions']['scope']


def test_real_duplicate_constraint_state_canonicalization():
    row = FIXTURE['rows'][4]
    from experiments.prompt_efficiency_v3.study import request
    from qihc.modeling.review_workflow import factory
    job = factory(Path('fixture.json'), request(row['case'], 'canonical_staged'))
    for step in row['steps']:
        if job.done:
            break
        assert job.task()['_kind'] == step['kind']
        job.accept(copy.deepcopy(step['generation']))
    assert job.done and job.result['mathematical_validity'] == 'valid', job.result
    text = job.result['response']
    model, _ = typed_sources.compile_math(text, row['case']['question'])
    assert len({c['id'] for c in model['constraints']+model['logic']}) == len(model['constraints'])+len(model['logic'])
    assert semantic_review.diagnostics(text, row['case']['question'])['structural_questions']['constant_false_constraints']


def test_affine_index_sums_exact_and_no_silent_index_repair():
    envelope = {'model': {'schema': 'bounded-integer-v2', 'variables': {'count': 9, 'prefix': 'x', 'lower': 0, 'upper': 1},
        'objective': {'sense': 'min', 'linear': {}, 'quadratic': []}, 'constraints': [], 'logic': []}}
    for prefix, gs, ts in [('r', 3, 1), ('c', 1, 3)]:
        envelope['model']['constraints'].append({'forall_affine': {'id_prefix': prefix, 'count': 3,
            'width': 3, 'start': 0, 'group_stride': gs, 'term_stride': ts, 'coefficient': 1, 'op': 'eq', 'rhs': 1}})
    model, _ = typed_sources.compile_math(json.dumps(envelope), '')
    for x in itertools.product(range(2), repeat=9):
        expected = all(sum(x[3*i+j] for j in range(3)) == 1 for i in range(3)) and all(sum(x[3*i+j] for i in range(3)) == 1 for j in range(3))
        assert bool(violations(model, x).sum() == 0) == expected
    consume(typed_schema.schema('flat', '', 9), json.dumps(envelope, separators=(',', ':')))
    for field, value in [('start', 1), ('term_stride', 0), ('count', 513)]:
        bad = copy.deepcopy(envelope); bad['model']['constraints'][0]['forall_affine'][field] = value
        with pytest.raises(ValueError):
            typed_sources.compile_math(json.dumps(bad), '')
