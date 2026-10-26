"""Shared-core parity, actual CPU search, resume integrity and paired scoring."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qihc.experiment_runtime import storage as s
from qihc.modeling import builder as b, protocol, workflow as w
from experiments.prompt_efficiency_v1 import study, report
from qihc.experiment_runtime import runner as run


QUESTION = 'Choose binary x0,x1,x2 with values [2,3,4]; select at most two.'


def plan():
    return {'variables': {'count': 3, 'prefix': 'x', 'lower': 0, 'upper': 1},
            'requirements': [{'id': rid, 'kind': kind, 'quote': QUESTION, 'explanation': 'meaning', 'expected_count': n}
                             for rid, kind, n in [('o', 'objective', 1), ('d', 'domain', 3), ('c', 'constraint', 1)]]}


def state():
    st = b.validate_plan(plan(), QUESTION, ['x0', 'x1', 'x2'])
    st = b.objective_step(st, {'objective': {'sense': 'max', 'linear': [2, 3, 4], 'quadratic': []}}, QUESTION)
    return b.rules_step(st, {'items': [{'requirement_id': 'c', 'constraints': [
        {'id': 'cap', 'a': [1, 1, 1], 'op': 'le', 'rhs': 2}], 'logic': []}]}, ['c'], QUESTION)


def case():
    return {'id': 'toy', 'dataset': 'fixture', 'question': QUESTION,
            'gold': protocol.compile_math(b.finish(state(), QUESTION), QUESTION)[0], 'reference_answer': 7}


def prepare_fixture(tmp_path):
    root, source, model = tmp_path/'new', tmp_path/'source', tmp_path/'model'
    s.save(model/'config.json', {'fixture': True})
    s.save(source/'cases.CPU_ONLY.json', [case()])
    s.save(source/'manifest.json', {'cases_sha256': s.digest(source/'cases.CPU_ONLY.json'), 'settings': {'model': str(model)}})
    study.prepare(root, source)
    return root, source


def fake_generation(generator, items, **kwargs):
    result = []
    for _, task in items:
        kind = task['_kind']
        obj = {'flat': b.assemble(state(), QUESTION), 'plan': plan(),
               'objective': {'objective': state()['objective']}, 'rules': {'items': state()['rules']}}[kind]
        result.append({'status': 'generated', 'response': b.dump(obj), 'input_tokens': len(task['prompt']),
                       'output_tokens': 50, 'finish_reason': 'json_complete'})
    return result


def generator():
    class OOM(Exception):
        pass
    return SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None)))


def generate_all(root, monkeypatch):
    from qihc.modeling import generation
    monkeypatch.setattr(generation, 'generate', fake_generation)
    for p in (root/'formal/requests').glob('*.json'):
        w.execute(generator(), [(p, s.load(p))], root/'formal', 0, p.stem)


@pytest.mark.parametrize('index', range(8))
def test_extracted_protocol_parity_on_real_failures(index):
    from tools.runtime.modeling_v6 import protocol as old
    cases = json.loads((Path(__file__).parent/'fixtures/modeling_v3_gate_regression.json').read_text(encoding='utf-8'))['cases']
    row = cases[index]
    def outcome(fn):
        try:
            return fn(row['response'], row['question'])
        except Exception as exc:
            return type(exc).__name__, str(exc)
    assert outcome(protocol.inspect) == outcome(old.inspect)
    assert protocol.modeling_prompt(row) == old.modeling_prompt(row)


def test_prompts_no_gold_leak_all_arms_and_smaller_inputs():
    c = case(); other = copy.deepcopy(c)
    other['gold']['objective']['linear'] = [991, 992, 993]
    other['gold']['constraints'] = []; other['reference_answer'] = 994
    for arm in study.ARMS:
        a, z = study.request(c, arm), study.request(other, arm)
        assert a == z
        assert QUESTION in a['prompt']
        assert 'gold' not in a and 'reference_answer' not in a
    assert len(study.request(c, 'concise_flat')['prompt']) < len(study.request(c, 'legacy_flat')['prompt'])
    assert len(study.request(c, 'concise_staged')['prompt']) < len(study.request(c, 'legacy_staged')['prompt'])
    assert study.request(c, 'semantic_staged')['frontend'] == 'staged'


def test_prepare_resume_conflict_and_model_identity(tmp_path):
    root, source = prepare_fixture(tmp_path)
    manifest = s.load(root/'manifest.json'); assert study.prepare(root, source) == manifest
    assert study.validate(root, model=True) == manifest
    assert manifest['phase_cases_sha256']['formal'] == s.digest(root/'formal/cases.CPU_ONLY.json')
    p = next((root/'formal/requests').glob('*.json'))
    assert manifest['request_hashes'][p.stem] == s.digest(p)
    row = s.load(p); row['prompt'] += 'tampered'; s.save(p, row)
    with pytest.raises(ValueError, match='Stored request changed'):
        study.prepare(root, source)


def test_answer_only_data_is_not_mislabeled_as_infeasible(tmp_path):
    _, source = prepare_fixture(tmp_path)
    c = case(); c['gold'] = None
    s.save(source/'cases.CPU_ONLY.json', [c])
    m = s.load(source/'manifest.json'); m['cases_sha256'] = s.digest(source/'cases.CPU_ONLY.json')
    s.save(source/'manifest.json', m)
    with pytest.raises(ValueError, match='requires standard mathematical models'):
        study.prepare(tmp_path/'answer-only', source)


def test_full_workflow_real_solvers_reports_and_resume(tmp_path, monkeypatch):
    root, _ = prepare_fixture(tmp_path); generate_all(root, monkeypatch)
    cfg = s.load(root/'manifest.json')['config']; cfg.update(budget=256, chains=2, seeds=[0, 1], solver_seconds=.2)
    # Test-only reduced budget; all arms still share identical settings.
    manifest = s.load(root/'manifest.json'); manifest['config'] = cfg; s.save(root/'manifest.json', manifest)
    for _, args in study.tasks(root/'formal', [case()], cfg):
        study.evaluate_one(*args)
    result = report.summarize(root, 'formal')
    assert result['status'] == 'complete' and result['completed_case_arms'] == 5
    assert run.gate_check(result)['passed']
    assert len(result['paired']) == 8
    assert all(v['paired_cases'] == 1 for v in result['paired'])
    assert all(v['backends']['cp_sat']['verified_feasible_rate'] == 1 for v in result['overall'].values())
    assert not list(study.tasks(root/'formal', [case()], cfg))
    # Generation replay must reuse exact substeps without executing another model call.
    from qihc.modeling import generation
    monkeypatch.setattr(generation, 'generate', lambda *_a, **_k: pytest.fail('Unexpected generation during resume'))
    p = next((root/'formal/requests').glob('*.json'))
    w.execute(generator(), [(p, s.load(p))], root/'formal', 0, 'resume')
    assert s.load(root/'formal/responses'/p.name)['mathematical_validity'] == 'valid'


def test_checkpoint_tampering_rejected(tmp_path, monkeypatch):
    root, _ = prepare_fixture(tmp_path); generate_all(root, monkeypatch)
    p = next((root/'formal/requests').glob('*.json'))
    checkpoint = root/'formal/builder_steps'/p.stem/'00.json'
    row = s.load(checkpoint); row['generation']['response'] = '{}'; s.save(checkpoint, row)
    with pytest.raises(ValueError, match='output integrity'):
        w.execute(generator(), [(p, s.load(p))], root/'formal', 0, 'bad')


def test_invalid_generation_counted_not_dropped(tmp_path):
    root, _ = prepare_fixture(tmp_path)
    cfg = s.load(root/'manifest.json')['config']
    for p in (root/'formal/requests').glob('*.json'):
        s.save(root/'formal/responses'/p.name, {'status': 'invalid_model_output', 'response': '{}',
               'request_sha256': s.digest(p), 'input_tokens': 10, 'output_tokens': 5,
               'generation_attempts': 1, 'seconds': .1, 'mathematical_validity': 'invalid'})
    for _, args in study.tasks(root/'formal', [case()], cfg):
        study.evaluate_one(*args)
    summary = report.summarize(root, 'formal')
    assert summary['status'] == 'complete' and not run.gate_check(summary)['passed']
    assert all(v['backends']['pbit']['verified_feasible_rate'] == 0 for v in summary['overall'].values())


def test_child_error_root_cause_visible(tmp_path, capsys):
    p = tmp_path/'gpu.log'; p.write_text('Traceback\nValueError: real root cause\n')
    with p.open('ab') as f:
        with pytest.raises(RuntimeError, match='GPU 0 exited 1'):
            run.child_failure([SimpleNamespace(poll=lambda: 1)], [f])
    assert 'real root cause' in capsys.readouterr().out


def test_core_no_legacy_runtime_imports():
    for folder in ('qihc/modeling', 'qihc/experiment_runtime'):
        for path in (Path(__file__).parents[1]/folder).glob('*.py'):
            assert 'from tools.' not in path.read_text(encoding='utf-8')


def test_launch_gate_to_formal_with_controlled_workers(tmp_path, monkeypatch):
    """Exercise supervisor phases/acceptance/reports without claiming a GPU run."""
    import itertools
    import torch
    from qihc.modeling import formatting, generation
    config_dir = tmp_path/'definition'
    cfg = s.load(study.HERE/'config.json')
    cfg.update(budget=256, chains=2, seeds=[0], solver_seconds=.2, cpu_workers=2)
    s.save(config_dir/'config.json', cfg)
    monkeypatch.setattr(study, 'HERE', config_dir)
    root, _ = prepare_fixture(tmp_path)
    manifest = s.load(root/'manifest.json')
    s.save(root/'preflight.json', {'status': 'passed', 'source_hash': manifest['source_hash']})
    monkeypatch.setattr(formatting, 'check_runtime', lambda: {'test': True})
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 4)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    monkeypatch.setattr(generation, 'generate', fake_generation)
    class Child:
        def __init__(self, *args, **kwargs):
            self.code = None
        def poll(self):
            if self.code is not None:
                return self.code
            directory = root/s.load(root/'phase.json')['phase']
            for p in (directory/'requests').glob('*.json'):
                if not (directory/'responses'/p.name).exists():
                    w.execute(generator(), [(p, s.load(p))], directory, 0, p.stem)
            return None
        def terminate(self): self.code = 0
        def kill(self): self.code = -9
        def wait(self): return self.code
    class Pool:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def submit(self, fn, *args):
            result = fn(*args)
            return SimpleNamespace(done=lambda: True, result=lambda: result)
    clock = itertools.count(step=31)
    monkeypatch.setattr(run.subprocess, 'Popen', Child)
    monkeypatch.setattr(run, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(run.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    assert run.launch(root, study) == 0
    assert s.load(root/'gate_acceptance.json')['passed']
    assert s.load(root/'formal/summary.json')['status'] == 'complete'
    assert s.load(root/'status.json')['status'] == 'complete'
