import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.public_solver_v1 import common, data, modeling, solve
from experiments.public_solver_v1.report import report
from qihc.experiment_runtime import storage as s
from qihc.generic.bounded import objective, violations


def graph_row(task):
    edges = [{'u': 1, 'v': 2}, {'u': 2, 'v': 3}, {'u': 3, 'v': 4}]
    return {'id': task+'_S_001', 'task_id': task, 'difficulty_tier': 'S',
        'instance_canonical_json': {'num_nodes': 4, 'edges': edges},
        'instance_surface_json': {'num_nodes': 4, 'edges': [
            {'u': 'A', 'v': 'B'}, {'u': 'B', 'v': 'C'}, {'u': 'C', 'v': 'D'}]},
        'reference_solution_canonical_json': [2, 3], 'reference_objective_value': 2,
        'prompt': 'Select graph vertices.', 'surface_format': 'json'}


@pytest.mark.parametrize('task', ['MCP', 'MDS'])
def test_new_public_adapters_exhaustive(task):
    c = data.convert(graph_row(task)); edges = {(0,1), (1,2), (2,3)}
    for x in itertools.product((0,1), repeat=4):
        chosen = [i for i,v in enumerate(x) if v]
        feasible = (all((u,v) in edges for u,v in itertools.combinations(chosen,2)) if task == 'MCP'
                    else all(x[u] or any(x[v] and tuple(sorted((u,v))) in edges for v in range(4)) for u in range(4)))
        assert bool(violations(c['gold'], x).sum() == 0) == feasible
        assert objective(c['gold'], x) == sum(x)


def test_adapter_rejects_label_and_reference_errors():
    r = graph_row('MCP'); r['instance_surface_json']['edges'][1]['u'] = 'C'
    with pytest.raises(ValueError, match='mapping'):
        data.convert(r)
    r = graph_row('MDS'); r['reference_objective_value'] = 99
    with pytest.raises(ValueError, match='witness'):
        data.convert(r)


def toy():
    return {'id': 'toy', 'dataset': 'test_S', 'original_id': 'test_S_001', 'question': 'Select exactly one item.',
        'gold': {'schema':'bounded-integer-v2', 'variables': [{'name':'x0','lower':0,'upper':1}, {'name':'x1','lower':0,'upper':1}],
                 'objective': {'sense':'max','linear':[2,3],'quadratic':[]},
                 'constraints':[{'id':'one','a':[1,1],'op':'eq','rhs':1}], 'logic': []},
        'reference_answer': 3}


def fixture(root, backends=('cp_sat','scip','highs','sa','pt','penalty_pt','pbit')):
    c = toy(); cfg = {'source_arms':['canonical',common.ARM], 'backends':list(backends),
        'seconds':[.05], 'seeds':[0,1], 'chains':4, 'reference_seconds':1,
        'report_interval_seconds':1, 'cpu_workers':2}
    r = common.request(c); rp = root/'requests'/(r['id']+'.json'); s.save(rp,r)
    m = {'config':cfg,'requests':{r['id']:s.value_digest(r)},'scope':'test', 'selected_before_validation':1,'excluded':[]}
    s.save(root/'manifest.json',m); s.save(root/'cases.CPU_ONLY.json',[c])
    s.save(root/'models'/(s.key(c['id'],'canonical')+'.json'),common.base_record(c,'canonical',c['gold'],'compiled',s.value_digest(c['gold'])))
    # Deliberately invalid generated model. It must terminate/count without blocking canonical.
    modeling.failure(root,rp,'test malformed output')
    return m,c,rp


def test_semantically_wrong_but_executable_is_admitted(tmp_path):
    m,c,rp = fixture(tmp_path)
    wrong = copy.deepcopy(c['gold']); wrong['constraints'] = []
    s.save(tmp_path/'responses'/rp.name, {'mathematical_validity':'valid','status':'generated',
        'response':json.dumps({'model':wrong}), 'request_sha256':s.digest(rp)})
    common.import_responses(tmp_path,m,[c])
    record = s.load(tmp_path/'models'/rp.name)
    assert record['status'] == 'compiled' and record['model']['constraints'] == []
    # Source changes remain fatal identity failures, not silently skipped semantics.
    row = s.load(tmp_path/'responses'/rp.name); row['response'] += ' '
    s.save(tmp_path/'responses'/rp.name,row)
    with pytest.raises(ValueError,match='Frozen'):
        common.import_responses(tmp_path,m,[c])


def test_requests_have_no_gold_or_reference():
    r = common.request(toy())
    assert not {'gold','reference_answer','reference_solution_canonical_json'} & set(r)
    assert not r['semantic_review']


def test_real_seven_backend_pipeline_invalid_models_and_resume(tmp_path, monkeypatch):
    m,c,rp = fixture(tmp_path)
    monkeypatch.setattr(solve,'validate',lambda *a,**kw:m)
    assert solve.run(tmp_path,workers=3) == 0
    summary = s.load(tmp_path/'summary.json')
    assert summary['all_terminal'] and summary['completed_scores'] == 28
    assert sum(x['runs'] for x in summary['failures'] if x['status']=='invalid_model') == 14
    assert summary['execution_errors'] == 0
    before = {p:s.digest(p) for p in (tmp_path/'scores').rglob('*.json')}
    assert solve.run(tmp_path,workers=3) == 0
    assert before == {p:s.digest(p) for p in (tmp_path/'scores').rglob('*.json')}
    assert '# E003-1.1' in (tmp_path/'report.md').read_text()


def test_process_crash_does_not_stop_other_backend(tmp_path,monkeypatch):
    m,c,rp = fixture(tmp_path,('highs','pbit'))
    monkeypatch.setattr(solve,'validate',lambda *a,**kw:m)
    real = subprocess.Popen
    def launch(command, **kwargs):
        job = s.load(Path(command[-1]))
        if job['backend']=='highs':
            return real([sys.executable,'-c','raise SystemExit(77)'],**kwargs)
        return real(command,**kwargs)
    monkeypatch.setattr(solve.subprocess,'Popen',launch)
    assert solve.run(tmp_path,workers=2) == 0
    result = report(tmp_path)
    assert result['completed_scores']==8 and result['execution_errors']==4
    assert any(x['backend']=='pbit' and x['status']=='searched' for x in result['failures'])


def test_timeout_marks_only_missing_and_preserves_checkpoint(tmp_path):
    m,c,rp = fixture(tmp_path,('pbit',)); identity = s.key(c['id'],'canonical')
    job = {'case':c,'identity':identity,'arm':'canonical','backend':'pbit'}
    solve.error_score(tmp_path,c,identity,'pbit',.05,0,'execution_error','first')
    path = solve.score_path(tmp_path,identity,'pbit',.05,0); sha = s.digest(path)
    solve.fail_remaining(tmp_path,job,m['config'],'hard_timeout','test watchdog')
    assert s.digest(path)==sha and not solve.missing(tmp_path,job,m['config'])
    assert s.load(solve.score_path(tmp_path,identity,'pbit',.05,1))['status']=='hard_timeout'


def test_failure_output_idempotent_and_assigned_replica(tmp_path):
    m,c,rp = fixture(tmp_path); sha=s.digest(tmp_path/'responses'/rp.name)
    modeling.failure(tmp_path,rp,'second error')
    assert s.digest(tmp_path/'responses'/rp.name)==sha
    assert not modeling.items(tmp_path,m,int(rp.stem,16)%2)


def test_gpu_worker_skips_bad_request_and_continues_without_gate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from qihc.modeling import backend, typed_workflow
    requests = []
    for i in range(20):
        c = toy(); c['id'] = str(i)
        r = common.request(c)
        if int(r['id'],16)%2 == 0:
            requests.append(r)
        if len(requests)==2:
            break
    m={'requests':{r['id']:s.value_digest(r) for r in requests},'config':{'batch_size':4},'model':'unused'}
    for r in requests:
        s.save(tmp_path/'requests'/(r['id']+'.json'),r)
    class OOM(RuntimeError):
        pass
    generator=SimpleNamespace(model=SimpleNamespace(hf_device_map={'':'0'}),
        torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))
    monkeypatch.setattr(backend,'LocalModel',lambda *a,**kw:generator)
    monkeypatch.setattr(modeling,'validate',lambda *a,**kw:m)
    bad=requests[0]['id']
    def execute(generator, items, root, rank, batch):
        if any(r['id']==bad for p,r in items):
            raise ValueError('bad individual output')
        for path,r in items:
            s.save(root/'responses'/path.name,{'status':'generated','request_sha256':s.digest(path)})
    monkeypatch.setattr(typed_workflow,'execute',execute)
    modeling.worker(tmp_path,0)
    assert s.load(tmp_path/'responses'/(bad+'.json'))['status']=='generation_error'
    assert s.load(tmp_path/'responses'/(requests[1]['id']+'.json'))['status']=='generated'
    assert s.load(tmp_path/'gpu-0.json')['status']=='complete'


def test_optional_complete_public_release():
    path = Path('artifacts/E003-NLCO')
    if not (path/'download_manifest.json').exists():
        pytest.skip('Pinned public release not downloaded')
    cases, excluded = data.load_cases(path)
    assert len(cases)==900 and not excluded
    from collections import Counter
    counts=Counter(c['dataset'] for c in cases)
    assert len(counts)==18 and set(counts.values())=={50}


def test_atomic_reader_lock_retry_is_bounded(tmp_path, monkeypatch):
    from qihc.experiment_runtime import resilient_storage as rs
    real=s.os.replace; calls=[]
    def flaky(a,b):
        calls.append(1)
        if len(calls)<3:
            raise PermissionError('transient reader lock')
        return real(a,b)
    monkeypatch.setattr(s.os,'replace',flaky)
    monkeypatch.setattr(rs.time,'sleep',lambda _:None)
    rs.save(tmp_path/'row.json',{'v':1})
    assert s.load(tmp_path/'row.json')=={'v':1} and len(calls)==3
    def denied(a,b):
        raise PermissionError('persistent permission denied')
    monkeypatch.setattr(s.os,'replace',denied)
    with pytest.raises(PermissionError):
        rs.save(tmp_path/'row.json',{'v':2})
    assert s.load(tmp_path/'row.json')=={'v':1}
    assert list(tmp_path.iterdir())==[tmp_path/'row.json']
