import asyncio
import copy
import json
from pathlib import Path
import subprocess
import sys

import httpx
import numpy as np

from experiments.feedback_active_v3.data import make_case, build, public_view, digest
from qihc.experiment_runtime.storage import save, load, key
from qihc.modeling.active_protocol import parse_model, parse_checks, guard, compact
from qihc.generic.feedback.verification import equivalent, enumerate_states, ordering_equal
from qihc.generic.bounded import objective


def config():
    c = load(Path('experiments/feedback_active_v3/config.json'))
    c.update(probe_seconds=.12, solver_seconds=.1, rounds=1, seeds=[17], extension_seeds=[17], cpu_workers=2)
    return c


def test_public_cohort_and_no_gold_in_public():
    from experiments.feedback_active_v3.validate_data import run
    data=Path('experiments/feedback_active_v3/assets/public_dataset')
    assert run(data)['synthetic_formal_cases']==0
    cases=load(data/'evaluation/cases.gold.json')
    assert set(public_view(cases[0])) == {'id','question','variable_names'}
    for c in cases:
        m = parse_model(json.dumps({'model':compact(c['gold'])}), public_view(c))
        assert equivalent(m,c['gold'])['status']=='verified_correct'


def test_weighted_maxsat_independent_literal_oracle():
    case = make_case('MAX2SAT', 8, 35)
    clauses = json.loads(case['question'].split('clauses = ')[1])
    xs = enumerate_states(case['gold']['variables']); expected = np.zeros(len(xs))
    for c in clauses:
        a = xs[:,c['i']] if c['positive_i'] else 1-xs[:,c['i']]
        b = xs[:,c['j']] if c['positive_j'] else 1-xs[:,c['j']]
        expected += c['weight']*(1-a)*(1-b)
    np.testing.assert_array_equal(expected, objective(case['gold'],xs)+case['objective_offset'])


def test_equivalence_preserves_ties_and_cross_chunk_order():
    assert not ordering_equal([0,0,1,1], [0,1,2,2])
    assert not ordering_equal([0,2,1], [0,1,2])
    assert ordering_equal([1,0,1], [5,3,5])
    case = make_case('KP',8,18); m = copy.deepcopy(case['gold'])
    assert equivalent(m,case['gold'])['status']=='verified_correct'
    m['constraints']=[]
    assert equivalent(m,case['gold'])['status']=='disproved'


def test_fallible_check_is_not_truth_or_patch_veto():
    case=make_case('KP',8,17); public=public_view(case)
    checks=parse_checks(json.dumps({'checks':[dict(id='wrong',quote='Choose items',confidence=.99,kind='constraint',a={'0':1},op='le',rhs=0)]}),public)
    assert checks['checks'][0]['trusted'] is False
    assert guard(public,case['gold'],json.dumps({'edits':[]}))['status']=='accepted'
    bad=parse_checks(json.dumps({'checks':[dict(id='fake',quote='invented quotation',confidence=1,kind='constraint',a={},op='le',rhs=0)]}),public)
    assert not bad['checks'] and bad['rejected']


def test_empirical_probe_and_real_replica_exchange():
    from qihc.generic.feedback.active import probe
    from qihc.generic.feedback.tempering import sample_pt
    case=make_case('KP',8,7); m=copy.deepcopy(case['gold']); m['constraints']=[]
    check=dict(case['gold']['constraints'][0],kind='constraint',quote='Total selected weight',confidence=1.)
    result=probe(m,{'checks':[check]},'P2',config(),17)
    assert result['trace'] and all('q' not in row for row in result['trace'])
    assert any(row['phase']=='confirmation' for row in result['trace'])
    assert result['cards']
    pt=sample_pt(m,.15,17,clamps={0:1})
    assert pt['replica_exchange_attempts']>0
    assert all(x[0]==1 for x in pt['final'])


def test_request_resume_does_not_repeat_spent_generation(tmp_path):
    from qihc.experiment_runtime.async_inference import InferencePool
    async def exercise():
        calls=[]
        def handler(request):
            calls.append(json.loads(request.content)); return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{}'}}],'usage':{'prompt_tokens':5,'completion_tokens':2}})
        pool=InferencePool(['http://local'],tmp_path,config()); await pool.client.aclose()
        pool.client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
        a,b=await asyncio.gather(pool.generate('same','hello',17),pool.generate('same','hello',17))
        assert a==b and len(calls)==1
        saved=load(tmp_path/'same.json');saved['status']='in_flight';save(tmp_path/'same.json',saved)
        assert (await pool.generate('same','hello',17))['status']=='interrupted_unknown'
        assert len(calls)==1
        await pool.close()
    asyncio.run(exercise())


def test_native_solvers_in_fresh_processes_and_trace(tmp_path):
    case=make_case('KP',8,11)
    for backend in ('scip','highs','cp_sat','pbit'):
        ip=tmp_path/(backend+'.in.json');op=tmp_path/(backend+'.out.json')
        save(ip,dict(stage='solve',model=case['gold'],backend=backend,seconds=.3,seed=17,chains=16))
        subprocess.run([sys.executable,'-m','experiments.feedback_active_v3.cpu',str(ip),str(op)],check=True,timeout=30)
        row=load(op)
        assert row.get('feasible'),row
        assert row.get('trace'),row
        assert all('x' in r and r['seconds']>=0 for r in row['trace'])


def test_partial_and_zero_report(tmp_path):
    from experiments.feedback_active_v3.report import report
    assert report(tmp_path)['status']=='partial'
    save(tmp_path/'stages/main.json',dict(trajectories=[dict(stage='main',arm='P2')]))
    row=report(tmp_path)
    assert row['expected']==1 and row['completed']==0
    assert row['groups'][0]['rounds_with_cards']==0


def test_ragged_source_sets_expand_without_task_dispatch():
    from qihc.modeling.sources import catalog
    case=make_case('WSC',8,23); public=public_view(case)
    sid=next(r['id'] for r in catalog(case['question']) if r['kind']=='table')
    model=compact(case['gold'])
    model['constraints']=[{'forall_sets':dict(source=sid,id_prefix='c',coefficient=1,index_offset=0,op='ge',rhs=1)}]
    assert parse_model(json.dumps({'model':model}),public)==case['gold']


def test_public_adapters_and_audit_are_not_invented_120_cases():
    from experiments.feedback_active_v3.public_data import cp_case
    path=Path('experiments/feedback_active_v3/assets/public_dataset/public_originals/cpbench_verified.jsonl')
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    accepted=[cp_case(row) for row in rows if cp_case(row) is not None]
    assert len(rows)==63 and len(accepted)==10
    from qihc.generic.feedback.verification import reference
    capital=next(c for c in accepted if c['id'].endswith('capital_budget'))
    chess=next(c for c in accepted if c['id'].endswith('chess_set'))
    assert reference(capital['gold'])['objective']==42
    assert reference(chess['gold'])['objective']==1330
    mismatch=next(r for r in rows if r['id'].endswith('maximal_independent_sets'))
    assert cp_case(mismatch) is None


def test_large_identity_and_scheduler_gpu_mapping(monkeypatch):
    from experiments.feedback_active_v3.launch import visible_device
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b,GPU-c,GPU-d')
    assert visible_device(2)=='GPU-c'
    gold=make_case('MIS',32,19)['gold']; model=copy.deepcopy(gold)
    model['constraints'].reverse()
    for i,row in enumerate(model['constraints']):
        row['id']=str(i);row['a']=[2*v for v in row['a']];row['rhs']*=2
    model['objective']['linear']=[3*v for v in model['objective']['linear']]
    assert equivalent(model,gold)['status']=='verified_correct'
    model['constraints']=[]
    assert equivalent(model,gold)['status']=='disproved'


def test_fake_llm_pipeline_resume_is_scaffolding_not_experiment(tmp_path):
    from experiments.feedback_active_v3.runtime import Pipeline
    from qihc.experiment_runtime.isolated_jobs import JobPool
    case=make_case('KP',8,9,split='development'); data=tmp_path/'data'; run=tmp_path/'run'
    save(data/'cases.public.json',[dict(public_view(case),family='KP',layer='direct',split='development')])
    save(data/'subsets.json',{'ablation':[],'audit':[],'replay':[]})
    class FixtureLLM:
        tokenizer=None
        def __init__(self): self.calls=0
        async def generate(self,k,prompt,seed,temperature=None):
            self.calls+=1
            text=json.dumps({'checks':[]}) if prompt.startswith('Extract') else json.dumps({'model':compact(case['gold'])})
            return dict(status='complete',text=text)
    async def exercise():
        llm=FixtureLLM(); pool=JobPool(run/'jobs',2,'experiments.feedback_active_v3.cpu')
        pipe=Pipeline(run,data,config(),llm,pool); await pipe.run('development'); calls=llm.calls
        await pipe.run('development'); assert llm.calls==calls
        states=[load(p) for p in (run/'trajectories').glob('*.json')]
        assert len(states)==7 and all(st['status']=='complete' for st in states)
        assert len([st for st in states if st['arm']=='C0'][0]['rounds'])==0
    asyncio.run(exercise())
