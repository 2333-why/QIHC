import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from qihc.experiment_runtime.storage import load,save
from qihc.generic.feedback.verification import equivalent
from experiments.pvcd_cycle_v1.evidence import cards_from_states,collect
from experiments.pvcd_cycle_v1.common import CFG

def problem():
    gold=dict(schema='bounded-integer-v2',variables=[dict(name='x0',lower=0,upper=1)],
              objective=dict(sense='max',linear=[1],quadratic=[]),logic=[],
              constraints=[dict(id='c0',a=[1],op='le',rhs=0)])
    wrong=copy.deepcopy(gold); wrong['constraints']=[]
    case=dict(id='fixture',variable_names=['x0'],question='Maximize binary x0 subject to x0 <= 0.')
    return case,gold,wrong

def test_bidirectional_cards_and_objective_ties():
    _,gold,wrong=problem()
    xs=[[0],[1]]
    loose=cards_from_states(wrong,gold,xs)
    tight=cards_from_states(gold,wrong,xs)
    assert loose==[dict(kind='feasibility',x=[1],correct_feasible=False,candidate_feasible=True)]
    assert tight==[dict(kind='feasibility',x=[1],correct_feasible=True,candidate_feasible=False)]
    tied=copy.deepcopy(wrong); tied['objective']['linear']=[0]
    cards=cards_from_states(tied,wrong,xs)
    assert cards[0]['kind']=='objective' and cards[0]['model_order']==0 and cards[0]['gold_order']!=0
    assert cards_from_states(gold,gold,xs)==[]
    renamed=copy.deepcopy(wrong);renamed['variables'][0]['name']='other'
    assert cards_from_states(renamed,gold,xs)==[]

def test_real_bidirectional_sources_have_same_assignment_budget():
    _,gold,wrong=problem();cfg=dict(CFG,probe_seconds_per_round=.12)
    traces=[]
    for uniform in (False,True):
        states,trace=collect(wrong,gold,17,cfg,uniform)
        traces.append(trace)
        assert len(states)==cfg['probe_calls']*cfg['chains']
        assert {r['side'] for r in trace}=={'candidate','gold'}
        assert cards_from_states(wrong,gold,states)
    assert [r['clamps'] for r in traces[0]]==[r['clamps'] for r in traces[1]]

def test_loop_refresh_and_unknown_never_distilled(monkeypatch):
    from experiments.pvcd_cycle_v1 import training as t
    from experiments.posttraining_v1.data import target
    case,gold,wrong=problem();mid=copy.deepcopy(wrong);mid['objective']['linear']=[0]
    calls=[]; outputs=iter([target(mid),target(gold)])
    def sample(prompt,seed): calls.append(prompt);text=next(outputs);return [1],text
    tok=lambda text,**kw: {'offset_mapping':[(0,len(text))]}
    e=SimpleNamespace(sample=sample,tok=tok)
    monkeypatch.setattr(t,'collect',lambda m,g,s,c,u: ([[0],[1]],[]))
    r,trace=t.repair_chain(e,case,gold,dict(model=wrong,text=target(wrong)),17,'cycle',CFG)
    assert r and len(trace)==2 and target(mid) in calls[1]
    assert trace[-1]['semantic']['status']=='verified_correct'
    # A finite witness set passing is not proof. An unknown verifier result must be rejected.
    monkeypatch.setattr(t,'equivalent',lambda m,g:dict(status='unknown'))
    e.sample=lambda p,s:([1],target(gold))
    r,trace=t.repair_chain(e,case,gold,dict(model=wrong,text=target(wrong)),17,'cycle',CFG)
    assert r is None
    # Static ablation re-presents the original attempted model after a failed repair.
    calls.clear();outputs=iter([target(mid),target(gold)]);e.sample=sample
    t.repair_chain(e,case,gold,dict(model=wrong,text=target(wrong)),17,'cycle_static',CFG)
    assert '\nAttempt: '+target(wrong)+'\n' in calls[1]

def test_parse_failure_keeps_evidence_and_attempt_aligned(monkeypatch):
    from experiments.pvcd_cycle_v1 import training as t
    from experiments.posttraining_v1.data import target
    case,gold,wrong=problem();prompts=[];answers=iter(['invalid-json',target(gold)])
    def sample(prompt,seed): prompts.append(prompt); return [1],next(answers)
    e=SimpleNamespace(sample=sample,tok=lambda text,**kw:{'offset_mapping':[(0,len(text))]})
    monkeypatch.setattr(t,'collect',lambda m,g,s,c,u:([[0],[1]],[]))
    result,trace=t.repair_chain(e,case,gold,dict(model=wrong,text=target(wrong)),17,'cycle',CFG)
    assert result and trace[0]['semantic']['status']=='invalid'
    assert '\nAttempt: '+target(wrong)+'\n' in prompts[1]
    assert 'Parse failure:' in prompts[1]

@pytest.fixture
def tiny_engine(tmp_path):
    import torch
    from tokenizers import Tokenizer,models,pre_tokenizers,trainers
    from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
    from experiments.posttraining_v1.engine import Engine
    from experiments.posttraining_v1.data import target
    from qihc.modeling.active_protocol_v2 import model_prompt
    torch.set_num_threads(1); torch.manual_seed(17)
    case,gold,wrong=problem()
    tok=Tokenizer(models.BPE(unk_token='<unk>'));tok.pre_tokenizer=pre_tokenizers.ByteLevel()
    tok.train_from_iterator([model_prompt(case),target(gold),target(wrong)],trainers.BpeTrainer(vocab_size=512,special_tokens=['<unk>','<eos>','<pad>']))
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='<unk>',eos_token='<eos>',pad_token='<pad>')
    tokenizer.save_pretrained(tmp_path)
    Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),hidden_size=24,num_hidden_layers=1,
        num_attention_heads=3,num_key_value_heads=1,intermediate_size=32,head_dim=8)).save_pretrained(tmp_path)
    return Engine(tmp_path,device='cpu')

def test_all_objectives_real_gradients_and_guard(tiny_engine,monkeypatch):
    import torch
    from experiments.pvcd_cycle_v1 import training as t
    from experiments.posttraining_v1.data import target
    case,gold,wrong=problem();row=dict(case=case,gold=gold,answer=target(gold))
    e=tiny_engine;reference=e.snapshot()
    monkeypatch.setattr(t,'collect',lambda m,g,s,c,u:([[0],[1]],[]))
    def output(p,s):
        text=target(gold if 'Training-only verified' in p or s%2 else wrong)
        return e.encode(text),text
    monkeypatch.setattr(e,'sample',output)
    for arm in ('cycle','cycle_uniform','cycle_static','cycle_no_guard'):
        e.restore(reference);opt=torch.optim.AdamW(e.params.values(),lr=.001);opt.zero_grad()
        r=t.step(e,arm,row,17,reference,gold)
        torch.nn.utils.clip_grad_norm_(list(e.params.values()),1.,error_if_nonfinite=True);opt.step()
        assert r['reward_update'] and r['verified_repairs']==1 and r['unique_targets']==1
        assert r['guard']==(arm!='cycle_no_guard')
        assert any(not torch.equal(v,reference[k]) for k,v in e.snapshot().items())

def test_checkpoint_resume_matches_uninterrupted(tiny_engine,tmp_path,monkeypatch):
    import torch
    from experiments.pvcd_cycle_v1 import training as t
    from experiments.posttraining_v1.data import target
    case,gold,_=problem();row=dict(case=case,gold=gold,answer=target(gold)); e=tiny_engine
    reference=e.snapshot(); cfg=dict(CFG,epochs=1,checkpoint_steps=1)
    monkeypatch.setattr(t,'CFG',cfg);monkeypatch.setattr(t,'RUN',tmp_path)
    monkeypatch.setattr(t,'training_rows',lambda mode:[row,row])
    save(tmp_path/'inputs/parents/17/fixture.json',dict(model=gold))
    counter=[0]; interrupt=[True]
    def simple_step(engine,arm,row,seed,ref,parent):
        counter[0]+=1
        if interrupt[0] and counter[0]==2:raise RuntimeError('simulated interruption')
        loss=engine.backward('fixture',engine.encode(row['answer']),'nll')
        return dict(verified_repairs=0,guard=False,loss=loss)
    monkeypatch.setattr(t,'step',simple_step)
    interrupted=tmp_path/'interrupted';interrupted.mkdir()
    with pytest.raises(RuntimeError,match='simulated'):t.fit(e,'cycle',17,'formal',interrupted,reference)
    assert torch.load(interrupted/'checkpoint.pt',weights_only=True)['step']==1
    interrupt[0]=False;t.fit(e,'cycle',17,'formal',interrupted,reference)
    resumed=e.snapshot()
    control=tmp_path/'control';control.mkdir(); t.fit(e,'cycle',17,'formal',control,reference)
    assert all(torch.equal(v,resumed[k]) for k,v in e.snapshot().items())

def test_report_coverage_full_transitions_and_failed_gate(tmp_path,monkeypatch):
    from experiments.pvcd_cycle_v1 import report as r
    monkeypatch.setattr(r,'RUN',tmp_path)
    monkeypatch.setattr(r,'cases',lambda seed,mode:[dict(id='a',stage='pilot'),dict(id='b',stage='pilot')])
    monkeypatch.setattr(r,'training_rows',lambda mode:[1])
    for arm in CFG['arms']:
        for cid in ('a','b'):
            status='verified_correct' if arm=='sft' else ('invalid' if cid=='a' else 'unknown')
            metric=dict(original_feasible=False,optimal_hit=False)
            save(tmp_path/f'pilot/models/17/{arm}/evaluations/{cid}.json',dict(case_id=cid,stage='pilot',semantic=dict(status=status),cross={b:dict(score=metric) for b in ('pbit','scip')}))
    result=r.report('pilot')
    assert result['complete']
    assert result['stages']['pilot']['transitions']['cycle']=={'verified_correct->invalid':1,'verified_correct->unknown':1}
    assert not load(tmp_path/'pilot/gate.json')['passed']
    (tmp_path/'pilot/models/17/pvcd/evaluations/a.json').unlink()
    partial=r.report('pilot')
    assert not partial['complete'] and not partial['stages']['pilot']['comparisons']

def test_same_model_score_cache(tmp_path,monkeypatch):
    from experiments.pvcd_cycle_v1 import score as s
    case,gold,_=problem();monkeypatch.setattr(s,'RUN',tmp_path);monkeypatch.setattr(s,'_GOLD',{'fixture':gold})
    calls=[]
    def execute(job):
        calls.append(job['stage']);return dict(x=[0],objective=0,optimality_proven=True,status='optimal')
    monkeypatch.setattr(s,'execute',execute)
    for arm in ('sft','cycle'):
        p=tmp_path/f'formal/models/17/{arm}/predictions/fixture.json'
        save(p,dict(case_id='fixture',stage='main',model=gold));s.one(p)
        assert load(p.parent.parent/'evaluations/fixture.json')['cross']['pbit']['score']['optimal_hit']
    assert calls==['reference','solve','solve']

def test_bootstrap_clusters_by_case():
    from experiments.pvcd_cycle_v1.report import paired_interval
    stat=paired_interval([1.,1.,1.],100)
    assert stat['cases']==3 and stat['ci98_75_pp']==[100.,100.]

def test_real_scip_and_pbit_scoring(tmp_path,monkeypatch):
    from experiments.pvcd_cycle_v1 import score as s
    _,gold,_=problem();monkeypatch.setattr(s,'RUN',tmp_path)
    monkeypatch.setattr(s,'_GOLD',{'fixture':gold});monkeypatch.setattr(s,'CFG',dict(CFG,solver_seconds=.2))
    p=tmp_path/'formal/models/17/cycle/predictions/fixture.json'
    save(p,dict(case_id='fixture',stage='main',model=gold));s.one(p)
    result=load(p.parent.parent/'evaluations/fixture.json')
    assert result['semantic']['status']=='verified_correct'
    assert all(result['cross'][b]['score']['optimal_hit'] for b in ('pbit','scip'))

def test_prepare_idempotence_and_input_tampering(tmp_path,monkeypatch):
    import hashlib
    from experiments.pvcd_cycle_v1 import prepare as p,launch
    case,gold,_=problem();other=copy.deepcopy(gold);other['variables'][0]['upper']=2
    hist=tmp_path/'history';run=tmp_path/'run';data=tmp_path/'data';model=tmp_path/'model'
    inputs={'train':[dict(case=dict(case,id=f't{i}'),gold=gold,answer='fixture') for i in range(90)],
            'validation':[dict(case=dict(case,id=f'v{i}'),gold=other,answer='fixture') for i in range(24)],
            'cohort':[dict(id=f'h{i}',stage='main') for i in range(910)]}
    save(data/'public.manifest.json',{'version':'fixture'})
    from qihc.experiment_runtime.storage import digest
    hashes={}
    for name,obj in inputs.items():
        save(hist/(name+'.json'),obj);hashes[name]=hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
    save(hist/'split.json',dict(experiment='E005-1.7',source=digest(data/'public.manifest.json'),files=hashes))
    save(hist/'prepared.json',dict(files=[]))
    for seed in CFG['seeds']:
        for arm in ('sft','pvcd'):
            folder=hist/f'formal/models/{seed}/{arm}';folder.mkdir(parents=True)
            (folder/'adapter.pt').write_bytes(b'fixture')
        if seed==17:  # Historical development predictions exist only for seed 17.
            for row in inputs['train']:
                save(hist/f'formal/models/{seed}/sft/predictions'/ (row['case']['id']+'.json'),dict(model=gold))
    for mod in (p,launch):
        monkeypatch.setattr(mod,'RUN',run);monkeypatch.setattr(mod,'HISTORY',hist);monkeypatch.setattr(mod,'MODEL',model)
        monkeypatch.setattr(mod,'check_paths',lambda:None);monkeypatch.setattr(mod,'source_hashes',lambda:{'fixture.py':'hash'})
        monkeypatch.setattr(mod,'model_identity',lambda path:[]);monkeypatch.setattr(mod,'environment_identity',lambda:{'python':'fixture'})
    monkeypatch.setattr(p,'DATA',data);monkeypatch.setattr(p,'restore',lambda path:None);monkeypatch.setattr(p,'check',lambda path:None)
    monkeypatch.setattr(p,'donor',lambda s,a:hist/f'formal/models/{s}/{a}')
    monkeypatch.setattr(p.subprocess,'check_output',lambda args,**kw:'abc123\n' if 'rev-parse' in args else '')
    p.prepare();first=load(run/'manifest.json');p.prepare()
    assert load(run/'manifest.json')==first
    assert first['parent_plan']['17']['missing_case_ids']==[]
    assert len(first['parent_plan']['29']['missing_case_ids'])==90
    assert len(first['parent_plan']['43']['missing_case_ids'])==90
    assert not (run/'derived').exists()  # CPU preparation does not perform inference.
    launch.validate()
    save(run/'inputs/train.json',[])
    with pytest.raises(ValueError,match='Input changed'):launch.validate()

def test_missing_parents_frozen_adapter_resume_and_identity(tmp_path):
    from experiments.pvcd_cycle_v1.parents import ensure_parents,parent_path
    from experiments.posttraining_v1.data import target
    case,gold,_=problem()
    rows=[dict(case=dict(case,id=cid),gold='MUST_NOT_ENTER_PROMPT',answer='MUST_NOT_ENTER_PROMPT') for cid in ('a','b')]
    save(tmp_path/'manifest.json',dict(code_sha='fixed',parent_plan={'29':dict(adapter_sha256='adapter29',missing_case_ids=['a','b'])}))
    calls=[];restores=[];fail=[True]
    def evaluate(prompt,seed):
        assert 'MUST_NOT_ENTER_PROMPT' not in prompt
        calls.append(prompt)
        if len(calls)==2 and fail[0]:raise RuntimeError('interrupted parent generation')
        return dict(choices=[dict(finish_reason='stop',message=dict(content=target(gold)))])
    engine=SimpleNamespace(restore=lambda ref:restores.append(ref),evaluate=evaluate)
    reference={'frozen':'seed29'}
    with pytest.raises(RuntimeError,match='interrupted'):ensure_parents(engine,reference,rows,29,tmp_path)
    assert parent_path(tmp_path,29,'a').exists() and not parent_path(tmp_path,29,'b').exists()
    fail[0]=False;ensure_parents(engine,reference,rows,29,tmp_path)
    ensure_parents(engine,reference,rows,29,tmp_path)
    assert len(calls)==3 and restores==[reference,reference]
    assert not (tmp_path/'inputs').exists()
    saved=load(parent_path(tmp_path,29,'a'))
    assert saved['parent_identity']['adapter_sha256']=='adapter29'
    assert equivalent(saved['model'],gold)['status']=='verified_correct'
    saved['parent_identity']['adapter_sha256']='wrong-adapter'
    save(parent_path(tmp_path,29,'a'),saved)
    with pytest.raises(ValueError,match='identity'):ensure_parents(engine,reference,rows,29,tmp_path)

def test_existing_invalid_parent_is_not_regenerated(tmp_path):
    from experiments.pvcd_cycle_v1.parents import ensure_parents,parent_path
    case,_,_=problem();save(tmp_path/'inputs/parents/17/fixture.json',dict(model=None))
    def forbidden(*args):raise AssertionError('Existing parent must stay frozen, even invalid')
    ensure_parents(SimpleNamespace(restore=forbidden,evaluate=forbidden),{},[dict(case=case)],17,tmp_path)
    assert load(parent_path(tmp_path,17,'fixture'))['model'] is None
