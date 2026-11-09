import copy
import json
import torch
from tokenizers import Tokenizer,models,pre_tokenizers,trainers
from transformers import Qwen3Config,Qwen3ForCausalLM,PreTrainedTokenizerFast
from experiments.posttraining_v1 import learn
from experiments.posttraining_v1.data import target
from experiments.posttraining_v1.engine import Engine
from experiments.posttraining_v1.witness import probe
from experiments.selective_v1 import training
from experiments.selective_v1.protocol import edit,protected,witnessed
from qihc.experiment_runtime.storage import load

def test_pair_training_and_local_guard(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    gold = dict(schema="bounded-integer-v2", variables=[dict(name="x0",lower=0,upper=1)],
                objective=dict(sense="max",linear=[1],quadratic=[]),logic=[],
                constraints=[dict(id="c0",a=[1],op="le",rhs=0)])
    wrong=copy.deepcopy(gold)
    wrong["constraints"]=[]
    case=dict(id="fixture",variable_names=["x0"],question="Maximize x0 with x0 binary and x0 <= 0.")
    row=dict(case=case,gold=gold,answer=target(gold))
    tok=Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel()
    tok.train_from_iterator([learn.model_prompt(case),target(gold),target(wrong)],
        trainers.BpeTrainer(vocab_size=512,special_tokens=["<unk>","<eos>","<pad>"]))
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token="<unk>",eos_token="<eos>",pad_token="<pad>")
    tokenizer.save_pretrained(tmp_path)
    Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer),hidden_size=24,num_hidden_layers=1,
        num_attention_heads=3,num_key_value_heads=1,intermediate_size=32,head_dim=8)).save_pretrained(tmp_path)
    engine=Engine(tmp_path,device="cpu")
    initial=engine.snapshot()

    check=dict(kind="constraint",id="s0",a=[1],op="le",rhs=0,quote="x0 <= 0")
    checks=dict(checks=[check])
    card=dict(check_id="s0",x=[1])
    patch=json.dumps(dict(action="append",row=dict(id="c0",a={"0":1},op="le",rhs=0)))
    assert witnessed(wrong,checks,[card])==[card]
    new=edit(patch,wrong,case)
    assert protected(wrong,new,checks,[card])
    assert not protected(wrong,wrong,checks,[card])
    monkeypatch.setattr(training,"probe",lambda m,g,s:probe(m,g,s,.3))
    def output(prompt,seed):
        if "Instead of a full model" in prompt:text=patch
        else:text=target(gold if seed%2 else wrong)
        return engine.encode(text),text
    monkeypatch.setattr(engine,"sample",output)
    for arm in ("pair_rl","pair_local"):
        engine.restore(initial)
        opt=torch.optim.Adam(engine.params.values(),lr=.001)
        opt.zero_grad()
        training.step(engine,arm,row,17,initial,tmp_path/arm,0)
        opt.step()
        assert any(not torch.equal(initial[k],v) for k,v in engine.snapshot().items())
        tr=load(tmp_path/arm/"trace/0000.json")
        assert len(set(s["reward"] for s in tr["samples"]))>1
        assert tr["verified"]==(2 if arm=="pair_local" else 0)

def test_feedback_routing_with_real_scip(tmp_path,monkeypatch):
    from experiments.selective_v1 import feedback
    from types import SimpleNamespace
    case=dict(id="example",variable_names=["x0"],question="Maximize x0 with x0 binary and x0 <= 0.")
    old=dict(schema="bounded-integer-v2",variables=[dict(name="x0",lower=0,upper=1)],
             objective=dict(sense="max",linear=[1],quadratic=[]),logic=[],constraints=[])
    corrected=copy.deepcopy(old)
    corrected["constraints"]=[dict(id="c0",a=[1],op="le",rhs=0)]
    row={"id":"c0","a":{"0":1},"op":"le","rhs":0}
    def answer(e,p,s):
        if "Extract at most" in p:
            return json.dumps({"checks":[dict(id="s0",kind="constraint",quote="x0 <= 0",confidence=1,
                                               a={"0":1},op="le",rhs=0)]})
        if "Instead of a full" in p:return json.dumps(dict(action="append",row=row))
        return target(corrected)
    monkeypatch.setattr(feedback,"generated",answer)
    cfg=dict(temperatures=[.25,.5,1.],chains=32,max_cards=4,candidate_actions=8,deep_actions=2,probe_seconds=1.)
    e=SimpleNamespace(encode=lambda x:list(x))
    for arm in ("sft","compile","scip_local","pbit_local"):
        result=feedback.decide(e,arm,case,{"model":None,"error":"parse"},tmp_path/"invalid",17,cfg)
        assert (result["model"] is None)==(arm=="sft")
        valid=feedback.decide(e,arm,case,{"model":old},tmp_path/"valid",17,cfg)
        if arm in ("sft","compile"): assert valid["model"]==old
        else:
            assert valid["decision"] in ("accepted","abstain")
            if arm=="scip_local": assert valid["decision"]=="accepted"
