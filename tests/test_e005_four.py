"""Exercise the four objectives with a tiny offline Qwen and controlled rollouts."""
import copy
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from experiments.posttraining_v1 import learn
from experiments.posttraining_v1.data import target
from experiments.posttraining_v1.engine import Engine
from experiments.posttraining_v1.witness import probe
from qihc.experiment_runtime.storage import load


def test_four_updates_and_verified_repair(tmp_path, monkeypatch):
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
    monkeypatch.setattr(learn,"probe",lambda m,g,s:probe(m,g,s,.3))
    for method in learn.MODEL_METHODS:
        engine.restore(initial)
        opt=torch.optim.AdamW(engine.params.values(),lr=.001)
        opt.zero_grad()
        def controlled(prompt,seed):
            text=target(gold if "Review this attempt" in prompt or seed%2 else wrong)
            return engine.encode(text),text
        monkeypatch.setattr(engine,"sample",controlled)
        learn.train_one(engine,method,row,17,initial,tmp_path/method,0)
        opt.step()
        assert any(not torch.equal(initial[k],v) for k,v in engine.snapshot().items())
    trace=load(tmp_path/"pvcd/trace/0000.json")
    assert any(s.get("cards") and s["repair_semantic"]["status"]=="verified_correct" for s in trace["samples"])
    ck=tmp_path/"resume.pt"
    learn.put_checkpoint(ck,{"weights":engine.snapshot(),"optimizer":opt.state_dict(),"step":1})
    restored=torch.load(ck,weights_only=True)
    engine.restore(restored["weights"])
    opt.load_state_dict(restored["optimizer"])

    # Both new learned arms use actual SFT and reference-based DPO updates.
    from experiments.posttraining_v1 import energy_learn as el
    from experiments.posttraining_v1.energy import diagnose, KEEP
    from qihc.experiment_runtime.storage import save
    spec=copy.deepcopy(gold)
    spec["constraints"].append(dict(id="c1",a=[1],op="ge",rhs=0))
    diag=diagnose(spec,3.,case["id"],17)
    data_root=tmp_path/"eight"
    save(data_root/"train.json",[dict(case=case,gold=spec)])
    save(data_root/"energy/settings.json",dict(base=3.,delta=1,pair_ids=["fixture"]))
    save(data_root/"energy/preferences/fixture.json",dict(diag=diag,pair=dict(chosen=KEEP,rejected={"transfers":[[0,1,1]]})))
    save(data_root/"pilot/models/17/sft/predictions/fixture.json",dict(model=spec))
    monkeypatch.setattr(el,"RUN",data_root)
    monkeypatch.setattr(el,"gold_inputs",lambda: {"fixture":spec})
    monkeypatch.setattr(el,"Engine",lambda _: Engine(tmp_path,device="cpu"))
    monkeypatch.setattr(el,"cohort",lambda m,s:[dict(case,stage="pilot")])
    monkeypatch.setattr(Engine,"evaluate",lambda *args:{"choices":[dict(finish_reason="stop",message=dict(content='{"transfers":[]}'))]})
    for arm in ("best_lm","trajectory_lm"):
        el.run(arm,17,"pilot")
        folder=data_root/"pilot/models/17"/arm
        assert load(folder/"predictions/fixture.json")["action"]==KEEP
        assert load(data_root/f"pilot/gold/models/17/{arm}/predictions/fixture.json")["model"]==spec
        ref=torch.load(folder/"energy_sft.pt",weights_only=True)
        result=torch.load(folder/"adapter.pt",weights_only=True)
        assert any(not torch.equal(ref[k],v) for k,v in result.items())



def test_paired_report_requires_all_seed_rows(tmp_path, monkeypatch):
    from experiments.posttraining_v1 import report as module
    from qihc.experiment_runtime.storage import load,save
    external=tmp_path/"e004"
    monkeypatch.setenv("QIHC_E004_RUN",str(external))
    monkeypatch.setattr(module,"RUN",tmp_path/"post")
    save(module.RUN/"prepared.json",{"identity":"same"})
    save(external/"manifest.json",dict(config=dict(experiment_id="E004_2.2",protocol_version="v2",
          solver_seconds=20,chains=32,max_output_tokens=4096),data_manifest=load(module.DATA/"manifest.json"),
          model_identity="same"))
    metric=dict(score=dict(original_feasible=True,optimal_hit=False))
    base=dict(stage="main",case_id="pair",semantic=dict(status="verified_correct"),cross=dict(scip=metric,pbit=metric))
    for m in module.METHODS:
        for seed in (17,29,43):
            save(module.RUN/f"formal/models/{seed}/{m}/evaluations/pair.json",base)
    for arm,seeds in (("C0",[0]),("P2",[17,29])):
        for seed in seeds:
            save(external/f"evaluation/trajectories/{arm}-{seed}.json",dict(base,arm=arm,seed=seed))
    module.report("formal")
    path=module.RUN/"formal/paired_cases.json"
    assert load(path)["cases"]==[]
    save(external/"evaluation/trajectories/P2-43.json",dict(base,arm="P2",seed=43))
    module.report("formal")
    assert load(path)["cases"]==["pair"]
