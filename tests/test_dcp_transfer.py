"""Regression checks for the DCP split, equal-budget ablation and repair acceptance."""

import pytest

from experiments.dcp_transfer_v1 import data, feedback, gpu
from experiments.posttraining_v1.data import target
from qihc.experiment_runtime.storage import load, save


def test_frozen_main_and_templates():
    from experiments.dcp_external_v2.common import CENSUS

    cases = [dict(id=str(i), dataset="nlco") for i in range(780)]
    for r in load(CENSUS)["templates"]:
        if r["objective"] != "satisfaction":
            cases += [
                dict(
                    dataset="dcp",
                    template=r["template"],
                    id=r["template"] + "__" + key,
                    main=key != "example" and r["history"] == "no_known_template_match",
                )
                for key in r["instances"]
            ]
    train, test = data.split(cases)
    assert len(train) == 25 and len(test) == 990
    assert {c["template"] for c in train}.isdisjoint(c.get("template") for c in test)
    assert sum(c.get("main", False) for c in test) == 159


def test_budget_and_schema_retry():
    corpus = dict(nlco=list(range(20)), dcp=[30, 40, 50])
    control = gpu.training_order(corpus, "dcp_replay")
    mixed = gpu.training_order(corpus, "dcp_mixed")
    assert len(control) == len(mixed) == 6 and control[::2] == mixed[::2]
    assert set(mixed[1::2]) == set(corpus["dcp"])
    gold = data.bounded_model(data.KNAPSACK, dict(values=[4], weights=[1], capacity=1))[
        0
    ]
    c = dict(
        id="repair",
        question="Pick a binary item of value 4 and weight 1 within capacity 1.",
        variable_names=["x0"],
    )
    with pytest.raises(ValueError):
        feedback.decode('{"edits":[{"target":"variables","value":[]}]}', c, gold)
    with pytest.raises(ValueError):
        feedback.decode('{"edits":[],"edits":[]}', c, gold)
    answers = iter(['{"edits":[]}', target(gold)])
    result = feedback.propose(
        c, dict(model=None, error="no model"), lambda *_: next(answers)
    )
    assert result["retry"] and result["model"] == gold
    result = feedback.propose(c, dict(model=gold), lambda *_: target(gold))
    assert result["retry"] and result["parse_failed"] and result["model"] == gold


def test_tiny_training_and_resume(tmp_path, monkeypatch):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from experiments.posttraining_v1.engine import Engine
    from qihc.modeling.active_protocol_v2 import model_prompt

    torch.set_num_threads(1)
    torch.manual_seed(17)
    gold = data.bounded_model(
        data.KNAPSACK, dict(values=[2, 1], weights=[1, 2], capacity=1)
    )[0]
    case = dict(
        id="tiny",
        question="Two binary items of value 2,1 and weight 1,2; total capacity 1.",
        variable_names=["x0", "x1"],
    )
    model, patch = feedback.supervise(case, gold, [1, 0], 0)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.train_from_iterator(
        [model_prompt(case), patch["prompt"], target(gold)],
        trainers.BpeTrainer(vocab_size=512, special_tokens=["<unk>", "<eos>", "<pad>"]),
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tok.save_pretrained(tmp_path)
    Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=24,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=1,
            intermediate_size=32,
            head_dim=8,
        )
    ).save_pretrained(tmp_path)
    engine = Engine(tmp_path, device="cpu")
    # Exercise the same nested checkpoint hooks used by the real Qwen feedback
    # trainer. Force the small fixture through the long-sequence path.
    prompt, answer = patch["prompt"], engine.encode(patch["target"]) + [
        engine.tok.eos_token_id
    ]
    reference = engine.snapshot()
    original = engine.backward(prompt, answer, "nll", reference=reference)
    gradients = {k: p.grad.clone() for k, p in engine.params.items()}
    engine.net.zero_grad(set_to_none=True)
    with gpu.offload_long_activations(
        engine.net.config.hidden_size, min_tokens=1
    ) as stats:
        recovered = engine.backward(prompt, answer, "nll", reference=reference)
    assert stats["offloaded_bytes"] > 0
    assert original == recovered
    for k, p in engine.params.items():
        torch.testing.assert_close(p.grad, gradients[k], rtol=0, atol=0)
    engine.net.zero_grad(set_to_none=True)
    base = tmp_path / "base.pt"
    torch.save(engine.snapshot(), base)
    m = dict(adapters=dict(nlco=str(base), dcp=str(base), feedback=str(base)))
    save(tmp_path / "manifest.json", m)
    save(
        tmp_path / "corpus.json",
        dict(nlco=[model], dcp=[model], nlco_patch=[patch], dcp_patch=[patch]),
    )
    monkeypatch.setattr(gpu, "RUN", tmp_path)
    for arm in gpu.TASKS:
        gpu.train(engine, m, arm)
        folder = tmp_path / "trained" / arm
        assert load(folder / "steps/00001.json")["gradient_norm"] > 0
        before = (folder / "adapter.pt").stat().st_mtime_ns
        gpu.train(engine, m, arm)
        assert (folder / "adapter.pt").stat().st_mtime_ns == before
