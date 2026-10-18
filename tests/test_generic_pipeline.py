"""Stage orchestration tests with fake subprocess workloads, not ML quality tests."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest

from scripts import generic_energy_pipeline as pipeline
from experiments import generic_energy as runner
from qihc.generic.benchmark import make_dataset


@pytest.mark.parametrize("failure", [None, "preflight", "weights"])
@pytest.mark.parametrize("semantic", [False, True])
def test_full_orchestration_and_failure_status(tmp_path, monkeypatch, failure, semantic):
    if semantic:
        pytest.importorskip("ortools")
    model = tmp_path / "model"
    model.mkdir()
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({"model_path": str(model)}))
    output = tmp_path / "run"
    commands = []
    monkeypatch.setattr(sys, "argv", ["pipeline", "--manifest", str(manifest), "--output", str(output)] +
                        (["--semantic-training", "--nproc", "2"] if semantic else []))

    def value(cmd, key):
        return cmd[cmd.index(key) + 1]

    def fake_solve(spec, policy, **kwargs):
        return {"feasible": True, "objective": {"fixed": 10, "adaptive": 8, "warm": 6, "cold": 4}[policy], "violation": 0}
    monkeypatch.setattr(runner, "solve", fake_solve)

    def execute(cmd, **kwargs):
        commands.append(cmd)
        if "experiments/train_s2e_lora.py" in cmd:
            stage = value(cmd, "--stage")
            target = Path(value(cmd, "--output"))
            target.mkdir()
            if stage == "dpo":
                expected = "smoke_adapter_sft" if target.name.startswith("smoke_") else "adapter_sft"
                assert value(cmd, "--adapter-path") == str(output / expected)
            if failure != "weights":
                (target / "adapter_model.safetensors").write_bytes(b"test fixture, not real weights")
            (target / "qihc_training_manifest.json").write_text(json.dumps({"training_metrics": [{"loss": 1.0}]}))
            return subprocess.CompletedProcess(cmd, 0)
        stage = cmd[2]
        if stage == "data":
            runner.write_rows(output / "cases.jsonl", make_dataset(48, expanded=semantic))
        elif stage == "teacher":
            runner.teacher(SimpleNamespace(output=output, holdout_family="balanced_cut", steps=2, chains=2, semantic_training=semantic), runner.read_rows(output / "cases.jsonl"))
        elif stage == "preflight":
            if failure == "preflight":
                raise subprocess.CalledProcessError(1, cmd)
            (output / "training_preflight.json").write_text(json.dumps({"longest_row": {"sft": 0, "dpo": 0}}))
        elif stage == "eval":
            mode, tag = value(cmd, "--mode"), value(cmd, "--tag")
            if tag in ("sft_eval", "dpo_eval"):
                assert cmd[cmd.index("--adapters") + 1:] == [str(output / ("adapter_sft" if tag == "sft_eval" else "adapter_dpo"))]
            cases = runner.read_rows(output / "cases.jsonl")
            lookup = {c["id"]: c for c in cases}
            records = []
            for cid, arm, seed in runner.expected_keys(cases, mode, tag, 3):
                c = lookup[cid]
                records.append({"case_id": cid, "arm": arm, "seed": seed, "family": c["family"], "split": c["split"],
                                "objective_sense": c["gold"]["objective"]["sense"], "feasible": False, "seconds": 0, "status": "UNKNOWN"})
            runner.write_rows(output / f"{tag}.jsonl", records)
            runner.audit_evaluation(output / f"{tag}.jsonl", cases, mode, tag, 3)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    monkeypatch.setattr(pipeline.subprocess, "check_output", lambda *a, **k: "test-commit\n")
    if failure:
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            pipeline.main()
    else:
        pipeline.main()
    report = json.loads((output / "run.json").read_text())
    assert report["status"] == ("failed" if failure else "complete")
    assert all(s["status"] != "running" for s in report["stages"])
    assert (output / "comparison.json").exists()
    if not failure:
        assert report["sft_rows"] == (96 if semantic else 192) and report["dpo_rows"] > 16
        assert [r["stage"] for r in report["stages"]][-2:] == ["train_dpo", "dpo_eval"]
        if semantic:
            dpos = [c for c in commands if "experiments/train_s2e_lora.py" in c and value(c, "--stage") == "dpo" and not value(c, "--output").endswith("smoke_adapter_dpo")]
            assert len(dpos) == 3
            assert all(value(c, "--max-steps") == "128" and "--nproc_per_node=2" in c for c in dpos)
    if semantic and failure == "weights":
        old_output = output
        output = tmp_path / "retry"
        failure = None
        commands.clear()
        monkeypatch.setattr(sys, "argv", ["pipeline", "--manifest", str(manifest), "--output", str(output),
                                         "--semantic-training", "--nproc", "2", "--reuse-preparation", str(old_output)])
        pipeline.main()
        retried = json.loads((output / "run.json").read_text())
        assert retried["status"] == "complete"
        assert {s["stage"] for s in retried["stages"] if s.get("reused")} == {"data", "oracle", "teacher"}
        assert not any(len(c) > 2 and c[2] in ("data", "teacher") for c in commands)
        assert json.loads((old_output / "run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("metrics", [[], [{"epoch": 1}], [{"loss": float("nan")}], [{"train_loss": float("inf")}]])
def test_training_requires_actual_finite_loss(tmp_path, metrics):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"fixture")
    (tmp_path / "qihc_training_manifest.json").write_text(json.dumps({"training_metrics": metrics}))
    with pytest.raises(RuntimeError):
        pipeline.check_training_output(tmp_path)


@pytest.mark.parametrize("limit,expect_error", [(20000, False), (32, True)])
def test_preflight_uses_real_prompts_and_model_context(tmp_path, monkeypatch, limit, expect_error):
    from qihc.generic.benchmark import make_case
    case = make_case("knapsack", 4, 1)
    args = SimpleNamespace(output=tmp_path, holdout_family="balanced_cut", chains=2, steps=2,
                           model_path="not-loaded", train_max_length=16384, max_new_tokens=12288)
    monkeypatch.setattr(runner, "solve", lambda spec, **k: {"feasible": True, "objective": 1, "violation": 0})
    runner.teacher(args, [case])
    class Tokenizer:
        chat_template = True
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"] + "assistant:"
        def encode(self, text, **kwargs):
            return list(range(len(text)))
    fake = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(model_type="test", max_position_embeddings=limit)),
                           AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()), AutoProcessor=None)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    if expect_error:
        with pytest.raises(ValueError, match="no truncation"):
            runner.training_preflight(args, [case])
    else:
        runner.training_preflight(args, [case])
        report = json.loads((tmp_path / "training_preflight.json").read_text())
        assert report["sft_rows"] == 2 and report["max_required_tokens"]["sft"] > 0
