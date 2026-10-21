import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.large_model_pipeline import score, summarize
from scripts.generic_energy_pipeline import training_command
from experiments.generic_energy import write_rows, read_rows
from qihc.generic.compositional import make_case
from qihc.generic.sparse import encode


def test_four_gpu_training_batch():
    cmd = training_command("model", "data", "out", "sft", 192, 4)
    assert "--nproc_per_node=4" in cmd
    assert cmd[cmd.index("--gradient-accumulation-steps") + 1] == "1"
    assert "--deepspeed-config" in cmd


def test_scoring_resume_and_summary(tmp_path, monkeypatch):
    import scripts.large_model_pipeline as pipeline
    case = make_case(6, 1, ("budget",), "test")
    write_rows(tmp_path / "cases.jsonl", [case])
    pred = tmp_path / "pred.jsonl"
    write_rows(pred, [{"case_id": case["id"], "response": json.dumps(encode(case["gold"]))}])
    calls = []
    def fake(spec, **kw):
        calls.append(kw)
        return {"x": [0] * 6}
    monkeypatch.setattr(pipeline, "solve", fake)
    monkeypatch.setattr(pipeline, "reference_solve", lambda *a: {"x": [0] * 6})
    a = SimpleNamespace(data=tmp_path, predictions=pred, output=tmp_path / "model_scores.jsonl", arm="modeling")
    score(a)
    assert len(calls) == 9
    assert len(read_rows(a.output)) == 10
    score(a)
    assert len(calls) == 9
    summarize(tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["metrics"]) == 8
    assert all(r["completed_cases"] == 1 for r in summary["metrics"].values())


def test_invalid_model_has_every_backend(tmp_path):
    case = make_case(6, 1, ("budget",), "test")
    write_rows(tmp_path / "cases.jsonl", [case])
    pred = tmp_path / "pred.jsonl"
    write_rows(pred, [{"case_id": case["id"], "response": "not json"}])
    output = tmp_path / "scores.jsonl"
    score(SimpleNamespace(data=tmp_path, predictions=pred, output=output, arm="modeling"))
    rows = read_rows(output)
    assert len(rows) == 4
    assert not any(r["feasible"] for r in rows)


def test_launchers_no_smi_or_outer_torchrun():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_large_4gpu.sh").read_text()
    assert "nvidia-smi" not in text
    assert "conda activate" in text
    assert "QIHC_LARGE_RESUME" in text
    assert "HF_HUB_OFFLINE=1" in text


def test_full_orchestration_and_resume(tmp_path, monkeypatch):
    import sys
    import scripts.large_model_pipeline as p
    prepared = tmp_path / "artifacts/large32b"
    prepared.mkdir(parents=True)
    manifest = {"source": "code", "env_prefix": sys.prefix, "packages": {}, "data": {},
                "model": {"model_path": "model", "files": {}}}
    (prepared / "prepared.json").write_text(json.dumps(manifest))
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 4,
        mem_get_info=lambda i: (79 * 2**30, 80 * 2**30), get_device_name=lambda i: "H100")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(p, "sources", lambda: "code")
    monkeypatch.setattr(p, "check_files", lambda *a: None)
    monkeypatch.setattr(p, "check_training_output", lambda *a: 1)
    commands = []
    monkeypatch.setattr(p.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    out = tmp_path / "run"
    p.run(SimpleNamespace(base=tmp_path, output=out))
    report = json.loads((out / "run.json").read_text())
    assert report["status"] == "gpu_complete_cpu_scoring_pending"
    assert len(commands) == 13  # NCCL + 2 smokes + 4 training + 6 inference, NO CPU search.
    assert not any("score" in c for c in commands)
    assert len([c for c in commands if "infer" in c]) == 6
    assert len([c for c in commands if "--nproc_per_node=4" in c]) == 7
    p.run(SimpleNamespace(base=tmp_path, output=out))
    assert len(commands) == 13


def test_legacy_resume_requires_unchanged_dependencies(monkeypatch):
    import scripts.large_model_pipeline as p
    monkeypatch.setattr(p, "sources", lambda exclude_runner=False: p.LEGACY_NONRUNNER if exclude_runner else "new")
    assert p.compatible_source(p.LEGACY_SOURCE)
    assert not p.compatible_source("unknown")
    monkeypatch.setattr(p, "sources", lambda exclude_runner=False: "changed-dependency")
    assert not p.compatible_source(p.LEGACY_SOURCE)


def test_prediction_checkpoint_recovers_only_incomplete_tail(tmp_path):
    from scripts.large_model_pipeline import prediction_checkpoint
    row = {"case_id": "a", "response": "{}", "input_tokens": 1, "output_tokens": 1, "seconds": 1}
    path = tmp_path / "pred.jsonl"
    raw = json.dumps(row).encode() + b'\n{"case_id":"b'
    path.write_bytes(raw)
    assert prediction_checkpoint(path) == [row]
    assert read_rows(path) == [row]
    assert next(tmp_path.glob("*.bak")).read_bytes() == raw
    path.write_bytes(b'not json\n' + json.dumps(row).encode())
    with pytest.raises(ValueError, match="Corrupt complete"):
        prediction_checkpoint(path)


def test_prediction_checkpoint_valid_without_final_newline(tmp_path):
    from scripts.large_model_pipeline import prediction_checkpoint
    row = {"case_id": "a", "response": "{}", "input_tokens": 1, "output_tokens": 1, "seconds": 1}
    path = tmp_path / "pred.jsonl"
    path.write_text(json.dumps(row))
    assert prediction_checkpoint(path) == [row]


def test_cpu_scoring_skips_unfinished_gpu_stages(tmp_path, monkeypatch):
    import scripts.large_model_pipeline as p
    prepared = tmp_path / "artifacts/large32b"
    prepared.mkdir(parents=True)
    (prepared / "prepared.json").write_text(json.dumps({"source": "ok", "data": {}}))
    out = tmp_path / "run"
    out.mkdir()
    state = {"prepared_sha256": p.digest(prepared / "prepared.json"), "stages": {
        "modeling_base_infer": {"status": "complete"}, "modeling_sft_infer": {"status": "running"}}}
    (out / "run.json").write_text(json.dumps(state))
    before = (out / "run.json").read_bytes()
    monkeypatch.setattr(p, "compatible_source", lambda s: True)
    calls = []
    monkeypatch.setattr(p, "score", lambda a: calls.append(a))
    p.cpu_score(SimpleNamespace(base=tmp_path, output=out))
    assert len(calls) == 1 and calls[0].arm == "modeling"
    assert (out / "run.json").read_bytes() == before
    cpu = json.loads((out / "cpu_run.json").read_text())
    assert cpu["status"] == "partial_waiting_for_gpu"


def test_extended_grid_evaluation_only(tmp_path):
    import scripts.large_model_pipeline as p
    out = tmp_path / "extended"
    p.prepare_extra(SimpleNamespace(output=out))
    rows = read_rows(out / "cases.jsonl")
    assert len(rows) == 240
    assert len({r["id"] for r in rows}) == 240
    assert {r["size"] for r in rows} == {20, 32, 48, 64}
    assert {r["split"] for r in rows} == {"test"}
    assert {r["evaluation_group"] for r in rows} == {"single", "triple"}
    with pytest.raises(FileExistsError):
        p.prepare_extra(SimpleNamespace(output=out))
