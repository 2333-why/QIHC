import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import generic_energy_pipeline as pipeline
from experiments.generic_energy import write_rows
from qihc.generic.benchmark import make_dataset


def make_reuse_fixture(tmp_path, monkeypatch):
    source, target = tmp_path / "old", tmp_path / "new"
    source.mkdir()
    target.mkdir()
    args = SimpleNamespace(per_family=8, steps=4096, chains=64, semantic_training=True, cpu_smoke=False)
    write_rows(source / "cases.jsonl", make_dataset(8, expanded=True))
    for name in ("oracle", "teacher_audit", "sft", "dpo", "dpo_semantic", "dpo_policy"):
        write_rows(source / f"{name}.jsonl", [{"fixture": name}])
    report = {"config": vars(args), "model_path": "model", "holdout_family": "balanced_cut",
              "dataset_sha256": hashlib.sha256((source / "cases.jsonl").read_bytes()).hexdigest(),
              "stages": [{"stage": s, "status": "complete"} for s in ("data", "oracle", "teacher")]}
    (source / "run.json").write_text(json.dumps(report))
    monkeypatch.setattr(pipeline, "audit_teacher", lambda *a: {})
    monkeypatch.setattr(pipeline, "audit_evaluation", lambda *a: {})
    return source, target, args, report


def test_reuse_copies_into_new_run_preserving_original(tmp_path, monkeypatch):
    source, target, args, _ = make_reuse_fixture(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = pipeline.reuse_preparation(source, target, args, "model")
    assert result["stages"] == ["data", "oracle", "teacher"]
    assert result["original_run_preserved"]
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert all((target / name).read_bytes() == before[name] for name in result["files_sha256"])


@pytest.mark.parametrize("problem", ["budget", "incomplete", "mined", "dataset", "hash", "model", "audit", "same_directory"])
def test_reuse_refuses_incompatible_or_unverified_inputs(tmp_path, monkeypatch, problem):
    source, target, args, report = make_reuse_fixture(tmp_path, monkeypatch)
    model = "model"
    if problem == "budget":
        report["config"]["steps"] = 42
    elif problem == "incomplete":
        report["stages"].pop()
    elif problem == "mined":
        report["stages"].append({"stage": "mine", "status": "failed"})
    elif problem == "dataset":
        write_rows(source / "cases.jsonl", make_dataset(8))
    elif problem == "hash":
        report["dataset_sha256"] = "wrong"
    elif problem == "model":
        model = "other"
    elif problem == "audit":
        def fail(*a):
            raise RuntimeError("Invalid teacher")
        monkeypatch.setattr(pipeline, "audit_teacher", fail)
    else:
        target = source
    (source / "run.json").write_text(json.dumps(report))
    # vars(args) was shared with fixture config; restore the intended current budget.
    args.steps = 4096
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.reuse_preparation(source, target, args, model)
