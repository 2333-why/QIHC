import json

from experiments.run_s2e_pipeline import main, read_checkpoint
from qihc.problems.cvrp.instance import generate_synthetic_instance, save_jsonl


def test_cpp_checkpoint_recovers_latest_complete_instance(tmp_path):
    path = tmp_path / "checkpoint_rank0.jsonl"
    failed = {"record": {"instance": "case-1", "status": "error"}, "feedback": {}}
    succeeded = {"record": {"instance": "case-1", "status": "ok"}, "feedback": {}}
    path.write_text(
        json.dumps(failed) + "\n" + json.dumps(succeeded) + "\n" + '{"record":',
        encoding="utf-8",
    )
    assert read_checkpoint(path) == {"case-1": succeeded}


def test_cpp_pipeline_resumes_completed_instance(tmp_path, monkeypatch):
    instance = generate_synthetic_instance(n_customers=5, seed=7)
    data = tmp_path / "instances.jsonl"
    output = tmp_path / "cpp"
    save_jsonl([instance], data)
    monkeypatch.setattr(
        "sys.argv",
        ["run_s2e_pipeline.py", "--data", str(data), "--output", str(output), "--backend", "heuristic", "--resume"],
    )
    assert main() == 0
    checkpoint = output / "checkpoint_rank0.jsonl"
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 1
    assert main() == 0
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 1
