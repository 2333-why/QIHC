import json

from experiments.run_s2e_pipeline import read_checkpoint


def test_cpp_checkpoint_recovers_latest_complete_instance(tmp_path):
    path = tmp_path / "checkpoint_rank0.jsonl"
    failed = {"record": {"instance": "case-1", "status": "error"}, "feedback": {}}
    succeeded = {"record": {"instance": "case-1", "status": "ok"}, "feedback": {}}
    path.write_text(
        json.dumps(failed) + "\n" + json.dumps(succeeded) + "\n" + '{"record":',
        encoding="utf-8",
    )
    assert read_checkpoint(path) == {"case-1": succeeded}
