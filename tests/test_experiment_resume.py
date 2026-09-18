import json

from experiments.run_cvrp_lns import (
    deduplicate_job_records,
    job_key,
    read_valid_jsonl,
)


def test_resume_reader_preserves_valid_rows_and_skips_partial_tail(tmp_path):
    path = tmp_path / "results_rank0.jsonl"
    valid = {
        "instance": "case-1",
        "method": "llm",
        "search_seed": 3,
        "status": "ok",
    }
    path.write_text(
        json.dumps(valid) + "\n" + '{"instance":"incomplete"',
        encoding="utf-8",
    )
    assert read_valid_jsonl(path) == [valid]
    assert job_key(valid) == ("case-1", "llm", 3)


def test_deduplicate_job_records_prefers_success_and_preserves_job_order():
    failed = {
        "instance": "case-1",
        "method": "llm",
        "search_seed": 3,
        "status": "error",
    }
    successful = {**failed, "status": "ok", "score": 1.0}
    other = {
        "instance": "case-2",
        "method": "llm",
        "search_seed": 0,
        "status": "ok",
    }

    assert deduplicate_job_records([failed, other, successful, successful]) == [
        successful,
        other,
    ]
