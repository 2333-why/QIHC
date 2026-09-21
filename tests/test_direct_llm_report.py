import json
import sys

from experiments.analyze_direct_llm_cvrp import main


def test_direct_llm_report_counts_infeasible_and_failed_jobs(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text(
        json.dumps({"name": "X-n101-k10-nl-hard", "customers": [{"id": 1}]}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "direct_llm"
    output.mkdir()
    (output / "results.json").write_text(
        json.dumps([
            {
                "instance": "X-n101-k10-nl-hard", "method": "llm_direct", "search_seed": 0,
                "reference_feasible": False, "reference_violations": [{"type": "capacity"}],
                "final_objective": 42.0, "total_elapsed_s": 1.5,
            }
        ]),
        encoding="utf-8",
    )
    (output / "failures.json").write_text(
        json.dumps([
            {"instance": "X-n101-k10-nl-hard", "method": "llm_direct", "search_seed": 1}
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "report", "--data", str(data), "--output", str(output), "--seeds", "0", "1",
    ])
    assert main() == 0
    report = json.loads((tmp_path / "direct_llm_report.json").read_text(encoding="utf-8"))
    assert report["coverage"]["expected_jobs"] == 2
    assert report["coverage"]["completed_jobs"] == 1
    assert report["outcomes"]["strict_feasible_rate"] == 0.0
    assert report["outcomes"]["strict_violation_type_counts"] == {"capacity": 1}
