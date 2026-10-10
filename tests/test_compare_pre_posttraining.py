import json
import sys

from experiments.compare_pre_posttraining import main


def test_compare_pre_posttraining_reports_cold_and_final_improvement(tmp_path, monkeypatch):
    before = tmp_path / "before" / "full"
    after = tmp_path / "after" / "full"
    before.mkdir(parents=True)
    after.mkdir(parents=True)
    common = {"instance": "case", "search_seed": 0, "feasible": True}
    (before / "results.json").write_text(json.dumps([
        {**common, "initial_objective": 120.0, "final_objective": 100.0}
    ]), encoding="utf-8")
    (after / "results.json").write_text(json.dumps([
        {**common, "initial_objective": 110.0, "final_objective": 90.0}
    ]), encoding="utf-8")
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(sys, "argv", [
        "compare_pre_posttraining.py", "--before", str(tmp_path / "before"),
        "--after", str(tmp_path / "after"), "--output", str(output),
    ])
    assert main() == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["mean_final_cost_improvement"] == 10.0
    assert summary["mean_cold_start_cost_improvement"] == 10.0
