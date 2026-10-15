import json

from scripts.posttrain_overnight import compare, select_training


def test_comparison_retains_failure_and_missing_counts():
    before = [{"instance": "a", "search_seed": 0, "status": "ok", "reference_feasible": True, "final_objective": 100},
              {"instance": "b", "search_seed": 0, "status": "ok", "reference_feasible": True, "final_objective": 80},
              {"instance": "c", "search_seed": 0, "status": "error"}]
    after = [{"instance": "a", "search_seed": 0, "status": "ok", "reference_feasible": True, "final_objective": 90},
             {"instance": "b", "search_seed": 0, "status": "error"}]
    result = compare(before, after)
    assert result["before_jobs"] == 3 and result["after_jobs"] == 2
    assert result["missing_after"] == 1 and result["paired_feasible"] == 1
    assert result["before_feasible"] == 2 and result["after_feasible"] == 1
    assert result["mean_paired_cost_improvement_percent"] == 10
    assert result["mean_paired_cost_after"] == 90


def test_no_paired_success_is_not_zero_improvement():
    result = compare([], [])
    assert result["mean_paired_cost_improvement_percent"] is None


def test_context_filter_reports_exclusion_without_truncating(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(text)

    source, output = tmp_path / "input.jsonl", tmp_path / "selected.jsonl"
    rows = [{"task": "proposal", "prompt": "x", "chosen": "yes", "rejected": "no"},
            {"task": "proposal", "prompt": "x" * 100, "chosen": "yes", "rejected": "no"}]
    source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    counts = select_training(source, output, Tokenizer(), "dpo", 10)
    assert counts["kept_rows"] == 1 and counts["excluded_over_context"] == 1
    assert json.loads(output.read_text()) == rows[0]
