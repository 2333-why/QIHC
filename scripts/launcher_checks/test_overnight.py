import json

from scripts.posttrain_overnight import compare, select_training, audit_summary, reliability_gate


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


def test_audit_does_not_count_recovery_or_fallback_as_strict_success(tmp_path):
    rows = [{"ok": True, "proposal_source": "llm", "normalization_filled_customers": 0, "normalization_changed": False},
            {"ok": True, "proposal_source": "llm", "recovered_truncated_logits": True},
            {"ok": True, "proposal_source": "llm_fallback_knn"},
            {"ok": True, "proposal_source": "llm", "normalization_filled_customers": 2},
            {"ok": False, "error_type": "ValueError", "error": "bad JSON",
             "generation": {"reached_token_limit": True}}]
    (tmp_path / "llm_audit_rank0.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = audit_summary(tmp_path)
    assert result["successful_calls"] == 4 and result["strict_model_calls"] == 1
    assert result["knn_fallback_calls"] == 2 and result["token_limit_calls"] == 1
    assert result["error_types"] == {"ValueError": 1}
    assert not reliability_gate(result)
    assert reliability_gate({"calls": 100, "strict_model_rate": 0.98}, {"strict_model_rate": 1.0})
    assert not reliability_gate({"calls": 100, "strict_model_rate": 0.95}, {"strict_model_rate": 1.0})


def test_completion_filter_respects_generation_budget(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(text)

    source, target = tmp_path / "source.jsonl", tmp_path / "target.jsonl"
    source.write_text(json.dumps({"prompt": "x", "chosen": "a" * 20, "rejected": "b"}))
    result = select_training(source, target, Tokenizer(), "dpo", 100, 10)
    assert result["kept_rows"] == 0 and result["excluded_over_completion"] == 1
    assert result["excluded_over_context"] == 0


def test_full_driver_builds_matched_knn_and_regularized_dpo_commands(tmp_path, monkeypatch):
    """Exercise orchestration without pretending this mock proves model quality."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from scripts import posttrain_overnight as driver

    output = tmp_path / "run"
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps({"model_path": "/local/model"}))
    monkeypatch.setattr(sys, "argv", ["driver", "--manifest", str(manifest), "--output", str(output)])
    monkeypatch.setattr(driver, "code_hash", lambda _: "test-hash")

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(text)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())))
    commands = {}

    class Process:
        def __init__(self, command, **kwargs):
            name = Path(kwargs["stdout"].name).stem
            commands[name] = command
            if name == "data":
                data = output / "data"
                data.mkdir()
                (data / "split_manifest.json").write_text(json.dumps({"splits": {
                    "train": ["train"], "validation": ["validation"], "test": ["test"]}}))
            if name in {"base", "legacy_base", "knn_pbit", "sft", "dpo", "sft_validation", "dpo_validation"}:
                directory = output / name
                directory.mkdir()
                instance = "validation" if name.endswith("validation") else "test"
                (directory / "results.json").write_text(json.dumps([
                    {"instance": instance, "search_seed": seed, "status": "ok",
                     "reference_feasible": True, "final_objective": 100} for seed in (0, 1)]))
                if name != "knn_pbit":
                    (directory / "llm_audit_rank0.jsonl").write_text(json.dumps({
                        "ok": True, "proposal_source": "llm", "normalization_changed": False}))
            if name == "training_data":
                directory = output / "training_data"
                directory.mkdir()
                (directory / "summary.json").write_text("{}")
                for stage, n in [("sft", 360), ("dpo", 64)]:
                    rows = [{"task": "proposal", "prompt": str(i), **(
                        {"completion": "a"} if stage == "sft" else {"chosen": "a", "rejected": "b"})} for i in range(n)]
                    (directory / f"{stage}.jsonl").write_text("\n".join(map(json.dumps, rows)))

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(driver.subprocess, "Popen", Process)
    assert driver.main() == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["dpo_steps"] == 16 and report["dpo_output_repair_validated"]
    assert set(report["comparisons_vs_knn_pbit"]) == {"base", "legacy_base", "sft", "dpo"}
    knn, base = commands["knn_pbit"], commands["base"]
    assert "--model-path" not in knn and knn[knn.index("--methods") + 1] == "knn"
    for flag in ("--initial-solutions", "--iterations", "--num-chains", "--sampling-steps", "--data"):
        assert knn[knn.index(flag) + 1] == base[base.index(flag) + 1]
    dpo = commands["train_dpo"]
    assert dpo[dpo.index("--dpo-sft-weight") + 1] == "0.2"
    assert dpo[dpo.index("--adapter-path") + 1] == str(output / "adapters/sft")
