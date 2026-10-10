import json
from pathlib import Path

import pytest

from experiments.prepare_research_data import geometry_key, prepare
from experiments.analyze_research_suite import analyze
from qihc.problems.cvrp.instance import generate_synthetic_instance
from qihc.s2e.cpp import ConstraintProgram, ConstraintProgramPackage
from qihc.s2e.validator import CPPValidator


def test_splits_keep_numerical_instances_together_and_generate_semantic_holdouts(tmp_path):
    summary = prepare(tmp_path, [], [12, 16], 4, 1)
    sets = [set(values) for values in summary["splits"].values()]
    assert all(sets)
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    variants = [json.loads(line) for line in (tmp_path / "semantic_test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert variants and all(row["metadata"]["semantic_parent"] in sets[2] for row in variants)
    assert all("known_feasible_routes" not in row["metadata"] for row in variants)
    instance = generate_synthetic_instance(4, 2, 40, seed=9)
    first = geometry_key(instance)
    instance.customers = dict(reversed(list(instance.customers.items())))
    assert geometry_key(instance) == first


def test_report_uses_instances_as_statistical_unit_and_keeps_missing_jobs(tmp_path):
    manifest = {"profile": "smoke", "test_instances": ["a", "b", "missing"],
                "search_seeds": [0, 1], "evaluation_arms": ["population", "no_feedback"],
                "search_time_limit_s": 1.0}
    (tmp_path / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for label, cost in (("population", 90), ("no_feedback", 100)):
        directory = tmp_path / "evaluation" / label
        directory.mkdir(parents=True)
        rows = [{"instance": name, "search_seed": seed, "initial_objective": 120,
                 "final_objective": cost, "reference_feasible": True, "total_elapsed_s": 1,
                 "records": []} for name in ("a", "b") for seed in (0, 1)]
        (directory / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    report = analyze(tmp_path)
    assert report["methods"]["population"]["strict_feasible_rate"] == pytest.approx(4 / 6)
    comparison = report["paired"]["no_feedback"]
    assert comparison["independent_instances"] == 2
    assert comparison["paired_jobs"] == 4
    assert comparison["complete_coverage"] is False


def test_large_validator_explicitly_records_skipped_exact_checks():
    instance = generate_synthetic_instance(10, 2, 60, seed=4)
    cpp = ConstraintProgramPackage.from_specs(instance.name, instance.description, instance.customer_ids, instance.constraints)
    report = CPPValidator().validate(cpp, instance)
    exact = next(stage for stage in report.stages if stage.stage == "A3_EXACT")
    assert exact.skipped and exact.checked == 0


def test_validator_executes_supplied_fixture_and_rejects_false_assertion():
    instance = generate_synthetic_instance(2, 2, 40, seed=4, semantic_constraints=False)
    program = ConstraintProgram("test", "same_resource", True, 1.0, {"entities": [1, 2]}, "same vehicle",
                                tests=({"routes": [[1], [2]], "expected": True},))
    cpp = ConstraintProgramPackage(instance.name, "same vehicle", [1, 2], [program])
    report = CPPValidator().validate(cpp, instance)
    assert not report.passed
    assert report.highest_stage == "A2_TESTS"
