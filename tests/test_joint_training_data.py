import json
import sys

from experiments.prepare_joint_training_data import main
from qihc.problems.cvrp.instance import generate_synthetic_instance, save_jsonl
from qihc.s2e.cpp import ConstraintProgramPackage


def test_joint_training_data_contains_constraint_and_pbit_proposal(tmp_path, monkeypatch):
    instance = generate_synthetic_instance(6, 3, 30, seed=19, semantic_constraints=False)
    instance.description = "Minimize distance while respecting capacity."
    data = tmp_path / "data.jsonl"
    save_jsonl([instance], data)
    cpp = ConstraintProgramPackage.from_specs(
        instance.name, instance.description, instance.customer_ids, instance.constraints
    )
    constraint_records = tmp_path / "constraints.jsonl"
    constraint_records.write_text(
        json.dumps({"instance": instance.name, "status": "ok", "cpp": cpp.to_dict()}) + "\n",
        encoding="utf-8",
    )
    solver_results = tmp_path / "results.json"
    solver_results.write_text(
        json.dumps([{
            "instance": instance.name,
            "method": "llm",
            "status": "ok",
            "problem_prompt": instance.description,
            "constraints": [],
            "records": [
                {"accepted": True, "objective_improvement": 3.0, "candidate_route_recall": 1.0,
                 "candidate_compression": 0.5, "proposal_payload": {"destroy_customers": [1], "candidate_routes": {"1": [0, 1]}}},
                {"accepted": False, "objective_improvement": 0.0, "candidate_route_recall": 0.0,
                 "candidate_compression": 0.5, "proposal_payload": {"destroy_customers": [2], "candidate_routes": {"2": [2]}}},
            ],
        }]), encoding="utf-8"
    )
    output = tmp_path / "training"
    monkeypatch.setattr(sys, "argv", [
        "prepare_joint_training_data.py", "--data", str(data),
        "--constraint-records", str(constraint_records),
        "--solver-results", str(solver_results), "--output", str(output),
    ])
    assert main() == 0
    sft = [json.loads(line) for line in (output / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    dpo = [json.loads(line) for line in (output / "dpo.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["task"] for row in sft} == {"constraint", "proposal"}
    assert any(row["task"] == "proposal" for row in dpo)
