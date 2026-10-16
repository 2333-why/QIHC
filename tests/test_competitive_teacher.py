import json

import pytest

from experiments import competitive_teacher as module
from experiments.prepare_joint_training_data import inference_domain
from qihc.problems.cvrp.instance import generate_synthetic_instance, RouteSolution
from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector
from qihc.problems.cvrp.neighborhood import KNNNeighborhoodSelector


def payload(customer):
    return {"destroy_customers": [customer], "candidate_routes": {str(customer): [0, 1]},
            "candidate_route_logits": {str(customer): {"0": 0, "1": 0}}, "confidence": 1}


def context():
    return {"routes": [[1, 2], [3]], "allowed_routes": {str(c): [0, 1] for c in (1, 2, 3)},
            "required_destroy_count": 1, "seed": 7, "solver_config": {"routes_per_customer": 2}}


def test_competition_uses_common_seeds_budget_and_can_choose_knn(monkeypatch):
    seen = []
    def reward(answer, ctx, *, chains, steps):
        c = answer["destroy_customers"][0]
        seen.append((c, ctx["seed"], chains, steps))
        return {1: 1.3, 2: 1.1, 3: 1.05}[c]
    monkeypatch.setattr(module, "rollout_reward", reward)
    result = module.select_teacher("KNN reference proposal: " + json.dumps(payload(1)), payload(2), payload(3), context())
    assert result["winner"]["source"] == "knn"
    assert result["pair"]["source"] == "llm"
    assert len(seen) == 9
    assert all({(s, ch, st) for c, s, ch, st in seen if c == customer} ==
               {(7, 64, 48), (104736, 64, 48), (209766, 64, 48)} for customer in (1, 2, 3))


def test_confirmation_seed_cannot_be_used_to_rerank(monkeypatch):
    def reward(answer, ctx, **kwargs):
        c = answer["destroy_customers"][0]
        return ({1: 1.3, 2: 1.2, 3: 1.1} if ctx["seed"] < 200000 else {1: 1.05, 2: 1.4, 3: 1.3})[c]
    monkeypatch.setattr(module, "rollout_reward", reward)
    result = module.select_teacher("KNN reference proposal: " + json.dumps(payload(1)), payload(2), payload(3), context())
    assert result["winner"]["source"] == "knn" and result["pair"] is None


def test_stagnant_samples_are_not_supervised(monkeypatch):
    monkeypatch.setattr(module, "rollout_reward", lambda *a, **k: 1.0)
    result = module.select_teacher("KNN reference proposal: " + json.dumps(payload(1)), payload(2), payload(3), context())
    assert result["winner"] is None and result["pair"] is None


def test_missing_exact_reference_is_not_silently_fabricated():
    with pytest.raises(ValueError, match="exact KNN"):
        module.select_teacher("legacy prompt", payload(2), payload(3), context())


def test_competitive_prompt_preserves_knn_domain_and_is_normalization_stable():
    instance = generate_synthetic_instance(60, 6, 10000, seed=10, semantic_constraints=False)
    solution = RouteSolution([list(range(i + 1, 61, 6)) for i in range(6)])
    selector = LocalLLMNeighborhoodSelector.__new__(LocalLLMNeighborhoodSelector)
    selector.destroy_size = 12
    selector.routes_per_customer = 4
    selector.fallback = KNNNeighborhoodSelector(12, 4)
    selector.neighborhood_strategy = "competitive"
    selector._proposal_iteration, selector._proposal_seed = 3, 9
    selector._pbit_logit_feedback = {}
    prompt = selector._prompt(instance, solution, 3)
    reference = selector._knn_reference(instance, solution)
    allowed = inference_domain(prompt, {})
    assert "coherent nearby customer cluster" in prompt
    for customer in reference["destroy_customers"]:
        assert set(reference["candidate_routes"][str(customer)]) <= set(allowed[str(customer)])
    normalized = selector._normalize(instance, solution, reference)
    assert normalized.destroy_customers == reference["destroy_customers"]
    assert {str(c): rs for c, rs in normalized.candidate_routes.items()} == reference["candidate_routes"]


def test_competition_real_cpu_rollout_runs():
    instance = generate_synthetic_instance(3, 2, 10000, seed=6, semantic_constraints=False)
    ctx = {**context(), "instance": instance.to_dict(), "reference_objective": 10000}
    result = module.select_teacher("KNN reference proposal: " + json.dumps(payload(1)), payload(2), payload(3), ctx,
                                   chains=4, steps=3)
    assert len(result["candidates"]) == 3
    assert all(len(candidate["rewards"]) == 3 for candidate in result["candidates"])


def test_builder_supervises_actual_winner_not_original_pbit_teacher(tmp_path, monkeypatch):
    import sys
    from experiments.prepare_joint_training_data import main
    from qihc.problems.cvrp.instance import save_jsonl
    instance = generate_synthetic_instance(3, 2, 10000, seed=6, semantic_constraints=False)
    data = tmp_path / "data.jsonl"
    save_jsonl([instance], data)
    constraints = tmp_path / "constraints.jsonl"
    constraints.write_text("")
    features = [{"customer": c, "current_route": int(c == 3), "moves": [{"route": int(c != 3)}]}
                for c in (1, 2, 3)]
    prompt = "Objective-aware move features: " + json.dumps(features) + "\nKNN reference proposal: " + json.dumps(payload(1))
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"instance": instance.name, "search_seed": 0, "iteration": 0,
                                 "ok": True, "prompt": prompt}))
    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"instance": instance.name, "search_seed": 0, "method": "llm",
        "status": "ok", "reference_feasible": True, "training_instance": instance.to_dict(),
        "config": {"routes_per_customer": 2}, "records": [{"iteration": 0, "accepted": False,
        "proposal_payload": payload(3), "teacher_payload": payload(2),
        "incumbent_routes_before": [[1, 2], [3]], "incumbent_before": 100}]}]))
    monkeypatch.setattr(module, "rollout_reward", lambda answer, ctx, **kw:
                        {1: 1.3, 2: 1.1, 3: 1.05}[answer["destroy_customers"][0]])
    output = tmp_path / "training"
    monkeypatch.setattr(sys, "argv", ["builder", "--data", str(data), "--constraint-records", str(constraints),
        "--solver-results", str(results), "--llm-audits", str(audit), "--output", str(output),
        "--competitive-teacher", "--verify-preferences"])
    assert main() == 0
    sft = [json.loads(row) for row in (output / "sft.jsonl").read_text().splitlines()]
    dpo = [json.loads(row) for row in (output / "dpo.jsonl").read_text().splitlines()]
    assert len(sft) == 2 and sft[1]["teacher_source"] == "knn"
    assert json.loads(sft[1]["completion"])["destroy_customers"] == [1]
    assert len(dpo) == 1 and dpo[0]["chosen_source"] == "knn" and dpo[0]["competition_verified"]
    assert json.loads((output / "summary.json").read_text())["teacher_wins"]["knn"] == 1
