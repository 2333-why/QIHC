"""Tests for BBH mini-set loader."""

from qihc.orchestrator.bbh import (
    evaluate_prediction,
    load_bbh_problems,
    load_bbh_tasks,
)
from qihc.orchestrator.encoder import greedy_top_k


def test_load_bbh_tasks():
    tasks = load_bbh_tasks()
    assert len(tasks) >= 30
    assert tasks[0].task_id
    assert len(tasks[0].candidates) >= tasks[0].top_k


def test_load_bbh_problems_gold_mask():
    problems = load_bbh_problems(limit=5)
    assert len(problems) == 5
    for p in problems:
        assert p.metadata.get("gold_indices") is not None
        mask = greedy_top_k(p.logits, p.top_k)
        ev = evaluate_prediction(p, mask)
        assert "feasible" in ev
        assert "exact_match" in ev
