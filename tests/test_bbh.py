"""Tests for BBH parser and loaders (no network)."""

from qihc.orchestrator.bbh import load_bbh_problems, load_bbh_tasks
from qihc.orchestrator.bbh_parser import (
    joschka_row_to_fields,
    lukaemon_row_to_fields,
    parse_option_lines,
    target_to_gold_index,
)
from qihc.orchestrator.encoder import greedy_top_k
from qihc.orchestrator.bbh import evaluate_prediction


JOSCHKA_ROW = {
    "question": (
        "On a branch, there are three birds: a blue jay, a quail, and a falcon. "
        "The falcon is to the right of the blue jay."
    ),
    "choices": {
        "label": ["A)", "B)", "C)"],
        "text": [
            "The blue jay is the second from the left",
            "The quail is the second from the left",
            "The falcon is the second from the left",
        ],
    },
    "target": "A",
}

LUKAEMON_ROW = {
    "input": (
        "Sentence: The patient was referred to the specialist because he had a rare skin condition.\n"
        "Options:\n"
        "(A) The patient had a skin condition\n"
        "(B) The specialist had a skin condition\n"
        "(C) Ambiguous"
    ),
    "target": "(A)",
}


def test_joschka_row_parse():
    stem, cands, target, labels = joschka_row_to_fields(JOSCHKA_ROW)
    assert len(cands) == 3
    assert target == "A"
    assert target_to_gold_index(target, cands, labels) == 0
    assert "birds" in stem


def test_lukaemon_row_parse():
    stem, cands, target, _ = lukaemon_row_to_fields(LUKAEMON_ROW)
    assert len(cands) == 3
    assert target_to_gold_index(target, cands) == 0
    assert "patient" in stem.lower()


def test_parse_dash_options():
    opts = parse_option_lines("- Yes\n- No")
    assert opts == ["Yes", "No"]
    assert target_to_gold_index("No", opts) == 1


def test_yes_no_implicit_options():
    row = {
        "question": "Does Elanor tell the truth?",
        "target": "No",
    }
    stem, cands, target, _ = joschka_row_to_fields(row)
    assert cands == ["Yes", "No"]
    assert target_to_gold_index(target, cands) == 1


def test_load_bbh_tasks_bundled():
    tasks = load_bbh_tasks(source="bundled")
    assert len(tasks) >= 30


def test_load_bbh_problems_bundled():
    problems = load_bbh_problems(source="bundled", limit=5)
    assert len(problems) == 5
    for p in problems:
        assert p.metadata.get("gold_indices") is not None
        mask = greedy_top_k(p.logits, p.top_k)
        ev = evaluate_prediction(p, mask)
        assert "feasible" in ev
        assert "exact_match" in ev
