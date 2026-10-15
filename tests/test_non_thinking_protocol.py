import ast
from pathlib import Path

import pytest

from qihc.problems.cvrp.json_output import extract_answer_json


@pytest.mark.parametrize("relative", [
    "qihc/problems/cvrp/llm_selector.py", "qihc/problems/cvrp/direct_llm.py",
    "experiments/train_s2e_lora.py", "experiments/check_research_environment.py",
])
def test_all_cvrp_chat_template_calls_disable_thinking(relative):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "apply_chat_template"]
    assert calls
    for call in calls:
        flags = {k.arg: k.value for k in call.keywords}
        assert isinstance(flags.get("enable_thinking"), ast.Constant)
        assert flags["enable_thinking"].value is False


def test_closed_empty_thinking_prefix_keeps_complete_answer():
    answer = '{"destroy_customers":[1],"candidate_routes":{"1":[0]},"candidate_route_logits":{"1":{"0":0}},"confidence":0.5}'
    assert extract_answer_json("<think>\n\n</think>\n" + answer)["destroy_customers"] == [1]


def test_unfinished_thinking_is_not_mistaken_for_a_proposal():
    with pytest.raises(ValueError):
        extract_answer_json('<think>Reasoning about {"destroy_customers":[1]}')
