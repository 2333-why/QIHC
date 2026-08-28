import pytest

from qihc.problems.cvrp.llm_selector import LocalLLMNeighborhoodSelector, _extract_json


def test_extract_json_from_fenced_response():
    value = _extract_json('answer\n```json\n{"destroy_customers":[1,2],"confidence":0.8}\n```')
    assert value["destroy_customers"] == [1, 2]


def test_extract_json_rejects_unstructured_text():
    with pytest.raises(ValueError):
        _extract_json("choose customers one and two")


def test_constraint_ir_normalization_without_loading_model():
    selector = LocalLLMNeighborhoodSelector.__new__(LocalLLMNeighborhoodSelector)
    selector._generate = lambda prompt: (
        '{"constraints":['
        '{"type":"same_resource","hard":true,"params":{"entities":[1,2]}},'
        '{"type":"precedence","hard":true,"params":{"before":3,"after":4}}]}'
    )
    constraints, audit = selector.parse_constraint_ir("demo", [1, 2, 3, 4])
    assert [constraint.type for constraint in constraints] == ["same_resource", "precedence"]
    assert audit["parsed"]["constraints"]
