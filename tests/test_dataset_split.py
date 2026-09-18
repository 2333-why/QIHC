from experiments.split_nlcvrp_jsonl import deterministic_split
from qihc.problems.cvrp.instance import generate_synthetic_instance


def test_deterministic_split_is_reproducible_and_disjoint():
    instances = [
        generate_synthetic_instance(8, 3, 30, seed=index)
        for index in range(20)
    ]
    first = deterministic_split(instances, seed=42)
    second = deterministic_split(list(reversed(instances)), seed=42)
    assert {name: [x.name for x in rows] for name, rows in first.items()} == {
        name: [x.name for x in rows] for name, rows in second.items()
    }
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 12,
        "validation": 4,
        "test": 4,
    }
    name_sets = [{instance.name for instance in rows} for rows in first.values()]
    assert not (name_sets[0] & name_sets[1])
    assert not (name_sets[0] & name_sets[2])
    assert not (name_sets[1] & name_sets[2])
