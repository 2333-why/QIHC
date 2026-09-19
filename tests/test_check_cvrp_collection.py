import json

from experiments.check_cvrp_collection import check_collection
from qihc.problems.cvrp.instance import generate_synthetic_instance, save_jsonl


def test_check_collection_counts_unique_jobs_not_rows(tmp_path):
    instance = generate_synthetic_instance(n_customers=5, seed=1)
    data = tmp_path / "train.jsonl"
    output = tmp_path / "solver"
    output.mkdir()
    save_jsonl([instance], data)
    one = {"instance": instance.name, "method": "llm", "search_seed": 0, "status": "ok"}
    two = {"instance": instance.name, "method": "llm", "search_seed": 1, "status": "ok"}
    (output / "results_rank0.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [one, one, two]) + "\n",
        encoding="utf-8",
    )
    report = check_collection(data, output, [0, 1], ["llm"])
    assert report["raw_rows"] == 3
    assert report["unique_success"] == 2
    assert report["duplicate_extra_rows"] == 1
    assert report["complete"] is True
