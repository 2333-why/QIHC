import json
import sys

from experiments.merge_joint_training_data import main


def test_merge_joint_training_data_deduplicates_rows(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    rows = {
        "sft.jsonl": {"prompt": "p", "completion": "c"},
        "dpo.jsonl": {"prompt": "p", "chosen": "c", "rejected": "r"},
        "grpo.jsonl": {"prompt": "p", "gold_completion": "c"},
    }
    for filename, row in rows.items():
        payload = json.dumps(row) + "\n"
        (left / filename).write_text(payload, encoding="utf-8")
        (right / filename).write_text(payload, encoding="utf-8")
    output = tmp_path / "merged"
    monkeypatch.setattr(sys, "argv", [
        "merge_joint_training_data.py", "--inputs", str(left), str(right),
        "--output", str(output),
    ])
    assert main() == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["sft"] == {"raw": 2, "unique": 1, "duplicates_removed": 1}
    assert len((output / "dpo.jsonl").read_text(encoding="utf-8").splitlines()) == 1
