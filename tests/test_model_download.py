import json

from experiments.nsfc_evidence.download_model_hf import _ready, model_inventory


def test_model_inventory_checks_every_indexed_shard(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"]}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}),
        encoding="utf-8",
    )
    (tmp_path / "model-1.safetensors").write_bytes(b"one")
    assert not _ready(tmp_path)
    assert model_inventory(tmp_path)["missing_weight_files"] == ["model-2.safetensors"]

    (tmp_path / "model-2.safetensors").write_bytes(b"two")
    inventory = model_inventory(tmp_path)
    assert inventory["ready"]
    assert inventory["model_type"] == "qwen3_moe"
    assert inventory["weight_file_count"] == 2
    assert inventory["weight_bytes"] == 6
