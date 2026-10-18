from types import SimpleNamespace
import ast
from pathlib import Path

import pytest

from experiments.train_s2e_lora import initialize_zero_runtime


def runtime(world=2):
    events = []
    state = {"ready": False}
    cuda = SimpleNamespace(device_count=lambda: 2, set_device=lambda rank: events.append(("device", rank)))
    comm = SimpleNamespace(is_initialized=lambda: state["ready"], get_world_size=lambda: world, get_rank=lambda: 1)
    def init(**kwargs):
        assert events == [("device", 1)]
        assert kwargs["dist_backend"] == "nccl" and kwargs["auto_mpi_discovery"] is False
        state["ready"] = True
        events.append(("initialized", world))
    return SimpleNamespace(cuda=cuda, distributed=comm), SimpleNamespace(init_distributed=init, comm=comm), events


ENV = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "localhost", "MASTER_PORT": "12345"}


def test_zero_initializes_before_batch_calculation():
    torch, ds, events = runtime()
    world = initialize_zero_runtime(torch, ds, ENV)
    assert world * 1 * 2 == 4
    assert events == [("device", 1), ("initialized", 2)]


def test_zero_rejects_backend_world_mismatch():
    torch, ds, _ = runtime(world=1)
    with pytest.raises(RuntimeError, match="disagrees"):
        initialize_zero_runtime(torch, ds, ENV)


@pytest.mark.parametrize("key", list(ENV))
def test_zero_missing_launcher_metadata_fails_early(key):
    torch, ds, events = runtime()
    with pytest.raises(RuntimeError, match="torchrun"):
        initialize_zero_runtime(torch, ds, {k: v for k, v in ENV.items() if k != key})
    assert not events


def test_main_initializes_zero_before_config_and_model_loading():
    tree = ast.parse(Path("experiments/train_s2e_lora.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
    init = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "initialize_zero_runtime")
    config = next(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "HfDeepSpeedConfig")
    load = next(n.lineno for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "from_pretrained"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "model_class")
    assert init < config < load
