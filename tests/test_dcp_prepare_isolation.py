import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_prepare_with_unloadable_highs(tmp_path):
    if importlib.util.find_spec("cpmpy") is None:
        pytest.skip("Requires the DCP CPU preparation dependencies")
    (tmp_path / "highspy.py").write_text(
        "raise ImportError('undefined symbol: Highs::releaseMemory')\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]),
    )
    # Exercise CLI dispatch in a fresh interpreter, independent of pytest's loaded solvers.
    code = """
import sys
from types import ModuleType
from experiments.dcp_transfer_v1.run import main

def certify():
    from ortools.sat.python import cp_model
    import cpmpy as cp
    x = cp.intvar(0, 2)
    model = cp.Model(x <= 1, maximize=x)
    assert model.solve(solver="ortools") and x.value() == 1

module = ModuleType("experiments.dcp_transfer_v1.prepare")
module.prepare = certify
sys.modules[module.__name__] = module
sys.argv = ["run", "prepare"]
main()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )
