import os
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
if not Path(BASH).is_file():
    pytest.skip("Bash unavailable", allow_module_level=True)
SCRIPT = (ROOT / "scripts/run_generic_energy.sh").as_posix()


def test_generic_launcher_syntax_and_plan():
    assert subprocess.run([BASH, "-n", SCRIPT], capture_output=True).returncode == 0
    result = subprocess.run([BASH, SCRIPT, "--print-plan"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "192 instances" in result.stdout and "No FPGA" in result.stdout


@pytest.mark.parametrize("metadata", [{"WORLD_SIZE": "2"}, {"LOCAL_RANK": "1"}])
def test_generic_launcher_rejects_multiple_workers(metadata):
    result = subprocess.run([BASH, SCRIPT], env={**os.environ, **metadata}, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Requires" in result.stderr


def test_generic_single_worker_activates_environment_and_runs_pipeline(tmp_path):
    script = tmp_path / "scripts/run_generic_energy.sh"
    script.parent.mkdir()
    script.write_text(Path(SCRIPT).read_text(), encoding="utf-8", newline="\n")
    base = tmp_path / "base"
    hook = base / "miniconda3/etc/profile.d/conda.sh"
    hook.parent.mkdir(parents=True)
    hook.write_text('conda() { export CONDA_PREFIX="$2"; export PATH="$(cd "$2/bin" && pwd -P):$PATH"; }\n', encoding="utf-8", newline="\n")
    python = base / "envs/qihc-cu128/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nset -eu\n'
                      '[[ "$CONDA_PREFIX" == "$CUDA_HOME" && "$HF_HUB_OFFLINE" == 1 && "$PIP_NO_INDEX" == 1 ]]\n'
                      '[[ -z "${RANK+x}" && -z "${WORLD_SIZE+x}" ]]\n'
                      'printf "CHECKED %s\\n" "$*"\n', encoding="utf-8", newline="\n")
    python.chmod(0o755)
    env = {**os.environ, "QIHC_BASE": base.as_posix(), "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
    for key in ("LOCAL_WORLD_SIZE", "GROUP_WORLD_SIZE", "ROLE_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
        env.pop(key, None)
    result = subprocess.run([BASH, script.as_posix()], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "refresh_non_thinking_preparation.py" in result.stdout
    assert "scripts/generic_energy_pipeline.py" in result.stdout
