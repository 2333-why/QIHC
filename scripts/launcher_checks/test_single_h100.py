"""Wrapper-only regressions kept outside the sealed preparation source set."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"
if not Path(BASH).is_file():
    pytest.skip("Bash unavailable", allow_module_level=True)


def run_wrapper(tmp_path, overrides=None, cli_present=False):
    wrapper = tmp_path / "scripts/run_single_h100.sh"
    child = tmp_path / "scripts/server/02_run_offline_gpu.sh"
    child.parent.mkdir(parents=True)
    wrapper.write_bytes((ROOT / "scripts/run_single_h100.sh").read_bytes())
    child.write_text(
        '#!/usr/bin/env bash\nset -eu\n'
        'for key in RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT; do\n'
        ' [[ -z "${!key+x}" ]] || exit 91\n'
        'done\n'
        'printf "nproc=%s visible=%s arg=%s\\n" "$QIHC_NPROC" "$CUDA_VISIBLE_DEVICES" "$1"\n'
        'nvidia-smi\n', encoding="utf-8", newline="\n",
    )
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE"):
        env.pop(key, None)
    # Use only Bash's directory plus an optional stub CLI, ensuring that the
    # unavailable-CLI case does not depend on the test machine's GPU tools.
    env["PATH"] = Path(BASH).parent.as_posix()
    env.update(QIHC_NPROC="2", CUDA_VISIBLE_DEVICES="3",
               MASTER_ADDR="platform-host", MASTER_PORT="12345")
    if cli_present:
        cli = tmp_path / "nvidia-smi"
        cli.write_text('#!/usr/bin/env bash\necho REAL_CLI\n', encoding="utf-8", newline="\n")
        cli.chmod(0o755)
        env["PATH"] = tmp_path.as_posix() + os.pathsep + env["PATH"]
    env.update(overrides or {})
    return subprocess.run([BASH, wrapper.as_posix(), "--preflight-only"],
                          env=env, capture_output=True, text=True)


@pytest.mark.parametrize("metadata", [{}, {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1",
                                          "LOCAL_WORLD_SIZE": "1", "GROUP_RANK": "0"}])
def test_single_worker_metadata_and_missing_cli(tmp_path, metadata):
    result = run_wrapper(tmp_path, metadata)
    assert result.returncode == 0, result.stderr
    assert "nproc=1 visible=3 arg=--preflight-only" in result.stdout
    assert "mandatory PyTorch CUDA/BF16 checks follow" in result.stdout


def test_real_cli_is_not_overridden(tmp_path):
    result = run_wrapper(tmp_path, cli_present=True)
    assert result.returncode == 0, result.stderr
    assert "REAL_CLI" in result.stdout
    assert "unavailable" not in result.stdout


@pytest.mark.parametrize("metadata", [{"WORLD_SIZE": "2"}, {"LOCAL_WORLD_SIZE": "2"},
                                      {"RANK": "1"}, {"WORLD_SIZE": "invalid"}])
def test_reject_duplicate_or_invalid_launch(tmp_path, metadata):
    result = run_wrapper(tmp_path, metadata)
    assert result.returncode == 2
    assert "Single-H100 entry requires" in result.stderr
    assert "nproc=" not in result.stdout
