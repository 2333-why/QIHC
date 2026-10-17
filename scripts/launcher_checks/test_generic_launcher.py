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
