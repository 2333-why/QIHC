"""Structured logging for NSFC evidence experiments."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ExperimentLogger:
    """Append human-readable logs and JSON artifacts under a run directory."""

    def __init__(self, run_dir: str | Path, run_name: str = "nsfc_evidence"):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.log_path = self.run_dir / "run.log"
        self.manifest_path = self.run_dir / "manifest.json"
        self.manifest: dict[str, Any] = {
            "run_name": run_name,
            "started_at": utc_now_iso(),
            "hostname": os.environ.get("HOSTNAME", "unknown"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "steps": [],
            "artifacts": [],
        }
        self._step_start: float | None = None
        self._current_step: str | None = None

    def log(self, message: str, level: str = "INFO") -> None:
        line = f"[{utc_now_iso()}] [{level}] {message}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def begin_step(self, name: str) -> None:
        self._current_step = name
        self._step_start = time.perf_counter()
        self.log(f"=== BEGIN {name} ===")

    def end_step(self, status: str = "ok", extra: dict | None = None) -> None:
        elapsed = time.perf_counter() - (self._step_start or time.perf_counter())
        record = {
            "name": self._current_step,
            "status": status,
            "elapsed_s": round(elapsed, 3),
            "finished_at": utc_now_iso(),
        }
        if extra:
            record.update(extra)
        self.manifest["steps"].append(record)
        self.log(f"=== END {self._current_step} ({status}, {elapsed:.1f}s) ===")
        self._current_step = None
        self._step_start = None
        self.flush_manifest()

    def save_json(self, filename: str, data: Any) -> Path:
        path = self.run_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.manifest["artifacts"].append(str(path.name))
        self.log(f"Saved artifact: {path.name}")
        self.flush_manifest()
        return path

    def flush_manifest(self) -> None:
        self.manifest["updated_at"] = utc_now_iso()
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def run_subprocess(self, name: str, argv: list[str], cwd: str | Path) -> int:
        import subprocess

        self.begin_step(name)
        log_file = self.run_dir / f"{name}.subprocess.log"
        self.log(f"Command: {' '.join(argv)}")
        self.log(f"Subprocess log: {log_file}")
        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(f"# {utc_now_iso()} {name}\n")
                lf.write(f"# cmd: {' '.join(argv)}\n\n")
                proc = subprocess.Popen(
                    argv,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    lf.write(line)
                code = proc.wait()
            self.end_step("ok" if code == 0 else "failed", {"exit_code": code})
            return code
        except Exception as exc:
            self.log(traceback.format_exc(), level="ERROR")
            self.end_step("error", {"error": str(exc)})
            return 1

    def finalize(self, status: str = "completed") -> None:
        self.manifest["status"] = status
        self.manifest["finished_at"] = utc_now_iso()
        self.flush_manifest()
        self.log(f"Run {status}. Manifest: {self.manifest_path}")
