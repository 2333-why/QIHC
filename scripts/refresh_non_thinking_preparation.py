#!/usr/bin/env python3
"""Offline, version-pinned migration; never download or accept arbitrary edits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.server_readiness import atomic_json, code_hash, verify

# Exact LF source fingerprints of the already prepared release and this patch.
PREVIOUS = "f981bda58233dace190b32155d2930a72aad030b91538605e926ba7f61bc79d2"
CURRENT = "86356606a82229a29fc206db10ea6f4b7b2e464cdcd5204e2c3085c29f9903ea"
PREVIOUS_NON_THINKING = "5a858b9fb8a87dd6ba0b17b05c0f98ab2e7282b3981dc09c1af611cedcb38c56"
PREVIOUS_REGULARIZED = "4e9d0c6217ba777988b9afb0df7d118425bab2ffead3589521b5cd7ab037666c"


def refresh(manifest: Path, code_dir: Path, env_prefix: Path) -> None:
    record = json.loads(manifest.read_text(encoding="utf-8"))
    actual = code_hash(code_dir)
    if record.get("source_sha256") == actual:
        return  # Normal launcher still verifies every input afterward.
    if record.get("source_sha256") not in {PREVIOUS, PREVIOUS_NON_THINKING, PREVIOUS_REGULARIZED} or actual != CURRENT:
        raise RuntimeError("Source is not the approved non-thinking migration; refusing to reseal arbitrary changes")
    if code_dir.resolve() != ROOT.resolve():
        raise RuntimeError("Migration and pipeline must use the same checkout")
    candidate = {**record, "source_sha256": actual}
    pending = manifest.with_name("prepared.non-thinking.pending.json")
    atomic_json(pending, candidate)
    try:
        # Recheck environment identity, every package, model hash/header and
        # benchmark digest before running tests or changing the real manifest.
        verify(SimpleNamespace(manifest=pending, code_dir=code_dir, env_prefix=env_prefix))
        offline = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", PIP_NO_INDEX="1", PYTHONNOUSERSITE="1",
                       PYTHONPATH=str(code_dir))
        print("OFFLINE PATCH REVALIDATION: tests and tiny SFT/DPO/GRPO; no downloads", flush=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=code_dir, env=offline, check=True)
        report = manifest.parent / "non-thinking-training-api"
        subprocess.run([sys.executable, "experiments/check_training_integration.py", "--output", str(report)],
                       cwd=code_dir, env=offline, check=True)
        training = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        if training.get("status") != "complete" or not training.get("adapter_inference_exercised"):
            raise RuntimeError("Patched training/adapter inference revalidation failed")
        # Abort if the checkout was modified while the tests were running.
        if code_hash(code_dir) != actual:
            raise RuntimeError("Source changed during patch revalidation")
        backup = manifest.with_name(f"prepared.before-{record['source_sha256'][:12]}.json")
        if not backup.exists():
            atomic_json(backup, record)
        candidate["source_migrations"] = [*record.get("source_migrations", []), {
            "name": "competitive_neighborhood_teacher", "previous_source_sha256": record["source_sha256"],
            "source_sha256": actual, "utc": datetime.now(timezone.utc).isoformat(),
            "checks": "all unit tests, actual tiny SFT/DPO/GRPO and adapter inference, original package/model/data verification",
            "training_smoke": str(report), "backup": str(backup),
            "deferred": "real-model generation and GPU training remain mandatory in launcher",
        }]
        atomic_json(manifest, candidate)
        print("Offline code revalidation passed; original model/data/environment reused", flush=True)
    finally:
        pending.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--env-prefix", type=Path, required=True)
    args = parser.parse_args()
    refresh(args.manifest, args.code_dir.resolve(), args.env_prefix.resolve())
