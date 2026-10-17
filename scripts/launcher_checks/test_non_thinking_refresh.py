import json
from pathlib import Path
import subprocess

import pytest

from scripts import refresh_non_thinking_preparation as migration


def setup_record(tmp_path, monkeypatch):
    manifest = tmp_path / "prepared.json"
    original = {"source_sha256": migration.PREVIOUS, "model_path": "/existing/model"}
    manifest.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    monkeypatch.setattr(migration, "code_hash", lambda _: migration.CURRENT)
    return manifest, original


@pytest.mark.parametrize("previous", [migration.PREVIOUS, migration.PREVIOUS_NON_THINKING, migration.PREVIOUS_REGULARIZED, migration.PREVIOUS_COMPETITIVE, migration.PREVIOUS_GENERIC])
def test_refresh_reuses_inputs_and_preserves_original_evidence(tmp_path, monkeypatch, previous):
    manifest, original = setup_record(tmp_path, monkeypatch)
    original["source_sha256"] = previous
    manifest.write_text(json.dumps(original))
    verified = []
    monkeypatch.setattr(migration, "verify", lambda args: verified.append(json.loads(args.manifest.read_text())))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["PIP_NO_INDEX"] == "1"
        if "experiments/check_training_integration.py" in command:
            report = Path(command[-1])
            report.mkdir()
            (report / "summary.json").write_text(json.dumps({"status": "complete", "adapter_inference_exercised": True}))

    monkeypatch.setattr(migration.subprocess, "run", run)
    migration.refresh(manifest, tmp_path, tmp_path)
    updated = json.loads(manifest.read_text())
    assert updated["source_sha256"] == migration.CURRENT
    assert updated["model_path"] == original["model_path"]
    assert json.loads((tmp_path / f"prepared.before-{previous[:12]}.json").read_text()) == original
    assert len(commands) == 3 and len(verified) == 1
    assert updated["source_migrations"][0]["previous_source_sha256"] == previous
    assert not (tmp_path / "prepared.non-thinking.pending.json").exists()


@pytest.mark.parametrize("failure_at", ["verify", "tests"])
def test_failed_revalidation_never_changes_manifest(tmp_path, monkeypatch, failure_at):
    manifest, original = setup_record(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise RuntimeError("test failure")

    monkeypatch.setattr(migration, "verify", fail if failure_at == "verify" else lambda _: None)
    monkeypatch.setattr(migration.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        migration.refresh(manifest, tmp_path, tmp_path)
    assert json.loads(manifest.read_text()) == original
    assert not (tmp_path / "prepared.before-non-thinking.json").exists()
    assert not (tmp_path / "prepared.non-thinking.pending.json").exists()


def test_unknown_checkout_is_not_resealed(tmp_path, monkeypatch):
    manifest, original = setup_record(tmp_path, monkeypatch)
    monkeypatch.setattr(migration, "code_hash", lambda _: "arbitrary-local-edits")
    with pytest.raises(RuntimeError, match="approved"):
        migration.refresh(manifest, tmp_path, tmp_path)
    assert json.loads(manifest.read_text()) == original


def test_current_preparation_is_not_repeated(tmp_path, monkeypatch):
    manifest, _ = setup_record(tmp_path, monkeypatch)
    manifest.write_text(json.dumps({"source_sha256": migration.CURRENT}))
    migration.refresh(manifest, tmp_path, tmp_path)
    assert not (tmp_path / "prepared.non-thinking.pending.json").exists()
