#!/usr/bin/env python3
"""
Download LLM weights for QIHC experiments (China-friendly).

Backends (in --backend auto order):
  1. modelscope  — recommended on mainland servers (Inspire / 4090 集群)
  2. huggingface — hf-mirror.com + HF_HUB_DISABLE_XET=1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path


def _ready(local_dir: Path) -> bool:
    metadata_ready = (local_dir / "config.json").is_file() and (
        (local_dir / "tokenizer_config.json").is_file()
        or (local_dir / "tokenizer.json").is_file()
    )
    if not metadata_ready:
        return False

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = local_dir / index_name
        if not index_path.is_file():
            continue
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(payload["weight_map"].values())
        except (KeyError, TypeError, ValueError, OSError):
            return False
        return bool(shards) and all(
            (local_dir / shard).is_file() and (local_dir / shard).stat().st_size > 0
            for shard in shards
        )

    return any(
        path.is_file() and path.stat().st_size > 0
        for path in (local_dir / "model.safetensors", local_dir / "pytorch_model.bin")
    )


def download_modelscope(repo: str, local_dir: Path, revision: str) -> Path:
    try:
        from modelscope import snapshot_download as ms_download
    except ImportError as exc:
        raise RuntimeError(
            "modelscope not installed. Run: pip install modelscope"
        ) from exc

    cache = os.environ.get("MODELSCOPE_CACHE", str(local_dir.parent / ".modelscope"))
    os.makedirs(cache, exist_ok=True)
    print(f"[modelscope] repo={repo} local_dir={local_dir} cache={cache}")
    path = ms_download(
        repo,
        cache_dir=cache,
        local_dir=str(local_dir),
        revision="master" if revision == "main" else revision,
    )
    return Path(path)


def download_huggingface(
    repo: str, local_dir: Path, cache_dir: str | None, revision: str
) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    from huggingface_hub import snapshot_download

    print(f"[huggingface] HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print(f"[huggingface] HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}")
    print(f"[huggingface] repo={repo} local_dir={local_dir}")

    kwargs: dict = {
        "repo_id": repo,
        "local_dir": str(local_dir),
        "resume_download": True,
        "max_workers": 4,
        "revision": revision,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    path = snapshot_download(**kwargs)
    return Path(path)


def model_inventory(local_dir: Path) -> dict:
    """Return a small, shareable completeness inventory without hashing huge weights."""

    config_path = local_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    except (OSError, ValueError):
        config = {}
    index_path = next(
        (local_dir / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
         if (local_dir / name).is_file()),
        None,
    )
    if index_path:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        weight_files = sorted(set(payload.get("weight_map", {}).values()))
    else:
        weight_files = [
            path.name for path in (local_dir / "model.safetensors", local_dir / "pytorch_model.bin")
            if path.is_file()
        ]
    missing = [name for name in weight_files if not (local_dir / name).is_file()]
    empty = [name for name in weight_files if (local_dir / name).is_file() and (local_dir / name).stat().st_size == 0]
    return {
        "ready": _ready(local_dir),
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "weight_file_count": len(weight_files),
        "weight_bytes": sum((local_dir / name).stat().st_size for name in weight_files if (local_dir / name).is_file()),
        "missing_weight_files": missing,
        "empty_weight_files": empty,
    }


def write_manifest(local_dir: Path, repo: str, revision: str, backend: str) -> Path:
    resolved_revision = None
    metadata_dir = local_dir / ".cache" / "huggingface" / "download"
    if metadata_dir.is_dir():
        for metadata_path in metadata_dir.rglob("*.metadata"):
            try:
                first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0].strip()
            except (OSError, IndexError, UnicodeError):
                continue
            if re.fullmatch(r"[0-9a-f]{40,64}", first_line, re.IGNORECASE):
                resolved_revision = first_line
                break
    resolution_errors = []
    if resolved_revision is None:
        from huggingface_hub import HfApi

        endpoints = [os.environ.get("HF_ENDPOINT"), "https://huggingface.co"]
        for endpoint in dict.fromkeys(item for item in endpoints if item):
            try:
                resolved_revision = HfApi(endpoint=endpoint).model_info(
                    repo_id=repo, revision=revision
                ).sha
                break
            except Exception as exc:  # Network metadata is useful but not required for validation.
                resolution_errors.append(f"{endpoint}: {exc!r}")
    if resolved_revision is None:
        resolved_revision = "unavailable: " + " | ".join(resolution_errors)
    manifest = {
        "model_id": repo,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "backend": backend,
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_dir": str(local_dir),
        "inventory": model_inventory(local_dir),
    }
    target = local_dir / "qihc_model_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model (ModelScope / HF mirror)")
    parser.add_argument("--repo", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Target directory (must contain config.json when done)",
    )
    parser.add_argument("--cache-dir", default=None, help="HF cache (huggingface backend only)")
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or immutable commit SHA")
    parser.add_argument(
        "--backend",
        choices=("auto", "modelscope", "huggingface"),
        default=os.environ.get("MODEL_DOWNLOAD_BACKEND", "auto"),
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if ready")
    parser.add_argument(
        "--write-manifest", action="store_true",
        help="Write qihc_model_manifest.json with revision and shard inventory",
    )
    parser.add_argument("--verify-only", action="store_true", help="Do not download; validate local files")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        inventory = model_inventory(local_dir)
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return 0 if inventory["ready"] else 1

    manifest_path = local_dir / "qihc_model_manifest.json"
    existing_manifest = {}
    if manifest_path.is_file():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing_manifest = {}
    identity_matches = (
        existing_manifest.get("model_id", args.repo) == args.repo
        and existing_manifest.get("requested_revision", args.revision) == args.revision
    )
    if not args.force and identity_matches and _ready(local_dir):
        if args.write_manifest and not manifest_path.is_file():
            write_manifest(local_dir, args.repo, args.revision, "existing")
        print(f"[skip] model ready: {local_dir}")
        return 0

    backends: list[str]
    if args.backend == "auto":
        backends = ["modelscope", "huggingface"]
    else:
        backends = [args.backend]

    last_err: Exception | None = None
    for name in backends:
        try:
            print(f"\n=== trying backend: {name} ===")
            if name == "modelscope":
                path = download_modelscope(args.repo, local_dir, args.revision)
            else:
                path = download_huggingface(args.repo, local_dir, args.cache_dir, args.revision)
            if _ready(local_dir):
                if args.write_manifest:
                    manifest = write_manifest(local_dir, args.repo, args.revision, name)
                    print(f"manifest={manifest}")
                print(f"OK ({name}): {path}")
                return 0
            print(f"WARN: {name} finished but config.json missing in {local_dir}")
        except Exception as exc:
            last_err = exc
            print(f"FAIL ({name}): {exc}", file=sys.stderr)

    if last_err is not None:
        raise SystemExit(f"All backends failed. Last error: {last_err}")
    raise SystemExit("Download failed: model files incomplete")


if __name__ == "__main__":
    raise SystemExit(main())
