#!/usr/bin/env python3
"""
Download LLM weights for QIHC experiments (China-friendly).

Backends (in --backend auto order):
  1. modelscope  — recommended on mainland servers (Inspire / 4090 集群)
  2. huggingface — hf-mirror.com + HF_HUB_DISABLE_XET=1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ready(local_dir: Path) -> bool:
    return (local_dir / "config.json").is_file() and (
        (local_dir / "tokenizer_config.json").is_file()
        or (local_dir / "tokenizer.json").is_file()
    )


def download_modelscope(repo: str, local_dir: Path) -> Path:
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
        revision="master",
    )
    return Path(path)


def download_huggingface(repo: str, local_dir: Path, cache_dir: str | None) -> Path:
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
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    path = snapshot_download(**kwargs)
    return Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model (ModelScope / HF mirror)")
    parser.add_argument("--repo", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Target directory (must contain config.json when done)",
    )
    parser.add_argument("--cache-dir", default=None, help="HF cache (huggingface backend only)")
    parser.add_argument(
        "--backend",
        choices=("auto", "modelscope", "huggingface"),
        default=os.environ.get("MODEL_DOWNLOAD_BACKEND", "auto"),
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if ready")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and _ready(local_dir):
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
                path = download_modelscope(args.repo, local_dir)
            else:
                path = download_huggingface(args.repo, local_dir, args.cache_dir)
            if _ready(local_dir):
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
