#!/usr/bin/env python3
"""Download HF models with China mirror + Xet disabled (avoids CAS 401)."""
from __future__ import annotations

import argparse
import os
import sys

# Must set BEFORE importing huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "300"))
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", os.environ.get("HF_HUB_ETAG_TIMEOUT", "60"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model via HF mirror (no Xet)")
    parser.add_argument("--repo", required=True, help="e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local-dir", default=None, help="If set, snapshot to this directory")
    parser.add_argument("--cache-dir", default=None, help="HF cache dir (HF_HOME)")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT"))
    print("HF_HUB_DISABLE_XET:", os.environ.get("HF_HUB_DISABLE_XET"))
    print("repo:", args.repo)
    print("local_dir:", args.local_dir)
    print("cache_dir:", args.cache_dir or os.environ.get("HF_HOME"))

    kwargs: dict = {"repo_id": args.repo}
    if args.local_dir:
        kwargs["local_dir"] = args.local_dir
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    path = snapshot_download(**kwargs)
    print("OK:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
