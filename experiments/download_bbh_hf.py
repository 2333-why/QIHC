# -*- coding: utf-8 -*-
"""Pre-download and cache Hugging Face BBH subset for offline use."""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qihc.orchestrator.bbh_hf import DEFAULT_HF_REPO, load_bbh_tasks_hf  # noqa: E402
from qihc.orchestrator.bbh_parser import DEFAULT_BBH_HF_TASKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download HF BBH cache")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    tasks = load_bbh_tasks_hf(
        repo=args.hf_repo,
        task_names=DEFAULT_BBH_HF_TASKS,
        limit_per_task=args.limit_per_task,
        use_cache=True,
        refresh_cache=args.refresh_cache,
    )
    print(f"Cached {len(tasks)} BBH tasks from {args.hf_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
