"""Load real BBH tasks from Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from qihc.orchestrator.bbh import BBHTask, DATA_DIR
from qihc.orchestrator.bbh_parser import (
    DEFAULT_BBH_HF_TASKS,
    joschka_row_to_fields,
    lukaemon_row_to_fields,
    target_to_gold_index,
)

DEFAULT_HF_REPO = "Joschka/big_bench_hard"
FALLBACK_HF_REPO = "lukaemon/bbh"
DEFAULT_CACHE_PATH = DATA_DIR / "bbh_hf_cache.json"


def _repo_style(repo: str) -> Literal["joschka", "lukaemon"]:
    if "lukaemon" in repo.lower():
        return "lukaemon"
    return "joschka"


def _rows_from_hf_dataset(repo: str, task_name: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    if _repo_style(repo) == "lukaemon":
        ds = load_dataset(repo, task_name, split="test")
        return [dict(row) for row in ds]

    ds_dict = load_dataset(repo, task_name)
    split_name = task_name if task_name in ds_dict else list(ds_dict.keys())[0]
    return [dict(row) for row in ds_dict[split_name]]


def _row_to_task(
    row: dict[str, Any],
    task_name: str,
    repo: str,
    example_idx: int,
) -> BBHTask | None:
    try:
        if _repo_style(repo) == "lukaemon":
            stem, candidates, target, labels = lukaemon_row_to_fields(row)
        else:
            stem, candidates, target, labels = joschka_row_to_fields(row)
        if len(candidates) < 2:
            return None
        gold = target_to_gold_index(target, candidates, labels)
        return BBHTask(
            task_id=f"{task_name}_{example_idx}",
            task_type=task_name,
            text=stem,
            candidates=candidates,
            top_k=1,
            gold_indices=[gold],
            exclusion_pairs=[],
            logits=None,
        )
    except (ValueError, KeyError, IndexError):
        return None


def load_bbh_tasks_hf(
    repo: str = DEFAULT_HF_REPO,
    task_names: list[str] | None = None,
    limit_per_task: int | None = None,
    cache_path: Path | str | None = DEFAULT_CACHE_PATH,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[BBHTask]:
    """
    Load BBH multiple-choice tasks from Hugging Face.

    Default repo: ``Joschka/big_bench_hard`` (structured ``choices`` field).
    Fallback: ``lukaemon/bbh``.
    """
    task_names = task_names or list(DEFAULT_BBH_HF_TASKS)
    cache_path = Path(cache_path) if cache_path else None

    if use_cache and cache_path and cache_path.is_file() and not refresh_cache:
        cached = _load_cache(cache_path)
        if cached.get("repo") == repo and cached.get("task_names") == task_names:
            if limit_per_task is None or cached.get("limit_per_task") == limit_per_task:
                return _tasks_from_cache_items(cached["tasks"])

    tasks: list[BBHTask] = []
    failed_tasks: list[str] = []

    for task_name in task_names:
        rows: list[dict[str, Any]]
        repo_used = repo
        try:
            rows = _rows_from_hf_dataset(repo, task_name)
        except Exception:
            if repo == FALLBACK_HF_REPO:
                failed_tasks.append(task_name)
                continue
            try:
                rows = _rows_from_hf_dataset(FALLBACK_HF_REPO, task_name)
                repo_used = FALLBACK_HF_REPO
            except Exception:
                failed_tasks.append(task_name)
                continue

        if limit_per_task is not None:
            rows = rows[:limit_per_task]

        n_ok = 0
        for idx, row in enumerate(rows):
            task = _row_to_task(row, task_name, repo_used, idx)
            if task is not None:
                tasks.append(task)
                n_ok += 1
        if n_ok == 0:
            failed_tasks.append(task_name)

    if not tasks:
        raise RuntimeError(
            f"No BBH tasks parsed from repo={repo!r}, tasks={task_names}, failed={failed_tasks}"
        )

    if use_cache and cache_path:
        _save_cache(
            cache_path,
            {
                "repo": repo,
                "task_names": task_names,
                "limit_per_task": limit_per_task,
                "n_tasks": len(tasks),
                "failed_tasks": failed_tasks,
                "tasks": [_task_to_dict(t) for t in tasks],
            },
        )

    return tasks


def _task_to_dict(task: BBHTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "text": task.text,
        "candidates": task.candidates,
        "top_k": task.top_k,
        "gold_indices": task.gold_indices,
        "exclusion_pairs": [list(p) for p in task.exclusion_pairs],
    }


def _tasks_from_cache_items(items: list[dict[str, Any]]) -> list[BBHTask]:
    out: list[BBHTask] = []
    for item in items:
        out.append(
            BBHTask(
                task_id=item["task_id"],
                task_type=item["task_type"],
                text=item["text"],
                candidates=item["candidates"],
                top_k=int(item["top_k"]),
                gold_indices=[int(x) for x in item["gold_indices"]],
                exclusion_pairs=[(int(a), int(b)) for a, b in item.get("exclusion_pairs", [])],
                logits=None,
            )
        )
    return out


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_cache(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
