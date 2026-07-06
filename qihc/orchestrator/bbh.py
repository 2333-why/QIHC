"""BBH-style subset-selection tasks for Case A (CR-aligned)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from qihc.orchestrator.reasoning import SubsetProblem, is_feasible

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_BBH_PATH = DATA_DIR / "bbh_subset.json"

SourceName = Literal["bundled", "hf"]


@dataclass
class BBHTask:
    task_id: str
    task_type: str
    text: str
    candidates: list[str]
    top_k: int
    gold_indices: list[int]
    exclusion_pairs: list[tuple[int, int]]
    logits: np.ndarray | None = None

    def to_subset_problem(self, seed: int = 0) -> SubsetProblem:
        logits = self.logits
        if logits is None:
            logits = _logits_from_candidates(self.candidates, self.gold_indices, seed=seed)
        gold_mask = np.zeros(len(self.candidates), dtype=bool)
        for i in self.gold_indices:
            gold_mask[int(i)] = True
        return SubsetProblem(
            text=self.text,
            logits=np.asarray(logits, dtype=float),
            top_k=self.top_k,
            exclusion_pairs=list(self.exclusion_pairs),
            metadata={
                "task_id": self.task_id,
                "task_type": self.task_type,
                "gold_indices": list(self.gold_indices),
                "gold_mask": gold_mask,
                "candidates": list(self.candidates),
            },
        )


def _logits_from_candidates(
    candidates: list[str],
    gold_indices: list[int],
    seed: int = 0,
    gold_bias: float = 1.2,
    noise_scale: float = 0.35,
) -> np.ndarray:
    """Deterministic pseudo-logits mimicking LLM preference scores."""
    rng = np.random.default_rng(seed)
    n = len(candidates)
    logits = rng.normal(0.0, noise_scale, size=n)
    for i in gold_indices:
        logits[int(i)] += gold_bias
    # slight hash-based tie-break
    for i, c in enumerate(candidates):
        h = int(hashlib.md5(c.encode()).hexdigest()[:6], 16) / 1e6
        logits[i] += 0.05 * (h - 0.5)
    return logits


def load_bbh_tasks(
    source: SourceName = "bundled",
    path: Path | str | None = None,
    hf_repo: str | None = None,
    hf_tasks: list[str] | None = None,
    limit_per_task: int | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[BBHTask]:
    """
    Load BBH tasks.

    - ``bundled``: local ``bbh_subset.json`` (synthetic, 40 tasks)
    - ``hf``: Hugging Face BIG-Bench Hard (default ``Joschka/big_bench_hard``)
    """
    if source == "bundled":
        return load_bbh_tasks_bundled(path)
    if source == "hf":
        from qihc.orchestrator.bbh_hf import DEFAULT_HF_REPO, load_bbh_tasks_hf

        return load_bbh_tasks_hf(
            repo=hf_repo or DEFAULT_HF_REPO,
            task_names=hf_tasks,
            limit_per_task=limit_per_task,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
    raise ValueError(f"Unknown BBH source: {source}")


def load_bbh_tasks_bundled(path: Path | str | None = None) -> list[BBHTask]:
    path = Path(path) if path else DEFAULT_BBH_PATH
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    tasks: list[BBHTask] = []
    for item in raw["tasks"]:
        logits = item.get("logits")
        tasks.append(
            BBHTask(
                task_id=item["id"],
                task_type=item["task_type"],
                text=item["text"],
                candidates=item["candidates"],
                top_k=int(item["top_k"]),
                gold_indices=[int(x) for x in item["gold_indices"]],
                exclusion_pairs=[(int(a), int(b)) for a, b in item.get("exclusion_pairs", [])],
                logits=np.asarray(logits, dtype=float) if logits is not None else None,
            )
        )
    return tasks


def load_bbh_problems(
    source: SourceName = "bundled",
    path: Path | str | None = None,
    seed: int = 0,
    limit: int | None = None,
    hf_repo: str | None = None,
    hf_tasks: list[str] | None = None,
    limit_per_task: int | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[SubsetProblem]:
    tasks = load_bbh_tasks(
        source=source,
        path=path,
        hf_repo=hf_repo,
        hf_tasks=hf_tasks,
        limit_per_task=limit_per_task,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    if limit is not None:
        tasks = tasks[:limit]
    return [t.to_subset_problem(seed=seed + i) for i, t in enumerate(tasks)]


def gold_mask_from_problem(problem: SubsetProblem) -> np.ndarray | None:
    gm = problem.metadata.get("gold_mask")
    if gm is not None:
        return np.asarray(gm, dtype=bool)
    indices = problem.metadata.get("gold_indices")
    if indices is None:
        return None
    mask = np.zeros(problem.logits.size, dtype=bool)
    for i in indices:
        mask[int(i)] = True
    return mask


def exact_match(pred: np.ndarray, gold: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(pred, dtype=bool), np.asarray(gold, dtype=bool)))


def jaccard(pred: np.ndarray, gold: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gold = np.asarray(gold, dtype=bool)
    inter = float(np.logical_and(pred, gold).sum())
    union = float(np.logical_or(pred, gold).sum())
    return inter / union if union > 0 else 0.0


def evaluate_prediction(problem: SubsetProblem, mask: np.ndarray) -> dict[str, Any]:
    gold = gold_mask_from_problem(problem)
    feasible = is_feasible(mask, problem.top_k, problem.exclusion_pairs)
    out: dict[str, Any] = {"feasible": feasible}
    if gold is not None:
        out["exact_match"] = exact_match(mask, gold)
        out["jaccard"] = jaccard(mask, gold)
    return out
