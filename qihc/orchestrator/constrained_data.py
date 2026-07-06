"""Synthetic constrained subset tasks and HF BBH constraint wrappers (P0–P2 cases)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np

from qihc.orchestrator.bbh import BBHTask, DATA_DIR, _logits_from_candidates, load_bbh_tasks

SourceName = Literal["synthetic", "constrained_hf"]

DEFAULT_SYNTHETIC_PATH = DATA_DIR / "constrained_synthetic_200.json"

_TASK_TYPES = (
    "logical_deduction",
    "reasoning_with_constraints",
    "multi_step_reasoning",
)

_TEMPLATES: dict[str, list[str]] = {
    "logical_deduction": [
        "Select {k} premises that logically entail: {claim}.",
        "Pick {k} facts supporting the conclusion: {claim}.",
        "Choose {k} statements needed to conclude: {claim}.",
    ],
    "reasoning_with_constraints": [
        "Select {k} compatible reasons for: {claim} (some options conflict).",
        "Pick {k} steps for a valid plan toward: {claim} without contradiction.",
        "Choose {k} evidence items supporting {claim} under mutual-exclusion rules.",
    ],
    "multi_step_reasoning": [
        "Select {k} intermediate steps required before concluding: {claim}.",
        "Pick {k} reasoning hops that bridge the premises to: {claim}.",
        "Choose {k} sub-goals that together justify: {claim}.",
    ],
}

_CLAIMS = [
    "All mammals are warm-blooded",
    "If it rains, the ground is wet",
    "A is greater than C",
    "The hypothesis H is consistent",
    "Route A is optimal under current traffic",
    "The proof plan is valid",
    "The causal chain holds",
    "No contradiction appears in the argument",
]

_CANDIDATE_POOL = [
    "Primary supporting fact",
    "Secondary supporting fact",
    "Tertiary supporting fact",
    "Conflicting alternative",
    "Irrelevant observation",
    "Contradictory claim",
    "Auxiliary lemma",
    "Background assumption",
    "Derived intermediate step",
    "External distraction",
]


def _build_candidates(n: int, rng: np.random.Generator) -> list[str]:
    pool = list(_CANDIDATE_POOL)
    rng.shuffle(pool)
    labels = [chr(ord("A") + i) for i in range(n)]
    return [f"({labels[i]}) {pool[i % len(pool)]}" for i in range(n)]


def generate_synthetic_tasks(
    n_tasks: int = 200,
    seed: int = 42,
) -> list[BBHTask]:
    """Generate reproducible constrained subset tasks (6–8 candidates, top_k 2–3, exclusions)."""
    rng = np.random.default_rng(seed)
    tasks: list[BBHTask] = []

    idx = 0
    while idx < n_tasks:
        task_type = _TASK_TYPES[idx % len(_TASK_TYPES)]
        templates = _TEMPLATES[task_type]
        n_cand = int(rng.integers(6, 9))
        top_k = int(rng.choice([2, 2, 3]))
        candidates = _build_candidates(n_cand, rng)
        gold = sorted(rng.choice(n_cand, size=top_k, replace=False).tolist())
        n_excl = int(rng.integers(1, 3))
        exclusion_pairs: list[tuple[int, int]] = []
        attempts = 0
        while len(exclusion_pairs) < n_excl and attempts < 30:
            a, b = int(rng.integers(0, n_cand)), int(rng.integers(0, n_cand))
            if a == b:
                attempts += 1
                continue
            pair = (min(a, b), max(a, b))
            if pair in exclusion_pairs:
                attempts += 1
                continue
            if a in gold and b in gold:
                attempts += 1
                continue
            exclusion_pairs.append(pair)
            attempts += 1

        claim = _CLAIMS[idx % len(_CLAIMS)]
        text = rng.choice(templates).format(k=top_k, claim=claim)
        logits = _logits_from_candidates(candidates, gold, seed=seed + idx)
        tasks.append(
            BBHTask(
                task_id=f"synthetic_{task_type}_{idx:04d}",
                task_type=task_type,
                text=text,
                candidates=candidates,
                top_k=top_k,
                gold_indices=gold,
                exclusion_pairs=exclusion_pairs,
                logits=logits,
            )
        )
        idx += 1

    return tasks


def save_synthetic_tasks(
    tasks: list[BBHTask],
    path: Path | str | None = None,
) -> Path:
    path = Path(path) if path else DEFAULT_SYNTHETIC_PATH
    payload = {
        "version": "1.0",
        "description": "Synthetic constrained subset-selection set for NSFC P0/P1/P2 cases.",
        "n_tasks": len(tasks),
        "tasks": [
            {
                "id": t.task_id,
                "task_type": t.task_type,
                "text": t.text,
                "candidates": t.candidates,
                "top_k": t.top_k,
                "gold_indices": t.gold_indices,
                "exclusion_pairs": [list(p) for p in t.exclusion_pairs],
                "logits": t.logits.tolist() if t.logits is not None else None,
            }
            for t in tasks
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_synthetic_tasks(
    path: Path | str | None = None,
    n_tasks: int | None = None,
    seed: int = 42,
    regenerate: bool = False,
) -> list[BBHTask]:
    path = Path(path) if path else DEFAULT_SYNTHETIC_PATH
    if regenerate or not path.is_file():
        tasks = generate_synthetic_tasks(n_tasks=n_tasks or 200, seed=seed)
        save_synthetic_tasks(tasks, path)
        return tasks

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
    if n_tasks is not None:
        tasks = tasks[:n_tasks]
    return tasks


def constrain_hf_task(task: BBHTask, seed: int = 0) -> BBHTask | None:
    """
    Wrap a HF single-choice BBH item as constrained subset selection.

    top_k=2 (or 3 if many candidates), synthetic exclusion pairs, expanded gold set.
    """
    n = len(task.candidates)
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    gold_main = int(task.gold_indices[0])
    top_k = 2 if n <= 6 else 3

    partners = [i for i in range(n) if i != gold_main]
    rng.shuffle(partners)
    gold_indices = [gold_main] + partners[: top_k - 1]
    gold_indices = sorted(gold_indices)

    exclusion_pairs: list[tuple[int, int]] = []
    contradictor = (gold_main + n // 2) % n
    if contradictor not in gold_indices:
        exclusion_pairs.append((min(gold_main, contradictor), max(gold_main, contradictor)))
    alt = int(rng.integers(0, n))
    if alt != gold_main and alt not in gold_indices:
        exclusion_pairs.append((min(gold_main, alt), max(gold_main, alt)))

    logits = task.logits
    if logits is None:
        logits = _logits_from_candidates(task.candidates, gold_indices, seed=seed)

    return BBHTask(
        task_id=f"constrained_{task.task_id}",
        task_type=task.task_type,
        text=f"[Subset k={top_k}] {task.text}",
        candidates=list(task.candidates),
        top_k=top_k,
        gold_indices=gold_indices,
        exclusion_pairs=exclusion_pairs[:2],
        logits=np.asarray(logits, dtype=float),
    )


def load_constrained_hf_tasks(
    hf_tasks: list[str] | None = None,
    limit_per_task: int = 50,
    seed: int = 0,
    use_cache: bool = True,
) -> list[BBHTask]:
    from qihc.orchestrator.bbh_parser import DEFAULT_BBH_HF_TASKS

    raw = load_bbh_tasks(
        source="hf",
        hf_tasks=hf_tasks or DEFAULT_BBH_HF_TASKS[:5],
        limit_per_task=limit_per_task,
        use_cache=use_cache,
    )
    out: list[BBHTask] = []
    for i, t in enumerate(raw):
        ct = constrain_hf_task(t, seed=seed + i)
        if ct is not None:
            out.append(ct)
    return out


def load_constrained_problems(
    source: SourceName = "synthetic",
    seed: int = 0,
    limit: int | None = None,
    hf_tasks: list[str] | None = None,
    limit_per_task: int = 50,
    use_cache: bool = True,
    n_tasks: int | None = None,
    regenerate: bool = False,
):
    """Return SubsetProblem list for synthetic or constrained_hf sources."""
    if source == "synthetic":
        tasks = load_synthetic_tasks(n_tasks=n_tasks or 200, regenerate=regenerate)
    elif source == "constrained_hf":
        tasks = load_constrained_hf_tasks(
            hf_tasks=hf_tasks,
            limit_per_task=limit_per_task,
            seed=seed,
            use_cache=use_cache,
        )
    else:
        raise ValueError(source)
    if limit is not None:
        tasks = tasks[:limit]
    return [t.to_subset_problem(seed=seed + i) for i, t in enumerate(tasks)]
