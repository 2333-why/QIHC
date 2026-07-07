"""
Combinatorial Reasoning (CR) protocol utilities — aligned with arXiv:2407.00071.

Maps sampled LLM reasons / answers to QUBO, solves with p-bit backend,
and supports linear vs quadratic CR baselines plus VCI refine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from qihc.orchestrator.encoder import greedy_top_k, refine_mask_to_top_k
from qihc.orchestrator.reasoning import SubsetProblem, is_feasible, subset_to_ising

CRMode = Literal["zeroshot", "random", "linear", "quadratic", "vci-1", "vci-2"]


@dataclass
class CRParams:
    """Hyper-parameters for CR QUBO mapping (tunable subset of CR Table 2)."""

    mu: float = 1.0
    alpha: float = 0.5
    beta: float = 1.0
    W: float = 2.0
    kappa: int = 2
    similarity_zeta: float = 0.90


@dataclass
class CRReasonSample:
    text: str
    answer_index: int | None = None
    candidate_hits: list[int] = field(default_factory=list)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def text_jaccard(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def deduplicate_reasons(
    samples: list[CRReasonSample],
    zeta: float = 0.90,
) -> tuple[list[CRReasonSample], list[int]]:
    """
    Merge semantically similar reason texts (CR semantic matching proxy).

    Returns distinct samples and mapping from original index -> distinct index.
    """
    distinct: list[CRReasonSample] = []
    mapping: list[int] = []
    for s in samples:
        placed = False
        for j, d in enumerate(distinct):
            if text_jaccard(s.text, d.text) >= zeta:
                mapping.append(j)
                placed = True
                break
        if not placed:
            mapping.append(len(distinct))
            distinct.append(CRReasonSample(text=s.text, answer_index=s.answer_index))
    return distinct, mapping


def extract_answer_index(completion: str, candidates: list[str]) -> int | None:
    """Heuristic: match candidate letter (A)/(B) or substring in completion."""
    text = completion.strip()
    letter_match = re.search(r"\b([A-E])\b", text.upper())
    if letter_match:
        letter = letter_match.group(1)
        idx = ord(letter) - ord("A")
        if 0 <= idx < len(candidates):
            return idx
    norm_completion = normalize_text(text)
    best_idx, best_len = None, 0
    for i, cand in enumerate(candidates):
        nc = normalize_text(cand)
        if nc and nc in norm_completion and len(nc) > best_len:
            best_idx, best_len = i, len(nc)
    return best_idx


def samples_from_completions(
    completions: list[str],
    candidates: list[str],
) -> list[CRReasonSample]:
    out: list[CRReasonSample] = []
    for c in completions:
        idx = extract_answer_index(c, candidates)
        out.append(CRReasonSample(text=c, answer_index=idx))
    return out


def build_frequency_logits(samples: list[CRReasonSample], n_candidates: int) -> np.ndarray:
    counts = np.zeros(n_candidates, dtype=float)
    for s in samples:
        if s.answer_index is not None and 0 <= s.answer_index < n_candidates:
            counts[s.answer_index] += 1.0
    return np.log1p(counts)


def build_cooccurrence_matrix(samples: list[CRReasonSample], n: int) -> np.ndarray:
    """Co-occurrence of answer indices within same completion (diagonal = frequency)."""
    co = np.zeros((n, n), dtype=float)
    for s in samples:
        if s.answer_index is None:
            continue
        i = int(s.answer_index)
        co[i, i] += 1.0
    return co


def build_reason_cooccurrence(distinct: list[CRReasonSample], mapping: list[int]) -> np.ndarray:
    """Pairwise co-occurrence among distinct reasons (same-sample proxy via mapping)."""
    n = len(distinct)
    co = np.zeros((n, n), dtype=float)
    bucket: dict[int, list[int]] = {}
    for orig_i, d_idx in enumerate(mapping):
        bucket.setdefault(d_idx, []).append(orig_i)
    for indices in bucket.values():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = mapping[indices[a]], mapping[indices[b]]
                co[i, j] += 1.0
                co[j, i] += 1.0
    for i in range(n):
        co[i, i] = max(co[i, i], 1.0)
    return co


def build_cr_qubo(
    n: int,
    counts: np.ndarray,
    cooccur: np.ndarray,
    params: CRParams,
    mode: Literal["linear", "quadratic"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Ising (Weight, Field) from CR-style frequency + correlation terms.

    Linear term favors high-frequency reasons; quadratic (CR) penalizes
    negatively correlated pairs per co-occurrence structure.
    """
    counts = np.asarray(counts, dtype=float).ravel()
    n_vars = int(counts.size)
    if n_vars != n:
        raise ValueError(f"counts size {n_vars} != n {n}")

    total = float(counts.sum()) + 1e-9
    freq = counts / total
    tilde_l = params.mu * (freq - 0.5 / max(n, 1))

    field = -tilde_l / 2.0
    weight = np.zeros((n, n), dtype=float)

    if mode == "quadratic":
        co = np.asarray(cooccur, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                n_i, n_j = max(counts[i], 1e-9), max(counts[j], 1e-9)
                if co[i, j] > 0:
                    corr = co[i, j] / np.sqrt(n_i * n_j)
                else:
                    corr = 0.0
                q_ij = params.alpha * (corr - params.beta)
                if mode == "quadratic":
                    weight[i, j] = -0.5 * q_ij
                    weight[j, i] = -0.5 * q_ij

    return weight, field


def cr_logits_from_samples(
    samples: list[CRReasonSample],
    n_candidates: int,
    params: CRParams,
    mode: Literal["linear", "quadratic"],
) -> np.ndarray:
    """Map CR samples to candidate logits (for top-k / MC selection)."""
    counts = np.zeros(n_candidates, dtype=float)
    for s in samples:
        if s.answer_index is not None and 0 <= s.answer_index < n_candidates:
            counts[s.answer_index] += 1.0
    co = build_cooccurrence_matrix(samples, n_candidates)

    if mode == "linear":
        return build_frequency_logits(samples, n_candidates)

    _, field = build_cr_qubo(n_candidates, counts, co, params, mode="quadratic")
    logits = -2.0 * field
    return logits.astype(float)


def select_mask_cr(
    logits: np.ndarray,
    top_k: int,
    exclusion_pairs: list[tuple[int, int]],
    backend,
    mode: CRMode,
    params: CRParams,
    samples: list[CRReasonSample] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Select candidate mask under CR / VCI modes.

    Returns mask and diagnostics dict (llm_calls, pbit_steps, etc.).
    """
    rng = rng or np.random.default_rng(0)
    n = logits.size
    diag: dict = {"mode": mode, "pbit_steps": 0, "llm_calls": 0}

    if mode == "zeroshot" or mode == "linear":
        # LEGACY: logits-greedy path — NOT used in CR paper benchmark (see cr_pipeline.py).
        if mode == "linear" and samples is not None:
            logits = cr_logits_from_samples(samples, n, params, mode="linear")
        mask = greedy_top_k(logits, top_k)
        return mask, diag

    if mode == "random":
        idx = rng.choice(n, size=top_k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask, diag

    if samples is not None and mode in ("quadratic", "vci-1", "vci-2"):
        logits = cr_logits_from_samples(samples, n, params, mode="quadratic")

    weight, field = subset_to_ising(
        logits,
        top_k=top_k,
        cardinality_penalty=2.0,
        exclusion_pairs=exclusion_pairs,
        exclusion_penalty=4.0,
    )
    mask, energy, elapsed = backend.solve(weight, field)
    diag["pbit_steps"] = backend.config.sampling_steps
    diag["ising_energy"] = float(energy)
    diag["pbit_elapsed_s"] = float(elapsed)
    mask = refine_mask_to_top_k(logits, mask, top_k)
    return mask, diag


@dataclass
class CRBenchmarkResult:
    task_id: str
    mode: str
    correct: bool
    feasible: bool
    exact_match: bool
    pred_indices: list[int]
    gold_indices: list[int]
    n_samples: int
    pbit_steps: int
    llm_calls: int
    free_energy_trace: list[float] = field(default_factory=list)


def evaluate_cr_on_problem(
    problem: SubsetProblem,
    mode: CRMode,
    backend,
    frontend,
    samples: list[CRReasonSample] | None,
    params: CRParams,
    config,
) -> CRBenchmarkResult:
    """Run one CR/VCI mode on a SubsetProblem with optional sampled reasons."""
    from qihc.orchestrator.free_energy import compute_free_energy
    from qihc.orchestrator.vci_scheduler import VCIOrchestrator

    gold = problem.metadata.get("gold_indices", [])
    gold_mask = problem.metadata.get("gold_mask")
    if gold_mask is not None:
        gold_set = list(np.flatnonzero(np.asarray(gold_mask, dtype=bool)))
    else:
        gold_set = [int(i) for i in gold]

    logits = np.asarray(problem.logits, dtype=float).ravel()
    q0 = logits.copy()
    f_trace: list[float] = []
    llm_calls = len(samples) if samples else 1
    pbit_steps = 0

    if mode in ("vci-1", "vci-2"):
        orch = VCIOrchestrator(config)
        if samples:
            cr_logits = cr_logits_from_samples(samples, logits.size, params, "quadratic")
            prob = SubsetProblem(
                text=problem.text,
                logits=cr_logits,
                top_k=problem.top_k,
                exclusion_pairs=problem.exclusion_pairs,
                metadata=dict(problem.metadata),
            )
        else:
            prob = problem
        res = orch.solve_subset(prob, mode=mode)  # type: ignore[arg-type]
        mask = res.final_mask
        f_trace = [r.free_energy.total for r in res.rounds]
        pbit_steps = int(config.sampling_steps * res.n_rounds)
        feasible = res.final_feasible
    else:
        mask, diag = select_mask_cr(
            logits,
            problem.top_k,
            problem.exclusion_pairs,
            backend,
            mode,
            params,
            samples=samples,
        )
        pbit_steps = int(diag.get("pbit_steps", 0))
        fe = compute_free_energy(
            logits,
            q0,
            mask,
            exclusion_pairs=problem.exclusion_pairs,
        )
        f_trace = [fe.total]
        feasible = is_feasible(mask, problem.top_k, problem.exclusion_pairs)

    pred_set = list(np.flatnonzero(mask))
    exact = set(pred_set) == set(gold_set)
    return CRBenchmarkResult(
        task_id=str(problem.metadata.get("task_id", "unknown")),
        mode=mode,
        correct=exact,
        feasible=feasible,
        exact_match=exact,
        pred_indices=pred_set,
        gold_indices=gold_set,
        n_samples=len(samples) if samples else 0,
        pbit_steps=pbit_steps,
        llm_calls=llm_calls,
        free_energy_trace=f_trace,
    )
