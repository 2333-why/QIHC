"""Parse BIG-Bench Hard (BBH) examples into multiple-choice subset tasks."""

from __future__ import annotations

import re
from typing import Any

# Case A default: reasoning / logic multiple-choice subtasks
DEFAULT_BBH_HF_TASKS: list[str] = [
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "disambiguation_qa",
    "causal_judgement",
    "formal_fallacies",
    "web_of_lies",
    "date_understanding",
    "hyperbaton",
    "movie_recommendation",
]

_OPTION_LINE_RE = re.compile(
    r"^\s*[\(\[]?([A-Za-z])[\)\]]?\s*[:\.\)]?\s*(.+?)\s*$"
)
_DASH_OPTION_RE = re.compile(r"^\s*[-•]\s*(.+?)\s*$")


def split_question_options(text: str) -> tuple[str, str | None]:
    """Split BBH prompt into stem and options block."""
    for marker in ("\nOptions:", "\nOPTIONS:", "\noptions:"):
        if marker in text:
            stem, opts = text.split(marker, 1)
            return stem.strip(), opts.strip()
    return text.strip(), None


def parse_option_lines(options_block: str) -> list[str]:
    """Parse (A).. / - Yes / - No style option lines."""
    candidates: list[str] = []
    for line in options_block.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _OPTION_LINE_RE.match(line)
        if m:
            candidates.append(m.group(2).strip())
            continue
        m = _DASH_OPTION_RE.match(line)
        if m:
            candidates.append(m.group(1).strip())
            continue
    return candidates


def target_to_gold_index(
    target: str,
    candidates: list[str],
    labels: list[str] | None = None,
) -> int:
    """Map BBH target string to 0-based candidate index."""
    raw = target.strip()
    compact = raw.strip("()").strip().upper()

    if len(compact) == 1 and compact.isalpha():
        return ord(compact) - ord("A")

    if labels:
        for i, lab in enumerate(labels):
            lab_norm = lab.strip().strip("()").upper()
            if lab_norm == compact or lab_norm.startswith(compact):
                return i

    raw_lower = raw.lower()
    for i, cand in enumerate(candidates):
        if cand.strip().lower() == raw_lower:
            return i

    raise ValueError(f"Cannot map target={target!r} to candidates={candidates!r}")


def joschka_row_to_fields(row: dict[str, Any]) -> tuple[str, list[str], str, list[str] | None]:
    """Normalize Joschka/big_bench_hard row to stem, candidates, target, labels."""
    question = str(row.get("question") or row.get("input") or "")
    target = str(row.get("target", "")).strip()
    choices = row.get("choices")

    if isinstance(choices, dict) and choices.get("text"):
        candidates = [str(t).strip() for t in choices["text"]]
        labels = [str(x) for x in choices.get("label", [])]
        return question.strip(), candidates, target, labels or None

    stem, opts_block = split_question_options(question)
    if opts_block:
        candidates = parse_option_lines(opts_block)
        if candidates:
            return stem, candidates, target, None

    # Yes/No style without explicit Options block (e.g. web_of_lies)
    tnorm = target.strip().lower()
    if tnorm in ("yes", "no"):
        return question.strip(), ["Yes", "No"], target, None

    raise ValueError("No parseable options in BBH row")


def lukaemon_row_to_fields(row: dict[str, Any]) -> tuple[str, list[str], str, list[str] | None]:
    """Normalize lukaemon/bbh row (input/target)."""
    text = str(row.get("input") or row.get("question") or "")
    target = str(row.get("target", "")).strip()
    stem, opts_block = split_question_options(text)
    if not opts_block:
        raise ValueError("No Options block in lukaemon row")
    candidates = parse_option_lines(opts_block)
    if not candidates:
        raise ValueError("Empty options after parse")
    return stem, candidates, target, None
