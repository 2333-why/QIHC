"""Convert verifier/solver traces into SFT, DPO and GRPO datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class FeedbackRecord:
    prompt: str
    response: str
    schema_valid: bool
    semantic_valid: bool
    feasible: bool
    gap: float | None = None
    solve_time_s: float = 0.0
    compile_cost: float = 0.0
    counterexamples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def reward(self) -> float:
        if not self.schema_valid: return -2.0
        if not self.semantic_valid: return -1.0 - 0.1 * len(self.counterexamples)
        reward = 1.0 + (1.0 if self.feasible else -1.0)
        if self.gap is not None: reward -= max(0.0, self.gap)
        reward -= 0.01 * self.solve_time_s + 0.001 * self.compile_cost
        return float(reward)


def build_sft_records(records: Iterable[FeedbackRecord]) -> list[dict[str, Any]]:
    return [{"prompt": r.prompt, "completion": r.response, "metadata": {"reward": r.reward}} for r in records if r.schema_valid and r.semantic_valid]


def build_dpo_pairs(records: Iterable[FeedbackRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[FeedbackRecord]] = {}
    for record in records: grouped.setdefault(record.prompt, []).append(record)
    pairs = []
    for prompt, group in grouped.items():
        ranked = sorted(group, key=lambda r: r.reward, reverse=True)
        if len(ranked) >= 2 and ranked[0].reward > ranked[-1].reward:
            pairs.append({"prompt": prompt, "chosen": ranked[0].response, "rejected": ranked[-1].response, "chosen_reward": ranked[0].reward, "rejected_reward": ranked[-1].reward})
    return pairs


def build_grpo_records(records: Iterable[FeedbackRecord]) -> list[dict[str, Any]]:
    return [{"prompt": r.prompt, "completion": r.response, "reward": r.reward, "signals": asdict(r)} for r in records]
