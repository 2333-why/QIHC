"""Shared data types for QIHC orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RoutingContext:
    """Output of the AI frontend for one routing request."""

    text: str
    logits: np.ndarray  # shape (num_experts,)
    hidden: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Routing outcome for one request."""

    text: str
    logits: np.ndarray
    expert_mask: np.ndarray  # bool, shape (num_experts,)
    expert_indices: list[int]
    energy: float | None = None
    elapsed_s: float = 0.0
    method: str = ""

    @property
    def load_vector(self) -> np.ndarray:
        return self.expert_mask.astype(float)


@dataclass
class RoutingBatchResult:
    """Aggregated comparison between two routing strategies."""

    greedy: list[RoutingDecision]
    pbit: list[RoutingDecision]
    metrics: dict[str, float]
    config_summary: dict[str, Any] = field(default_factory=dict)
