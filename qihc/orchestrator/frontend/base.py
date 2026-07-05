"""AI frontend abstractions (mock / transformers / future LLM tiers)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qihc.orchestrator.config import QIHCConfig
from qihc.orchestrator.types import RoutingContext


class BaseFrontend(ABC):
    """Upgrade path: implement this to plug in larger LLMs or custom routers."""

    def __init__(self, config: QIHCConfig):
        self.config = config

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[RoutingContext]:
        """Map input texts to per-request expert routing logits."""

    def encode_one(self, text: str) -> RoutingContext:
        return self.encode_batch([text])[0]

    def refine(
        self,
        context: RoutingContext,
        mask: np.ndarray,
        feedback: dict,
        q0_logits: np.ndarray | None = None,
    ) -> RoutingContext:
        """
        VCI q-step: adjust semantic logits from discrete feedback.

        Default implementation down-weights candidates involved in
        constraint violations (minimal refine for Case A / IF).
        """
        logits = np.asarray(context.logits, dtype=float).copy()
        penalty = float(feedback.get("refine_penalty", 1.5))
        for i, j in feedback.get("violation_pairs", []):
            logits[int(i)] -= penalty
            logits[int(j)] -= penalty

        meta = dict(context.metadata)
        meta["vci_refined"] = True
        meta["refine_round"] = int(meta.get("refine_round", 0)) + 1
        return RoutingContext(
            text=context.text,
            logits=logits,
            hidden=context.hidden,
            metadata=meta,
        )


def build_frontend(config: QIHCConfig) -> BaseFrontend:
    """Factory: select frontend implementation by config."""
    if config.frontend == "mock":
        from qihc.orchestrator.frontend.mock import MockFrontend

        return MockFrontend(config)
    if config.frontend == "transformers":
        from qihc.orchestrator.frontend.transformers import TransformersFrontend

        return TransformersFrontend(config)
    raise ValueError(f"Unknown frontend: {config.frontend}")
