"""Mock frontend for CI and Tier-A (no torch/transformers)."""

from __future__ import annotations

import hashlib

import numpy as np

from qihc.orchestrator.frontend.base import BaseFrontend
from qihc.orchestrator.types import RoutingContext


class MockFrontend(BaseFrontend):
    """
    Deterministic pseudo-logits from text hash.

    Useful when GPU / transformers are unavailable.
    """

    def encode_batch(self, texts: list[str]) -> list[RoutingContext]:
        rng = np.random.default_rng(self.config.seed)
        out: list[RoutingContext] = []
        for i, text in enumerate(texts):
            digest = hashlib.sha256(f"{self.config.seed}:{text}".encode()).digest()
            local_rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            logits = local_rng.normal(size=self.config.num_experts)
            # mild token-position prior for diversity across batch
            logits += 0.1 * rng.normal(size=self.config.num_experts)
            out.append(RoutingContext(text=text, logits=logits, hidden=None, metadata={"tier": "mock"}))
        return out
