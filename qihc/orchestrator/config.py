"""QIHC runtime configuration with tiered presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SamplerName = Literal["gibbs", "parallel_tempering", "sqa", "sa_sync", "sa_async"]


@dataclass
class QIHCConfig:
    """
    QIHC orchestrator configuration.

    Tier presets
    ------------
    - ``tier_a`` : mock frontend, CPU-only, fastest smoke test
    - ``tier_b`` : small HuggingFace model (DistilGPT-2 default), optional GPU
    - ``tier_c`` : larger model placeholder for future upgrade
    """

    num_experts: int = 8
    top_k: int = 2
    frontend: Literal["mock", "transformers"] = "transformers"
    model_name: str = "distilgpt2"
    device: str | None = None
    cardinality_penalty: float = 2.0
    batch_joint: bool = True
    expert_capacity: int | None = 2
    capacity_penalty: float = 6.0
    overflow_penalty: float = 4.0
    sampler: SamplerName = "parallel_tempering"
    sampling_steps: int = 800
    T_start: float = 10.0
    T_end: float = 0.01
    boltzmann_k: float = 1.0
    pt_replicas: int = 6
    pt_swap_interval: int = 15
    sqa_slices: int = 6
    sqa_gamma_start: float = 3.0
    sqa_gamma_end: float = 0.01
    seed: int = 0

    def to_summary(self) -> dict:
        return asdict(self)

    def resolve_capacity(self, n_requests: int) -> int:
        """Per-expert max assignments in a batch (auto if unset)."""
        if self.expert_capacity is not None:
            return int(self.expert_capacity)
        return max(1, (n_requests * self.top_k) // self.num_experts)

    @classmethod
    def tier_a(cls, **overrides) -> "QIHCConfig":
        base = cls(
            frontend="mock",
            model_name="mock",
            sampling_steps=400,
            batch_joint=True,
            expert_capacity=2,
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    @classmethod
    def tier_b(cls, **overrides) -> "QIHCConfig":
        base = cls(
            frontend="transformers",
            model_name="distilgpt2",
            sampling_steps=1200,
            sampler="parallel_tempering",
            batch_joint=True,
            expert_capacity=2,
            capacity_penalty=8.0,
            overflow_penalty=5.0,
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    @classmethod
    def tier_c(cls, **overrides) -> "QIHCConfig":
        base = cls(
            frontend="transformers",
            model_name="gpt2",
            sampling_steps=1200,
            sampler="sqa",
            num_experts=16,
            top_k=4,
        )
        for k, v in overrides.items():
            setattr(base, k, v)
        return base
