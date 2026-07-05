"""p-bit probability backend wrapping Ising samplers."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from qihc import IsingModel
from qihc.orchestrator.config import QIHCConfig, SamplerName
from qihc.orchestrator.encoder import (
    batch_spins_to_masks,
    mask_to_expert_indices,
    spins_to_mask,
)


class PBitBackend:
    """Run p-bit / Ising sampling on a pre-encoded routing problem."""

    def __init__(self, config: QIHCConfig):
        self.config = config

    def solve(
        self,
        weight: np.ndarray,
        field: np.ndarray,
        logits_batch: np.ndarray | None = None,
    ) -> tuple[np.ndarray | list[np.ndarray], float, float]:
        """
        Returns (expert_mask or mask list, ising_energy, elapsed_seconds).
        """
        n = weight.shape[0]
        model = IsingModel(size=n, Weight=weight.copy(), Field=field.copy())
        j_dict = self._dense_to_j_dict(weight)

        t0 = time.perf_counter()
        spins, energy_trace, _ = self._run_sampler(model, j_dict)
        elapsed = time.perf_counter() - t0

        energy = float(energy_trace[-1]) if energy_trace else model._energy_of(model.State)

        if logits_batch is not None:
            capacity = self.config.resolve_capacity(logits_batch.shape[0])
            masks = batch_spins_to_masks(
                spins,
                logits_batch,
                self.config.top_k,
                expert_capacity=capacity,
            )
            return masks, energy, elapsed

        mask = spins_to_mask(spins, self.config.top_k)
        return mask, energy, elapsed

    def _run_sampler(self, model: IsingModel, j_dict: dict):
        cfg = self.config
        common = dict(
            J=j_dict,
            steps=cfg.sampling_steps,
            T_start=cfg.T_start,
            T_end=cfg.T_end,
            k=cfg.boltzmann_k,
        )
        name: SamplerName = cfg.sampler

        if name == "gibbs":
            return model.gibbs_sampling_Maxcut(**common, sequential=True)
        if name == "parallel_tempering":
            return model.parallel_tempering_Maxcut(
                **common,
                n_replicas=cfg.pt_replicas,
                swap_interval=cfg.pt_swap_interval,
                sequential=True,
            )
        if name == "sqa":
            return model.simulated_quantum_annealing_Maxcut(
                **common,
                Gamma_start=cfg.sqa_gamma_start,
                Gamma_end=cfg.sqa_gamma_end,
                m_slices=cfg.sqa_slices,
            )
        if name == "sa_sync":
            return model.ising_simulated_annealing_Maxcut_Syn(**common)
        if name == "sa_async":
            return model.ising_simulated_annealing_Maxcut_Asyn(**common)
        raise ValueError(f"Unknown sampler: {name}")

    @staticmethod
    def _dense_to_j_dict(weight: np.ndarray) -> dict:
        j_dict: dict = {}
        n = weight.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if weight[i, j] != 0.0:
                    j_dict[(i, j)] = float(weight[i, j])
                    j_dict[(j, i)] = float(weight[i, j])
        return j_dict
