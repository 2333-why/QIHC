"""Batched p-bit samplers for CPU smoke tests and H100 formal experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PBitSampleBatch:
    bits: np.ndarray
    energies: np.ndarray
    elapsed_s: float
    metadata: dict = field(default_factory=dict)


def ising_energy(states: np.ndarray, weight: np.ndarray, field: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    return -0.5 * np.einsum("bi,ij,bj->b", states, weight, states) - states @ field


class NumpyPBitSampler:
    """Parallel p-bit update simulator used when torch/CUDA is unavailable."""

    def __init__(
        self,
        num_chains: int = 64,
        steps: int = 300,
        temperature_start: float = 5.0,
        temperature_end: float = 0.05,
        update_fraction: float = 0.25,
        top_k: int = 16,
        seed: int = 0,
    ):
        self.num_chains = int(num_chains)
        self.steps = int(steps)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.update_fraction = float(update_fraction)
        self.top_k = int(top_k)
        self.seed = int(seed)

    def solve(
        self,
        weight: np.ndarray,
        field: np.ndarray,
        initial_bits: np.ndarray | None = None,
    ) -> PBitSampleBatch:
        rng = np.random.default_rng(self.seed)
        n = int(field.size)
        states = rng.choice([-1.0, 1.0], size=(self.num_chains, n))
        if initial_bits is not None and len(initial_bits) == n:
            states[0] = 2.0 * np.asarray(initial_bits, dtype=float) - 1.0
        best_states = states.copy()
        best_energies = ising_energy(states, weight, field)
        t0 = time.perf_counter()
        for step in range(self.steps):
            progress = step / max(self.steps - 1, 1)
            temperature = self.temperature_start * (
                self.temperature_end / self.temperature_start
            ) ** progress
            beta = 1.0 / max(temperature, 1e-8)
            local = states @ weight.T + field
            probabilities = 1.0 / (1.0 + np.exp(np.clip(-2.0 * beta * local, -60.0, 60.0)))
            proposed = np.where(rng.random(states.shape) < probabilities, 1.0, -1.0)
            mask = rng.random(states.shape) < self.update_fraction
            states = np.where(mask, proposed, states)
            energies = ising_energy(states, weight, field)
            improved = energies < best_energies
            best_states[improved] = states[improved]
            best_energies[improved] = energies[improved]
        elapsed = time.perf_counter() - t0
        order = np.argsort(best_energies)
        unique: list[int] = []
        seen: set[bytes] = set()
        for idx in order:
            key = np.packbits(best_states[idx] > 0).tobytes()
            if key not in seen:
                unique.append(int(idx))
                seen.add(key)
            if len(unique) >= self.top_k:
                break
        selected = np.asarray(unique, dtype=int)
        return PBitSampleBatch(
            bits=(best_states[selected] > 0).astype(np.int8),
            energies=best_energies[selected],
            elapsed_s=elapsed,
            metadata={
                "backend": "numpy",
                "num_chains": self.num_chains,
                "steps": self.steps,
                "update_fraction": self.update_fraction,
            },
        )

class TorchPBitSampler:
    """Dense batched p-bit updates on one CUDA device; one process per GPU."""

    def __init__(
        self,
        num_chains: int = 2048,
        steps: int = 1000,
        temperature_start: float = 5.0,
        temperature_end: float = 0.05,
        update_fraction: float = 0.25,
        top_k: int = 32,
        seed: int = 0,
        device: str = "cuda:0",
        dtype: str = "float32",
    ):
        self.num_chains = int(num_chains)
        self.steps = int(steps)
        self.temperature_start = float(temperature_start)
        self.temperature_end = float(temperature_end)
        self.update_fraction = float(update_fraction)
        self.top_k = int(top_k)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

    def solve(
        self,
        weight: np.ndarray,
        field: np.ndarray,
        initial_bits: np.ndarray | None = None,
    ) -> PBitSampleBatch:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("TorchPBitSampler requires qihc[llm] or a CUDA torch wheel") from exc

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested on {self.device}, but torch.cuda.is_available() is false")
        torch_dtype = {"float32": torch.float32, "float64": torch.float64}[self.dtype]
        device = torch.device(self.device)
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        w = torch.as_tensor(weight, device=device, dtype=torch_dtype)
        h = torch.as_tensor(field, device=device, dtype=torch_dtype)
        n = int(h.numel())
        random = torch.rand((self.num_chains, n), device=device, generator=generator)
        states = torch.where(random < 0.5, -torch.ones_like(random), torch.ones_like(random)).to(torch_dtype)
        if initial_bits is not None and len(initial_bits) == n:
            warm = torch.as_tensor(initial_bits, device=device, dtype=torch_dtype)
            states[0] = 2.0 * warm - 1.0

        def energies(value):
            return -0.5 * torch.einsum("bi,ij,bj->b", value, w, value) - value @ h

        best_states = states.clone()
        best_energies = energies(states)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for step in range(self.steps):
                progress = step / max(self.steps - 1, 1)
                temperature = self.temperature_start * (
                    self.temperature_end / self.temperature_start
                ) ** progress
                beta = 1.0 / max(temperature, 1e-8)
                local = states @ w.T + h
                probabilities = torch.sigmoid(2.0 * beta * local)
                proposed = torch.where(
                    torch.rand(states.shape, device=device, generator=generator) < probabilities,
                    torch.ones_like(states),
                    -torch.ones_like(states),
                )
                mask = (
                    torch.rand(states.shape, device=device, generator=generator)
                    < self.update_fraction
                )
                states = torch.where(mask, proposed, states)
                current_energies = energies(states)
                improved = current_energies < best_energies
                best_states[improved] = states[improved]
                best_energies = torch.minimum(best_energies, current_energies)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        order = torch.argsort(best_energies).detach().cpu().numpy()
        states_cpu = best_states.detach().cpu().numpy()
        energies_cpu = best_energies.detach().cpu().numpy()
        unique: list[int] = []
        seen: set[bytes] = set()
        for idx in order:
            key = np.packbits(states_cpu[idx] > 0).tobytes()
            if key not in seen:
                unique.append(int(idx))
                seen.add(key)
            if len(unique) >= self.top_k:
                break
        selected = np.asarray(unique, dtype=int)
        return PBitSampleBatch(
            bits=(states_cpu[selected] > 0).astype(np.int8),
            energies=energies_cpu[selected],
            elapsed_s=elapsed,
            metadata={
                "backend": "torch",
                "device": str(device),
                "dtype": self.dtype,
                "num_chains": self.num_chains,
                "steps": self.steps,
                "update_fraction": self.update_fraction,
            },
        )
