"""Executable p-dit sampler with multiplier-feedback constraints for LNS assignments."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qihc.problems.cvrp.instance import CVRPInstance, RouteSolution
from qihc.problems.cvrp.neighborhood import NeighborhoodProposal
from qihc.problems.cvrp.verifier import best_insertion


@dataclass
class HybridSampleBatch:
    assignments: list[dict[int, int]]
    energies: np.ndarray
    elapsed_s: float
    metadata: dict = field(default_factory=dict)


class PDitMFCSampler:
    """Categorical Gibbs updates; capacity multipliers adapt from population violation."""

    def __init__(self, num_chains=256, steps=200, temperature_start=3.0, temperature_end=0.05, multiplier_lr=0.1, quadratic_penalty=2.0, semantic_penalty=20.0, top_k=16, seed=0, proposal_bias=1.0):
        self.num_chains, self.steps, self.temperature_start, self.temperature_end = int(num_chains), int(steps), float(temperature_start), float(temperature_end)
        self.multiplier_lr, self.quadratic_penalty, self.semantic_penalty, self.top_k, self.seed = float(multiplier_lr), float(quadratic_penalty), float(semantic_penalty), int(top_k), int(seed)
        self.proposal_bias = float(proposal_bias)

    def _context(self, instance: CVRPInstance, solution: RouteSolution, proposal: NeighborhoodProposal):
        destroyed = list(proposal.destroy_customers); removed = set(destroyed)
        fixed = [[c for c in r if c not in removed] for r in solution.routes]
        fixed += [[] for _ in range(instance.vehicle_count - len(fixed))]
        base_load = np.asarray([sum(instance.customers[c].demand for c in r) for r in fixed], dtype=float)
        candidates = [list(proposal.candidate_routes[c]) for c in destroyed]
        insertion = [[best_insertion(instance, fixed[k], c)[1] - self.proposal_bias * proposal.candidate_route_logits.get(c, {}).get(k, 0.0) for k in ks] for c, ks in zip(destroyed, candidates)]
        return destroyed, candidates, np.asarray([instance.customers[c].demand for c in destroyed], dtype=float), base_load, insertion

    def solve(self, instance: CVRPInstance, solution: RouteSolution, proposal: NeighborhoodProposal) -> HybridSampleBatch:
        destroyed, candidates, demands, base_load, insertion = self._context(instance, solution, proposal)
        rng = np.random.default_rng(self.seed); n = len(destroyed)
        states = np.asarray([[rng.integers(len(candidates[i])) for i in range(n)] for _ in range(self.num_chains)], dtype=np.int32)
        current_route = {c: r for r, route in enumerate(solution.routes) for c in route}
        for i, c in enumerate(destroyed):
            if c in current_route and current_route[c] in candidates[i]: states[0, i] = candidates[i].index(current_route[c])
        multipliers = np.zeros(instance.vehicle_count, dtype=float)

        def energy(s):
            routes = np.asarray([[candidates[i][s[b, i]] for i in range(n)] for b in range(len(s))], dtype=np.int32)
            e = np.asarray([sum(insertion[i][s[b, i]] for i in range(n)) for b in range(len(s))], dtype=float)
            loads = np.repeat(base_load[None, :], len(s), axis=0)
            for i in range(n): loads[np.arange(len(s)), routes[:, i]] += demands[i]
            violation = np.maximum(loads - instance.vehicle_capacity, 0.0)
            e += violation @ multipliers + 0.5 * self.quadratic_penalty * np.square(violation).sum(axis=1)
            index = {c: i for i, c in enumerate(destroyed)}
            for spec in instance.constraints:
                if spec.type not in {"same_resource", "same_vehicle", "mutual_exclusion", "precedence"}: continue
                entities = ([int(spec.params["before"]), int(spec.params["after"])] if spec.type == "precedence"
                            else [int(x) for x in spec.params.get("entities", [])])
                if len(entities) < 2: continue
                a, b = entities[:2]
                ra = routes[:, index[a]] if a in index else current_route.get(a, -1)
                rb = routes[:, index[b]] if b in index else current_route.get(b, -1)
                bad = (ra != rb) if spec.type in {"same_resource", "same_vehicle", "precedence"} else (ra == rb)
                e += self.semantic_penalty * spec.weight * bad
            return e, violation

        t0 = time.perf_counter(); best_states = states.copy(); best_energy, _ = energy(states)
        for step in range(self.steps):
            temperature = self.temperature_start * (self.temperature_end / self.temperature_start) ** (step / max(self.steps - 1, 1))
            for i in rng.permutation(n):
                choices = []
                for state_idx in range(len(candidates[i])):
                    trial = states.copy(); trial[:, i] = state_idx; trial_e, _ = energy(trial); choices.append(trial_e)
                logits = -np.stack(choices, axis=1) / max(temperature, 1e-8); logits -= logits.max(axis=1, keepdims=True)
                probs = np.exp(np.clip(logits, -60, 0)); probs /= probs.sum(axis=1, keepdims=True)
                draws = rng.random(self.num_chains); states[:, i] = (draws[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
            energies, violation = energy(states); multipliers = np.maximum(0.0, multipliers + self.multiplier_lr * violation.mean(axis=0))
            improved = energies < best_energy; best_states[improved] = states[improved]; best_energy[improved] = energies[improved]
        order = np.argsort(best_energy)[: self.top_k]
        assignments = [{c: candidates[i][int(best_states[b, i])] for i, c in enumerate(destroyed)} for b in order]
        return HybridSampleBatch(assignments, best_energy[order], time.perf_counter() - t0, {"backend": "pdit-mfc-numpy", "multipliers": multipliers.tolist(), "steps": self.steps, "num_chains": self.num_chains})


class TorchPDitMFCSampler(PDitMFCSampler):
    """CUDA implementation used one process per H100 under torchrun."""

    def __init__(self, *args, device="cuda:0", **kwargs):
        super().__init__(*args, **kwargs); self.device = device

    def solve(self, instance: CVRPInstance, solution: RouteSolution, proposal: NeighborhoodProposal) -> HybridSampleBatch:
        import torch
        if self.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        destroyed, candidates, demands_np, base_np, insertion = self._context(instance, solution, proposal)
        device = torch.device(self.device); bsz, n = self.num_chains, len(destroyed); max_k = max(map(len, candidates))
        cand = torch.full((n, max_k), -1, device=device, dtype=torch.long); ins = torch.full((n, max_k), 1e6, device=device)
        valid = torch.zeros((n, max_k), device=device, dtype=torch.bool)
        for i, values in enumerate(candidates):
            cand[i, :len(values)] = torch.tensor(values, device=device); ins[i, :len(values)] = torch.tensor(insertion[i], device=device); valid[i, :len(values)] = True
        gen = torch.Generator(device=device); gen.manual_seed(self.seed)
        states = torch.stack([torch.randint(0, len(candidates[i]), (bsz,), device=device, generator=gen) for i in range(n)], dim=1)
        compute_dtype = ins.dtype
        demands = torch.as_tensor(demands_np, device=device, dtype=compute_dtype)
        base = torch.as_tensor(base_np, device=device, dtype=compute_dtype)
        multipliers = torch.zeros(instance.vehicle_count, device=device, dtype=compute_dtype)
        current = {c: r for r, route in enumerate(solution.routes) for c in route}; index = {c: i for i, c in enumerate(destroyed)}; rows = torch.arange(n, device=device)
        def energy(s):
            routes = cand[rows[None, :], s]; e = ins[rows[None, :], s].sum(1); loads = base[None, :].expand(len(s), -1).clone(); loads.scatter_add_(1, routes, demands[None, :].expand(len(s), -1))
            violation = torch.relu(loads - instance.vehicle_capacity); e = e + violation @ multipliers + 0.5 * self.quadratic_penalty * violation.square().sum(1)
            for spec in instance.constraints:
                if spec.type not in {"same_resource", "same_vehicle", "mutual_exclusion", "precedence"}: continue
                entities = ([int(spec.params["before"]), int(spec.params["after"])] if spec.type == "precedence"
                            else [int(x) for x in spec.params.get("entities", [])])
                if len(entities) < 2: continue
                a, b = entities[:2]
                ra = (
                    routes[:, index[a]]
                    if a in index
                    else torch.full(
                        (len(s),), current.get(a, -1), device=device, dtype=routes.dtype
                    )
                )
                rb = (
                    routes[:, index[b]]
                    if b in index
                    else torch.full(
                        (len(s),), current.get(b, -1), device=device, dtype=routes.dtype
                    )
                )
                bad = (ra != rb) if spec.type in {"same_resource", "same_vehicle", "precedence"} else (ra == rb); e = e + self.semantic_penalty * spec.weight * bad.float()
            return e, violation
        if device.type == "cuda": torch.cuda.synchronize(device)
        t0 = time.perf_counter(); best = states.clone(); best_e, _ = energy(states)
        with torch.inference_mode():
            for step in range(self.steps):
                temp = self.temperature_start * (self.temperature_end / self.temperature_start) ** (step / max(self.steps - 1, 1))
                for i in torch.randperm(n, device=device, generator=gen).tolist():
                    options = []
                    for k in range(max_k):
                        if not bool(valid[i, k]): options.append(torch.full((bsz,), 1e6, device=device)); continue
                        trial = states.clone(); trial[:, i] = k; options.append(energy(trial)[0])
                    logits = -torch.stack(options, 1) / max(temp, 1e-8); states[:, i] = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).squeeze(1)
                e, v = energy(states); multipliers = torch.relu(multipliers + self.multiplier_lr * v.mean(0)); improved = e < best_e; best[improved] = states[improved]; best_e = torch.minimum(best_e, e)
        if device.type == "cuda": torch.cuda.synchronize(device)
        order = torch.argsort(best_e)[:self.top_k].cpu().numpy(); best_np = best.cpu().numpy(); e_np = best_e.cpu().numpy()
        assignments = [{c: candidates[i][int(best_np[b, i])] for i, c in enumerate(destroyed)} for b in order]
        return HybridSampleBatch(assignments, e_np[order], time.perf_counter() - t0, {"backend": "pdit-mfc-torch", "device": str(device), "multipliers": multipliers.cpu().tolist(), "steps": self.steps, "num_chains": self.num_chains})
