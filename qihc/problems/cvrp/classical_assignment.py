"""Classical controls using exactly the local assignment QUBO used by p-bits."""

from __future__ import annotations

import time

import numpy as np

from qihc.ising.batched import PBitSampleBatch, ising_energy


def solve_classical_assignment(problem, backend: str, *, seed: int, num_chains: int,
                               steps: int, top_k: int, time_limit_s: float) -> PBitSampleBatch:
    if backend == "cp-sat":
        return _solve_cp_sat(problem, seed, top_k, time_limit_s)
    if backend not in {"metropolis", "uniform"}:
        raise ValueError(f"Unknown classical assignment backend: {backend}")
    weight, field, _ = problem.model.to_ising()
    rng = np.random.default_rng(seed)
    n = len(field)
    states = rng.choice([-1.0, 1.0], size=(num_chains, n))
    states[0] = 2 * problem.initial_bits.astype(float) - 1
    energies = ising_energy(states, weight, field)
    best, best_e = states.copy(), energies.copy()
    start = time.perf_counter()
    updates = 0
    if backend == "metropolis":
        for step in range(steps):
            temperature = 5.0 * (0.05 / 5.0) ** (step / max(steps - 1, 1))
            for column in rng.permutation(n):
                if time.perf_counter() - start >= time_limit_s:
                    break
                local = states @ weight[column] + field[column]
                delta = 2 * states[:, column] * local
                accept = rng.random(num_chains) < np.exp(np.clip(-delta / temperature, -60, 0))
                states[accept, column] *= -1
                energies[accept] += delta[accept]
                improved = energies < best_e
                best[improved], best_e[improved] = states[improved], energies[improved]
                updates += num_chains
            if time.perf_counter() - start >= time_limit_s:
                break
    order = np.argsort(best_e)[:top_k]
    return PBitSampleBatch(
        (best[order] > 0).astype(np.int8), best_e[order], time.perf_counter() - start,
        {"backend": backend, "state_updates": updates, "time_limit_s": time_limit_s,
         "same_local_qubo": True}, (states > 0).astype(np.int8),
    )


def _solve_cp_sat(problem, seed: int, top_k: int, time_limit_s: float) -> PBitSampleBatch:
    from ortools.sat.python import cp_model

    start = time.perf_counter()
    model = cp_model.CpModel()
    bits = [model.new_bool_var(f"b{i}") for i in range(problem.num_variables)]
    terms = []
    scale = 1000
    for idx, coefficient in problem.model.linear.items():
        terms.append(int(round(scale * coefficient)) * bits[idx])
    for (left, right), coefficient in problem.model.quadratic.items():
        product = model.new_bool_var(f"p{left}_{right}")
        model.add_multiplication_equality(product, [bits[left], bits[right]])
        terms.append(int(round(scale * coefficient)) * product)
    model.minimize(sum(terms))
    for variable, value in zip(bits, problem.initial_bits):
        model.add_hint(variable, int(value))
    pool = []

    class Collector(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            pool.append(np.asarray([self.value(variable) for variable in bits], dtype=np.int8))
            if len(pool) > max(top_k, 128):
                pool.pop(0)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = int(seed % (2**31 - 1))
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(1e-3, time_limit_s - (time.perf_counter() - start))
    status = solver.solve(model, Collector())
    if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        pool.append(np.asarray([solver.value(variable) for variable in bits], dtype=np.int8))
    population = np.asarray(pool, dtype=np.int8).reshape((-1, problem.num_variables))
    values = np.asarray([problem.model.energy(row) for row in population])
    selected = np.argsort(values)[:top_k]
    return PBitSampleBatch(
        population[selected], values[selected], time.perf_counter() - start,
        {"backend": "cp-sat", "same_local_qubo": True, "integer_scale": scale,
         "status": solver.status_name(status), "time_limit_s": time_limit_s,
         "population_kind": "improving_solution_pool_not_posterior"}, population,
    )
