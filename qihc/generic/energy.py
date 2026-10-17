"""Restricted IR, exact squared-slack compiler, and original-constraint verifier.

No generated code is executed. All original variables are binary; constraint
coefficients are integers, which supplies a nonzero penalty gap of at least one.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def integer(value, limit=10000):
    if type(value) is not int or abs(value) > limit:
        raise ValueError("Expected bounded integer, not bool/float/string")
    return value


def validate(spec: dict) -> dict:
    if not isinstance(spec, dict) or set(spec) != {"n", "objective", "constraints"}:
        raise ValueError("IR requires exactly n, objective, constraints")
    n = integer(spec["n"], 96)
    if n < 1:
        raise ValueError("n must be positive")
    obj = spec["objective"]
    if not isinstance(obj, dict) or set(obj) != {"sense", "linear", "quadratic"}:
        raise ValueError("Invalid objective fields")
    if obj["sense"] not in ("min", "max") or not isinstance(obj["linear"], list) or len(obj["linear"]) != n:
        raise ValueError("Invalid sense or linear dimension")
    for c in obj["linear"]:
        integer(c)
    if not isinstance(obj["quadratic"], list) or len(obj["quadratic"]) > n * (n - 1) // 2:
        raise ValueError("Too many quadratic terms")
    seen = set()
    for term in obj["quadratic"]:
        if not isinstance(term, list) or len(term) != 3:
            raise ValueError("Quadratic terms are [i,j,coefficient]")
        i, j, c = map(integer, term)
        if not 0 <= i < j < n or (i, j) in seen:
            raise ValueError("Quadratic pairs must be unique and i<j")
        integer(c)
        seen.add((i, j))
    if not isinstance(spec["constraints"], list) or len(spec["constraints"]) > 128:
        raise ValueError("At most 128 constraints")
    ids = set()
    for con in spec["constraints"]:
        if not isinstance(con, dict) or set(con) != {"id", "a", "op", "rhs"}:
            raise ValueError("Invalid constraint fields")
        if not isinstance(con["id"], str) or not 0 < len(con["id"]) <= 64 or con["id"] in ids:
            raise ValueError("Constraint IDs must be unique bounded strings")
        ids.add(con["id"])
        if con["op"] not in ("le", "ge", "eq") or not isinstance(con["a"], list) or len(con["a"]) != n:
            raise ValueError("Invalid relation or dimension")
        for c in con["a"]:
            integer(c)
        integer(con["rhs"])
    return spec


def objective(spec, x):
    x = np.asarray(x)
    out = x @ np.asarray(spec["objective"]["linear"], dtype=float)
    for i, j, c in spec["objective"]["quadratic"]:
        out = out + c * x[..., i] * x[..., j]
    return out


def violations(spec, x):
    """Evaluate ORIGINAL constraints, never auxiliary slack or compiled energy."""
    x = np.asarray(x)
    values = []
    for con in spec["constraints"]:
        r = x @ np.asarray(con["a"], dtype=float) - con["rhs"]
        values.append(abs(r) if con["op"] == "eq" else np.maximum(0, r if con["op"] == "le" else -r))
    return np.stack(values, axis=-1) if values else np.zeros((*x.shape[:-1], 0))


def bounded_weights(upper):
    result, left, power = [], int(upper), 1
    while left > 0:
        w = min(power, left)
        result.append(w)
        left -= w
        power *= 2
    return result


@dataclass
class Energy:
    spec: dict
    objective_q: np.ndarray  # x'Qx convention: off-diagonal coefficient split in half
    residual_a: np.ndarray
    rhs: np.ndarray
    bound: float

    @property
    def bits(self):
        return self.objective_q.shape[0]

    def qubo(self, weights):
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.rhs.shape or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("One finite positive penalty weight per constraint")
        q = self.objective_q + self.residual_a.T @ (weights[:, None] * self.residual_a)
        q = q.copy()
        q[np.diag_indices(self.bits)] -= 2 * self.residual_a.T @ (weights * self.rhs)
        return q, float(weights @ (self.rhs ** 2))

    def ising(self, weights):
        """E(s)=-0.5*s'Ws-h's+offset, x=(s+1)/2."""
        q, constant = self.qubo(weights)
        w = -q / 2
        np.fill_diagonal(w, 0)
        h = -q.sum(axis=1) / 2
        offset = constant + (q.sum() + np.trace(q)) / 4
        return w, h, float(offset)


def compile_energy(spec):
    validate(spec)
    n = spec["n"]
    slacks, total = [], n
    for con in spec["constraints"]:
        lo = sum(min(0, c) for c in con["a"])
        hi = sum(max(0, c) for c in con["a"])
        upper = max(0, con["rhs"] - lo) if con["op"] == "le" else max(0, hi - con["rhs"])
        weights = [] if con["op"] == "eq" else bounded_weights(upper)
        slacks.append((total, weights))
        total += len(weights)
    if total > 512:
        raise ValueError("Compiled problem exceeds software limit of 512 bits")
    obj = spec["objective"]
    bound = max(1, sum(abs(c) for c in obj["linear"]) + sum(abs(c) for _, _, c in obj["quadratic"]))
    sign = 1 if obj["sense"] == "min" else -1
    q = np.zeros((total, total), dtype=float)
    q[np.arange(n), np.arange(n)] = np.asarray(obj["linear"]) * sign / bound
    for i, j, c in obj["quadratic"]:
        q[i, j] = q[j, i] = c * sign / (2 * bound)
    a = np.zeros((len(slacks), total))
    for row, (con, (start, weights)) in enumerate(zip(spec["constraints"], slacks)):
        a[row, :n] = con["a"]
        a[row, start:start + len(weights)] = np.asarray(weights) * (1 if con["op"] == "le" else -1)
    return Energy(spec, q, a, np.asarray([c["rhs"] for c in spec["constraints"]], dtype=float), float(bound))


POLICIES = {
    "fixed": {"weight": 2.0, "adaptive": False, "temperature": 1.0},
    "adaptive": {"weight": 0.1, "adaptive": True, "temperature": 1.0},
    "warm": {"weight": 0.5, "adaptive": True, "temperature": 4.0},
    "cold": {"weight": 2.0, "adaptive": False, "temperature": 0.25},
}


def solve(spec, *, policy="adaptive", seed=0, chains=64, steps=4096, method="pbit", feedback=True):
    """Exact single-site conditional updates, parallel ONLY across independent chains.

    One step means one proposed bit per chain, not a full sweep. Random initial
    states may be infeasible. No construction, repair, or family-specific move.
    """
    import time
    started = time.perf_counter()
    if policy not in POLICIES or method not in ("pbit", "metropolis", "uniform"):
        raise ValueError("Unknown policy or sampler")
    if not 1 <= chains <= 4096 or not 1 <= steps <= 1000000:
        raise ValueError("Invalid sampling budget")
    energy = compile_energy(spec)
    cfg = POLICIES[policy]
    init_rng, site_rng, draw_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    states = init_rng.integers(0, 2, (chains, energy.bits)).astype(float)
    weights = np.full(len(spec["constraints"]), cfg["weight"])
    archive, best_key, history = None, None, []
    sign = 1 if spec["objective"]["sense"] == "min" else -1

    def inspect_states():
        nonlocal archive, best_key
        x = states[:, :spec["n"]]
        residuals = violations(spec, x)
        v = residuals.sum(axis=1)
        f = objective(spec, x)
        idx = min(range(chains), key=lambda i: (v[i] > 0, v[i], sign * f[i]))
        key = (bool(v[idx] > 0), float(v[idx]), float(sign * f[idx]))
        if best_key is None or key < best_key:
            best_key, archive = key, x[idx].astype(int).tolist()
        return residuals

    inspect_states()
    q, _ = energy.qubo(weights)
    fields = 2 * states @ q - np.diag(q)
    # Rebuild after bounded blocks. Archive is ranked by ORIGINAL constraints,
    # so changing penalty scales never invalidates previous comparisons.
    block = max(1, steps // 8)
    for step in range(steps):
        if cfg["adaptive"] and method != "uniform" and step == int(.75 * steps):
            # Apply BEFORE sampling, including very short budgets. Identical in
            # feedback and no-feedback arms: this is scheduled hardening, not feedback.
            weights = np.maximum(2, weights)
            q, _ = energy.qubo(weights)
            fields = 2 * states @ q - np.diag(q)
        i = int(site_rng.integers(energy.bits))
        draws = draw_rng.random(chains)
        if method == "uniform":
            # Random-scan fair-bit refresh, NOT independent full-vector samples.
            new = (draws < .5).astype(float)
        else:
            # field excluding the current bit: E(x_i=1)-E(x_i=0)
            delta = fields[:, i] + 2 * q[i, i] * (1 - states[:, i])
            temp = cfg["temperature"] * (0.02 ** (step / max(1, steps - 1)))
            if method == "pbit":
                prob = 1 / (1 + np.exp(np.clip(delta / temp, -60, 60)))
                new = (draws < prob).astype(float)
            else:
                change = (1 - 2 * states[:, i]) * delta
                accept = draws < np.exp(np.minimum(0, -change / temp))
                new = np.where(accept, 1 - states[:, i], states[:, i])
        diff = new - states[:, i]
        states[:, i] = new
        fields += 2 * diff[:, None] * q[i][None, :]
        # Every updated state is eligible, not just block endpoints.
        v = inspect_states()
        if (step + 1) % block == 0 or step + 1 == steps:
            rates = (v > 0).mean(axis=0)
            history.append({"step": step + 1, "violation_rates": rates.tolist(), "weights": weights.tolist()})
            if feedback and cfg["adaptive"] and method != "uniform" and step + 1 < steps:
                weights = np.minimum(8, weights * (1 + 0.75 * rates))
                q, _ = energy.qubo(weights)
                fields = 2 * states @ q - np.diag(q)
    return {"x": archive, "feasible": not best_key[0], "violation": best_key[1],
            "objective": float(objective(spec, np.asarray(archive))), "bits": energy.bits,
            "steps": steps, "chains": chains, "bit_updates": steps * chains,
            "seconds": time.perf_counter() - started, "history": history,
            "method": method, "policy": policy, "seed": seed, "feedback_enabled": bool(feedback),
            "final_weights": weights.tolist(), "rng_protocol": "separate_init_site_draw_streams_v2"}
