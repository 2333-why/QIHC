"""Conditional <=128-bit software p-bit blocks with global original-model feedback."""
from __future__ import annotations

import time
import numpy as np
from .bounded import compile_energy, objective, violations


def conditional_qubo(q, state, block):
    """Exact energy restricted to block, retaining all cross-block interactions."""
    q, state = np.asarray(q, dtype=float), np.asarray(state, dtype=float)
    block = np.asarray(block)
    if q.shape != (len(state), len(state)) or not np.all(np.isfinite(q)) or not np.allclose(q, q.T):
        raise ValueError('Expected finite symmetric Q')
    if not np.all((state == 0) | (state == 1)) or block.ndim != 1 or block.dtype.kind not in 'iu' or not 1 <= len(block) <= 128:
        raise ValueError('Invalid state/block')
    if len(set(block.tolist())) != len(block) or np.any(block < 0) or np.any(block >= len(state)):
        raise ValueError('Invalid block indices')
    outside = state.copy()
    outside[block] = 0
    local = q[np.ix_(block, block)].copy()
    local[np.diag_indices(len(block))] += 2*q[block] @ outside
    return local, float(outside @ q @ outside)


def solve(spec, *, seed=0, chains=8, steps=4096, block_size=128,
          block_updates=128, feedback=True, method='pbit', max_bits=2048):
    for value, maximum in ((chains, 256), (steps, 1000000), (block_size, 128), (block_updates, 1000000)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError('Invalid search budget')
    if method not in ('pbit', 'metropolis', 'uniform') or type(feedback) is not bool:
        raise ValueError('Invalid method/feedback')
    started = time.perf_counter()
    compiled = compile_energy(spec, max_bits=max_bits)
    spec = compiled.spec
    init, schedule, draws = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    states = init.integers(0, 2, (chains, compiled.bits)).astype(float)
    weights = np.full(len(compiled.row_sources), .1)
    q, _ = compiled.qubo(weights)
    history, archive, best = [], None, None
    sign = 1 if spec['objective']['sense'] == 'min' else -1
    def inspect():
        nonlocal archive, best
        x = compiled.decode(states)
        residual = violations(spec, x)
        v = residual.sum(axis=1)
        f = objective(spec, x)
        i = min(range(chains), key=lambda k: (v[k] > 0, v[k], sign*f[k]))
        key = (bool(v[i] > 0), float(v[i]), float(sign*f[i]))
        if best is None or key < best:
            best, archive = key, x[i].astype(int).tolist()
        return residual
    inspect()
    blocks, cursor, count = [], 0, 0
    interval = max(1, steps//8)
    for step in range(steps):
        if step == int(.75*steps) and method != 'uniform':
            weights = np.maximum(weights, 2.)
            q, _ = compiled.qubo(weights)
        if step % block_updates == 0:
            if cursor == len(blocks):
                order = schedule.permutation(compiled.bits)
                blocks = [order[i:i+block_size] for i in range(0, compiled.bits, block_size)]
                cursor = 0
            active = blocks[cursor]
            cursor += 1
            count += 1
        # Only active variables change. The full field retains the exact boundary bias.
        i = int(schedule.choice(active))
        delta = 2*states @ q[:, i] + q[i, i]*(1-2*states[:, i])
        u = draws.random(chains)
        temperature = .02 ** (step/max(1, steps-1))
        if method == 'pbit':
            new = (u < 1/(1+np.exp(np.clip(delta/temperature, -60, 60)))).astype(float)
        elif method == 'metropolis':
            change = (1-2*states[:, i])*delta
            new = np.where(u < np.exp(np.minimum(0, -change/temperature)), 1-states[:, i], states[:, i])
        else:
            new = (u < .5).astype(float)
        states[:, i] = new
        original = inspect()
        if (step+1) % interval == 0 or step+1 == steps:
            encoded = np.abs(states @ compiled.energy.residual_a.T - compiled.energy.rhs) > 1e-9
            original_rates = (original[:, 1:] > 0).mean(axis=0)
            rates = encoded.mean(axis=0)
            if len(rates):
                rates = np.maximum(rates, original_rates[compiled.row_sources])
            history.append({'step': step+1, 'original_violation_rates': original_rates.tolist(),
                            'compiled_violation_rates': rates.tolist(), 'weights': weights.tolist()})
            if feedback and method != 'uniform' and step+1 < steps:
                weights = np.minimum(8, weights*(1+.75*rates))
                q, _ = compiled.qubo(weights)
    return {'schema': spec['schema'], 'x': archive, 'objective': float(objective(spec, archive)),
            'feasible': not best[0], 'violation': best[1], 'bits': compiled.bits,
            'decision_bits': compiled.decision_bits, 'gate_bits': compiled.gate_bits,
            'block_size': min(block_size, compiled.bits), 'block_calls': count,
            'steps': steps, 'chains': chains, 'bit_updates': steps*chains, 'seed': seed,
            'method': method, 'feedback_enabled': feedback, 'history': history,
            'final_weights': weights.tolist(), 'seconds': time.perf_counter()-started,
            'backend': 'software_float64_block_reference', 'optimality_proven': False}


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--chains', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--block-updates', type=int, default=128)
    parser.add_argument('--method', choices=['pbit', 'metropolis', 'uniform'], default='pbit')
    parser.add_argument('--no-feedback', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.model.read_text(encoding='utf-8'))
    result = solve(spec, steps=args.steps, chains=args.chains, seed=args.seed,
                   block_size=args.block_size, block_updates=args.block_updates,
                   method=args.method, feedback=not args.no_feedback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'history'}))


if __name__ == '__main__':
    main()
