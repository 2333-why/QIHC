"""Real replica exchanges for the mechanism-only PT feedback control."""
import time
import numpy as np

from .search import Energy, encode_states, summarize
from ..factor_search import encoding


def sample_pt(spec, seconds, seed, chains=32, clamps=None, **unused):
    start = time.monotonic(); deadline = start + seconds; rng = np.random.default_rng(seed)
    energy = Energy(spec); sites, lower = encoding(spec)
    xs = np.column_stack([rng.integers(v['lower'], v['upper'] + 1, chains) for v in spec['variables']])
    for i, value in (clamps or {}).items(): xs[:, int(i)] = value
    bits = encode_states(spec, xs); free = [k for k, (i, _) in enumerate(sites) if i not in (clamps or {})]
    temperatures = np.geomspace(.25, 1., chains); f, residual, e = energy.values(xs)
    evaluations = chains; exchanges = accepted = steps = 0
    best_x = None; best_key = None
    while free and time.monotonic() < deadline:
        k = int(rng.choice(free)); i, w = sites[k]; proposal = xs.copy()
        proposal[:, i] += (1 - 2 * bits[:, k]) * w
        nf, nr, ne = energy.values(proposal); evaluations += chains
        take = np.log(np.maximum(rng.random(chains), 1e-300)) < np.minimum(0, (e - ne) / temperatures)
        xs[take] = proposal[take]; bits[take, k] = 1 - bits[take, k]
        f[take], residual[take], e[take] = nf[take], nr[take], ne[take]
        steps += 1
        if steps % 8 == 0:
            for a in range((steps // 8) % 2, chains - 1, 2):
                b = a + 1; exchanges += 1
                ratio = (1 / temperatures[a] - 1 / temperatures[b]) * (e[a] - e[b])
                if np.log(max(rng.random(), 1e-300)) < min(0, ratio):
                    xs[[a, b]] = xs[[b, a]]; bits[[a, b]] = bits[[b, a]]
                    f[[a, b]] = f[[b, a]]; residual[[a, b]] = residual[[b, a]]; e[[a, b]] = e[[b, a]]; accepted += 1
        r = residual.sum(axis=1); j = min(range(chains), key=lambda j: (r[j] > 0, r[j], energy.sign * f[j]))
        candidate = (bool(r[j] > 0), float(r[j]), float(energy.sign * f[j]))
        if best_key is None or candidate < best_key: best_key = candidate; best_x = xs[j].tolist()
    result = summarize(spec, xs, xs, best_x, evaluations=evaluations, seconds=time.monotonic() - start)
    result.update(replica_exchange_attempts=exchanges, replica_exchange_accepts=accepted,
                  sampling_scope='finite_time_temperature_mixture_with_replica_exchange', temperatures=temperatures.tolist())
    return result
