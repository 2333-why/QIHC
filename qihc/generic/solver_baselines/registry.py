"""All backends accept the same validated IR; no generated code or task dispatch."""
import math
import time
from ..bounded import validate
from .isolation import ensure_backend

BACKENDS = ('cp_sat', 'scip', 'highs', 'sa', 'pt', 'penalty_pt', 'pbit')


def solve(spec, backend, seconds, seed=0, chains=16):
    validate(spec)
    if backend not in BACKENDS or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Unknown backend or invalid time budget')
    ensure_backend(backend)
    # Native library initialization is a separately reported one-off cost,
    # not allowed to consume just the first solver/seed's search budget.
    loading = time.perf_counter()
    if backend == 'cp_sat':
        from ortools.sat.python import cp_model  # noqa: F401
    elif backend == 'scip':
        import pyscipopt  # noqa: F401
    elif backend == 'highs':
        import highspy  # noqa: F401
    import_seconds = time.perf_counter()-loading
    start = time.perf_counter()
    if backend == 'cp_sat':
        from .cp_sat import solve as cp
        result = cp(spec, seconds, seed=seed, include_build=True)
    elif backend in ('scip', 'highs'):
        from .milp import solve as milp, UnsupportedModel
        try:
            result = milp(spec, backend, seconds, seed)
        except UnsupportedModel as exc:
            result = {'status': 'UNSUPPORTED', 'x': None, 'feasible': False,
                      'optimality_proven': False, 'reason': str(exc)}
    else:
        from .sampling import solve as sample
        result = sample(spec, backend, seconds, seed, chains)
    result.update(backend=backend, requested_seconds=seconds, wall_seconds=time.perf_counter()-start,
                  threads=1, seed=seed, import_seconds=import_seconds)
    result['time_limit_overrun_seconds'] = max(0., result['wall_seconds']-seconds)
    return result
