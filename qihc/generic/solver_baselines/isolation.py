"""Fresh, backend-pinned processes: native solver wheels must not share a loader."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys

NATIVE_MODULES = {'cp_sat': 'ortools', 'scip': 'pyscipopt', 'highs': 'highspy'}
_BOUND_BACKEND = None


def ensure_backend(backend):
    if _BOUND_BACKEND is not None and backend != _BOUND_BACKEND:
        raise RuntimeError(f'Worker pinned to {_BOUND_BACKEND}, cannot execute {backend}')
    if backend in NATIVE_MODULES:
        foreign = [name for key, name in NATIVE_MODULES.items() if key != backend and
                   any(m == name or m.startswith(name+'.') for m in sys.modules)]
        if foreign:
            raise RuntimeError(f'Native solver isolation required for {backend}; already loaded {foreign}. '
                               'Use BackendPools with spawn; do not reorder imports or fork.')


def bind_backend(backend):
    global _BOUND_BACKEND
    _BOUND_BACKEND = backend
    ensure_backend(backend)


class BackendPools:
    """Bounded scheduler with one spawn pool per backend, never fork inherited libs.

    Normal runs split the total worker budget across backends. If the total is
    smaller than the backend count, each pool has one worker but the scheduler
    still admits at most total_workers simultaneous tasks.
    """
    def __init__(self, backends, total_workers):
        self.backends = tuple(dict.fromkeys(backends))
        if not self.backends or type(total_workers) is not int or total_workers < 1:
            raise ValueError('Nonempty backends and positive integer worker budget required')
        self.total_workers = total_workers
        base, extra = divmod(total_workers, len(self.backends))
        self.capacity = {b: max(1, base+(i < extra)) for i, b in enumerate(self.backends)}
        self.pools = {}

    def __enter__(self):
        return self

    def submit(self, backend, function, *args):
        if backend not in self.capacity:
            raise ValueError('Unconfigured solver backend: '+backend)
        if backend not in self.pools:
            self.pools[backend] = ProcessPoolExecutor(max_workers=self.capacity[backend],
                mp_context=multiprocessing.get_context('spawn'),
                initializer=bind_backend, initargs=(backend,))
        return self.pools[backend].submit(function, *args)

    def __exit__(self, *exc):
        for pool in self.pools.values():
            pool.shutdown(wait=True, cancel_futures=True)
