import sys
from types import ModuleType

import pytest

from experiments.solver_comparison_v1.smoke import check_one
from experiments.solver_comparison_v1.staging import find_staging
from qihc.generic.solver_baselines import isolation
from qihc.generic.solver_baselines.registry import BACKENDS


def test_same_native_worker_is_reused_but_never_between_backends():
    with isolation.BackendPools(BACKENDS, 7) as pools:
        first = {name: pools.submit(name, check_one, name) for name in BACKENDS}
        first = {name: f.result() for name, f in first.items()}
        second = {name: pools.submit(name, check_one, name).result() for name in reversed(BACKENDS)}
        assert len({r['worker_pid'] for r in first.values()}) == len(BACKENDS)
        for name in BACKENDS:
            assert first[name]['worker_pid'] == second[name]['worker_pid']
            expected = {name} if name in isolation.NATIVE_MODULES else set()
            assert set(first[name]['loaded_solver_modules']) == expected
        with pytest.raises(RuntimeError, match='pinned'):
            pools.submit('cp_sat', check_one, 'highs').result()


def test_native_import_guard_stops_conflict_before_loading(monkeypatch):
    monkeypatch.setattr(isolation, '_BOUND_BACKEND', None)
    monkeypatch.setitem(sys.modules, 'ortools', ModuleType('ortools'))
    with pytest.raises(RuntimeError, match='isolation required'):
        isolation.ensure_backend('highs')
    with pytest.raises(RuntimeError, match='isolation required'):
        isolation.ensure_backend('scip')


def test_production_worker_budget_allocation():
    pools = isolation.BackendPools(BACKENDS, 24)
    assert sum(pools.capacity.values()) == 24
    assert all(v >= 1 for v in pools.capacity.values())
    assert isolation.BackendPools(BACKENDS, 2).total_workers == 2


def test_reuse_complete_staging_without_importing_extensions(tmp_path):
    incomplete = tmp_path/'.e002-solvers-incomplete'; incomplete.mkdir()
    assert find_staging(tmp_path) is None
    ready = tmp_path/'.e002-solvers-FTw2es'; ready.mkdir()
    for name, version in [('highspy', '1.15.1'), ('pyscipopt', '6.3.0')]:
        (ready/name).mkdir()
        (ready/name/'__init__.py').write_text('raise RuntimeError("MUST NOT IMPORT")', encoding='utf-8')
        info = ready/f'{name}-{version}.dist-info'; info.mkdir()
        (info/'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n', encoding='utf-8')
    assert find_staging(tmp_path) == ready
