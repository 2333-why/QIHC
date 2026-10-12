import json
from pathlib import Path
import sys

import pytest

from experiments.prepare_hard_nlcvrp import prepare_cases
from qihc.problems.cvrp.instance import load_jsonl


def write_pair(directory: Path, n: int, fleet: int) -> str:
    """Reference uses three-customer routes; capacity permits other fleet sizes."""
    name = f'X-n{n + 1}-k{fleet}'
    lines = [f'NAME: {name}', 'TYPE: CVRP', f'DIMENSION: {n + 1}',
             'EDGE_WEIGHT_TYPE: EUC_2D', 'CAPACITY: 100', 'NODE_COORD_SECTION']
    lines.extend(f'{i} {i} {i % 3}' for i in range(1, n + 2))
    lines.append('DEMAND_SECTION')
    lines.extend(f'{i} {0 if i == 1 else 1}' for i in range(1, n + 2))
    lines.extend(['DEPOT_SECTION', '1', '-1', 'EOF'])
    (directory / f'{name}.vrp').write_text('\n'.join(lines), encoding='utf-8')
    routes = [list(range(start, min(start + 3, n + 2))) for start in range(2, n + 2, 3)]
    text = '\n'.join(f'Route #{i}: ' + ' '.join(map(str, route)) for i, route in enumerate(routes, 1))
    (directory / f'{name}.sol').write_text(text + '\nCost 100\n', encoding='utf-8')
    return name


def test_fill_failed_primary_without_relaxing_fleet_or_leaking_routes(tmp_path):
    primary = [write_pair(tmp_path, 9, 3), write_pair(tmp_path, 12, 3),
               write_pair(tmp_path, 15, 5), write_pair(tmp_path, 18, 6)]
    reserves = [write_pair(tmp_path, 12, 4), write_pair(tmp_path, 21, 7)]
    unrelated = write_pair(tmp_path, 24, 8)
    manifest = tmp_path / 'download.json'
    manifest.write_text(json.dumps([{'name': name, 'candidate_role': role} for role, names in
                                   [('primary', primary), ('reserve', reserves)] for name in names]), encoding='utf-8')
    output = tmp_path / 'cases.jsonl'
    result = prepare_cases(tmp_path, output, min_customers=1, max_customers=100, limit=4,
                           clauses_per_type=1, selection='spread', input_manifest=manifest)
    assert len(result['selected']) == 4
    assert result['replacements'] == [{'excluded_primary': primary[1], 'replacement': reserves[0]}]
    assert all(name + '-nl-hard' in result['selected'] for name in [primary[0], primary[2], primary[3]])
    rejected = next(row for row in result['skipped'] if row['name'] == primary[1])
    assert rejected['vehicle_limit'] == 3 and rejected['witness_routes'] == 4
    assert rejected['violations'] == [{'type': 'vehicle_count', 'actual': 4, 'limit': 3}]
    assert unrelated not in result['candidate_pool']
    assert result['vehicle_policy'] == 'preserve_original_limit'
    for instance in load_jsonl(output):
        assert instance.vehicle_count == int(instance.metadata['headers']['NAME'].split('-k')[1])
        assert 'known_feasible_routes' not in instance.metadata
        assert len(instance.constraints) == 3
    first = output.read_bytes()
    prepare_cases(tmp_path, output, min_customers=1, max_customers=100, limit=4,
                  clauses_per_type=1, selection='spread', input_manifest=manifest)
    assert output.read_bytes() == first


def test_insufficient_valid_pool_fails_with_diagnostic_manifest(tmp_path):
    write_pair(tmp_path, 9, 2)
    write_pair(tmp_path, 12, 4)
    output = tmp_path / 'cases.jsonl'
    with pytest.raises(RuntimeError, match='enlarge candidate pool'):
        prepare_cases(tmp_path, output, min_customers=1, max_customers=100, limit=2, clauses_per_type=1)
    result = json.loads(output.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    assert len(result['selected']) == 1 and result['requested'] == 2
    assert result['skipped'][0]['violations'][0]['type'] == 'vehicle_count'


def test_spread_selection_happens_after_verification(tmp_path):
    for n, k in [(9, 3), (12, 4), (15, 4), (18, 6), (21, 7)]:
        write_pair(tmp_path, n, k)
    result = prepare_cases(tmp_path, tmp_path / 'cases.jsonl', min_customers=1,
                           max_customers=100, limit=3, clauses_per_type=1, selection='spread')
    assert len(result['selected']) == 3 and len(result['valid_candidates']) == 4
    assert not any('X-n16-k4' in name for name in result['selected'])


def test_reserve_downloader_caps_pool_and_preserves_primary(tmp_path, monkeypatch):
    import experiments.download_hard_cvrplib as download
    names = [f'X-n{n}-k3' for n in range(10, 16)]
    html = ''.join(f'<a href="/cvrplib/en/download/instance/{i}">{name}</a>' for i, name in enumerate(names, 1))
    def fetch(url):
        if url.endswith('/en/instances'):
            return html.encode()
        return b'NODE_COORD_SECTION\n' if '/download/instance/' in url else b'Route #1: 1 2 3\n'
    monkeypatch.setattr(download, 'fetch', fetch)
    monkeypatch.setattr(sys, 'argv', ['download', str(tmp_path), '--min-customers', '9', '--max-customers', '14',
                                    '--limit', '4', '--reserve-factor', '2', '--selection', 'spread'])
    assert download.main() == 0
    result = json.loads((tmp_path / 'download_manifest.json').read_text(encoding='utf-8'))
    assert len(result) == 6
    assert [row['name'] for row in result if row['candidate_role'] == 'primary'] == [names[i] for i in [0, 2, 3, 5]]
    assert len([row for row in result if row['candidate_role'] == 'reserve']) == 2
