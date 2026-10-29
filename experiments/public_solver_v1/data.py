"""Pinned public evaluation data; domain adapters are CPU-only, not solver logic."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
import tarfile
import urllib.request

from qihc.experiment_runtime import resilient_storage as s
from qihc.generic.bounded import VERSION, validate, objective, violations
from tools.benchmarks.nlco import convert as existing_convert

REVISION = '6a7dff75171906e0054ba6c92aa1ef8015b47c36'
REPO = 'summer142857jiang/NLCO'
TASKS = ('KP', 'MIS', 'MVC', 'MAXCUT', 'MCP', 'MDS')
ASSETS = Path(__file__).resolve().parent/'assets'


def validate_csv(payload, task, tier, expected_sha):
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError('Pinned CSV checksum mismatch')
    parsed = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
    if len(parsed) != 50 or any(r['task_id'] != task or r['difficulty_tier'] != tier for r in parsed):
        raise ValueError('Unexpected pinned dataset content')


def bundled_files(lock):
    path = ASSETS/lock['bundle']
    if not path.exists():
        return {}  # Source-only packaging may omit the optional bundle.
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != lock['bundle_sha256']:
        raise ValueError('Bundled public dataset checksum mismatch; restore checkout')
    result = {}
    # Never extract arbitrary paths or links. Read only exact locked regular members.
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name not in lock['files_sha256'] or member.name in result:
                raise ValueError('Unexpected/duplicate bundled dataset member')
            if member.size > 10_000_000:
                raise ValueError('Oversized bundled CSV')
            result[member.name] = archive.extractfile(member).read()
    if set(result) != set(lock['files_sha256']):
        raise ValueError('Incomplete bundled public dataset')
    return result


def convert(row):
    if row['task_id'] not in ('MCP', 'MDS'):
        return existing_convert(row)
    a, surface = row['instance_canonical_json'], row['instance_surface_json']
    n = a['num_nodes']
    if surface['num_nodes'] != n or len(surface['edges']) != len(a['edges']):
        raise ValueError('Surface/canonical size mismatch')
    mapping, edges = {}, set()
    for e, shown in zip(a['edges'], surface['edges']):
        u, v = e['u']-1, e['v']-1
        if not 0 <= u < n or not 0 <= v < n or u == v or tuple(sorted((u, v))) in edges:
            raise ValueError('Invalid/duplicate edge')
        edges.add(tuple(sorted((u, v))))
        for k in ('u', 'v'):
            if e[k] in mapping and mapping[e[k]] != shown[k]:
                raise ValueError('Inconsistent label mapping')
            mapping[e[k]] = shown[k]
    if set(mapping) != set(range(1, n+1)) or len(set(mapping.values())) != n:
        raise ValueError('Incomplete label mapping; no guessed isolated labels')
    constraints = []
    if row['task_id'] == 'MCP':
        for u in range(n):
            for v in range(u+1, n):
                if (u, v) not in edges:
                    c = [int(i in (u, v)) for i in range(n)]
                    constraints.append({'id': f'nonedge{u}_{v}', 'a': c, 'op': 'le', 'rhs': 1})
    else:
        for u in range(n):
            c = [int(v == u or tuple(sorted((u, v))) in edges) for v in range(n)]
            constraints.append({'id': f'dominate{u}', 'a': c, 'op': 'ge', 'rhs': 1})
    gold = validate({'schema': VERSION, 'variables': [{'name': f'x{i}', 'lower': 0, 'upper': 1} for i in range(n)],
        'objective': {'sense': 'max' if row['task_id'] == 'MCP' else 'min', 'linear': [1]*n, 'quadratic': []},
        'constraints': constraints, 'logic': []})
    chosen = row['reference_solution_canonical_json']
    x = [int(i+1 in chosen) for i in range(n)]
    ref = float(row['reference_objective_value'])
    if violations(gold, x).sum() or float(objective(gold, x)) != ref:
        raise ValueError('Published witness inconsistent with independent adapter')
    question = row['prompt']+'\nEvaluation output convention: x0,...,x'+str(n-1)+' denote vertex selection in this identifier order: '+json.dumps([mapping[i+1] for i in range(n)])+'. Follow the requested modeling schema, not the original solution output example.'
    return {'id': 'nlco_'+row['id'], 'dataset': 'nlco_'+row['task_id']+'_'+row['difficulty_tier'],
        'question': question, 'gold': gold, 'reference_answer': ref,
        'reference_status': 'published_reference_not_claimed_proven_optimal',
        'surface_format': row['surface_format'], 'original_id': row['id']}


def rows(path):
    data = list(csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig'))))
    for row in data:
        for key in ('instance_canonical_json', 'instance_surface_json', 'reference_solution_canonical_json'):
            row[key] = json.loads(row[key])
    return data


def download(root):
    """Verified cache, byte-identical Git snapshot, then mirror/official fallback."""
    lock = s.load(ASSETS/'nlco.lock.json')
    if lock['repo'] != REPO or lock['revision'] != REVISION:
        raise ValueError('Data lock revision differs')
    bundled = None
    endpoints = list(dict.fromkeys([os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'), 'https://huggingface.co']))
    evidence = {}
    for task in TASKS:
        for tier in ('S', 'M', 'L'):
            name = f'{task}/{task}_{tier}_context.csv'; path = root/name
            expected_sha = lock['files_sha256'][name]
            if path.exists():
                validate_csv(path.read_bytes(), task, tier, expected_sha)
            else:
                if bundled is None:
                    bundled = bundled_files(lock)
                if name in bundled:
                    payload = bundled[name]
                    validate_csv(payload, task, tier, expected_sha)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix('.part'); tmp.write_bytes(payload); tmp.replace(path)
                    print('RESTORED FROM VERIFIED GIT SNAPSHOT', name, flush=True)
            if not path.exists():
                errors = []
                for endpoint in endpoints:
                    for retry in range(3):
                        try:
                            url = f'{endpoint.rstrip("/")}/datasets/{REPO}/resolve/{REVISION}/{name}'
                            with urllib.request.urlopen(url, timeout=90) as response:
                                payload = response.read()
                            validate_csv(payload, task, tier, expected_sha)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            tmp = path.with_suffix('.part'); tmp.write_bytes(payload); tmp.replace(path)
                            break
                        except Exception as exc:
                            errors.append(f'{endpoint}: {type(exc).__name__}: {exc}')
                            if getattr(exc, 'code', None) in (401, 403, 404):
                                break  # Do not repeat a nontransient refusal three times.
                            time.sleep(retry+1)
                    if path.exists():
                        break
                if not path.exists():
                    raise RuntimeError(f'Download failed {name}: {errors}')
            parsed = rows(path)
            if len(parsed) != 50:
                raise ValueError('Each published task/tier must contain 50 rows')
            evidence[name] = s.digest(path)
            print('DATA READY', name, flush=True)
    provenance = {'repo': REPO, 'revision': REVISION, 'files_sha256': evidence, 'rows': 900}
    previous = root/'download_manifest.json'
    if previous.exists() and s.load(previous) != provenance:
        raise ValueError('Cached dataset changed; refusing overwrite')
    s.save(previous, provenance)


def load_cases(root):
    cases, excluded = [], []
    for task in TASKS:
        for tier in ('S', 'M', 'L'):
            data = rows(root/task/f'{task}_{tier}_context.csv')
            if len(data) != 50:
                raise ValueError('Incomplete public task/tier')
            for row in data:
                try:
                    cases.append(convert(row))
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    excluded.append({'id': row.get('id'), 'reason': str(exc)})
    if len({c['id'] for c in cases}) != len(cases) or not cases:
        raise ValueError('Empty/duplicate cases')
    return cases, excluded


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('directory', type=Path)
    download(p.parse_args().directory)
