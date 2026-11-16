"""Count distinct DCP-Bench instances using the pinned upstream evaluator."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REVISION = '3f7e121301e0ea4565e20d8ee0660e799686c194'
ASSET_SHA256 = 'bf4f50f435c03a5600194306937e37039cdfa2e002dfa1c6548cd815c8601385'
ROOT = Path(__file__).resolve().parents[2]
PILOT = {'knapsack', 'csplib_022_bus_driver_scheduling', 'csplib_074_maximum_clique'}


def objective(source):
    directions = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                directions.add(node.func.attr)
            directions.update(keyword.arg for keyword in node.keywords)
    declared = directions & {'maximize', 'minimize'}
    if len(declared) > 1:
        raise ValueError('Multiple objective directions')
    return next(iter(declared), 'satisfaction')


def inventory(upstream, asset):
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('Expected DCP-Bench v0.3.0')
    subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD', '--', 'evaluation', 'dataset'], check=True)
    raw = asset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ASSET_SHA256:
        raise ValueError('Unexpected release asset')
    sys.path.insert(0, str(upstream.resolve()))
    from evaluation.reference import available_instances

    history = ROOT/'experiments/feedback_active_v3/assets/public_dataset/evaluation/cases.public_external.gold.json'
    previous = set()
    for row in json.loads(history.read_text(encoding='utf8')):
        collection, name = row['provenance']['original_id'].split('__')
        previous.add('session'+name if collection == 'aplai_course' else name)
    templates = []
    totals = Counter()
    for row in map(json.loads, raw.splitlines()):
        name = row['id']
        source = (upstream/'dataset'/name/(name+'.cpmpy.py')).read_text(encoding='utf8')
        instances = [key for key, _ in available_instances(source, row['instances'])]
        direction = objective(row['model'])
        exposure = 'E005-1.16' if name in PILOT else 'CP-Bench-10' if name in previous else 'no_known_template_match'
        templates.append(dict(template=name, objective=direction, history=exposure, instances=instances))
        totals[direction] += len(instances)
    if len({r['template'] for r in templates}) != 164 or not previous.union(PILOT).issubset(r['template'] for r in templates):
        raise ValueError('Unexpected dataset or unmatched prior template')
    return dict(release='v0.3.0', revision=revision, asset_sha256=ASSET_SHA256, counts=dict(totals), templates=templates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path, help='Clean official repository checkout')
    parser.add_argument('--asset', required=True, type=Path, help='dcp-bench-open.jsonl release asset')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = inventory(args.upstream, args.asset)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(result['counts'])
