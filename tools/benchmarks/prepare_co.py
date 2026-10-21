"""Pinned CO sources and audited NLCO subset. Never execute third-party generated code."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmarks.comparison import save, load
from tools.benchmarks.nlco import cases_from_repo


def checkout(item, destination):
    target = destination/(item['name']+'-'+item['revision'][:12])
    if not target.exists():
        target.mkdir(parents=True)
        subprocess.run(['git', 'init', str(target)], check=True)
    if not (target/'.git').exists():
        raise ValueError(f'Not a managed Git checkout: {target}')
    actual = subprocess.run(['git', '-C', str(target), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    if actual.returncode or actual.stdout.strip() != item['revision']:
        dirty = subprocess.run(['git', '-C', str(target), 'status', '--porcelain'], capture_output=True, text=True, check=True)
        if dirty.stdout.strip(): raise ValueError(f'Dirty source checkout: {target}')
        subprocess.run(['git', '-C', str(target), 'fetch', '--depth', '1',
                        'https://github.com/'+item['repo']+'.git', item['revision']], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', item['revision']], check=True)
    dirty = subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True)
    if dirty.strip(): raise ValueError(f'Modified reference source: {target}')
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--per-tier', type=int, default=20)
    args = p.parse_args()
    if (args.run/'manifest.json').exists():
        from tools.benchmarks.comparison import validate_run
        validate_run(args.run)
        coverage = args.run/'coverage.json'
        if coverage.exists() and load(coverage)['requested_per_task_tier'] != args.per_tier:
            raise ValueError('NLCO selection changed; use a new QIHC_CO_RUN directory')
        print('Existing experiment verified; resume without downloading.'); return
    resources = load(Path(__file__).with_name('co_resources.lock.json'))
    paths = {}
    for item in resources['repositories']:
        # Deferred paper baselines must not block the currently requested experiment.
        if item['name'] != 'NLCO':
            continue
        paths[item['name']] = checkout(item, args.base/'data/co_sources')
    cases, report = cases_from_repo(paths['NLCO'], args.per_tier)
    prepared = args.base/'data/co_prepared'/f'nlco-{args.per_tier}'
    save(prepared/'cases.CPU_ONLY.json', cases)
    save(prepared/'coverage.json', report)
    save(prepared/'baseline_availability.json', resources)
    save(args.run/'coverage.json', report)
    save(args.run/'baseline_availability.json', resources)
    subprocess.run([sys.executable, str(ROOT/'tools/benchmarks/prepare_comparison.py'),
                    '--base', str(args.base), '--run', str(args.run),
                    '--cases-file', str(prepared/'cases.CPU_ONLY.json')], check=True)
    print(json.dumps({'nlco_cases': len(cases), 'excluded': len(report['excluded']),
                      'synthetic_cases': 60, 'paper_baselines': 'NOT executed; see baseline_availability.json'}), flush=True)


if __name__ == '__main__': main()
