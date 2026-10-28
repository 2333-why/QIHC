"""Independent CPU-only E002 launch after the frozen model export is complete."""
import argparse
import os
from pathlib import Path
import signal

from qihc.experiment_runtime import storage as s
from . import run


def require_models(source):
    cfg = s.load(run.HERE/'config.json')
    manifest_path = source/'manifest.json'
    cases_path = source/'formal/cases.CPU_ONLY.json'
    if not manifest_path.exists() or not cases_path.exists():
        raise ValueError('E002 needs a prepared frozen-model source: '+str(source)+
                         '. Run E001 separately first; E002 will not start it.')
    manifest = s.load(manifest_path)
    if s.digest(cases_path) != manifest['phase_cases_sha256']['formal']:
        raise ValueError('Source case identity mismatch')
    cases = s.load(cases_path)
    if any(arm not in manifest['config']['arms'] for arm in cfg['source_arms']):
        raise ValueError('Missing predeclared modeling arm in source')
    missing = []
    for case in cases:
        for arm in cfg['source_arms']:
            identity = s.key(case['id'], arm)
            if not (source/'formal/responses'/(identity+'.json')).exists():
                missing.append(case['id']+'/'+arm)
    if missing:
        total = len(cases)*len(cfg['source_arms'])
        raise ValueError(f'Frozen source incomplete: {total-len(missing)}/{total} model outputs; '
                         f'missing examples: {missing[:5]}. E002 stopped before computing; '
                         'finish E001 modeling separately, then repeat this command.')
    # Failed modeling responses are included, not selectively dropped. The
    # existing importer verifies every request/response hash and records failures.
    return len(cases)*len(cfg['source_arms'])


def main():
    base = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path(os.environ.get('QIHC_PROMPT_V3_RUN', str(base/'results/E001-1.3_prompt_fixed'))))
    p.add_argument('--run', type=Path, default=Path(os.environ.get('QIHC_E002_RUN', str(base/'results/E002-1.1_solver_standalone'))))
    args = p.parse_args()
    source, root = args.source.resolve(), args.run.resolve()
    count = require_models(source)
    print(f'E002 independent CPU run: {count} frozen model outputs; no GPU or E001 launch', flush=True)
    signal.signal(signal.SIGTERM, run.stop); signal.signal(signal.SIGINT, run.stop)
    return run.run(root, source, follow=False)


if __name__ == '__main__':
    raise SystemExit(main())
