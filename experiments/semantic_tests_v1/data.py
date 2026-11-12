import hashlib
import os
import subprocess
from collections import Counter
from pathlib import Path
from qihc.experiment_runtime.storage import load, save, digest, value_digest, worker_lock
from qihc.experiment_runtime.identity import model_identity
from experiments.optimal_feedback_v1.data import BASE, DATA, MODEL, SEEDS, wait_lock

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = Path(os.getenv('QIHC_PREVIOUS_RUN', BASE/'results/E004_2.6_boundary_feedback_fix1'))
RUN = Path(os.getenv('QIHC_SEMANTIC_RUN', BASE/'results/E004_2.7_semantic_tests'))
ARMS = ('A', 'B', 'C', 'R', 'S', 'P', 'Q')
SHARED_COST_ARMS = ('R', 'S', 'P', 'Q')
EXPERIMENT = 'E004_2.7'
AUDIT_MODULE = 'experiments.semantic_tests_v1.audit'

def sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    for folder in ('qihc', 'experiments/feedback_active_v3', 'experiments/optimal_feedback_v1', 'experiments/boundary_feedback_v1'):
        paths.extend((ROOT/folder).rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}

def prepare():
    from experiments.feedback_active_v3.assets import restore
    from experiments.feedback_active_v3.check_local_model import check
    from experiments.feedback_active_v3.cpu import execute
    if RUN.resolve() == PREVIOUS.resolve():
        raise ValueError('Use a distinct output directory')
    restore(DATA)
    previous = load(PREVIOUS/'manifest.json')
    items = load(PREVIOUS/'inputs.json')
    if previous['experiment'] != 'E004_2.6' or previous['cohort'] != value_digest(items):
        raise ValueError('Previous input identity mismatch')
    if previous['dataset'] != digest(DATA/'public.manifest.json'):
        raise ValueError('Public dataset mismatch')
    if len(items) != 2180 or len({x['case']['id'] for x in items}) != 910:
        raise ValueError('Expected the full previous cohort')
    counts = Counter((x['stage'], x['seed']) for x in items)
    if counts != {(stage, seed): size for stage, size in [('development', 120), ('main', 480), ('scale', 300), ('public', 10)] for seed in SEEDS[stage]}:
        raise ValueError('Stage/seed counts differ from the registered cohort')
    seen = set()
    for item in items:
        cid, seed = item['case']['id'], item['seed']
        if seed not in SEEDS[item['stage']] or (cid, seed) in seen:
            raise ValueError('Invalid seed or duplicate item')
        seen.add((cid, seed))
        row = load(PREVIOUS/'formal/models'/str(seed)/'B/predictions'/f'{cid}.json')
        if (row['case_id'], row['seed'], row['stage'], row['arm']) != (cid, seed, item['stage'], 'B'):
            raise ValueError('Previous prediction identity mismatch')
        if item['model'] is not None and row['model'] != item['model']:
            raise ValueError('Compile-only baseline altered an executable initial model')
        item['baseline'] = row['model']
        reviewed = load(PREVIOUS/'formal/models'/str(seed)/'C/predictions'/f'{cid}.json')
        if (reviewed['case_id'], reviewed['seed'], reviewed['stage'], reviewed['arm']) != (cid, seed, item['stage'], 'C'):
            raise ValueError('Previous C prediction identity mismatch')
        item['reviewed'] = reviewed['model']
        # Prediction-only reuse. No prior evaluation, gold or reference enters feedback.
    check(MODEL)
    files = model_identity(MODEL)
    for row in files:
        if row['sha256'] is None:
            with (MODEL/row['path']).open('rb') as f:
                row['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    if files != previous['model_files']:
        raise ValueError('Frozen base model changed')
    g = load(DATA/'evaluation/cases.gold.json')[0]['gold']
    if execute(dict(stage='solve', model=g, backend='scip', seconds=2, seed=17, chains=32)).get('x') is None:
        raise RuntimeError('Native SCIP check failed')
    manifest = dict(experiment=EXPERIMENT, source=sources(), cohort=value_digest(items),
                    model_files=files, dataset=previous['dataset'], arms=list(ARMS),
                    previous_manifest=digest(PREVIOUS/'manifest.json'), previous=str(PREVIOUS),
                    probe_seconds=20, terminal_seconds=20, chains=32, max_tokens=4096,
                    protocol="single_semantic_choice_v1", max_hypotheses=3)
    manifest['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    with worker_lock(RUN/'prepare.lock'):
        for name, obj in [('inputs', items), ('manifest', manifest)]:
            p = RUN/f'{name}.json'
            if p.exists() and load(p) != obj:
                raise ValueError(f'Immutable prepared run changed: {p}')
            save(p, obj)
    print(EXPERIMENT+'_CPU_READY', RUN, flush=True)

if __name__ == '__main__':
    prepare()
