import os
import sys
from importlib.metadata import version
from pathlib import Path
from qihc.experiment_runtime.storage import load, digest
from experiments.posttraining_v1.data import BASE, DATA, MODEL, STAGE_SEEDS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG = load(HERE / 'config.json')
HISTORY = Path(os.environ.get('QIHC_HISTORY', BASE / 'results/E005-1.7_eight'))
RUN = Path(os.environ.get('QIHC_CYCLE_RUN', BASE / 'results/E005-1.9_pvcd_cycle'))
TRAIN_ARMS = tuple(a for a in CFG['arms'] if a not in ('sft', 'pvcd'))

def check_paths():
    a, b = RUN.resolve(), HISTORY.resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('New results must be separate from the historical run')

def donor(seed, arm):
    return HISTORY / 'formal/models' / str(seed) / arm

def cases(seed, mode):
    if mode == 'pilot':
        return [dict(r['case'], stage='pilot') for r in load(RUN/'inputs/validation.json')]
    return [r for r in load(RUN/'inputs/cohort.json') if seed in STAGE_SEEDS[r['stage']]]

def training_rows(mode):
    from experiments.feedback_active_v3.runtime import seed_for
    rows = load(RUN/'inputs/train.json')
    if mode == 'pilot':
        rows = sorted(rows, key=lambda r: seed_for(r['case']['id'], 'pilot'))[:CFG['pilot_train_cases']]
    return rows

def source_hashes():
    paths = list((REPO/'qihc').rglob('*.py'))
    for name in ('posttraining_v1','feedback_active_v3','feedback_active_v4','pvcd_cycle_v1'):
        paths += list((REPO/'experiments'/name).glob('*.py'))
    paths += list(HERE.glob('*.sh')) + [HERE/'config.json']
    paths += list((REPO/'experiments/feedback_active_v3').glob('requirements-*.txt'))
    return {p.relative_to(REPO).as_posix(): digest(p) for p in sorted(paths)}

def environment_identity():
    return dict(python=sys.version,executable=str(Path(sys.executable).resolve()),
                packages={name:version(name) for name in ('torch','transformers','accelerate','numpy','pyscipopt')})
