import importlib
import os
from pathlib import Path
import signal
import sys
from qihc.experiment_runtime.storage import load
from qihc.experiment_runtime import posttrain_screen as runtime
from .training import step
from .report import build_report

HERE=Path(__file__).resolve().parent
BASE=Path('/inspire/hdd/global_user/czxs25210022/why')
RUN=Path(os.environ.get('QIHC_VERIFIED_RUN',BASE/'results/E005-1.10_verified'))
PREVIOUS=Path(os.environ.get('QIHC_CYCLE_RUN',BASE/'results/E005-1.9_pvcd_cycle'))
CFG=load(HERE/'config.json')
def report(mode):return build_report(importlib.import_module('experiments.pvcd_verified_v1.run'),mode)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    job=importlib.import_module('experiments.pvcd_verified_v1.run')
    action=sys.argv[1];mode=sys.argv[2] if len(sys.argv)>2 else 'pilot'
    if action=='prepare':runtime.prepare(job)
    elif action=='worker':runtime.worker(job,mode,sys.argv[3])
    elif action=='gpu':
        for phase in (('pilot','screen') if mode=='all' else (mode,)):runtime.launch(job,phase)
    elif action=='score':runtime.validate(job);runtime.score_run(job,mode)
    elif action=='report':report(mode)
    else:raise ValueError('Unknown action')
