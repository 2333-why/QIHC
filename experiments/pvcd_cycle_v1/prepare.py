"""CPU/login preparation: immutable input snapshot; never install or modify an environment."""
import hashlib
import json
import subprocess
from pathlib import Path
from qihc.experiment_runtime.storage import load, save, digest, worker_lock
from qihc.experiment_runtime.identity import model_identity
from experiments.feedback_active_v3.assets import restore
from experiments.feedback_active_v3.check_local_model import check
from .common import CFG, RUN, HISTORY, DATA, MODEL, REPO, donor, check_paths, source_hashes, environment_identity

def stream_hash(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def prepare():
    check_paths()
    if not (RUN/'manifest.json').exists() and any((RUN/mode).exists() for mode in ('pilot','formal')):
        raise ValueError('Unbound result directory already contains runs')
    from . import training
    from qihc.generic.solver_baselines.registry import solve
    from .evidence import collect
    restore(DATA); check(MODEL)
    split=load(HISTORY/'split.json'); prepared=load(HISTORY/'prepared.json')
    if split['experiment']!='E005-1.7' or split['source']!=digest(DATA/'public.manifest.json'):
        raise ValueError('Wrong historical data revision')
    snapshots={}
    for name,h in split['files'].items():
        obj=load(HISTORY/(name+'.json'))
        if hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()!=h: raise ValueError('Changed split '+name)
        snapshots[name+'.json']=obj
    train=snapshots['train.json']; validation=snapshots['validation.json']; cohort=snapshots['cohort.json']
    if len(train)!=90 or len(validation)!=24 or len(cohort)!=910: raise ValueError('Unexpected split size')
    # Explicit leakage guard in addition to the frozen donor split hashes.
    from experiments.posttraining_v1.data import structure
    train_ids={r['case']['id'] for r in train}; val_ids={r['case']['id'] for r in validation}
    held={r['id'] for r in cohort if r['stage']!='development'}
    if train_ids & (val_ids|held): raise ValueError('Training ID leakage')
    if {structure(r) for r in train} & {structure(r) for r in validation}: raise ValueError('Duplicate validation model')
    files=model_identity(MODEL)
    for r in files:
        if r['sha256'] is None: r['sha256']=stream_hash(MODEL/r['path'])
    if files!=prepared['files']: raise ValueError('Model differs from original SFT base')
    frozen={}; predictions={}
    for seed in CFG['seeds']:
        for arm in ('sft','pvcd'):
            adapter=donor(seed,arm)/'adapter.pt'; frozen[str(adapter)]=stream_hash(adapter)
            for p in sorted((donor(seed,arm)/'predictions').glob('*.json')):
                frozen[str(p)]=digest(p); predictions[str(p)]=True
        for row in train:
            p=donor(seed,'sft')/'predictions'/(row['case']['id']+'.json')
            if not p.exists(): raise ValueError('Missing training-only SFT parent '+str(p))
            snapshots[f'parents/{seed}/{p.name}']=load(p)
    gold=train[0]['gold']
    if solve(gold,'scip',2,17).get('x') is None: raise RuntimeError('SCIP runtime preflight failed')
    collect(gold,gold,17,dict(CFG,probe_seconds_per_round=.1))
    source=source_hashes()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    # Tracked source edits must be committed before the run is bound.
    changed=subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=REPO,text=True).splitlines()
    changed += subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=REPO,text=True).splitlines()
    if any(p in source for p in changed): raise ValueError('Commit experiment source before preparation')
    manifest=dict(experiment_id=CFG['experiment_id'],config=CFG,code_sha=sha,source=source,environment=environment_identity(),
        history=str(HISTORY),model=str(MODEL),fingerprints=files,donor_files=frozen,
        donor_predictions=predictions,data_files={str(p):digest(p) for p in sorted(DATA.rglob('*.json'))},
        split=split['files'])
    with worker_lock(RUN/'prepare.lock'):
        path=RUN/'manifest.json'
        if path.exists() and {k:v for k,v in load(path).items() if k!='input_files'}!=manifest:
            raise ValueError('Run identity changed; do not overwrite old results')
        for name,obj in snapshots.items():
            p=RUN/'inputs'/name
            if p.exists() and load(p)!=obj: raise ValueError('Input snapshot changed '+name)
            save(p,obj)
        manifest['input_files']={str(p.relative_to(RUN)):digest(p) for p in sorted((RUN/'inputs').rglob('*.json'))}
        if path.exists():
            # Compare includes the snapshot hashes on repeated preparation.
            if load(path)!=manifest: raise ValueError('Prepared inputs changed')
        save(path,manifest)
    print('PVCD_CYCLE_READY',RUN,'code',sha,flush=True)

if __name__=='__main__': prepare()
