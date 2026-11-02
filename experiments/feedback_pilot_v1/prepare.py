"""Freeze 18 performance-blind cases and historical comparators; no new solving."""
import argparse
from collections import defaultdict
import copy
import hashlib
import os
from pathlib import Path
from qihc.experiment_runtime import resilient_storage as s
from qihc.experiment_runtime.identity import model_identity
from qihc.generic.bounded import validate as validate_math
from experiments.feedback_study_v1 import common
from experiments.feedback_comparison_v1.report import metrics
from .adapter import EXPERIMENT, install, fingerprint, default_root


def select_cases(cases):
    groups = defaultdict(list)
    for case in cases:
        groups[case['dataset']].append(case)
    expected = {f'nlco_{task}_{tier}' for task in ('KP','MIS','MVC','MAXCUT','MCP','MDS') for tier in ('S','M','L')}
    if set(groups) != expected or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Expected the 18 E003 task/tier strata')
    # dataset is the original nlco_TASK_TIER field, never a metric/validity label.
    strata = sorted(groups, key=lambda name: ('SML'.index(name.rsplit('_',1)[1]), name))
    return [min(groups[name], key=lambda c: hashlib.sha256(('20261102\0'+c['id']).encode()).hexdigest())
            for name in strata]


def configuration():
    cfg = s.load(common.HERE/'config.json')
    cfg.update(experiment_id=EXPERIMENT, public_arms=['C0', 'C1', 'U2', 'B2'],
               extension_arms=['C0', 'C1', 'U2', 'B2'], diagnostic_arms=[], seeds=[0], rounds=2,
               cpu_workers=24, inflight_trajectories=12, review_output_tokens=2048,
               patch_output_tokens=4096, trajectory_input_tokens=131072,
               development_cases=0, selection_seed=20261102,
               pilot_scope='exploratory retrospective stratified pilot; no formal quality acceptance claim')
    return cfg


def prepare(root, source, model, smoke=False):
    install()
    if root.resolve() == source.resolve() or source.resolve() in root.resolve().parents:
        raise ValueError('Pilot output must be outside E003')
    if root.exists() and (root/'manifest.json').exists():
        # Preparation is an idempotent validation, never a metadata rewrite.
        m = common.validate(root)
        if m['source_manifest_sha256'] != s.digest(source/'manifest.json') or m['model'] != str(model.resolve()):
            raise ValueError('Prepared source/model identity changed')
        return m
    src = s.load(source/'manifest.json')
    cases = s.load(source/'cases.CPU_ONLY.json')
    if src['cases_digest'] != s.value_digest(cases):
        raise ValueError('E003 case identity mismatch')
    if not smoke and len(cases) != 900:
        raise ValueError('Expected original 900-case E003 population')
    if 0 not in src['config']['seeds'] or 20 not in src['config']['seconds']:
        raise ValueError('Missing predeclared E003 seed=0, 20-second baseline')
    chosen = select_cases(cases)
    gold_cases, public, initial, references, baselines, evidence = [], [], {}, {}, [], {}
    for rank, original in enumerate(chosen):
        c = copy.deepcopy(original)
        cid = c['id']
        c.update(split='pilot', size=len(c['gold']['variables']), generation_seed=rank,
                 variable_names=[v['name'] for v in c['gold']['variables']])
        validate_math(c['gold'])
        gold_cases.append(c)
        public.append({k: c[k] for k in ('id','dataset','split','size','generation_seed','question','variable_names')})
        modelpath = source/'models'/(s.key(cid, 'llm_typed_flat')+'.json')
        frozen = s.load(modelpath)
        if frozen.get('model') is not None:
            validate_math(frozen['model'])
            if frozen['model_sha256'] != s.value_digest(frozen['model']):
                raise ValueError('Frozen model hash mismatch')
        initial[cid] = {'model': frozen.get('model'), 'error': frozen.get('error'),
                        'source_record_sha256': s.digest(modelpath), 'generation_cost': frozen.get('generation_cost', {})}
        refpath = source/'references'/(s.key(cid)+'.json')
        ref = s.load(refpath)
        if ref['gold_sha256'] != s.value_digest(c['gold']):
            raise ValueError('Reference mathematics mismatch')
        references[cid] = ref
        for track in ('llm_typed_flat', 'canonical'):
            source_model = s.load(source/'models'/(s.key(cid, track)+'.json'))
            if source_model.get('model') is not None and s.value_digest(source_model['model']) != source_model.get('model_sha256'):
                raise ValueError('Historical mathematics checksum mismatch')
            for backend in src['config']['backends']:
                path = source/'scores'/s.key(cid, track)/f'{backend}-20-0.json'
                row = s.load(path)
                if (row['case_id'],row['source_arm'],row['backend'],row['budget_seconds'],row['seed']) != (cid,track,backend,20,0):
                    raise ValueError('Historical score identity mismatch')
                if row['response_sha256'] != source_model['response_sha256'] or row.get('model_sha256') != source_model.get('model_sha256'):
                    raise ValueError('Historical model/score mismatch')
                baselines.append({'case_id':cid, 'method':f'E003/{track}/{backend}',
                                  'metrics':metrics(row.get('verified_feasible'),row.get('verified_objective'),c,ref),
                                  'solver_wall_seconds':row.get('raw_metrics',{}).get('wall_seconds'),
                                  'source_path':str(path), 'source_sha256':s.digest(path)})
        evidence[cid] = {'initial':s.digest(modelpath), 'reference':s.digest(refpath)}
    cfg = configuration()
    manifest = {'experiment_id':EXPERIMENT, 'config':cfg, 'config_digest':s.value_digest(cfg),
                'source_hash':fingerprint(), 'code_commit':common.commit(),
                'source_run':str(source.resolve()), 'source_manifest_sha256':s.digest(source/'manifest.json'),
                'model':str(model.resolve()), 'model_files':model_identity(model) if not smoke else [],
                'dependencies':common.versions() if not smoke else {}, 'smoke':smoke, 'cases':len(chosen),
                'selection':{'seed':20261102,'rule':'min sha256(seed NUL case_id) per dataset; no outcome/validity filtering',
                             'ids':[c['id'] for c in chosen]}, 'source_evidence':evidence,
                'scope':'Exploratory pilot; no formal acceptance bypass or algorithm superiority claim'}
    with s.worker_lock(root/'prepare.lock'):
        common.immutable(root/'cases.public.json', public)
        common.immutable(root/'evaluation/cases.CPU_ONLY.json', gold_cases)
        common.immutable(root/'evaluation/e003_baselines.json', baselines)
        manifest['public_digest'] = s.digest(root/'cases.public.json')
        manifest['evaluation_digest'] = s.digest(root/'evaluation/cases.CPU_ONLY.json')
        manifest['baselines_digest'] = s.digest(root/'evaluation/e003_baselines.json')
        common.immutable(root/'manifest.json', manifest)
        for cid in initial:
            common.immutable(root/'initial'/(s.key(cid)+'.json'), initial[cid])
            common.immutable(root/'evaluation/references'/(s.key(cid)+'.json'), references[cid])
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=default_root())
    p.add_argument('--source',type=Path,default=Path(os.environ.get('QIHC_E003_RUN', common.BASE/'results/E003-1.1_public_solvers')))
    p.add_argument('--model',type=Path,default=Path(os.environ.get('QIHC_MODEL', common.BASE/'models/Qwen3-32B')))
    a = p.parse_args()
    prepare(a.run,a.source,a.model)
    from experiments.feedback_study_v1.prepare import preflight
    preflight(a.run)
    print('E004-1.2 PILOT READY: 18 cases, 72 trajectories, <=216 new LLM calls. Not a formal quality acceptance.')
