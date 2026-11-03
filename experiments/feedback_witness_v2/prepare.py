"""Import frozen E004-1.2 inputs, without calling or modifying its runtime."""
import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path
from .common import HERE, ROOT, BASE, s, immutable, fingerprint, validate, configuration
from .budget import assess


def prepare(root, source, model, smoke=False, config=None):
    from qihc.generic.bounded import validate as validate_model
    from qihc.experiment_runtime.identity import model_identity
    root, source, model = map(lambda p: Path(p).resolve(), (root, source, model))
    if root == source or source in root.parents or root in source.parents:
        raise ValueError('Source and run directories must be disjoint')
    if (root/'manifest.json').exists():
        m = validate(root)
        if m['source_run'] != str(source) or m['model'] != str(model) or m['smoke'] != smoke:
            raise ValueError('Resume arguments differ from frozen manifest')
        return m
    start = time.monotonic()
    previous = s.load(source/'manifest.json')
    if previous['experiment_id'] != 'E004-1.2':
        raise ValueError('Expected prepared E004-1.2 run as frozen data source')
    config = config or configuration()
    selection = s.load(ROOT/config['selection_file'])
    if not smoke and assess(config, selection)['status'] != 'capacity_check_passed':
        raise ValueError('6-hour capacity check failed')
    files = {}
    for name, field in [('cases.public.json', 'public_digest'),
                        ('evaluation/cases.CPU_ONLY.json', 'evaluation_digest'),
                        ('evaluation/e003_baselines.json', 'baselines_digest')]:
        if s.digest(source/name) != previous[field]:
            raise ValueError('Source identity mismatch: '+name)
        value=s.load(source/name)
        if config['experiment_id']=='E004-1.6' and name in ('cases.public.json','evaluation/cases.CPU_ONLY.json'):
            ids={c['id'] for c in selection['cases']}
            value=[c for c in value if c['id'] in ids]
        immutable(root/name, value)
        files[name] = s.digest(root/name)
    cases = s.load(root/'cases.public.json')
    gold_cases={c['id']:c for c in s.load(root/'evaluation/cases.CPU_ONLY.json')}
    if not smoke and [c['id'] for c in cases] != [c['id'] for c in selection['cases']]:
        raise ValueError('Frozen 18-case selection differs')
    for c in cases:
        gold=gold_cases[c['id']]
        if gold['question']!=c['question'] or [v['name'] for v in gold['gold']['variables']]!=c['variable_names']:
            raise ValueError('Public/evaluator source mapping differs')
        reference=s.load(source/'evaluation/references'/(s.key(c['id'])+'.json'))
        if reference['gold_sha256']!=s.value_digest(gold['gold']):
            raise ValueError('Reference belongs to different mathematics')
        for name in ['initial/'+s.key(c['id'])+'.json', 'evaluation/references/'+s.key(c['id'])+'.json']:
            value = s.load(source/name)
            if name.startswith('initial/') and value.get('model') is not None:
                validate_model(value['model'])
            immutable(root/name, value)
            files[name] = s.digest(root/name)
    dependencies = {}
    if not smoke:
        from experiments.feedback_study_v1.common import versions
        from qihc.modeling.formatting import check_runtime
        import importlib.metadata
        check_runtime()
        dependencies = versions()
        dependencies['highspy'] = importlib.metadata.version('highspy')
        from .common import cpu_job
        tiny = {'schema':'bounded-integer-v2', 'variables':[{'name':'x0','lower':0,'upper':1}],
                'objective':{'sense':'max','linear':[1],'quadratic':[]}, 'constraints':[], 'logic':[]}
        for backend in config['cross_backends']:
            trial=dict(config,cross_solver_seconds=.1)
            answer = cpu_job(root,{'stage':'cross','config':trial,'model':tiny,'backend':backend,'seed':0},30,time.time()+30)
            if not answer.get('feasible'):
                raise ValueError('Solver preflight failed: '+backend)
    m = {'experiment_id':config['experiment_id'], 'source_hash':fingerprint(), 'config':config,
         'config_digest':s.value_digest(config), 'source_run':str(source),
         'source_manifest_sha256':s.digest(source/'manifest.json'), 'model':str(model),
         'model_files':[] if smoke else model_identity(model), 'smoke':smoke,
         'files':files, 'dependencies':dependencies, 'python':sys.version, 'platform':platform.platform(),
         'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'prepare_seconds':time.monotonic()-start}
    immutable(root/'manifest.json',m)
    return m


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source',type=Path,default=BASE/'results/E004-1.2_feedback_pilot')
    p.add_argument('--model',type=Path,default=BASE/'models/Qwen3-32B')
    a=p.parse_args();prepare(a.run,a.source,a.model)
    print('Prepared E004-1.5 immutable inputs; this command does not start the 6-hour run.')
