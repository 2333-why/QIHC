"""Immutable E003 preparation, separate public requests and CPU-only references."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess

from qihc.experiment_runtime import resilient_storage as s
from qihc.experiment_runtime.identity import model_identity
from qihc.modeling import typed_builder, typed_sources
from . import data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARM = 'llm_typed_flat'


def fingerprint():
    h = hashlib.sha256()
    for folder in ('qihc/modeling', 'qihc/generic', 'qihc/experiment_runtime',
                   'experiments/public_solver_v1', 'experiments/solver_comparison_v1'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.suffix in ('.py', '.json', '.sh'):
                h.update(path.relative_to(ROOT).as_posix().encode())
                h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    h.update((ROOT/'tools/benchmarks/nlco.py').read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def versions():
    return {n: importlib.metadata.version(n) for n in ('numpy', 'ortools', 'highspy', 'pyscipopt')}


def request(case):
    names = [v['name'] for v in case['gold']['variables']]
    return {'id': s.key(case['id'], ARM), 'case_id': case['id'], 'phase': ARM,
        'queue_group': case['dataset'], 'queue_order': case['original_id'].rsplit('_', 1)[-1],
        'question': case['question'], 'question_sha256': hashlib.sha256(case['question'].encode()).hexdigest(),
        'prompt': typed_builder.prompt(case['question'], names, 'flat'), 'frontend': 'flat',
        'prompt_style': 'legacy', 'variable_names': names, 'contract': 'typed',
        'local_repair': False, 'semantic_review': False, 'execution_label': 'E003-1.1'}


def base_record(case, arm, model, status, response_hash):
    return {'case_id': case['id'], 'dataset': case['dataset'], 'arm': arm,
        'status': status, 'model': model, 'model_sha256': s.value_digest(model) if model else None,
        'response_sha256': response_hash, 'generation_cost': {}}


def prepare(root, dataset, model):
    cfg = s.load(HERE/'config.json')
    provenance = s.load(dataset/'download_manifest.json')
    if provenance['revision'] != data.REVISION:
        raise ValueError('Unexpected dataset revision')
    for name, checksum in provenance['files_sha256'].items():
        if s.digest(dataset/name) != checksum:
            raise ValueError('Dataset checksum mismatch: '+name)
    cases, excluded = data.load_cases(dataset)
    requests = [request(c) for c in cases]
    manifest = {'experiment_id': 'E003-1.1', 'config': cfg,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_hash': fingerprint(), 'dataset': provenance, 'selected_before_validation': 900,
        'excluded': excluded, 'cases': len(cases), 'cases_digest': s.value_digest(cases),
        'model': str(model.resolve()), 'model_files': model_identity(model), 'dependencies': versions(),
        'requests': {r['id']: s.value_digest(r) for r in requests},
        'scope': 'Canonical public mathematics and frozen LLM mathematics are separate tracks. '
                 'Six complete NLCO task families (900 selected), not the full 43-task leaderboard. '
                 'All seven CPU backends receive identical mathematics within each track. No semantic gate.'}
    with s.worker_lock(root/'prepare.lock'):
        if (root/'manifest.json').exists() and s.load(root/'manifest.json') != manifest:
            raise ValueError('Prepared identity changed. Preserve existing run and use a new path.')
        s.save(root/'manifest.json', manifest)
        s.save(root/'cases.CPU_ONLY.json', cases)
        for c, r in zip(cases, requests):
            s.save(root/'requests'/(r['id']+'.json'), r)
            canonical = base_record(c, 'canonical', c['gold'], 'compiled', s.value_digest(c['gold']))
            s.save(root/'models'/(s.key(c['id'], 'canonical')+'.json'), canonical)
    print(f'PREPARED {len(cases)}/900 public cases; {len(excluded)} adapter exclusions; no gate.', flush=True)
    return manifest


def validate(root, *, cpu=False, model=False):
    m = s.load(root/'manifest.json')
    if m['source_hash'] != fingerprint() or m['config'] != s.load(HERE/'config.json'):
        raise ValueError('Pinned code/config changed; refusing mixed experiment')
    if cpu and (versions() != m['dependencies'] or s.value_digest(s.load(root/'cases.CPU_ONLY.json')) != m['cases_digest']):
        raise ValueError('CPU case/environment identity mismatch')
    if cpu:
        for c in s.load(root/'cases.CPU_ONLY.json'):
            r = s.load(root/'models'/(s.key(c['id'], 'canonical')+'.json'))
            if r['model'] != c['gold'] or r['model_sha256'] != s.value_digest(c['gold']):
                raise ValueError('Canonical model identity changed')
    if model and model_identity(Path(m['model'])) != m['model_files']:
        raise ValueError('Model files changed')
    return m


def import_responses(root, manifest, cases):
    for c in cases:
        identity = s.key(c['id'], ARM); path = root/'responses'/(identity+'.json')
        if not path.exists():
            continue
        reqpath = root/'requests'/(identity+'.json'); req = s.load(reqpath)
        if s.value_digest(req) != manifest['requests'][identity]:
            raise ValueError('Request changed')
        row = s.load(path); rh = s.digest(path); dest = root/'models'/(identity+'.json')
        if row['request_sha256'] != s.digest(reqpath):
            raise ValueError('Response/request mismatch')
        if dest.exists():
            saved = s.load(dest)
            if saved['response_sha256'] != rh:
                raise ValueError('Frozen model response changed')
            if saved['status'] == 'compiled' and saved['model_sha256'] != s.value_digest(saved['model']):
                raise ValueError('Frozen mathematics changed')
            continue
        record = base_record(c, ARM, None, 'invalid_model', rh)
        try:
            if row.get('mathematical_validity') != 'valid':
                raise ValueError(row.get('validation_error', row.get('status')))
            model, reason = typed_sources.compile_math(row['response'], c['question'])
            if model is None:
                raise ValueError(reason)
            if [v['name'] for v in model['variables']] != req['variable_names']:
                raise ValueError('Public variable mapping differs')
            record.update(model=model, model_sha256=s.value_digest(model), status='compiled')
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            record['error'] = str(exc)
        record['generation_cost'] = {k: row.get(k) for k in ('input_tokens', 'output_tokens', 'seconds', 'generation_attempts')}
        s.save(dest, record)


def default_root():
    base = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))
    return Path(os.environ.get('QIHC_E003_RUN', str(base/'results/E003-1.1_public_solvers')))
