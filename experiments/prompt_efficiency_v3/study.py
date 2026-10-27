"""Study definition, reproducible requests, and CPU-only per-seed evaluation."""
import hashlib
import math
import subprocess
from pathlib import Path

from qihc.experiment_runtime import storage as s
from qihc.experiment_runtime.identity import model_identity
from qihc.modeling import builder, protocol
from qihc.modeling import typed_builder, typed_sources
from qihc.modeling.review_workflow import execute  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = 'prompt_efficiency_v3'
RUN_ENV = 'QIHC_PROMPT_V3_RUN'
DEFAULT_SOURCE = 'co_comparison_v1'
ENTRY_MODULE = 'experiments.prompt_efficiency_v3.run'
ARMS = {
    'legacy_flat': ('flat', 'legacy'), 'typed_flat': ('flat', 'typed'),
    'typed_staged': ('staged', 'typed'), 'canonical_staged': ('staged', 'typed'),
    'semantic_flat': ('flat', 'typed'), 'semantic_staged': ('staged', 'typed'),
}
PAIRS = [('legacy_flat', 'typed_flat'), ('typed_staged', 'canonical_staged'),
         ('typed_flat', 'semantic_flat'), ('canonical_staged', 'semantic_staged'),
         ('semantic_flat', 'semantic_staged')]


def summarize(root, phase):
    from .report import summarize as report
    return report(root, phase)


def gate_check(report):
    from qihc.experiment_runtime.runner import gate_check as execution_gate
    result = execution_gate(report)
    failures = [arm for arm in ('semantic_flat', 'semantic_staged')
                if report['overall'][arm]['semantic_review_statuses'].get('accepted_self_consistent', 0) < 1]
    result['semantic_review_execution_failures'] = failures
    result['failed_arms'] = sorted(set(result['failed_arms']) | set(failures))
    result['passed'] = result['passed'] and not failures
    result['review_scope'] = 'Each review arm must complete at least one compiled self-consistent review; no gold-based selection.'
    return result


def prepare_runtime(root, source):
    from .preflight import prepare as preflight
    return preflight(root, source)


def source_hash():
    h = hashlib.sha256()
    files = []
    for folder in ('qihc/modeling', 'qihc/experiment_runtime', 'qihc/generic', 'experiments/prompt_efficiency_v3'):
        files += list((ROOT/folder).rglob('*.py')) + list((ROOT/folder).glob('*.json')) + list((ROOT/folder).glob('*.sh'))
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def gate_cases(cases):
    groups = {}
    for case in cases:
        dataset = case['dataset']
        family = dataset.rsplit('_', 1)[0] if dataset.startswith('nlco_') else dataset
        groups.setdefault(family, []).append(case)
    return [max(rows, key=lambda c: (len(c['question']), c['id'])) for _, rows in sorted(groups.items())]


def request(case, arm):
    frontend, style = ARMS[arm]
    names = [v['name'] for v in case['gold']['variables']]
    if style == 'legacy':
        text = protocol.modeling_prompt(case)
    elif style == 'audit':
        text = builder.common(case['question'], names)
    else:
        text = typed_builder.prompt(case['question'], names, frontend)
    return {'id': s.key(case['id'], arm), 'case_id': case['id'], 'phase': arm,
            'question': case['question'], 'question_sha256': hashlib.sha256(case['question'].encode()).hexdigest(),
            'prompt': text, 'frontend': frontend, 'prompt_style': 'legacy', 'variable_names': names,
            'contract': 'legacy' if style == 'legacy' else 'audit' if style == 'audit' else 'typed',
            'local_repair': False, 'canonical_ids': arm in ('canonical_staged', 'semantic_staged'),
            'semantic_review': arm in ('semantic_flat', 'semantic_staged'), 'execution_label': 'semantic_contract_study_v3'}


def validate(root, *, model=False):
    manifest = s.load(root/'manifest.json')
    if manifest['source_hash'] != source_hash():
        raise ValueError('Experiment source differs; use pinned checkout or a new run directory')
    if manifest['config'] != s.load(HERE/'config.json'):
        raise ValueError('Saved settings differ from the pinned experiment config')
    if model and model_identity(Path(manifest['model'])) != manifest['model_files']:
        raise ValueError('Prepared local model metadata changed')
    return manifest


def prepare(root, source):
    cfg = s.load(HERE/'config.json')
    if (cfg['thinking'] is not False or cfg['feedback'] != 'none' or cfg['selection'] != 'fixed'
            or cfg['replicas'] != 2 or cfg['total_output_tokens'] != builder.TOTAL_OUTPUT_TOKENS):
        raise ValueError('This study fixes thinking, feedback, selection and output budget; use a new study to change them')
    for key, low, high in [('chains', 1, 256), ('budget', 256, 1000000), ('cpu_workers', 1, 64), ('batch_size', 1, 16),
                          ('max_idle_seconds', 180, 3600), ('generation_watchdog_seconds', 900, 14400)]:
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError('Invalid setting: '+key)
    if (not cfg['arms'] or len(set(cfg['arms'])) != len(cfg['arms']) or any(a not in ARMS for a in cfg['arms'])
            or not cfg['seeds'] or len(set(cfg['seeds'])) != len(cfg['seeds']) or len(cfg['seeds']) > 20
            or any(type(v) is not int or v < 0 for v in cfg['seeds'])
            or not math.isfinite(cfg['solver_seconds']) or cfg['solver_seconds'] <= 0):
        raise ValueError('Invalid arms, seeds or solver time')
    previous = s.load(source/'manifest.json')
    case_path = source/'cases.CPU_ONLY.json'
    if s.digest(case_path) != previous['cases_sha256']:
        raise ValueError('Source case checksum differs from original manifest')
    cases = s.load(case_path)
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Empty or duplicate case set')
    if any(not c.get('gold') for c in cases):
        raise ValueError('This study requires standard mathematical models for independent feasibility; answer-only data needs a separate adapter')
    manifest = {'experiment': cfg['experiment'], 'source_hash': source_hash(), 'config': cfg,
                'source_run': str(source.resolve()), 'source_cases_sha256': s.digest(case_path),
                'model': previous['settings']['model'],
                'model_files': model_identity(Path(previous['settings']['model'])),
                'case_count': len(cases), 'gate_count': len(gate_cases(cases)),
                'assistance': 'original_variable_names_and_order_from_gold_when_available',
                'scope': 'previously inspected diagnostic benchmark, not an unseen test claim'}
    manifest['code_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest['phase_cases_sha256'] = {phase: s.value_digest(rows) for phase, rows in
                                    [('gate', gate_cases(cases)), ('formal', cases)]}
    manifest['request_hashes'] = {s.key(c['id'], a): s.value_digest(request(c, a)) for c in cases for a in cfg['arms']}
    if (root/'manifest.json').exists() and s.load(root/'manifest.json') != manifest:
        raise ValueError('Settings/source/model changed; preserve old run and choose a new QIHC_PROMPT_V3_RUN')
    for phase, selected in [('gate', gate_cases(cases)), ('formal', cases)]:
        directory = root/phase
        target = directory/'cases.CPU_ONLY.json'
        if target.exists() and s.load(target) != selected:
            raise ValueError('Stored phase cases changed')
        s.save(target, selected)
        for case in selected:
            for arm in cfg['arms']:
                row = request(case, arm)
                path = directory/'requests'/(row['id']+'.json')
                if path.exists() and s.load(path) != row:
                    raise ValueError('Stored request changed')
                if not path.exists():
                    s.save(path, row)
    s.save(root/'manifest.json', manifest)
    return manifest


def tasks(directory, cases, cfg):
    """Yield only available CPU jobs; at most one model/seed per future."""
    for case in cases:
        for arm in cfg['arms']:
            identity = s.key(case['id'], arm)
            response = directory/'responses'/(identity+'.json')
            if not response.exists():
                continue
            for backend, seed in [('cp_sat', 0)]+[('pbit', seed) for seed in cfg['seeds']]:
                target = directory/'scores'/identity/f'{backend}-{seed}.json'
                if not target.exists():
                    yield str(target), (directory, case, arm, backend, seed, cfg)


def evaluate_one(directory, case, arm, backend, seed, cfg):
    # CPU-only imports. The model/answers are never passed to GPU requests.
    from qihc.experiment_runtime.evaluation import evaluate
    from qihc.generic.bounded_reference import solve as exact
    from qihc.generic.distribution.search import solve
    identity = s.key(case['id'], arm)
    response = directory/'responses'/(identity+'.json')
    req = directory/'requests'/(identity+'.json')
    row = s.load(response)
    if row['request_sha256'] != s.digest(req):
        raise ValueError('Response/request hash mismatch')
    result = {'case_id': case['id'], 'dataset': case['dataset'], 'arm': arm,
              'backend': backend, 'seed': seed, 'response_sha256': s.digest(response),
              'status': 'invalid_model', 'verified_feasible': False, 'seconds': 0.0}
    try:
        if row.get('status') != 'generated':
            raise ValueError(row.get('validation_error', row.get('status')))
        inspector = typed_sources.inspect if ARMS[arm][1].startswith('typed') else protocol.inspect
        spec, reason, audits = inspector(row['response'], case['question'])
        if spec is None:
            raise ValueError(reason)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result['error'] = str(exc)
    else:
        def search(model):
            return (exact(model, cfg['solver_seconds']) if backend == 'cp_sat' else
                    solve(model, seed=seed, chains=cfg['chains'], budget=cfg['budget'],
                          method='pbit', interventions=False, selection='fixed'))
        raw = search(spec)
        verified = evaluate(case, spec, raw)
        result.update(verified)
        result.update(status='searched', backend=backend, engine_backend=raw.get('backend'),
                      solver_status=raw.get('status'), audits=audits,
                      structural_match=(spec == case['gold']) if case.get('gold') else None)
        result['failure_attribution'] = failure_attribution(case, spec, verified)
        if row.get('initial_response'):
            initial, _ = typed_sources.compile_math(row['initial_response'], case['question'])
            if initial == spec:
                initial_result = dict(verified)
                extra_seconds = 0.0
            else:
                initial_result = evaluate(case, initial, search(initial))
                extra_seconds = initial_result.get('seconds', 0.0)
            result['initial_evaluation'] = initial_result
            result['diagnostic_extra_search_seconds'] = extra_seconds
            result['review_transition'] = ('improved' if result['verified_feasible'] and not initial_result['verified_feasible']
                else 'regressed' if initial_result['verified_feasible'] and not result['verified_feasible'] else 'unchanged')
        if result.get('verified_feasible') and case.get('reference_answer') is not None:
            ref = float(case['reference_answer'])
            sign = 1 if case['gold']['objective']['sense'] == 'min' else -1
            result['reference_gap'] = sign*(result['verified_objective']-ref)/max(abs(ref), 1.0)
    s.save(directory/'scores'/identity/f'{backend}-{seed}.json', result)
    return result


def failure_attribution(case, spec, result):
    """CPU-only witnesses, never returned to the semantic reviewer."""
    from qihc.generic.bounded import objective, violations
    if result.get('verified_feasible'):
        status = 'original_feasible_semantics_not_proven'
    elif result.get('predicted_model_feasible'):
        status = 'predicted_feasible_original_infeasible_witness'
    else:
        status = 'no_original_feasible_solution_search_or_model_unresolved'
    out = {'category': status}
    if result.get('x') is not None:
        try:
            values = {v['name']: x for v, x in zip(spec['variables'], result['x'])}
            x = [values[v['name']] for v in case['gold']['variables']]
            residual = violations(case['gold'], x).tolist()
            labels = ['domains']+[c['id'] for c in case['gold']['constraints']+case['gold']['logic']]
            out['original_violations'] = [{'id': k, 'magnitude': v} for k, v in zip(labels, residual) if v > 0]
            out['original_objective_at_candidate'] = float(objective(case['gold'], x))
            out['predicted_objective_at_candidate'] = float(objective(spec, result['x']))
        except (ValueError, KeyError, TypeError):
            out['mapping_error'] = True
    return out
