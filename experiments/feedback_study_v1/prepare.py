"""Read-only E003 import plus isolated deterministic E004 data/preflight."""
import argparse
import copy
import json
import os
from pathlib import Path
from qihc.experiment_runtime.identity import model_identity
from qihc.generic.bounded import validate as validate_math
from .common import HERE, BASE, s, immutable, fingerprint, commit, default_root, versions
from .data import extension, development


def prepare(root, source, model, cfg=None, smoke=False):
    if root.resolve() == source.resolve() or source.resolve() in root.resolve().parents:
        raise ValueError('E004 must be outside the E003 source tree')
    cfg = copy.deepcopy(cfg or s.load(HERE/'config.json'))
    src = s.load(source/'manifest.json')
    path = source/'cases.CPU_ONLY.json'; cases = s.load(path)
    if src.get('cases_digest') != s.value_digest(cases):
        raise ValueError('E003 source case hash mismatch')
    if not smoke and len(cases) != 900:
        raise ValueError('Formal E004 requires all 900 E003 cases, including invalid initial models')
    public = []; initial = {}; references = {}
    for case in cases:
        p = source/'models'/(s.key(case['id'], 'llm_typed_flat')+'.json')
        frozen = s.load(p)
        if frozen.get('model') is not None:
            validate_math(frozen['model'])
            if s.value_digest(frozen['model']) != frozen['model_sha256']:
                raise ValueError('Frozen model hash mismatch: '+case['id'])
        case = copy.deepcopy(case); case['split'] = 'public'; case['size'] = len(case['gold']['variables'])
        case['variable_names'] = [v['name'] for v in case['gold']['variables']]
        public.append(case)
        initial[case['id']] = {'model': frozen.get('model'), 'error': frozen.get('error'),
                              'source_record_sha256': s.digest(p), 'generation_cost': frozen.get('generation_cost', {})}
        ref = source/'references'/(s.key(case['id'])+'.json')
        if ref.exists():
            references[case['id']] = s.load(ref)
            if references[case['id']].get('gold_sha256') != s.value_digest(case['gold']):
                raise ValueError('E003 reference/model identity mismatch')
    cases = development(cfg)+public+extension(cfg)
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    for c in cases:
        validate_math(c['gold'])
    # Exact mathematics duplicates across development/formal splits are rejected.
    groups = {}
    for c in cases:
        digest = s.value_digest(c['gold'])
        group = 'development' if c['split'].startswith('development') else 'formal'
        if digest in groups and groups[digest] != group:
            raise ValueError('Mathematical instance leakage between development and formal data')
        groups[digest] = group
    sanitized = [{k: v for k,v in c.items() if k in ('id','dataset','size','split','question','variable_names','generation_seed','diagnostic_subset')}
                 for c in cases]
    manifest = {'experiment_id': 'E004-1.1', 'config': cfg, 'source_hash': fingerprint(),
                'code_commit': commit(), 'source_run': str(source.resolve()),
                'source_manifest_sha256': s.digest(source/'manifest.json'),
                'model': str(model.resolve()), 'model_files': model_identity(model) if not smoke else [],
                'dependencies':versions() if not smoke else {},
                'config_digest': s.value_digest(cfg), 'cases': len(cases), 'smoke': smoke,
                'scope': 'No gold/reference enters feedback. E003 immutable read-only source.'}
    with s.worker_lock(root/'prepare.lock'):
        immutable(root/'cases.public.json', sanitized)
        immutable(root/'evaluation'/'cases.CPU_ONLY.json', cases)
        manifest['public_digest'] = s.digest(root/'cases.public.json')
        manifest['evaluation_digest'] = s.digest(root/'evaluation'/'cases.CPU_ONLY.json')
        immutable(root/'manifest.json', manifest)
        for case_id, row in initial.items():
            immutable(root/'initial'/(s.key(case_id)+'.json'), row)
        for case_id, row in references.items():
            immutable(root/'evaluation'/'references'/(s.key(case_id)+'.json'), row)
    return manifest


def preflight(root):
    from transformers import AutoTokenizer
    from types import SimpleNamespace
    from qihc.modeling.formatting import check_runtime, parser, prefix_function
    from .protocol import build_request, schemas
    m = s.load(root/'manifest.json'); print(check_runtime(), flush=True)
    tok = AutoTokenizer.from_pretrained(m['model'], local_files_only=True)
    lengths = []; oversized = []; probes = set()
    for c in s.load(root/'cases.public.json'):
        p = root/'initial'/(s.key(c['id'])+'.json')
        state = dict(s.load(p) if p.exists() else {'model': None}, round=0, arm='U2')
        req = build_request(c, state, 'review')
        text = tok.apply_chat_template([{'role':'user','content':req['prompt']}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False)
        n = len(tok.encode(text, add_special_tokens=False)); lengths.append(n)
        if n > m['config']['max_input_tokens']:
            oversized.append({'case_id':c['id'], 'tokens':n})
        if c['dataset'] in probes:
            continue
        _, review, patch = schemas(c['question'], c['variable_names'])
        for syntax, value in ((review, {'hypotheses':[], 'notes':'ok'}), (patch, {'edits':[], 'quote':'', 'reason':'retain'})):
            parser('flat', syntax)
            fn = prefix_function(SimpleNamespace(tokenizer=tok), prompt_width=0, kind='flat', schema_override=syntax)
            seq = []
            for token in tok.encode(json.dumps(value, separators=(',', ':')), add_special_tokens=False):
                if token not in fn(0, seq):
                    raise RuntimeError('Actual tokenizer/schema probe rejected token')
                seq.append(token)
            if tok.eos_token_id not in fn(0, seq):
                raise RuntimeError('Schema EOS probe failed')
        probes.add(c['dataset'])
    row = {'status':'passed', 'source_hash':m['source_hash'], 'checked':len(lengths),
           'max_review_tokens':max(lengths), 'oversized':oversized, 'schema_groups':sorted(probes),
           'scope':'CPU tokenizer only, real-model execution acceptance remains; oversize cases retained'}
    s.save(root/'preflight.json', row); print(json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, default=default_root())
    p.add_argument('--source', type=Path, default=Path(os.environ.get('QIHC_E003_RUN', BASE/'results/E003-1.1_public_solvers')))
    p.add_argument('--model', type=Path, default=Path(os.environ.get('QIHC_MODEL', BASE/'models/Qwen3-32B')))
    p.add_argument('--gpu-replicas',type=int,choices=[2,4],default=4)
    p.add_argument('--cpu-workers',type=int,default=48)
    a = p.parse_args()
    if not 1<=a.cpu_workers<=64:
        p.error('--cpu-workers must be 1..64')
    cfg=s.load(HERE/'config.json');cfg.update(gpu_replicas=a.gpu_replicas,cpu_workers=a.cpu_workers,
                                          batch_size=1 if a.gpu_replicas==4 else 2)
    prepare(a.run, a.source, a.model,cfg); preflight(a.run)
    print('E004 CPU READY. Online CPU can stop. Launch 01_run_4gpu.sh in one four-GPU worker.', flush=True)


if __name__ == '__main__':
    main()
