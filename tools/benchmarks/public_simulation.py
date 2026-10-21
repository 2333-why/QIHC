#!/usr/bin/env python3
"""Separated GPU inference / CPU search for quarantined public optimization tests."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qihc.generic.public_modeling import prompt, parse, search_response, review_prompt, answer_agreement
from tools.benchmarks.download_public import audit, atomic_write, json_bytes, LOCK


def read(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate ids: {path}')
    return rows


def write(path, rows):
    atomic_write(path, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode('utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hash():
    h = hashlib.sha256()
    files = list((ROOT/'qihc/generic').glob('*.py')) + [Path(__file__), ROOT/'qihc/problems/cvrp/direct_llm.py']
    for p in sorted(files):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def model_identity(directory):
    """Local metadata fingerprint; weight content hashes remain the preparation manifest's job."""
    if not directory.is_dir():
        raise ValueError(f'Missing local model/adapter: {directory}')
    files = []
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.jinja'):
            stat = p.stat()
            files.append({'path': p.relative_to(directory).as_posix(), 'bytes': stat.st_size,
                          'mtime_ns': stat.st_mtime_ns,
                          'sha256': digest(p) if p.suffix in ('.json', '.jinja') else None})
    if not files:
        raise ValueError('Empty model/adapter directory')
    return files


def seal(path, config):
    meta = path.with_name(path.name+'.run.json')
    config = dict(config, source_sha256=source_hash())
    if meta.exists():
        if json.loads(meta.read_text()) != config:
            raise ValueError('Resume inputs/settings/source changed; use a new output path')
    else:
        if path.exists():
            raise ValueError('Unsealed existing output; refusing to reuse')
        atomic_write(meta, json_bytes(config))


def prepare(args):
    lock = json.loads(LOCK.read_text())
    report, questions, answers = audit(lock, args.data/'raw')
    if args.per_dataset < 0:
        raise ValueError('per-dataset must be nonnegative; 0 means all')
    selected = []
    for dataset in sorted({r['dataset'] for r in questions}):
        rows = [r for r in questions if r['dataset'] == dataset]
        count = min(args.per_dataset or len(rows), len(rows))
        indices = [0] if count == 1 else [i*(len(rows)-1)//(count-1) for i in range(count)]
        selected.extend(rows[i] for i in indices)
    # No eligibility/success/answer-based filtering; coverage failures remain in denominator.
    requests = [{'id': r['id'], 'dataset': r['dataset'], 'question': r['question'],
                 'prompt': prompt(r['question']), 'round': 0, 'branch': 'base'} for r in selected]
    if args.output.exists():
        raise ValueError('Preparation output must be a new directory')
    args.output.mkdir(parents=True)
    ids = {r['id'] for r in selected}
    write(args.output/'requests.jsonl', requests)
    write(args.output/'answers.DO_NOT_PROMPT.jsonl', [r for r in answers if r['id'] in ids])
    atomic_write(args.output/'selection.json', json_bytes({
        'source': lock, 'selected': len(requests), 'available': report['total_rows'],
        'selection': 'deterministic spread within every dataset; no outcome filtering',
        'per_dataset': args.per_dataset, 'scope': 'external_test_only',
        'selected_unique_questions': len({r['question_sha256'] for r in selected}),
        'semantic_validation': 'pending_independent_review'}))
    print(json.dumps({'requests': len(requests), 'output': str(args.output)}))


def requests(path):
    rows = read(path)
    expected = {'id', 'dataset', 'question', 'prompt', 'round', 'branch'}
    if any(set(r) != expected for r in rows):
        raise ValueError('Request file must contain only model-visible request fields')
    return rows


def infer(args):
    rows = requests(args.requests)
    seal(args.output, {'stage': 'infer', 'requests': digest(args.requests),
                       'model': str(args.model.resolve()), 'adapter': str(args.adapter.resolve()) if args.adapter else None,
                       'model_files': model_identity(args.model),
                       'adapter_files': model_identity(args.adapter) if args.adapter else None,
                       'context': args.context, 'tokens': args.tokens})
    saved = read(args.output) if args.output.exists() else []
    ids = {r['id'] for r in rows}
    if any(r['id'] not in ids for r in saved):
        raise ValueError('Unexpected checkpoint ID')
    done = {r['id'] for r in saved}
    if done == ids:
        print('Inference already complete'); return
    from qihc.problems.cvrp.direct_llm import LocalDirectLLMGenerator
    gen = LocalDirectLLMGenerator(str(args.model), device='auto', temperature=0,
                                 max_input_tokens=args.context, max_new_tokens=args.tokens)
    if args.adapter:
        from peft import PeftModel
        gen.model = PeftModel.from_pretrained(gen.model, str(args.adapter), local_files_only=True)
        gen.model.eval()
    for row in rows:
        if row['id'] in done:
            continue
        print('GENERATE '+row['id'], flush=True)
        started = time.perf_counter()
        try:
            response = gen.generate(row['prompt'])
            result = {'id': row['id'], 'response': response, 'status': 'generated', **gen.last_usage}
        except ValueError as exc:
            # Explicit oversized context etc.; never silently truncate the NL problem.
            result = {'id': row['id'], 'response': '', 'status': 'generation_error', 'error': str(exc)}
        result['seconds'] = time.perf_counter()-started
        saved.append(result)
        write(args.output, saved)


def score(args):
    rows = requests(args.requests)
    predictions = {r['id']: r for r in read(args.predictions)}
    if set(predictions) != {r['id'] for r in rows}:
        raise ValueError('Prediction set incomplete/extra IDs')
    if not 1 <= args.seeds <= 20:
        raise ValueError('seeds must be 1..20')
    import math
    timeout = getattr(args, 'reference_seconds', 0)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('reference-seconds must be finite and nonnegative')
    labels = {r['id']: r['reference_answer'] for r in read(args.answers)} if args.answers else {}
    if args.answers and set(labels) != set(predictions):
        raise ValueError('Reference IDs differ from requests')
    seal(args.output, {'stage': 'score', 'requests': digest(args.requests),
                       'predictions': digest(args.predictions), 'answers': digest(args.answers) if args.answers else None,
                       'steps': args.steps, 'chains': args.chains, 'seeds': args.seeds,
                       'reference_seconds': getattr(args, 'reference_seconds', 0)})
    saved = read(args.output) if args.output.exists() else []
    done = {r['id'] for r in saved}
    if not done <= set(predictions):
        raise ValueError('Unexpected checkpoint ID')
    for row in rows:
        if row['id'] in done:
            continue
        pred = predictions[row['id']]
        result = search_response(row['question'], pred['response'], steps=args.steps,
                                 chains=args.chains, seeds=tuple(range(args.seeds)))
        result.update(id=row['id'], dataset=row['dataset'], round=row['round'], branch=row['branch'])
        result['generation'] = {k: v for k, v in pred.items() if k not in ('response', 'id')}
        if result['status'] == 'searched' and getattr(args, 'reference_seconds', 0) > 0:
            from qihc.generic.bounded_reference import solve as reference
            spec, _ = parse(pred['response'], row['question'])
            result['cp_sat'] = reference(spec, seconds=args.reference_seconds)
            # Never feed the reference solution back to only one experimental arm.
        # References enter scoring only, not the semantic feedback diagnostic.
        for run in result['runs']:
            run['reference_answer_agreement'] = answer_agreement(run, labels[row['id']]) if row['id'] in labels else None
        saved.append(result)
        write(args.output, saved)
        print(f"SCORED {row['id']} {result['status']}", flush=True)
    indexed = {r['id']: r for r in saved}
    for feedback, name in [(True, 'feedback'), (False, 'self_review')]:
        next_rows = [{**r, 'round': r['round']+1, 'branch': name,
                      'prompt': review_prompt(r['question'], predictions[r['id']]['response'],
                                              indexed[r['id']], with_search_feedback=feedback)} for r in rows]
        write(args.output.with_name(args.output.stem+'.'+name+'.requests.jsonl'), next_rows)
    summary = {'scope': 'public_answer_agreement_not_verified_semantic_accuracy',
               'cases': len(rows), 'searched': sum(r['status'] == 'searched' for r in saved),
               'invalid': sum(r['status'] == 'invalid_model' for r in saved),
               'unsupported': sum(r['status'] == 'unsupported' for r in saved),
               'generation_seconds': sum(r['generation'].get('seconds', 0) for r in saved),
               'generation_input_tokens': sum(r['generation'].get('input_tokens', 0) for r in saved),
               'generation_output_tokens': sum(r['generation'].get('output_tokens', 0) for r in saved),
               'cp_sat_optimal_predicted_models': sum(r.get('cp_sat', {}).get('optimality_proven', False) for r in saved),
               'numeric_reference_cases': sum(answer_agreement({'feasible': True, 'objective': 0}, a) is not None for a in labels.values()),
               'arms': {}}
    for arm in ('pbit_feedback', 'pbit_no_feedback', 'metropolis', 'uniform'):
        by_case = [[v for v in r['runs'] if v['arm'] == arm] for r in saved]
        summary['arms'][arm] = {
            'predicted_model_feasible_rate_all_cases': sum(sum(v['feasible'] for v in vs)/len(vs) if vs else 0 for vs in by_case)/max(1, len(rows)),
            'reference_agreement_rate_all_cases': sum(sum(v['reference_answer_agreement'] is True for v in vs)/len(vs) if vs else 0 for vs in by_case)/max(1, len(rows)) if labels else None,
            'search_seconds': sum(v['seconds'] for vs in by_case for v in vs)}
    atomic_write(args.output.with_name(args.output.stem+'.summary.json'), json_bytes(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data', type=Path, required=True, help='Verified revision directory containing raw/')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--per-dataset', type=int, default=40, help='0=all; default 40 per dataset, deterministic spread')
    p = sub.add_parser('infer')
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--context', type=int, default=16384)
    p.add_argument('--tokens', type=int, default=8192)
    p = sub.add_parser('score')
    for name in ('requests', 'predictions', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--answers', type=Path)
    p.add_argument('--steps', type=int, default=4096)
    p.add_argument('--chains', type=int, default=16)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--reference-seconds', type=float, default=5, help='CP-SAT on same predicted model; 0 disables')
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'score': score}[args.stage](args)


if __name__ == '__main__':
    main()
