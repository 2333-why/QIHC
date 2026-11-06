"""Streaming DAG: independent cases overlap LLM, feedback and offline scoring."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import signal
import time

from qihc.experiment_runtime.storage import load, save, key, value_digest, worker_lock
from qihc.experiment_runtime.async_inference import InferencePool
from qihc.experiment_runtime.isolated_jobs import JobPool
from qihc.modeling.active_protocol import model_prompt, source_prompt, patch_prompt, parse_model, parse_checks, guard
from .data import public_view

HERE = Path(__file__).parent


def seed_for(*values):
    return int(key(*map(str, values))[:8], 16) % (2**31 - 1)


def schedule(cases, subsets, stage, cfg):
    arms, seeds = cfg['arms'], cfg['seeds']
    if stage == 'development': arms, seeds = cfg['arms'], [17]
    elif stage == 'ablation': arms, seeds = ['P0', 'P1', 'PR'], cfg['extension_seeds']
    elif stage == 'scale': arms, seeds = ['C0', 'X', 'R', 'P2'], cfg['extension_seeds']
    elif stage == 'replay': arms, seeds = ['replay_P2', 'replay_R'], [17]
    elif stage == 'public': arms, seeds = ['C0', 'P2', 'R', 'B'], cfg['extension_seeds']
    else:
        if stage != 'main': raise ValueError('Unsupported automatic stage: ' + stage)
    chosen = [c for c in cases if (c['id'] in subsets[stage] if stage in ('ablation', 'replay') else c['split'] == stage)]
    rows = [dict(case_id=c['id'], arm=arm, seed=s, stage=stage) for c in chosen for arm in arms for s in ([0] if arm == 'C0' and stage != 'public' else seeds)]
    return chosen, rows


def public_cases(data):
    cases = load(data / 'cases.public.json')
    extra = data / 'cases.public_external.json'
    if extra.exists(): cases += load(extra)
    return cases


class Pipeline:
    def __init__(self, root, data, cfg, llm, cpus):
        self.root = root; self.data = data; self.cfg = cfg; self.llm = llm; self.cpus = cpus
        from qihc.modeling import active_protocol, active_protocol_v2
        self.protocol = active_protocol_v2 if cfg.get('protocol_version')=='v2' else active_protocol
        self.initials = {}; self.case_slots = asyncio.Semaphore(cfg['active_trajectories'])

    async def initialize(self, c, meta=None):
        cid = c['id'] if meta is None else key('public', c['id'], meta['arm'], str(meta['seed']))
        if cid not in self.initials: self.initials[cid] = asyncio.create_task(self._initialize(c, meta))
        return await self.initials[cid]

    async def _initialize(self, c, meta=None):
        identity = c['id'] if meta is None else key('public', c['id'], meta['arm'], str(meta['seed']))
        case = public_view(c); initial_key = key(identity); path = self.root / 'initial' / (initial_key + '.json')
        if path.exists(): return load(path)
        prompt = self.protocol.model_prompt(case)
        if meta and meta['arm'] == 'R': prompt = 'Internally understand, formalize, synthesize, and verify completeness, then emit only the requested JSON.\n' + prompt
        source = dict(status='complete', text='{"checks":[]}')
        async def source_call():
            if meta and meta['arm'] == 'C0': return source
            return await self.llm.generate(key('sources', identity), self.protocol.source_prompt(case), seed_for(c['id'], meta['seed'] if meta else 0, 'sources'), 0)
        responses = await asyncio.gather(
            self.llm.generate(key('initial', identity), prompt, seed_for(c['id'], meta['seed'] if meta else 0, 'initial'), 0), source_call())
        row = dict(case_id=c['id'], model=None, checks={'checks': [], 'rejected': []}, error=None, source_error=None,
                   initial_key=initial_key, initial_request=key('initial', identity), source_request=key('sources', identity),
                   workflow_scope='budget_matched_IR_adaptation_not_native_paper_reproduction' if meta else 'shared_initial_model')
        try:
            if responses[0]['status'] != 'complete': raise ValueError(responses[0]['status'])
            row['model'] = self.protocol.parse_model(responses[0]['text'], case)
        except (ValueError, TypeError, KeyError) as exc: row['error'] = str(exc)
        try:
            if responses[1]['status'] != 'complete': raise ValueError(responses[1]['status'])
            row['checks'] = self.protocol.parse_checks(responses[1]['text'], case)
        except (ValueError, TypeError, KeyError) as exc: row['source_error'] = str(exc)
        save(path, row); return row

    def feedback_view(self, result):
        """Enforce card token cap using the actual model tokenizer, not chars."""
        result = dict(cards=copy.deepcopy(result.get('cards', [])), status=result.get('status'),
                      abstained=result.get('abstained'), method_scope=result.get('method_scope'))
        for card in result['cards']:
            x = card.get('x')
            if isinstance(x,list) and all(v in (0,1) for v in x):
                card['assignment_bits'] = ''.join(map(str,x)); card['mapping'] = 'character i is xi'; card.pop('x')
        def length():
            text = json.dumps(result, separators=(',', ':'))
            return len(self.llm.tokenizer.encode(text)) if self.llm.tokenizer else len(text.encode())
        while result['cards'] and length() > self.cfg['card_tokens']: result['cards'].pop()
        return result

    async def trajectory(self, meta, c):
        tid = key(meta['stage'], c['id'], meta['arm'], str(meta['seed']))
        path = self.root / 'trajectories' / (tid + '.json')
        if path.exists() and load(path).get('status') == 'complete': return
        async with self.case_slots:
            initial = await self.initialize(c, meta if meta['stage'] == 'public' else None); case = public_view(c)
            state = load(path) if path.exists() else dict(meta, id=tid, status='running', model=initial['model'],
                initial_key=initial.get('initial_key', key(c['id'])), initial_hash=value_digest(initial['model']), rounds=[], error=initial['error'])
            if meta['arm'] != 'C0':
                for round_no in range(len(state['rounds']), self.cfg['rounds']):
                    if meta['stage'] == 'replay':
                        source_id = key('main', c['id'], meta['arm'].removeprefix('replay_'), str(meta['seed']))
                        source = load(self.root / 'trajectories' / (source_id + '.json'))
                        probe = source['rounds'][round_no]['probe']; evidence = source['rounds'][round_no]['presented_evidence']
                    else:
                        probe = await self.cpus.run(dict(stage='probe', model=state['model'], checks=initial['checks'],
                            arm=meta['arm'], config=self.cfg, seed=seed_for(c['id'], meta['seed'], round_no, 'probe')),
                            self.cfg['probe_seconds'] + 30)
                        evidence = self.feedback_view(probe)
                    prompt = self.protocol.patch_prompt(case, state['model'], initial['checks'], evidence, state.get('error'))
                    request_key = key('patch', tid, str(round_no))
                    response = await self.llm.generate(request_key, prompt, seed_for(c['id'], meta['seed'], round_no, 'decode'))
                    if response['status'] == 'complete': decision = self.protocol.guard(case, state['model'], response['text'])
                    else: decision = dict(status='rejected', model=state['model'], error=response['status'], candidate=None)
                    state['model'] = decision['model']; state['error'] = decision.get('error')
                    state['rounds'].append(dict(round=round_no, probe=probe, presented_evidence=evidence,
                        request_key=request_key, guard={k: v for k, v in decision.items() if k != 'model'},
                        resulting_hash=value_digest(state['model'])))
                    save(path, state)
            state.update(status='complete', final_hash=value_digest(state['model']), completed_at=time.time())
            save(path, state)

    async def run(self, stage):
        # Scheduling and inference consume only the public dataset.
        cases = public_cases(self.data); subsets = load(self.data / 'subsets.json')
        chosen, rows = schedule(cases, subsets, stage, self.cfg)
        index = dict(stage=stage, expected=len(rows), cases=[c['id'] for c in chosen], trajectories=rows)
        p = self.root / 'stages' / (stage + '.json')
        if p.exists() and load(p) != index: raise ValueError('Stage cohort changed during resume')
        save(p, index); byid = {c['id']: c for c in chosen}
        # Prefill can continue while already initialized cases produce CPU work.
        init_slots = asyncio.Semaphore(self.cfg['active_trajectories'])
        async def initialize(c):
            async with init_slots: await self.initialize(c)
        initial_tasks = [] if stage == 'public' else [asyncio.create_task(initialize(c)) for c in chosen]
        try:
            await asyncio.gather(*(self.trajectory(m, byid[m['case_id']]) for m in rows), *initial_tasks)
        finally:
            from .report import report
            report(self.root)


async def main_async(args):
    cfg = load(args.config)
    if args.cpu_workers: cfg['cpu_workers'] = args.cpu_workers
    if args.concurrency: cfg['requests_per_gpu'] = args.concurrency
    if args.smoke:
        cfg.update(probe_seconds=.08, solver_seconds=.08, rounds=1, seeds=[17], extension_seeds=[17], active_trajectories=8)
    args.run.mkdir(parents=True, exist_ok=True)
    public_manifest = args.data / 'public.manifest.json'
    from .identity import source_identity
    manifest = dict(config=cfg, data_manifest=load(args.data / 'manifest.json'), model_identity=args.model_identity, smoke=args.smoke,
                    source_identity=source_identity(), second_model_enabled=False,
                    public_manifest=load(public_manifest) if public_manifest.exists() else None)
    if not manifest['data_manifest'].get('formal_public_only'):
        raise ValueError('Runtime accepts public-only formal datasets; synthetic fixtures belong in tests')
    frozen = args.run / 'manifest.json'
    if frozen.exists() and load(frozen) != manifest: raise ValueError('Run configuration/model/data changed; choose a new run directory')
    save(frozen, manifest)
    # A previous stage completion marker must not terminate a new watch evaluator.
    (args.run / 'inference_complete.json').unlink(missing_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    llm = InferencePool(args.endpoints.split(','), args.run / 'requests', cfg, tokenizer=tokenizer)
    cpus = JobPool(args.run / 'jobs', cfg['cpu_workers'], 'experiments.feedback_active_v3.cpu')
    pipeline = Pipeline(args.run, args.data, cfg, llm, cpus)
    async def heartbeat():
        from .report import report
        while True:
            await asyncio.sleep(cfg['report_seconds']); report(args.run)
    beat = asyncio.create_task(heartbeat())
    try:
        all_cases = public_cases(args.data); subsets = load(args.data / 'subsets.json')
        for stage in args.stages.split(','):
            chosen, rows = schedule(all_cases, subsets, stage, cfg)
            index = dict(stage=stage, expected=len(rows), cases=[c['id'] for c in chosen], trajectories=rows)
            path = args.run / 'stages' / (stage + '.json')
            if path.exists() and load(path) != index: raise ValueError('Registered stage changed')
            save(path, index)
        for stage in args.stages.split(','):
            await pipeline.run(stage)
            if stage=='development' and cfg.get('development_gate'):
                from experiments.feedback_active_v4.gate import check
                check(args.run, args.data, cfg)
        save(args.run / 'inference_complete.json', dict(stages=args.stages.split(','), completed_at=time.time()))
    finally:
        beat.cancel(); await llm.close()
        from .report import report
        report(args.run)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--data', type=Path, default=HERE / 'assets/public_dataset')
    p.add_argument('--config', type=Path, default=HERE / 'config.json'); p.add_argument('--model', required=True)
    p.add_argument('--model-identity', required=True); p.add_argument('--stages', default='development,main,ablation,scale')
    p.add_argument('--endpoints', default='http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103')
    p.add_argument('--cpu-workers', type=int); p.add_argument('--concurrency', type=int); p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    def interrupted(*_): raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM, interrupted)
    with worker_lock(args.run / 'coordinator.lock'): asyncio.run(main_async(args))


if __name__ == '__main__': main()
