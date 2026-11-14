"""E005-1.14. Run CPU preparation, four GPU workers, then CPU scoring."""
import argparse
import importlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from qihc.experiment_runtime import posttrain_screen as runtime
from qihc.experiment_runtime.storage import load, save, digest, value_digest, worker_lock
from qihc.experiment_runtime.pbit_control import seed
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.pbit_joint_v1 import search, modeling, run as v1
from .signals import examples, pair_loss, control_prompt

HERE = Path(__file__).resolve().parent
CFG = load(HERE / 'config.json')
BASE = Path('/inspire/hdd/global_user/czxs25210022/why')
RUN = Path(os.getenv('QIHC_FOUR_RUN', BASE / 'results/E005-1.14_four'))
PREVIOUS = Path(os.getenv('QIHC_CYCLE_RUN', BASE / 'results/E005-1.9_pvcd_cycle'))
MODELS = ('base', 'weighted', 'paired')
CONTROLS = ('schedule', 'rule', 'terminal', 'trajectory')
ARMS = MODELS[1:] + CONTROLS[2:]


def job():
    return importlib.import_module('experiments.pbit_joint_v2.run')


def cases():
    return [c for c in load(RUN/'inputs/cohort.json') if c['stage'] in ('main', 'scale')]


def prepare_case(task):
    row, candidate = task
    cid = row['case']['id']
    path = RUN/'evidence'/f'{cid}.json'
    if path.exists():
        return
    gold = row['gold']
    evidence = modeling.collect(candidate, gold, cid, CFG)
    specs = {value_digest(s): s for s in (candidate, gold) if s is not None}
    samples = [e for spec in specs.values() for e in examples(spec, cid, CFG)]
    save(path, dict(candidate=candidate, cards=evidence['cards'], samples=samples))


def prepare():
    runtime.prepare(job())
    manifest = runtime.validate(job())
    rows = load(RUN/'inputs/train.json')
    tasks = [(r, load(Path(manifest['predictions'][r['case']['id']]))['model']) for r in rows]
    with worker_lock(RUN/'cpu.lock'), ProcessPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(prepare_case, tasks), 1):
            print('prepared', i, '/', len(rows), flush=True)
        files = {r['case']['id']: digest(RUN/'evidence'/(r['case']['id']+'.json')) for r in rows}
        labels = {a: Counter() for a in CONTROLS[2:]}
        gain_cases = {a: set() for a in CONTROLS[2:]}
        differing_states = 0
        for cid in files:
            for sample in load(RUN/'evidence'/f'{cid}.json')['samples']:
                differing_states += sample['labels']['terminal'] != sample['labels']['trajectory']
                for arm in labels:
                    labels[arm][sample['labels'][arm]] += 1
                    if sample['labels'][arm] != 'cool':
                        gain_cases[arm].add(cid)
        diagnostics = dict(action_counts=labels, differing_states=differing_states,
                           noncool_cases={arm:len(ids) for arm,ids in gain_cases.items()})
        save(RUN/'ready.json', dict(files=files, shared_source=shared_hashes(), **diagnostics))
        print('Training labels:', json.dumps(diagnostics), flush=True)


def shared_hashes():
    return {p.name: digest(p) for p in v1.HERE.iterdir() if p.suffix in ('.py','.json')}


def prediction(model, cid, manifest):
    if model == 'base':
        return Path(manifest['predictions'][cid])
    return RUN/'predictions'/model/f'{cid}.json'


def train(engine, arm, manifest, ready):
    import torch
    from experiments.posttraining_v1.data import target
    rows = load(RUN/'inputs/train.json')
    records = {}
    for cid, h in ready['files'].items():
        path = RUN/'evidence'/f'{cid}.json'
        if digest(path) != h:
            raise ValueError(f'Training evidence changed: {cid}')
        records[cid] = load(path)
    if arm in MODELS:
        engine.restore(torch.load(manifest['adapter'], map_location='cpu', weights_only=True))
        count = len(rows)
        def backward(i):
            row = rows[i]
            cid = row['case']['id']
            evidence = records[cid]
            gold = row['gold']
            prompt = model_prompt(row['case'])
            answer = target(gold)
            candidate = evidence['candidate']
            weights = modeling.token_weights(engine.tok, answer, candidate, gold, evidence['cards'])
            preferred = engine.encode(answer) + [engine.tok.eos_token_id]
            loss = engine.backward(prompt, preferred, 'nll', weights=weights+[1.])
            ranking = 0.
            use_pair = bool(arm == 'paired' and evidence['cards'])
            if use_pair:
                rejected = engine.encode(target(candidate)) + [engine.tok.eos_token_id]
                extra = CFG['pair_weight'] * pair_loss(engine, prompt, preferred, rejected)
                extra.backward()
                ranking = float(extra.detach())
            return dict(case_id=cid, loss=loss+ranking, pair=use_pair, ranking=ranking,
                        weighted_tokens=sum(w > 1 for w in weights))
    else:
        samples = [(cid, s) for cid, record in records.items() for s in record['samples']]
        count = len(samples)
        ids = v1.action_ids(engine)
        actions = Counter(s['labels'][arm] for _, s in samples)
        cases_count = Counter(cid for cid, _ in samples)
        weights = np.array([1/(cases_count[cid]*np.sqrt(actions[s['labels'][arm]])) for cid,s in samples])
        weights /= weights.mean()
        def backward(i):
            cid, sample = samples[i]
            action = sample['labels'][arm]
            scores = v1.logits(engine, sample['prompt'], ids)
            label = torch.tensor([search.ACTIONS.index(action)], device=scores.device)
            loss = torch.nn.functional.cross_entropy(scores[None], label)*float(weights[i])
            loss.backward()
            return dict(case_id=cid, action=action, loss=float(loss.detach()))
    # v1 helpers use seed 17 and checkpoint interval 8, fixed for this protocol.
    v1.optimize(engine, RUN/'trained'/arm, v1.order_for(count, CFG['epochs']),
                CFG['model_lr'] if arm in MODELS else CFG['control_lr'], backward,
                dict(arm=arm, evidence=value_digest(ready), manifest=digest(RUN/'manifest.json')))


def trajectory(spec, cid, policy, engine=None, ids=None):
    if spec is None:
        return dict(x=None, actions=[], episode_seconds=0., decision_seconds=0.,
                    search_seconds=0., heat_bath_attempts=0, intervention_writes=0)
    def choose(spec, state, remaining):
        if policy == 'schedule':
            return 'cool'
        if policy == 'rule':
            return search.rule(spec, state)
        import torch
        with torch.inference_mode():
            return search.ACTIONS[int(v1.logits(engine, control_prompt(spec,state,remaining),ids).argmax())]
    return search.episode(spec, cid, CFG, choose)


def tasks():
    for shard in range(CFG['shards']):
        for model in MODELS[1:]:
            yield ('predict', model, model, shard)
        for model in MODELS:
            for policy in CONTROLS[2:]:
                yield ('search', model, policy, shard)


def marker(task):
    op, model, arm, shard = task
    return RUN/'shards'/f'{op}_{model}_{arm}_{shard}.json'


def eligible(task):
    op, model, arm, shard = task
    if not (RUN/'trained'/arm/'adapter.pt').exists():
        return False
    return op == 'predict' or model == 'base' or marker(('predict', model, model, shard)).exists()


def run_shard(task, subset, engine, manifest):
    from experiments.posttraining_v1.learn import parsed
    op, model, arm, shard = task
    engine.net.eval()
    ids = v1.action_ids(engine) if op == 'search' else None
    for case in subset[shard::CFG['shards']]:
        cid = case['id']
        path = prediction(model, cid, manifest)
        if op == 'predict':
            if path.exists():
                continue
            start = time.monotonic()
            response = engine.evaluate(model_prompt(case), seed(cid, 'model') & 0x7fffffff)
            choice = response['choices'][0]
            spec, error = parsed(choice['message']['content'],case) if choice['finish_reason']=='stop' else (None,'length')
            save(path,dict(model=spec,error=error,generation_seconds=time.monotonic()-start))
        else:
            dest = RUN/'episodes'/f'{model}__{arm}'/f'{cid}.json'
            if not dest.exists():
                save(dest,trajectory(load(path)['model'],cid,arm,engine,ids))


def worker(arm):
    import torch
    from experiments.posttraining_v1.engine import Engine
    manifest = runtime.validate(job())
    ready = load(RUN/'ready.json')
    if ready['shared_source'] != shared_hashes():
        raise ValueError('Shared p-bit source changed')
    if torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported() or torch.cuda.get_device_properties(0).total_memory<75*1024**3:
        raise RuntimeError('Each worker requires one visible BF16 80GB GPU')
    torch.manual_seed(CFG['seed'])
    engine = Engine(Path(manifest['model']))
    with worker_lock(RUN/'trained'/arm/'train.lock'):
        train(engine,arm,manifest,ready)
    current = arm
    subset = cases()
    while True:
        pending = [t for t in tasks() if not marker(t).exists()]
        if not pending:
            return
        worked = False
        for task in sorted(pending, key=lambda t: t[2] != current):
            if not eligible(task):
                continue
            path = marker(task)
            try:
                with worker_lock(path.with_suffix('.lock')):
                    if path.exists():
                        continue
                    if current != task[2]:
                        current = task[2]
                        engine.restore(torch.load(RUN/'trained'/current/'adapter.pt',map_location='cpu',weights_only=True))
                    print('working', task, flush=True)
                    run_shard(task,subset,engine,manifest)
                    save(path,dict(task=task))
                    worked = True
                break
            except BlockingIOError:
                continue
        if not worked:
            time.sleep(3)


def gpu():
    import torch
    if torch.cuda.device_count()!=4:
        raise RuntimeError('Request four visible GPUs')
    ids = os.getenv('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    if len(ids)!=4 or len(set(ids))!=4:
        raise ValueError('Invalid visible GPU list')
    # Historical launcher logs failures and terminates sibling process groups.
    from subprocess import Popen, STDOUT, run, TimeoutExpired
    from experiments.pvcd_cycle_v1.launch import stop_process
    live = []
    with worker_lock(RUN/'gpu.lock'):
        try:
            for arm, device in zip(ARMS, ids):
                path = RUN/'logs'/f'{arm}.log'
                path.parent.mkdir(parents=True, exist_ok=True)
                log = path.open('a')
                proc = Popen([sys.executable,'-m',job().__name__,'worker',arm],stdout=log,stderr=STDOUT,
                             env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),start_new_session=True)
                live.append((proc,log,path))
            next_sample = time.monotonic()
            while any(proc.poll() is None for proc,_,_ in live):
                for proc,_,path in live:
                    if proc.poll() not in (None,0):
                        with path.open('rb') as tail:
                            tail.seek(max(0,path.stat().st_size-4096))
                            raise RuntimeError(f'{path}: {proc.returncode}\n'+tail.read().decode(errors='replace'))
                if time.monotonic() >= next_sample:
                    try:
                        sample = run(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,memory.used',
                                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                        if sample.returncode == 0:
                            with (RUN/'logs/gpu_usage.csv').open('a') as out:
                                out.write(sample.stdout)
                    except (OSError,TimeoutExpired):
                        pass
                    next_sample = time.monotonic()+60
                time.sleep(2)
            if any(proc.returncode for proc,_,_ in live):
                raise RuntimeError('Worker failed; inspect '+str(RUN/'logs'))
        finally:
            for proc,log,_ in live:
                stop_process(proc)
                log.close()


def main():
    from .report import report, score_all
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','gpu','worker','score','report'])
    parser.add_argument('arm',nargs='?',choices=ARMS)
    args = parser.parse_args()
    if args.stage == 'worker':
        if args.arm is None:
            parser.error('worker requires a training arm')
        worker(args.arm)
    else:
        dict(prepare=prepare,gpu=gpu,score=score_all,report=report)[args.stage]()


if __name__=='__main__':
    main()
