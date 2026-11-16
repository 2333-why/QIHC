"""Offline inference with frozen E005 checkpoints and CPU scoring against original models."""
import argparse
import json
import os
import time
from pathlib import Path

from qihc.experiment_runtime.pbit_control import seed
from qihc.experiment_runtime.storage import digest, load, save, worker_lock
from qihc.generic.bounded import objective, violations
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.pbit_joint_v1.search import episode
from .prepare import SEARCH

BASE = Path('/inspire/hdd/global_user/czxs25210022/why')
ARMS = ('sft', 'weighted')


def infer(out, donor):
    import torch
    from experiments.posttraining_v1.engine import Engine
    from experiments.posttraining_v1.learn import parsed

    manifest = load(donor/'manifest.json')
    if manifest['experiment_id'] != 'E005-1.14':
        raise ValueError('Expected E005-1.14 donor')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Inference requires a BF16-capable GPU')
    public = load(out/'public.json')
    engine = Engine(Path(manifest['model']))
    for arm, path in zip(ARMS, (Path(manifest['adapter']), donor/'trained/weighted/adapter.pt')):
        folder = out/'predictions'/arm
        with worker_lock(folder/'infer.lock'):
            fingerprint = dict(adapter=digest(path), donor=digest(donor/'manifest.json'), cases=digest(out/'public.json'))
            if (folder/'identity.json').exists() and load(folder/'identity.json') != fingerprint:
                raise ValueError('Inputs changed. Use a fresh result directory.')
            save(folder/'identity.json', fingerprint)
            engine.restore(torch.load(path, map_location='cpu', weights_only=True))
            for case in public:
                cid = case['id']
                dest = folder/(cid + '.json')
                if dest.exists():
                    continue
                tick = time.monotonic()
                response = engine.evaluate(model_prompt(case), seed(cid, 'model') & 0x7fffffff)
                choice = response['choices'][0]
                spec, error = (parsed(choice['message']['content'], case)
                               if choice['finish_reason'] == 'stop' else (None, choice['finish_reason']))
                save(dest, dict(model=spec, error=error, seconds=time.monotonic()-tick))


def score(out):
    refs = load(out/'reference.json')
    report = {}
    audit_rows = {r['id']: r for r in load(out/'audit.json')['rows']}
    for arm in ARMS:
        rows = []
        for ref in refs:
            cid = ref['id']
            prediction = load(out/'predictions'/arm/(cid + '.json'))
            spec = prediction['model']
            row = dict(id=cid, example=audit_rows[cid]['example'], valid=spec is not None,
                       feasible=False, hit=False, gap=None, error=prediction['error'])
            if spec is not None:
                ep = episode(spec, cid, SEARCH, lambda *_: 'cool')
                x, gold = ep['x'], ref['gold']
                row['x'] = x
                row['feasible'] = not bool(violations(gold, x).sum())
                if row['feasible']:
                    obj = float(objective(gold, x))
                    row.update(objective=obj, hit=obj == ref['optimum'],
                               gap=abs(obj-ref['optimum']) / max(1, abs(ref['optimum'])))
            rows.append(row)
        report[arm] = dict(completed=len(rows), valid=sum(r['valid'] for r in rows),
                           feasible=sum(r['feasible'] for r in rows), hit=sum(r['hit'] for r in rows), rows=rows)
    save(out/'frozen_report.json', report)
    print(json.dumps({arm: {k: v for k, v in r.items() if k != 'rows'} for arm, r in report.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('infer', 'score'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--donor', type=Path, default=BASE/'results/E005-1.14_four')
    args = parser.parse_args()
    if args.stage == 'infer':
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        infer(args.out, args.donor)
    else:
        score(args.out)
