"""Read-only paired statistics and ETA; no solver/inference/run-manifest mutations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.benchmarks.distribution.experiment import ARMS
from tools.benchmarks.comparison import key, worker_lock
from tools.benchmarks.download_public import atomic_write, json_bytes
from tools.analysis.distribution_feedback.statistics import compare
from tools.analysis.distribution_feedback.eta import estimate

STOP = False


class Reader:
    """Cache immutable checkpoint projections; repeated watching avoids reparsing."""
    def __init__(self):
        self.cache = {}

    def read(self, path, kind='plain'):
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cached = self.cache.get((path, kind))
        if cached is not None and cached[0] == signature:
            return cached[1]
        data = path.read_bytes()
        value = json.loads(data.decode('utf-8'))
        if kind == 'request':
            value = {k: value.get(k) for k in ('id', 'case_id', 'phase')}
            value['_sha256'] = hashlib.sha256(data).hexdigest()
        elif kind == 'response':
            if not isinstance(value.get('response'), str):
                raise ValueError('Invalid response text: '+str(path))
            value['_response_text_hash'] = hashlib.sha256(value['response'].encode()).hexdigest()
            value.pop('response')
        elif kind == 'record':
            if not isinstance(value.get('runs'), list) or len(value['runs']) > 1:
                raise ValueError('Expected at most one run per seed checkpoint: '+str(path))
            value = {k: value.get(k) for k in ('arm', 'seed', 'status', 'input_response_sha256', 'runs')}
            value['runs'] = [{k: r.get(k) for k in ('verified_feasible', 'reference_answer_agreement',
                                'verified_objective', 'seconds', 'candidate_evaluations')} for r in value['runs']]
        if kind != 'plain':
            value['_mtime'] = stat.st_mtime
        self.cache[(path, kind)] = (signature, value)
        return value


def load_snapshot(run, reader=None):
    reader = reader or Reader()
    started = time.time()
    manifest = reader.read(run/'manifest.json')
    if manifest.get('experiment') != 'distribution_feedback_v1':
        raise ValueError('This reporter only supports distribution_feedback_v1; no migration is performed')
    cfg = manifest['settings']
    if (not cfg['arms'] or len(set(cfg['arms'])) != len(cfg['arms']) or any(a not in ARMS for a in cfg['arms'])
            or type(cfg['seeds']) is not int or not 1 <= cfg['seeds'] <= 20):
        raise ValueError('Invalid sealed arm/seed configuration')
    case_bytes = (run/'cases.CPU_ONLY.json').read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != manifest['cases_sha256']:
        raise ValueError('CPU case file checksum mismatch')
    cases = json.loads(case_bytes)
    if len(cases) != manifest['cases'] or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Case count or IDs mismatch')
    # No need to retain questions, gold coefficients, or solution assignments.
    cases = [{**{k: c.get(k) for k in ('id', 'dataset', 'reference_answer', 'reference_status')},
              'gold': {'objective': {'sense': c['gold']['objective']['sense']}} if c.get('gold') else None} for c in cases]
    requests, responses, complete_stages, tasks = {}, {}, {}, []
    for case in cases:
        cid = case['id']
        for phase in ['model']+['review_'+arm for arm in cfg['arms']]:
            filename = key(cid, phase)+'.json'
            req, resp = run/'requests'/filename, run/'responses'/filename
            if req.exists():
                request = reader.read(req, 'request')
                if (request['case_id'], request['phase'], request['id']) != (cid, phase, key(cid, phase)):
                    raise ValueError('Request identity mismatch: '+str(req))
                requests[(cid, phase)] = request
            if resp.exists():
                row = reader.read(resp, 'response')
                if (cid, phase) not in requests or row.get('request_sha256') != requests[(cid, phase)]['_sha256']:
                    raise ValueError('Response/request checksum mismatch: '+str(resp))
                responses[(cid, phase)] = row
        for arm in cfg['arms']:
            seeds = 1 if ARMS[arm]['method'] == 'cp_sat' else cfg['seeds']
            for stage in ('initial', 'revised'):
                records = []
                phase = 'model' if stage == 'initial' else 'review_'+arm
                for seed in range(seeds):
                    path = run/'progress'/key(cid, 'case')/f'{arm}.{stage}.{seed}.json'
                    record = None
                    if path.exists():
                        record = reader.read(path, 'record')
                        rsp = responses.get((cid, phase))
                        # A producer may have written response then checkpoint
                        # after this snapshot's request pass: refresh that pair.
                        if rsp is None:
                            name = key(cid, phase)+'.json'
                            req = reader.read(run/'requests'/name, 'request')
                            rsp = reader.read(run/'responses'/name, 'response')
                            if (req['case_id'], req['phase'], req['id']) != (cid, phase, key(cid, phase)) or rsp.get('request_sha256') != req['_sha256']:
                                raise ValueError('Concurrent response integrity mismatch')
                            requests[(cid, phase)], responses[(cid, phase)] = req, rsp
                        if (record['arm'] != arm or record['seed'] != seed
                                or record['input_response_sha256'] != rsp['_response_text_hash']):
                            raise ValueError('Checkpoint input identity mismatch: '+str(path))
                        if record['status'] not in ('searched', 'unsupported', 'invalid_model'):
                            raise ValueError('Unknown checkpoint status: '+str(path))
                        if (record['status'] == 'searched') != bool(record['runs']):
                            raise ValueError('Checkpoint status/runs mismatch: '+str(path))
                        records.append(record)
                    tasks.append({'case_id': cid, 'dataset': case['dataset'], 'arm': arm, 'stage': stage,
                                  'seed': seed, 'backend': ARMS[arm]['method'], 'record': record})
                if len(records) == seeds:
                    complete_stages[(cid, arm, stage)] = records
    cpu = run/'cpu_status.json'
    states = [reader.read(p) for p in sorted(run.glob('gpu_replica_*.json')) if not p.name.endswith('_placement.json')]
    return {'cases': cases, 'settings': cfg, 'requests': requests, 'responses': responses,
            'complete_stages': complete_stages, 'tasks': tasks,
            'final_cases': sum((run/'final'/(key(c['id'], 'case')+'.json')).exists() for c in cases),
            'cpu_state': reader.read(cpu) if cpu.exists() else None, 'gpu_states': states,
            'events': [reader.read(p) for p in sorted((run/'inference_events').glob('*.json'))],
            'snapshot_started': started, 'snapshot_finished': time.time(),
            'manifest_sha256': hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest(),
            'experiment_source_sha256': cfg['source']}


def hours(seconds):
    return '待校准' if seconds is None else f'{seconds/3600:.2f} 小时'


def render(stats, eta, metadata):
    lines = ['# 分布反馈：关键配对统计与剩余时间', '',
             f"更新（UTC）：{metadata['generated_at_utc']}", '',
             '## 进度与 ETA', '',
             f"- 状态：`{eta['state']}`；最终完成实例：{metadata['final_cases']}/{metadata['expected_cases']}。",
             f"- CPU 搜索检查点：{eta['cpu']['tasks_completed']}/{eta['cpu']['tasks_expected']}；假设 {eta['assumed_cpu_workers']} 个进程。",
             f"- GPU 回答：{eta['gpu']['responses_completed']}/{eta['gpu']['requests_expected']}；其中复用 {eta['gpu']['reused_responses_excluded_from_speed']} 条，不参与速度校准。",
             f"- 尚需 GPU 请求：{eta['gpu']['requests_remaining']}；已排队 {eta['gpu']['queued_pending']}，尚未排队 {eta['gpu']['future_not_yet_enqueued']}。",
             f"- CPU 剩余条件估算：{hours(eta['cpu']['remaining_wall_seconds_conditional'])}。",
             f"- GPU 按实际 batch 耗时的剩余条件估算：{hours(eta['gpu']['remaining_busy_wall_seconds_conditional'])}（{eta['assumed_gpu_replicas']} 个副本）。",
             f"- 按近期队列实际吞吐估算的 GPU 剩余时间：{hours(eta['gpu']['remaining_seconds_at_observed_queue_throughput'])}。"]
    scenario = eta['overlap_scenarios']
    if scenario:
        lines.append(f"- 完全重叠/完全串行两种情景：{hours(scenario['full_overlap_seconds'])} / {hours(scenario['no_overlap_seconds'])}；不是置信区间或完成期限。")
    lines += ['- '+v for v in eta['resume_required']]
    lines += ['', '## 关键配对', '',
              'Δ = 实验组 − 对照组；可行率/答案一致率越大越好，gap 越小越好。所有率和 gap 用百分点（pp）显示。', '',
              '| 数据集 | 阶段 | 对照（实验 − 基准） | 配对/全部 | 独立指标 Δ pp [95% CI] | 配对胜/平/负 | sign p（Holm） | 共同可行 gap Δ pp（实例数） |',
              '|---|---|---|---:|---|---|---:|---|']
    def delta(result):
        value, ci = result['mean_delta_treatment_minus_control'], result['bootstrap_95pct']
        if value is None:
            return '—'
        interval = '样本不足' if ci is None else f'{100*ci[0]:+.2f}, {100*ci[1]:+.2f}'
        return f'{100*value:+.2f} [{interval}]'
    for row in stats['comparisons']:
        r, g = row['metrics'][row['primary_metric']], row['metrics']['common_feasible_signed_gap']
        p = r['sign_test_p_holm']
        lines.append(f"| {row['group']} | {row['stage']} | {row['treatment']} − {row['control']} | "
                     f"{row['paired_instances']}/{row['expected_instances']} | {row['primary_metric']}: {delta(r)} | "
                     f"{r['wins']}/{r['ties']}/{r['losses']} | "+('—' if p is None else f'{p:.4g}')+
                     f" | {delta(g)} (n={g['instances']}) |")
    lines += ['', '## 解释限制', '', *['- '+w for w in stats['warnings']+eta['warnings']],
              '- JSON 同时保存搜索时间/token 配对差、完整 Holm 校正、共同可行种子数及每组配对实例 ID。',
              '- 本工具只读实验目录，不修改旧 summary、请求、响应、manifest、检查点或运行进程。']
    return '\n'.join(lines)+'\n'


def output_guard(run, output):
    run, output = run.resolve(), output.resolve()
    if output == run or run in output.parents:
        raise ValueError('Report output must be OUTSIDE the experiment directory')
    if any((output/name).is_symlink() for name in ('paired_statistics.json', 'eta.json', 'report.md', 'snapshot.json')):
        raise ValueError('Refusing report output symlinks')


def generate(run, output, *, reader=None, cpu_workers=8, replicas=2, window_seconds=3600, bootstrap=2000):
    output_guard(run, output)
    snapshot = load_snapshot(run, reader)
    stats = compare(snapshot, bootstrap=bootstrap)
    eta = estimate(snapshot, now=time.time(), cpu_workers=cpu_workers, replicas=replicas, window_seconds=window_seconds)
    metadata = {k: snapshot[k] for k in ('manifest_sha256', 'experiment_source_sha256', 'snapshot_started', 'snapshot_finished', 'final_cases')}
    metadata.update(generated_at_utc=datetime.now(timezone.utc).isoformat(), expected_cases=len(snapshot['cases']),
                    run=str(run.resolve()), reporter_source_sha256=hashlib.sha256(b''.join(
                        p.read_bytes().replace(b'\r\n', b'\n') for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest())
    stats['snapshot'] = metadata
    eta['snapshot'] = metadata
    for name, value in (('paired_statistics.json', stats), ('eta.json', eta), ('snapshot.json', metadata)):
        atomic_write(output/name, json_bytes(value))
    atomic_write(output/'report.md', render(stats, eta, metadata).encode('utf-8'))
    print(f"REPORT {output/'report.md'} | {metadata['final_cases']}/{metadata['expected_cases']} final | "
          f"{eta['state']} | CPU {hours(eta['cpu']['remaining_wall_seconds_conditional'])} | "
          f"GPU {hours(eta['gpu']['remaining_busy_wall_seconds_conditional'])}", flush=True)
    return eta


def stop(signum, frame):
    global STOP
    STOP = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-workers', type=int, default=8)
    p.add_argument('--replicas', type=int, default=2)
    p.add_argument('--window-seconds', type=int, default=3600)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--watch-seconds', type=int, default=0, help='0 = once; otherwise refresh every >=30 seconds until complete')
    args = p.parse_args()
    if args.watch_seconds != 0 and args.watch_seconds < 30:
        p.error('watch-seconds must be 0 or at least 30')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    output_guard(args.run, args.output)
    with worker_lock(args.output/'.report.lock'):
        reader = Reader()
        while not STOP:
            eta = generate(args.run, args.output, reader=reader, cpu_workers=args.cpu_workers, replicas=args.replicas,
                           window_seconds=args.window_seconds, bootstrap=args.bootstrap)
            if not args.watch_seconds or eta['state'] == 'complete':
                break
            deadline = time.monotonic()+args.watch_seconds
            while not STOP and time.monotonic() < deadline:
                time.sleep(min(1., max(.01, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
