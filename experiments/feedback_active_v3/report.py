"""Partial-safe denominators and paired instance-level statistics."""
from __future__ import annotations

import argparse
import os
import tempfile
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from qihc.experiment_runtime.storage import load, save


def paired_statistics(rows, comparisons, draws=10000):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['stage'] == 'main': by[r['arm']][r['case_id']].append(float(r['semantic']['status'] == 'verified_correct'))
    rng = np.random.default_rng(20261108); result = []
    for arm in comparisons:
        ids = sorted(set(by['P2']) & set(by[arm]))
        if not ids: continue
        delta = np.array([np.mean(by['P2'][i]) - np.mean(by[arm][i]) for i in ids])
        boots, nulls = [], []
        for _ in range(draws // 100):
            boots.extend(delta[rng.integers(0, len(delta), (100, len(delta)))].mean(axis=1))
            nulls.extend((delta * rng.choice([-1, 1], (100, len(delta)))).mean(axis=1))
        p = (1 + np.count_nonzero(np.abs(nulls) >= abs(delta.mean()) - 1e-12)) / (1 + len(nulls))
        result.append(dict(comparison='P2-' + arm, paired_instances=len(ids), difference=float(delta.mean()),
            ci95=np.quantile(boots, [.025, .975]).tolist(), p_raw=float(p), unit='independent_instance'))
    ordered = sorted(result, key=lambda r: r['p_raw']); previous = 0.
    for i, r in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - i) * r['p_raw'])); r['p_holm'] = previous
    return result


def report(root, statistics=False):
    root = Path(root); expected = defaultdict(int); done = defaultdict(list); evaluated = defaultdict(list)
    for p in (root / 'stages').glob('*.json'):
        for row in load(p)['trajectories']: expected[(row['stage'], row['arm'])] += 1
    for p in (root / 'trajectories').glob('*.json'):
        row = load(p)
        if row.get('status') == 'complete': done[(row['stage'], row['arm'])].append(row)
    evaluations = []
    for p in (root / 'evaluation/trajectories').glob('*.json'):
        row = load(p); evaluations.append(row); evaluated[(row['stage'], row['arm'])].append(row)
    groups = []; solver_groups = []
    design_path = root / 'evaluation_design.json'
    audit_ids = set(load(design_path)['audit_case_ids']) if design_path.exists() else set()
    planned = defaultdict(list)
    for p in (root / 'stages').glob('*.json'):
        for row in load(p)['trajectories']: planned[(row['stage'], row['arm'])].append(row)
    for group in sorted(expected):
        states = done[group]; ev = evaluated[group]; denominator = expected[group]
        counts = defaultdict(int)
        for row in ev:
            status = row['semantic'].get('status', 'unknown'); counts[status] += 1
            if row['initial_semantic'] == 'invalid' and row['semantic'].get('compiled'): counts['recovered_invalid'] += 1
            if row['initial_semantic'] == 'verified_correct' and status == 'disproved': counts['regressed_correct'] += 1
            if row['initial_semantic'] == 'disproved' and status == 'verified_correct': counts['repaired_wrong'] += 1
        groups.append(dict(stage=group[0], arm=group[1], expected_trajectories=denominator, completed=len(states), evaluated=len(ev),
            compiled=sum(st.get('model') is not None for st in states), counts=dict(counts),
            rounds_with_cards=sum(bool(rd.get('probe', {}).get('cards')) for st in states for rd in st.get('rounds', []))))
        for backend in ('scip','pbit','cp_sat','highs','sa','pt','penalty_pt'):
            full = backend in ('scip','pbit') or (group[0] == 'main' and group[1] == 'C0')
            wanted = denominator if full else sum(r.get('case_id') in audit_ids for r in planned[group]) if group[0]=='main' else 0
            if not wanted: continue
            present=[r['cross'][backend] for r in ev if backend in r.get('cross',{})]
            solver_groups.append(dict(stage=group[0],arm=group[1],backend=backend,expected=wanted,evaluated=len(present),
                original_feasible=sum(bool(v['score']['original_feasible']) for v in present),
                optimal_hit=sum(bool(v['score']['optimal_hit']) for v in present),
                invalid_inputs=sum(v.get('status')=='input_invalid' for v in present),
                reference_proven=sum(bool(v['score']['reference_proven']) for v in present),
                coverage='full_cohort' if full else 'prespecified_120_case_subset'))
    requests = [load(p) for p in (root / 'requests').glob('*.json')]
    complete = bool(expected) and all(len(evaluated[g]) == n for g, n in expected.items())
    experiment_id=load(root/'manifest.json')['config']['experiment_id'] if (root/'manifest.json').exists() else 'E004_2.1'
    result = dict(experiment_id=experiment_id, status='complete' if complete else 'partial', generated_at=time.time(),
        scope='registered_stages_only', expected=sum(expected.values()), completed=sum(len(v) for v in done.values()),
        evaluated=len(evaluations), groups=groups, solver_groups=solver_groups, llm=dict(attempts=sum(r.get('attempts', 0) for r in requests),
        input_tokens=sum(r.get('input_tokens', 0) for r in requests), output_tokens=sum(r.get('output_tokens', 0) for r in requests),
        failed=sum(r.get('status') != 'complete' for r in requests)))
    if statistics:
        cfg = load(root / 'manifest.json')['config']
        main_complete = all(len(evaluated[g]) == n for g, n in expected.items() if g[0] == 'main') and any(g[0] == 'main' for g in expected)
        result['statistics'] = paired_statistics(evaluations, cfg['primary_comparisons']) if main_complete else []
        result['statistics_status'] = 'full_main_only' if main_complete else 'pending_complete_main'
    save(root / 'summary.json', result)
    lines = ['# '+experiment_id+' 实验报告', '', f'状态：{result["status"]}；推理 {result["completed"]}/{result["expected"]}，评价 {result["evaluated"]}/{result["expected"]}。',
        '范围仅含已登记阶段；缺失、unknown、失败不删除。以下为轨迹计数；正式检验以题为单位平均种子。', '',
        '所有正式题目来自公开数据：NLCO 六任务 900 题分开发/主测/规模；CP-Bench 10 题补充。第二模型关闭，人工审计另行记录。', '',
        '| 阶段 | 方法 | 推理完成/计划 | 已评价 | 可执行 | 完整正确 | 已证错误 | unknown | 初始失败恢复 | 正确退化 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        c = g['counts']; lines.append(f'| {g["stage"]} | {g["arm"]} | {g["completed"]}/{g["expected_trajectories"]} | {g["evaluated"]} | {g["compiled"]} | {c.get("verified_correct",0)} | {c.get("disproved",0)} | {c.get("unknown",0)} | {c.get("recovered_invalid",0)} | {c.get("regressed_correct",0)} |')
    lines += ['', '| 阶段 | 反馈 | 终端求解器 | 已评价/计划 | 原题可行 | 已证最优命中 | 无效输入 | 覆盖范围 |',
              '|---|---|---|---:|---:|---:|---:|---|']
    for g in solver_groups:
        lines.append(f'| {g["stage"]} | {g["arm"]} | {g["backend"]} | {g["evaluated"]}/{g["expected"]} | {g["original_feasible"]} | {g["optimal_hit"]} | {g["invalid_inputs"]} | {g["coverage"]} |')
    main_path=root/'stages/main.json'
    main_ids=set(load(main_path).get('cases',[])) if main_path.exists() else set()
    fixed=defaultdict(list)
    for path in (root/'evaluation/initial').glob('*.json'):
        row=load(path)
        if row['case_id'] not in main_ids: continue
        for label,field in [('M0_20s','cross'),('GOLD_20s','gold_cross'),('M0_40s_independent','compensated')]:
            for backend,solved in row.get(field,{}).items(): fixed[(label,backend)].append(solved)
    result['fixed_model_solver_comparison']=[]
    lines += ['', '固定模型对照：各后端使用完全相同的初稿 M0 或标准模型 GOLD；40 秒为独立重跑。', '',
              '| 模型/预算 | 求解器 | 已评价/计划 | 原题可行 | 已证最优命中 | 无效输入 |',
              '|---|---|---:|---:|---:|---:|']
    for (label,backend),values in sorted(fixed.items()):
        r=dict(model=label,backend=backend,evaluated=len(values),expected=len(main_ids),
               original_feasible=sum(bool(v['score']['original_feasible']) for v in values),
               optimal_hit=sum(bool(v['score']['optimal_hit']) for v in values),
               invalid_inputs=sum(v.get('status')=='input_invalid' for v in values))
        result['fixed_model_solver_comparison'].append(r)
        lines.append(f'| {label} | {backend} | {r["evaluated"]}/{r["expected"]} | {r["original_feasible"]} | {r["optimal_hit"]} | {r["invalid_inputs"]} |')
    save(root/'summary.json',result)
    lines += ['', '最优命中只使用已证明参考；完整值及检查点在 evaluation/trajectories/*.json 的 cross/score；全部模型 hash 可追溯。',
              '', f'LLM 调用尝试：{result["llm"]["attempts"]}；输入 token：{result["llm"]["input_tokens"]}；输出 token：{result["llm"]["output_tokens"]}。']
    target = root / 'report.md'
    fd, name = tempfile.mkstemp(prefix='.report.', dir=root)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f: f.write('\n'.join(lines) + '\n')
        os.replace(name, target)
    finally:
        if os.path.exists(name): os.unlink(name)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--statistics', action='store_true')
    a = p.parse_args(); print(report(a.run, a.statistics)['status'])
