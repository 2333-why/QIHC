"""All-case denominators, paired case-level tests and offline judge auditing."""
import collections
import sys
import numpy as np
from qihc.experiment_runtime.storage import load, save
from .data import RUN, ARMS, SEEDS, EXPERIMENT

LABELS = dict(A='原初稿', B='仅修编译', C='无采样双复核', S='SCIP多解反馈',
              P='普通p-bit反馈', Q='主动p-bit反馈')

def report(mode='formal'):
    from qihc.experiment_runtime import feedback_job as job
    job.configure(__package__)
    items = job.cohort(mode)
    root = RUN/mode
    records = collections.defaultdict(dict)
    for path in root.glob('models/*/*/evaluations/*.json'):
        row = load(path)
        records[row['arm']][row['case_id'], row['seed']] = row
    lines = [f'# {EXPERIMENT} {mode}', '',
             '主指标：原题已证最优命中。计划分母保留全部题；缺失单列，未完成时不是最终性能。', '']
    summary = dict(experiment=EXPERIMENT, expected=len(items)*len(ARMS),
                   completed=sum(len(v) for v in records.values()), stages={})
    for stage, seeds in SEEDS.items():
        expected = [(r['case']['id'], r['seed']) for r in items if r['stage'] == stage]
        if not expected:
            continue
        n = len(expected)
        ids = sorted({cid for cid, _ in expected if all((cid, s) in records[a] for s in seeds for a in ARMS)})
        lines += [f'## {stage}，计划{n}轨迹，六组配齐种子共同题{len(ids)}', '',
                  '|组|完成/计划|完整正确|p-bit可行|p-bit最优|SCIP最优|对B新增/丢失|接受修订|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        stats = dict(planned=n, common_cases=ids, arms={}, comparisons={})
        for arm in ARMS:
            rows = [records[arm][k] for k in expected if k in records[arm]]
            scores = [r['scores']['20'] for r in rows]
            metrics = dict(completed=len(rows), missing=n-len(rows),
                correct=sum(r['semantic'] == 'verified_correct' for r in rows),
                feasible=sum(r['pbit']['original_feasible'] for r in scores),
                optimal=sum(r['pbit']['optimal_hit'] for r in scores),
                scip_optimal=sum(r['scip']['optimal_hit'] for r in scores),
                actions=dict(collections.Counter(r['action'] for r in rows)))
            diff = [int(records[arm][k]['scores']['20']['pbit']['optimal_hit'])-int(records['B'][k]['scores']['20']['pbit']['optimal_hit']) for k in expected if k in records[arm] and k in records['B']]
            metrics['gained'], metrics['lost'] = diff.count(1), diff.count(-1)
            for field in ('known_labels', 'wrong_labels', 'unknown_labels', 'search_seconds', 'unique_states', 'sampled_states'):
                metrics[field] = sum(r.get('audit', {}).get(field, 0) for r in rows)
            for field in ('prompt_tokens', 'completion_tokens', 'calls'):
                metrics[field] = sum(r.get('usage', {}).get(field, 0) for r in rows)
            stats['arms'][arm] = metrics
            pct = lambda key: f'{metrics[key]}/{n} ({metrics[key]/n:.2%})'
            lines.append('| '+' | '.join([arm+' '+LABELS[arm], f'{len(rows)}/{n}', pct('correct'), pct('feasible'), pct('optimal'), pct('scip_optimal'), f'{metrics["gained"]}/{metrics["lost"]}', str(metrics['actions'].get('accepted', 0))])+' |')
        for other in ('B', 'C', 'S', 'P'):
            if not ids:
                continue
            differences = np.array([np.mean([int(records['Q'][cid, s]['scores']['20']['pbit']['optimal_hit'])-int(records[other][cid, s]['scores']['20']['pbit']['optimal_hit']) for s in seeds]) for cid in ids])
            rng = np.random.default_rng(17)
            boot = rng.choice(differences, (5000, len(ids))).mean(axis=1)
            perm = (rng.choice([-1, 1], (5000, len(ids)))*differences).mean(axis=1)
            stats['comparisons']['Q-'+other] = dict(delta=float(differences.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(), p=float((1+(abs(perm) >= abs(differences.mean())-1e-12).sum())/5001))
        ordered = sorted(stats['comparisons'], key=lambda k: stats['comparisons'][k]['p'])
        last = 0.
        for i, key in enumerate(ordered):
            r = stats['comparisons'][key]
            last = max(last, min(1., (len(ordered)-i)*r['p']))
            r['holm'] = last
            lines += ['', f'{key} 最优差 {100*r["delta"]:+.2f}百分点，95% CI [{100*r["ci95"][0]:.2f}, {100*r["ci95"][1]:.2f}]，Holm p={last:.4f}。']
        for arm in ARMS:
            r = stats['arms'][arm]
            lines += ['', f'{arm} 决策：{r["actions"]}；离线核对已判断标签错误 {r["wrong_labels"]}/{r["known_labels"]}，未知 {r["unknown_labels"]}。']
            lines += [f'增量成本（包含本组依赖的共享候选调用）：{r["calls"]} 次调用，输入/输出 tokens {r["prompt_tokens"]}/{r["completion_tokens"]}，探针 {r["search_seconds"]:.1f} 秒。复用历史初稿和编译修复的既有成本未包含。']
        summary['stages'][stage] = stats
    summary['complete'] = summary['completed'] == summary['expected']
    lines[2] += f' 完成 {summary["completed"]}/{summary["expected"]}。'
    root.mkdir(parents=True, exist_ok=True)
    save(root/'summary.json', summary)
    text = '\n'.join(lines)+'\n'
    (root/'report.md').write_text(text, encoding='utf-8')
    print(text)
    return summary

if __name__ == '__main__':
    report(sys.argv[1] if len(sys.argv) > 1 else 'formal')
