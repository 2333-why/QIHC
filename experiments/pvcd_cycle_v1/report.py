"""Complete paired cohorts, case-cluster bootstrap, real training funnel and full transitions."""
import collections
import sys
import numpy as np
from qihc.experiment_runtime.storage import load,save
from .common import CFG,RUN,TRAIN_ARMS,STAGE_SEEDS,cases,training_rows

def signals(folder):
    result={}
    for arm in TRAIN_ARMS:
        records=[load(p) for p in folder.glob(f'models/*/{arm}/trace/*.json')]
        steps=[load(p) for p in folder.glob(f'models/*/{arm}/steps/*.json')]
        samples=[s for r in records for s in r['samples']]
        attempts=[a for s in samples for a in s.get('repair_trace',[])]
        result[arm]=dict(groups=len(records),reward_updates=sum(r['reward_update'] for r in records),
            verified_repairs=sum(r['verified_repairs'] for r in records),unique_targets=sum(r['unique_targets'] for r in records),
            distinct_repaired_cases=len({r['case_id'] for r in records if r['verified_repairs']}),
            guards=sum(r['guard'] for r in records),
            witnessed_candidates=sum(any(a['cards'] for a in s.get('repair_trace',[])) for s in samples),
            generated_repairs=sum(a['status']=='generated' for a in attempts),
            repair_outcomes=dict(collections.Counter(a.get('semantic',{}).get('status',a['status']) for a in attempts)),
            card_types=dict(collections.Counter(('objective' if c['kind']=='objective' else 'too_tight' if c['correct_feasible'] else 'too_loose') for a in attempts for c in a['cards'])),
            rollout_tokens=sum(len(s['ids']) for s in samples),repair_tokens=sum(a.get('tokens',0) for a in attempts),
            training_seconds=sum(s['seconds'] for s in steps),probe_seconds=sum(t['seconds'] for a in attempts for t in a['search']),
            nonzero_grad_steps=sum(s['gradient_norm']>0 for s in steps))
    return result

def metrics(row):
    return [row['semantic']['status']=='verified_correct',row['semantic']['status']=='unknown',
            *[row['cross'][b]['score'][k] for b in ('pbit','scip') for k in ('original_feasible','optimal_hit')]]

def paired_interval(differences, repeats=5000):
    d=np.asarray(differences,dtype=float)
    if not len(d): raise ValueError('Empty paired cohort')
    rng=np.random.default_rng(20261112)
    boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(repeats)])*100
    # Four prespecified comparisons per stage: Bonferroni percentile intervals.
    return dict(delta_pp=float(d.mean()*100),ci95_pp=np.percentile(boot,[2.5,97.5]).tolist(),
                ci98_75_pp=np.percentile(boot,[.625,99.375]).tolist(),cases=len(d))

def report(mode):
    if mode not in ('pilot','formal'): raise ValueError('Unknown mode')
    folder=RUN/mode; stages={'pilot':(17,)} if mode=='pilot' else STAGE_SEEDS
    data=collections.defaultdict(lambda:collections.defaultdict(dict))
    for p in folder.glob('models/*/*/evaluations/*.json'):
        row=load(p); data[row['stage']][p.parts[-3]][(row['case_id'],int(p.parts[-4]))]=row
    summary=dict(mode=mode,experiment_id=CFG['experiment_id'],stages={},training=signals(folder))
    lines=[f'# E005-1.9 {mode}', '开发含训练题；历史已查看测试集属于探索性复测。所有组本轮重新评分，同模型共享缓存。',
           '按题内平均种子、再按题平均；原题为bootstrap单位。unknown和invalid不算正确，原题最优分母为全题。','']
    all_complete=True
    for stage,seeds in stages.items():
        expected={(c['id'],s) for s in seeds for c in cases(s,mode) if c['stage']==stage}
        actual={a:set(data[stage][a]) for a in CFG['arms']}
        full=all(v==expected for v in actual.values()); all_complete &= full
        common=set.intersection(*(set(cid for cid,s in v if all((cid,t) in v for t in seeds)) for v in actual.values()))
        ids=sorted(common & {c for c,s in expected})
        entry=dict(expected_trajectories_per_arm=len(expected),complete=full,common_cases=len(ids),arms={},comparisons={},transitions={},subgroups={})
        lines += [f'## {stage}：共同题{len(ids)}，覆盖'+('完整' if full else '不完整，禁止总体排名'),
                  '|组|实际轨迹/应有轨迹|完整正确|unknown|p-bit可行|p-bit最优|SCIP可行|SCIP最优|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        vectors={}
        for arm in CFG['arms']:
            vectors[arm]=np.array([np.mean([metrics(data[stage][arm][cid,s]) for s in seeds],axis=0) for cid in ids])
            means=vectors[arm].mean(axis=0).tolist() if ids else None
            entry['arms'][arm]=dict(scored=len(actual[arm]),missing=len(expected-actual[arm]),unexpected=len(actual[arm]-expected),means=means)
            cells=[f'{v*100:.2f}%' for v in means] if means else ['—']*6
            lines.append('|'+ '|'.join([arm,f'{len(actual[arm])}/{len(expected)}',*cells])+'|')
            transition=collections.Counter()
            for key in sorted(expected & actual[arm] & actual['sft']):
                before=data[stage]['sft'][key]['semantic']['status'];after=data[stage][arm][key]['semantic']['status']
                transition[before+'->'+after]+=1
            entry['transitions'][arm]=dict(transition)
        if ids:
            family_groups=collections.defaultdict(list)
            size_groups=collections.defaultdict(list)
            for i,cid in enumerate(ids):
                baseline=data[stage]['sft'][cid,seeds[0]]
                family_groups[baseline.get('family','unspecified')].append(i)
                size_groups[str(baseline.get('n_variables','unspecified'))].append(i)
            for axis,groups in (('family',family_groups),('n_variables',size_groups)):
                entry['subgroups'][axis]={name:dict(cases=len(indices),means={a:vectors[a][indices].mean(axis=0).tolist() for a in CFG['arms']}) for name,indices in groups.items()}
            entry['subgroups']['seed']={str(s):{a:np.mean([metrics(data[stage][a][cid,s]) for cid in ids],axis=0).tolist() for a in CFG['arms']} for s in seeds}
        if full and stage in ('main','scale','pilot'):
            lines += ['', 'cycle 的 p-bit 最优命中差值（百分点）：']
            for arm in ('pvcd','cycle_uniform','cycle_static','cycle_no_guard'):
                stat=paired_interval(vectors['cycle'][:,3]-vectors[arm][:,3],CFG['bootstrap_repeats'])
                entry['comparisons'][arm]=stat
                lo,hi=stat['ci98_75_pp']
                lines.append(f"- 对 {arm}: {stat['delta_pp']:+.2f}，98.75%区间[{lo:.2f}, {hi:.2f}]（每阶段四项比较校正；探索性）。")
        summary['stages'][stage]=entry
    lines += ['', '## 真实训练信号', '|组|训练组|奖励更新组|有证据候选|生成修复|已验证修复事件|独立修复题|保持锚次数|训练秒|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary['training'].items():
        lines.append('|'+ '|'.join(map(str,[arm,s['groups'],s['reward_updates'],s['witnessed_candidates'],s['generated_repairs'],s['verified_repairs'],s['distinct_repaired_cases'],s['guards'],round(s['training_seconds'],1)]))+'|')
    lines += ['', '训练秒为已记录步骤的worker墙钟秒之和，不包括中断后丢失/重算的耗时，也不等于实测GPU利用时长；修复事件可重复原题。完整状态转移、分题型/变量数/种子、证据类别、token、搜索秒见summary.json。',
              '最终推理不使用gold、不搜索训练反例，也不进行多轮修复。原P-VCD历史训练预算更小；来源消融仅匹配返回赋值数和生成上限，不声称等FLOPs。']
    summary['complete']=all_complete
    save(folder/'summary.json',summary)
    if mode=='pilot':
        reasons=[]
        if not all_complete: reasons.append('pilot evaluation incomplete')
        for arm,s in summary['training'].items():
            if s['groups']!=len(training_rows('pilot')): reasons.append(arm+': training incomplete')
            if s['nonzero_grad_steps']==0: reasons.append(arm+': no nonzero gradient')
        full=summary['training']['cycle']
        for field in ('reward_updates','verified_repairs','guards'):
            if not full[field]: reasons.append('cycle: no '+field+'; inspect trace before spending formal budget')
        save(folder/'gate.json',dict(passed=not reasons,reasons=reasons))
        lines.append('Pilot gate: '+str(not reasons)+' '+str(reasons))
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)
    return summary

if __name__=='__main__': report(sys.argv[1] if len(sys.argv)>1 else 'formal')
