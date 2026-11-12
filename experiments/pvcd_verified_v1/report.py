from collections import Counter
import numpy as np
from qihc.experiment_runtime.storage import load,save
from qihc.experiment_runtime.posttrain_screen import case_list,train_list
from experiments.pvcd_cycle_v1.report import metrics

def build_report(job,mode):
    if mode not in ('pilot','screen'):raise ValueError('Unsupported phase')
    folder=job.RUN/mode;seed=job.CFG['seed'];arms=job.CFG['arms'];cases=case_list(job,mode)
    data={};signals={};complete=True;lines=[f'# E005-1.10 {mode}',
        '同种子17、相同训练题/步数，三个训练组从同一SFT开始。历史完整训练P-VCD不参与本轮排序。',
        '额外正样本NLL及解释反馈具有额外计算/训练监督成本，不声称等GPU小时。', '']
    for arm in arms:
        out=folder/'models'/str(seed)/arm
        data[arm]={p.stem:load(p) for p in (out/'evaluations').glob('*.json')}
        complete &= set(data[arm])=={c['id'] for c in cases}
        if arm=='sft':continue
        rows=[load(p) for p in (out/'trace').glob('*.json')];steps=[load(p) for p in (out/'steps').glob('*.json')]
        samples=[s for r in rows for s in r['samples']]
        repairs=[s for s in samples if 'repair_semantic' in s]
        signals[arm]=dict(groups=len(rows),positive_groups=sum(r['positive_used'] for r in rows),
            positive_cases=len({r['case_id'] for r in rows if r['positive_used']}),
            positive_candidates=sum(r['positive_candidates'] for r in rows),
            reward_updates=sum(r['reward_update'] for r in rows),verified_repairs=sum(r['verified_repairs'] for r in rows),
            repair_cases=len({r['case_id'] for r in rows if r['verified_repairs']}),
            repair_outcomes=dict(Counter(s['repair_semantic']['status'] for s in repairs)),
            original_witnesses_cleared=sum(s.get('witness_remaining')==0 for s in repairs),
            generated_repairs=len(repairs),nonzero_steps=sum(s['gradient_norm']>0 for s in steps),
            seconds=sum(s['seconds'] for s in steps),rollout_tokens=sum(len(s['ids']) for s in samples),
            repair_tokens=sum(s.get('repair_tokens',0) for s in samples))
    summary=dict(experiment_id='E005-1.10',mode=mode,complete=bool(complete),training=signals,stages={})
    for stage in sorted({c['stage'] for c in cases}):
        expected={c['id'] for c in cases if c['stage']==stage}
        ids=sorted(expected.intersection(*(set(data[a]) for a in arms)))
        entry=dict(expected_cases=len(expected),paired_cases=len(ids),arms={},comparisons={},transitions={},families={})
        lines += [f'## {stage}: {len(ids)}/{len(expected)}道共同题',
                  '|组|完整正确|unknown|p-bit可行|p-bit最优|SCIP可行|SCIP最优|','|---|---:|---:|---:|---:|---:|---:|']
        vectors={}
        for a in arms:
            vectors[a]=np.array([metrics(data[a][cid]) for cid in ids],dtype=float)
            means=vectors[a].mean(axis=0).tolist() if ids else None;entry['arms'][a]=means
            lines.append('|'+a+'|'+'|'.join(f'{x*100:.2f}%' for x in means)+'|' if means else '|'+a+'|—|—|—|—|—|—|')
            entry['transitions'][a]=dict(Counter(data['sft'][cid]['semantic']['status']+'->'+data[a][cid]['semantic']['status'] for cid in ids))
        if ids:
            for family in {data['sft'][cid].get('family','public') for cid in ids}:
                indices=[i for i,cid in enumerate(ids) if data['sft'][cid].get('family','public')==family]
                entry['families'][family]=dict(cases=len(indices),means={a:vectors[a][indices].mean(axis=0).tolist() for a in arms})
        if complete and stage in ('pilot','main','scale'):
            for a,b in (('pvcd','sft'),('positive','pvcd'),('localized','positive')):
                diff=vectors[a][:,3]-vectors[b][:,3];rng=np.random.default_rng(20261112)
                boot=[rng.choice(diff,len(diff),replace=True).mean()*100 for _ in range(job.CFG['bootstrap_repeats'])]
                ci=np.percentile(boot,[100*.05/6,100*(1-.05/6)]).tolist()
                entry['comparisons'][a+'-'+b]=dict(delta_pp=float(diff.mean()*100),ci98_33_pp=ci)
                lines.append(f'{a} - {b}: {diff.mean()*100:+.2f}pp, 98.33% case-bootstrap interval {ci}')
        summary['stages'][stage]=entry
    lines+=['','## 训练机制','|组|步数|正样本学习题数|修复生成|验证修复|原反例已消除|训练秒|','|---|---:|---:|---:|---:|---:|---:|']
    for a,s in signals.items():lines.append(f"|{a}|{s['groups']}|{s['positive_cases']}|{s['generated_repairs']}|{s['verified_repairs']}|{s['original_witnesses_cleared']}|{s['seconds']:.1f}|")
    lines+=['原反例消除不等于完整正确；仅verified_correct可进入修复蒸馏。开发含训练题，不作为泛化证据。',
            'screen只有一个种子且主测历史已查看，属于探索性筛查；不自动启动三种子正式实验。']
    reasons=[]
    if not complete:reasons.append('incomplete evaluation')
    for a,s in signals.items():
        if s['groups']!=len(train_list(job,mode)) or s['nonzero_steps']!=s['groups']:reasons.append(a+': incomplete/non-updating training')
    for a in ('positive','localized'):
        if signals[a]['positive_cases']<2:reasons.append(a+': fewer than two positive training cases')
    gate=dict(passed=not reasons,reasons=reasons,repair_active=signals['localized']['verified_repairs']>0,
              scope='execution and verified-positive signal only; does not certify repair efficacy or accuracy gains')
    if mode=='pilot':save(folder/'gate.json',gate)
    summary['gate']=gate;save(folder/'summary.json',summary)
    lines += [json_dump(gate)]
    (folder/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines),flush=True)
    return summary

def json_dump(obj):
    import json
    return json.dumps(obj,ensure_ascii=False)
