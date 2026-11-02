"""Full-denominator modeling and solver reports; independent from feedback."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import tarfile
import time
import numpy as np
from .common import s,validate


def score(case,model,x,reference):
    from qihc.generic.bounded import objective,violations
    from experiments.feedback_comparison_v1.report import metrics
    feasible=False;value=None;predicted=False;error=None
    if x is not None:
        try:
            feasible=bool(violations(case['gold'],x).sum()==0)
            value=float(objective(case['gold'],x))
            predicted=model is not None and bool(violations(model,x).sum()==0)
        except (ValueError,TypeError,IndexError) as exc:error=repr(exc)
    try:row=metrics(feasible,value,case,reference)
    except ValueError as exc:return {'status':'evaluation_error','error':str(exc)}
    row.update(status='scored',objective=value,predicted_feasible=predicted,candidate_error=error)
    return row


def report(root):
    m=validate(root);cfg=m['config'];cases=s.load(root/'evaluation/cases.CPU_ONLY.json');n=len(cases)
    refs={c['id']:s.load(root/'evaluation/references'/(s.key(c['id'])+'.json')) for c in cases}
    index=s.load(root/'evaluation/index.json') if (root/'evaluation/index.json').exists() else {'semantic':{},'cross':{}}
    semantics={};semstatuses=Counter()
    for key,meta in index['semantic'].items():
        path=root/'evaluation/semantic'/(key+'.json')
        if path.exists():
            row=s.load(path);semstatuses[row['status']]+=1
            identity=(meta['case_id'],meta['model_sha256'])
            # Prefer an actual conclusive check to a timeout on an identical cached model.
            if identity not in semantics or row['status'] in ('verified_correct','disproved','invalid'):semantics[identity]=row
    def semantic(cid,model):
        if model is None:return 'invalid'
        row=semantics.get((cid,s.value_digest(model)),{})
        return row.get('status') if row.get('status') in ('verified_correct','disproved') else 'unknown'
    allrows=[];groups=[];crosstable=[];state_status=Counter();round_status=Counter()
    for arm in cfg['arms']:
        aggregate=Counter();transitions=Counter();initial_counts=Counter();final_counts=Counter()
        for c in cases:
            tid=s.key(c['id'],arm,'0');path=root/'states'/(tid+'.json')
            initial=s.load(root/'initial'/(s.key(c['id'])+'.json'))['model']
            st=s.load(path) if path.exists() else {'round':0,'model':initial,'x':None,'rounds':[]}
            model=st['model'];before=semantic(c['id'],initial);after=semantic(c['id'],model)
            initial_counts[before]+=1;final_counts[after]+=1;transitions[before+' -> '+after]+=1
            aggregate['initial_compiled']+=initial is not None;aggregate['final_compiled']+=model is not None
            aggregate['invalid_initial_recovered_compilable']+=initial is None and model is not None
            aggregate['invalid_initial_recovered_correct']+=initial is None and after=='verified_correct'
            aggregate['semantic_repairs']+=before=='disproved' and after=='verified_correct'
            aggregate['semantic_regressions']+=before=='verified_correct' and after in ('disproved','invalid')
            aggregate['correct_became_unknown']+=before=='verified_correct' and after=='unknown'
            aggregate['rounds_processed']+=st['round']==cfg['rounds']
            quality=score(c,model,st.get('x'),refs[c['id']]);row={'case_id':c['id'],'arm':arm,
                       'rounds_processed':st['round'],'initial_semantic':before,'final_semantic':after,'metrics':quality}
            for field in ('original_feasible','optimal_hit','quality1','quality5'):
                aggregate[field]+=bool(quality.get(field))
            aggregate['scoring_errors']+=quality['status']=='evaluation_error'
            allrows.append(row)
            for r in st['rounds']:
                for field in ('base','probe','guard','final'):round_status[field+'/'+str(r[field].get('status'))]+=1
            for backend in cfg['cross_backends']:
                key=s.key(tid,backend);p=root/'evaluation/cross'/(key+'.json')
                result=s.load(p) if p.exists() else {'status':'not_completed','x':None}
                meta=index['cross'].get(key)
                if meta and meta['model_sha256']!=s.value_digest(model):raise ValueError('Cross matrix model hash mismatch')
                crossquality=score(c,model,result.get('x'),refs[c['id']])
                crosstable.append({'case_id':c['id'],'arm':arm,'backend':backend,'status':result.get('status'),
                                   'model_sha256':s.value_digest(model),'raw_solver':result,'metrics':crossquality})
        groups.append({'arm':arm,'expected':n,'counts':dict(aggregate),'initial_semantic':dict(initial_counts),
                       'final_semantic':dict(final_counts),'transitions':dict(transitions),
                       'rates':{'compiled':aggregate['final_compiled']/n,'verified_correct':final_counts['verified_correct']/n,
                                'quality5':aggregate['quality5']/n,
                                'invalid_recovery':aggregate['invalid_initial_recovered_compilable']/initial_counts['invalid'] if initial_counts['invalid'] else None,
                                'semantic_repair':aggregate['semantic_repairs']/initial_counts['disproved'] if initial_counts['disproved'] else None,
                                'semantic_regression':aggregate['semantic_regressions']/initial_counts['verified_correct'] if initial_counts['verified_correct'] else None}})
    costs=Counter();responses=Counter()
    for p in (root/'responses').glob('*.json'):
        row=s.load(p);responses[row['status']]+=1
        for k in ('input_tokens','output_tokens','generation_attempts','attributed_seconds'):costs[k]+=row.get(k,0)
        costs['cost_incomplete_requests']+=bool(row.get('cost_incomplete'))
    cpu_status=Counter();cpu_seconds=0.
    for p in (root/'outputs').glob('*.json'):
        row=s.load(p);cpu_status[row['status']]+=1;cpu_seconds+=row.get('process_seconds',row.get('worker_seconds',0))
    costs['cpu_attributed_seconds']=cpu_seconds
    paired=[];lookup={(r['case_id'],r['arm']):r['metrics'] for r in allrows}
    for other in cfg['arms']:
        if other=='P':continue
        fields={}
        for field in ('original_feasible','optimal_hit','quality5'):
            counts=Counter()
            for c in cases:
                a,b=lookup[c['id'],'P'],lookup[c['id'],other]
                if a['status']!='scored' or b['status']!='scored':counts['unscored']+=1;continue
                av,bv=bool(a[field]),bool(b[field]);counts['wins' if av>bv else 'losses' if av<bv else 'ties']+=1
            fields[field]=dict(counts)
        paired.append({'left':'P','right':other,'denominator':n,'metrics':fields})
    clock=s.load(root/'clock.json') if (root/'clock.json').exists() else {}
    processed=sum(g['counts']['rounds_processed'] for g in groups)
    matrix_done=sum(r['status']!='not_completed' for r in crosstable)
    summary={'experiment_id':'E004-1.4','smoke':m['smoke'],'scope':'18 observed development cases; no formal superiority claim',
             'status':'all_stages_processed' if processed==n*len(cfg['arms']) and matrix_done==n*len(cfg['arms'])*len(cfg['cross_backends']) else 'partial',
             'expected_trajectories':n*len(cfg['arms']),'rounds_processed_trajectories':processed,
             'cross_rows_terminal':matrix_done,'semantic_status_counts':dict(semstatuses),'response_status_counts':dict(responses),
             'cpu_status_counts':dict(cpu_status),'round_status_counts':dict(round_status),'groups':groups,'per_case':allrows,
             'paired':paired,'costs':dict(costs),'elapsed_since_original_start':time.time()-clock['started_at'] if clock else None,
             'failure_accounting':'processed includes recorded failures; not a successful solve count',
             'semantic_scope':'factor identity or exhaustive feasible-domain checks; positive objective scale allowed; incomplete checks unknown',
             'historical_baselines':s.load(root/'evaluation/e003_baselines.json')}
    s.save(root/'summary.json',summary);s.save(root/'cross_matrix.json',crosstable)
    lines=['# E004-1.4 结果（探索性）','',f"状态：{summary['status']}；处理轨迹 {processed}/{n*len(cfg['arms'])}。处理终结不等于求解成功。",'',
           '| 组 | 可编译 / 全量 | 已核验正确 / 全量 | 语义未判定 | 原题可行 | 最优命中 | gap≤5% |','|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        a=g['counts'];f=g['final_semantic']
        lines.append(f"| {g['arm']} | {a['final_compiled']}/{n} | {f.get('verified_correct',0)}/{n} | {f.get('unknown',0)} | {a['original_feasible']}/{n} | {a['optimal_hit']}/{n} | {a['quality5']}/{n} |")
    lines+=['','完整状态转移、分母、失败、成本见 summary.json；固定模型的七后端比较见 cross_matrix.json。',
            '无效初稿保留在总体比较中；not_run_invalid_model 不算后端自身求解失败。',
            '核验超时或只通过有限测试仍为未判定。历史 E003 参照不能直接证明同硬件端到端速度优势。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def pack(root):
    target=root.parent/(root.name+'_analysis.tar.gz');tmp=target.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as archive:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.md','.log'):
                archive.add(p,arcname=root.name+'/'+p.relative_to(root).as_posix())
    tmp.replace(target);return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--pack',action='store_true');a=p.parse_args()
    r=report(a.run);print(r['status'],r['rounds_processed_trajectories'],'/',r['expected_trajectories'])
    if a.pack:print(pack(a.run))
