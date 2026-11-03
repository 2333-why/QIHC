"""Three-track report with complete denominators and explicit operational failures."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time
from .common import s, validate
from experiments.feedback_witness_v1.report import score, pack


def report(root):
    m=validate(root);cfg=m['config'];cases=s.load(root/'evaluation/cases.CPU_ONLY.json')
    byid={c['id']:c for c in cases};n=len(cases)
    refs={cid:s.load(root/'evaluation/references'/(s.key(cid)+'.json')) for cid in byid}
    initial={cid:s.load(root/'initial'/(s.key(cid)+'.json'))['model'] for cid in byid}
    index=s.load(root/'evaluation/index.json') if (root/'evaluation/index.json').exists() else {'semantic':{},'cross':{}}
    sem={};semstatuses=Counter()
    for key,meta in index['semantic'].items():
        p=root/'evaluation/semantic'/(key+'.json')
        if p.exists():
            row=s.load(p);semstatuses[row['status']]+=1
            sem[meta['case_id'],meta['model_sha256']]=row
    def semantic(cid,model):
        if model is None:return 'invalid'
        row=sem.get((cid,s.value_digest(model)),{})
        return row.get('status') if row.get('status') in ('verified_correct','disproved') else 'unknown'
    states={};groups=[];details=[];stages=Counter()
    for arm in cfg['arms']:
        counts=Counter();before=Counter();after=Counter();transitions=Counter()
        # These counters otherwise do not exist until the first saved round.
        counts.update({key:0 for key in ('probe_rounds','compatible_probe_rounds','rounds_with_cards',
                                         'evidence_cards','accepted_hypothesis_conflicts')})
        for cid,c in byid.items():
            tid=s.key(cid,arm,'0');p=root/'states'/(tid+'.json')
            st=s.load(p) if p.exists() else {'round':0,'model':initial[cid],'x':None,'rounds':[]}
            states[cid,arm]=st
            a=semantic(cid,initial[cid]);b=semantic(cid,st['model']);before[a]+=1;after[b]+=1;transitions[a+' -> '+b]+=1
            counts['initial_compiled']+=initial[cid] is not None;counts['final_compiled']+=st['model'] is not None
            counts['rounds_processed']+=st['round']==cfg['rounds']
            counts['semantic_repairs']+=a=='disproved' and b=='verified_correct'
            counts['semantic_regressions']+=a=='verified_correct' and b in ('invalid','disproved')
            counts['invalid_initial_recovered_compilable']+=a=='invalid' and st['model'] is not None
            counts['invalid_initial_recovered_correct']+=a=='invalid' and b=='verified_correct'
            counts['correct_became_unknown']+=a=='verified_correct' and b=='unknown'
            metrics=score(c,st['model'],st.get('x'),refs[cid])
            for field in ('original_feasible','optimal_hit','quality1','quality5'):counts[field]+=bool(metrics.get(field))
            details.append({'case_id':cid,'arm':arm,'initial_semantic':a,'final_semantic':b,'metrics':metrics})
            for r in st['rounds']:
                for field in ('base','probe','guard','final'):stages[field+'/'+str(r[field].get('status'))]+=1
                counts['probe_rounds']+=1
                counts['compatible_probe_rounds']+=r['probe'].get('status')=='complete'
                counts['rounds_with_cards']+=bool(r['probe'].get('cards'))
                counts['evidence_cards']+=len(r['probe'].get('cards',[]))
                counts['accepted_hypothesis_conflicts']+=any(r['guard'].get('hypothesis_diagnostics',{}).get('mismatch_counts',[]))
        groups.append({'arm':arm,'expected':n,'counts':dict(counts),'initial_semantic':dict(before),
                       'final_semantic':dict(after),'transitions':dict(transitions)})
    cross=[]
    for cid,c in byid.items():
        for arm in ['C0','GOLD']+cfg['arms']:
            kind='initial' if arm=='C0' else 'canonical'
            model=initial[cid] if arm=='C0' else c['gold'] if arm=='GOLD' else states[cid,arm]['model']
            for backend in cfg['cross_backends']:
                key=s.key(cid,kind,backend) if arm in ('C0','GOLD') else s.key(s.key(cid,arm,'0'),backend)
                p=root/'evaluation/cross'/(key+'.json');row=s.load(p) if p.exists() else {'status':'not_completed','x':None}
                meta=index['cross'].get(key)
                if meta and meta['model_sha256']!=s.value_digest(model):raise ValueError('Cross model identity mismatch')
                cross.append({'case_id':cid,'arm':arm,'backend':backend,'model_sha256':s.value_digest(model),
                              'semantic':semantic(cid,model),'status':row['status'],'raw_solver':row,
                              'metrics':score(c,model,row.get('x'),refs[cid])})
    crossgroups=[]
    for arm in ['C0','GOLD']+cfg['arms']:
        for backend in cfg['cross_backends']:
            rows=[r for r in cross if r['arm']==arm and r['backend']==backend]
            valid=[r for r in rows if r['semantic']!='invalid'];verified=[r for r in rows if r['semantic']=='verified_correct']
            crossgroups.append({'arm':arm,'backend':backend,'full_denominator':n,
                                'valid_model_denominator':len(valid),'verified_model_denominator':len(verified),
                                'status_counts':dict(Counter(r['status'] for r in rows)),
                                'original_feasible':sum(bool(r['metrics'].get('original_feasible')) for r in rows),
                                'optimal_hit':sum(bool(r['metrics'].get('optimal_hit')) for r in rows),
                                'quality5':sum(bool(r['metrics'].get('quality5')) for r in rows),
                                'generated_model_feasible':sum(bool(r['metrics'].get('predicted_feasible')) for r in valid),
                                'generated_model_optimality_proven':sum(bool(r['raw_solver'].get('optimality_proven')) for r in valid),
                                'verified_subset_optimal_hit':sum(bool(r['metrics'].get('optimal_hit')) for r in verified)})
    paired=[];lookup={(r['case_id'],r['arm'],r['backend']):r for r in cross}
    for backend in cfg['cross_backends']:
        for other in ['C0','S','R','X','A','M']:
            counts={}
            for metric in ('original_feasible','optimal_hit','quality5'):
                tally=Counter()
                for cid in byid:
                    a=lookup[cid,'P',backend]['metrics'];b=lookup[cid,other,backend]['metrics']
                    if a['status']!='scored' or b['status']!='scored':tally['unscored']+=1;continue
                    av,bv=bool(a[metric]),bool(b[metric]);tally['wins' if av>bv else 'losses' if av<bv else 'ties']+=1
                counts[metric]=dict(tally)
            paired.append({'left':'P','right':other,'final_backend':backend,'denominator':n,'metrics':counts})
    contract_status=Counter();contract_semantics=Counter()
    for cid in byid:
        p=root/'contracts'/(s.key(cid)+'.json');row=s.load(p) if p.exists() else {'status':'not_completed'}
        contract_status[row['status']]+=1
        contract_semantics[semantic(cid,row.get('model'))]+=1
    costs=Counter();responses=Counter()
    for p in (root/'responses').glob('*.json'):
        row=s.load(p);responses[row['status']]+=1
        for field in ('input_tokens','output_tokens','generation_attempts','attributed_seconds'):costs[field]+=row.get(field,0)
        costs['cost_incomplete_requests']+=bool(row.get('cost_incomplete'))
    cpu=Counter()
    for p in (root/'outputs').glob('*.json'):
        row=s.load(p);cpu[row['status']]+=1
        costs['cpu_attributed_seconds']+=row.get('process_seconds',row.get('worker_seconds',0))
    clock=s.load(root/'clock.json') if (root/'clock.json').exists() else {}
    status=s.load(root/'status.json') if (root/'status.json').exists() else {}
    stopped=status.get('time') if status.get('stage')=='stopped' else None
    computation=s.load(root/'computation_stopped.json').get('time') if (root/'computation_stopped.json').exists() else None
    processed=sum(g['counts']['rounds_processed'] for g in groups);terminal=sum(r['status']!='not_completed' for r in cross)
    result={'experiment_id':m['experiment_id'],'smoke':m['smoke'],
            'scope':str(n)+' observed development cases; paired diagnostics, not held-out superiority evidence',
            'status':'all_stages_processed' if processed==n*len(cfg['arms']) and terminal==len(cross) and sum(semstatuses.values())==len(index['semantic']) else 'partial',
            'expected_trajectories':n*len(cfg['arms']),'rounds_processed_trajectories':processed,
            'cross_rows_expected':len(cross),'cross_rows_terminal':terminal,'groups':groups,'per_case':details,
            'semantic_rows_expected':len(index['semantic']),'semantic_rows_terminal':sum(semstatuses.values()),
            'cross_groups':crossgroups,'paired_same_final_backend':paired,
            'contract_status_counts':dict(contract_status),'semantic_status_counts':dict(semstatuses),
            'contract_semantic_counts':dict(contract_semantics),
            'round_status_counts':dict(stages),'response_status_counts':dict(responses),'cpu_status_counts':dict(cpu),
            'costs':dict(costs),'report_generated_at':time.time(),
            'run_wall_seconds':stopped-clock['started_at'] if stopped and clock else None,
            'compute_and_load_seconds':computation-clock['started_at'] if computation and clock else None,
            'wall_note':'run_wall_seconds covers launch through computation and first report, excludes archive export; regenerating reports never extends it',
            'denominator_note':'All failures retained; invalid-model rows are upstream failures, not solver failures. Same model hashes are shared jobs, not independent trials.',
            'unique_valid_model_hashes':len({r['model_sha256'] for r in cross if r['semantic']!='invalid'}),
            'historical_baselines':s.load(root/'evaluation/e003_baselines.json')}
    s.save(root/'summary.json',result);s.save(root/'cross_matrix.json',cross)
    lines=['# '+m['experiment_id']+' 对比（开发诊断集）','',f'状态：{result["status"]}；轨迹 {processed}/{n*len(cfg["arms"])}；交叉行 {terminal}/{len(cross)}。','',
           '| 组 | 可编译 | 核验正确 | 未判定 | 原题可行 | 最优命中 | 有反馈卡轮数 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        a=g['counts'];b=g['final_semantic']
        lines.append(f'| {g["arm"]} | {a["final_compiled"]}/{n} | {b.get("verified_correct",0)}/{n} | {b.get("unknown",0)} | {a["original_feasible"]}/{n} | {a["optimal_hit"]}/{n} | {a["rounds_with_cards"]}/{cfg["rounds"]*n} |')
    lines+=['',f'此表末次 p-bit 搜索为 {cfg["final_seconds"]} 秒。同模型七后端比较均为 {cfg["cross_solver_seconds"]} 秒，见 cross_matrix.json / summary.cross_groups。',
            'C0 是同一冻结初稿，GOLD 是标准模型；两者只在反馈冻结后求解。模型失败始终留在端到端分母中。',
            '同一模型的缓存复用不是独立重复；语义修复、未知、回归、反馈覆盖和配对胜负另列于 summary.json。',
            'all_stages_processed 包含失败，不能当作全成功。独立约定只是可错的模型假设，不能替代隐藏标准核验。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--pack',action='store_true');a=p.parse_args()
    result=report(a.run);print(result['status'],result['rounds_processed_trajectories'],'/',result['expected_trajectories'])
    if a.pack:print(pack(a.run))
