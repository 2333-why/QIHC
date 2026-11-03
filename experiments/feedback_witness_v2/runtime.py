"""Fixed-cohort, two-round scheduler with persistent deadlines and no retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import BASE, HERE, ENTRY, s, immutable, validate, public_case, cpu_job, job_paths, configuration
from .prepare import prepare
from .protocol import request


from experiments.feedback_witness_v1.runtime import GPUWorkers


def prepare_process(root, source, model, timeout):
    # A timed-out preparation must not leave its native-solver descendants running.
    cmd=[sys.executable,'-u','-m',ENTRY+'.prepare','--run',str(root),'--source',str(source),'--model',str(model)]
    proc=subprocess.Popen(cmd,start_new_session=os.name!='nt')
    try:
        code=proc.wait(timeout=timeout)
        if code:raise subprocess.CalledProcessError(code,cmd)
    except BaseException:
        if os.name!='nt':
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        elif proc.poll() is None:proc.kill()
        proc.wait(timeout=5)
        p=root/'prepare_progress.json'
        print('PREPARATION FAILED; last stage:',s.load(p) if p.exists() else 'before progress initialization',flush=True)
        raise


def process_jobs(root,jobs,timeout,deadline,workers,inline=False):
    unique={};aliases={}
    for key,job in jobs:
        digest=s.value_digest(job)
        if key in aliases:raise ValueError('Duplicate logical job key')
        aliases[key]=digest;unique.setdefault(digest,job)
    def one(item):
        digest,job=item
        if inline:
            from .cpu import execute
            ip,op=job_paths(root,job);immutable(ip,job)
            if op.exists():
                row=s.load(op)
                if row.get('job_sha256')!=s.digest(ip):raise ValueError('Cached job mismatch')
            else:
                try:row=execute(job)
                except Exception as exc:row={'status':'execution_error','error':str(exc)[:1600],'x':None}
                row['job_sha256']=s.digest(ip);s.save(op,row)
        else:row=cpu_job(root,job,timeout,deadline)
        return digest,row
    # Submit once per mathematical job, then fan out immutable result snapshots.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results=dict(pool.map(one,unique.items()))
    return {key:copy.deepcopy(results[digest]) for key,digest in aliases.items()}


def run_pipeline(root,m,gpus,deadline,inline=False):
    cfg=m['config'];cases=s.load(root/'cases.public.json');byid={c['id']:public_case(c) for c in cases}
    workers=cfg['cpu_workers'];cleanup=cfg['cpu_stage_cleanup_seconds']
    feedback_deadline=deadline-cfg['stage_reservations_seconds']['offline_evaluation']
    if hasattr(gpus,'deadline'):gpus.deadline=min(gpus.deadline,feedback_deadline)
    def cpus(jobs,limit):return process_jobs(root,jobs,limit,feedback_deadline,workers,inline)
    contracts=[]
    for c in cases:
        case=public_case(c);req=request(case,'contract',cfg)
        contracts.append((s.key('contract',c['id']),req))
    responses=gpus.batch(contracts)
    compiled=cpus([(c['id'],{'stage':'contract_compile','config':cfg,'case':byid[c['id']],
                            'response':responses[s.key('contract',c['id'])]}) for c in cases],cfg['guard_seconds']+cleanup)
    for cid,row in compiled.items():s.save(root/'contracts'/(s.key(cid)+'.json'),row)
    states={}
    for c in cases:
        for arm in cfg['arms']:
            tid=s.key(c['id'],arm,'0');path=root/'states'/(tid+'.json')
            if path.exists():st=s.load(path)
            else:
                initial=s.load(root/'initial'/(s.key(c['id'])+'.json'))
                cr=responses[s.key('contract',c['id'])]
                st={'id':tid,'case_id':c['id'],'arm':arm,'seed':0,'round':0,'model':initial.get('model'),
                    'initial_model_sha256':s.value_digest(initial.get('model')),'initial_error':initial.get('error'),
                    'input_tokens':cr.get('input_tokens',0),'rounds':[],'x':None}
                s.save(path,st)
            states[tid]=st
    for round_no in range(cfg['rounds']):
        active={k:st for k,st in states.items() if st['round']==round_no}
        if not active:continue
        s.save(root/'status.json',{'stage':'feedback','round':round_no,'time':time.time(),'trajectories':len(states)})
        print('FEEDBACK ROUND',round_no+1,'active',len(active),flush=True)
        def payload(tid,stage):
            st=active[tid]
            return {'stage':stage,'config':cfg,'model':st['model'],'seed':st['seed']+round_no*100,
                    'arm':st['arm'],'trajectory_id':tid,'round':round_no}
        bases=cpus([(k,payload(k,'base')) for k in active],cfg['base_seconds']+cleanup)
        probes=cpus([(k,dict(payload(k,'probe'),base=bases[k],
                            contract=compiled[st['case_id']].get('model'),quotes=compiled[st['case_id']].get('quotes',[])))
                     for k,st in active.items()],cfg['probe_seconds']+cleanup)
        reqs=[];request_keys={}
        for k,st in active.items():
            cr=compiled[st['case_id']]
            req=request(byid[st['case_id']],'patch',cfg,model=st['model'],contract=cr,
                        evidence=probes[k],error=st.get('error',st.get('initial_error')),
                        remaining=max(0,cfg['trajectory_input_tokens']-st['input_tokens']))
            req.update(trajectory_id=k,round=round_no)
            rk=s.key('patch',k,str(round_no));request_keys[k]=rk;reqs.append((rk,req))
        rows=gpus.batch(reqs)
        gates=cpus([(k,dict(payload(k,'guard'),case=byid[st['case_id']],response=rows[request_keys[k]],
                           contract=compiled[st['case_id']].get('model'),evidence=probes[k])) for k,st in active.items()],cfg['guard_seconds']+cleanup)
        final_jobs=[]
        for k,st in active.items():
            gate=gates[k]
            if gate.get('status')=='accepted':st['model']=gate['model']
            final_jobs.append((k,dict(payload(k,'final'),model=st['model'])))
        finals=cpus(final_jobs,cfg['final_seconds']+cleanup)
        for k,st in active.items():
            response=rows[request_keys[k]];gate=gates[k]
            st['rounds'].append({'round':round_no,'base':bases[k],'probe':probes[k],
                                 'request_key':request_keys[k],'guard':gate,'final':finals[k],
                                 'accepted_model_sha256':s.value_digest(st['model'])})
            st.update(round=round_no+1,x=finals[k].get('x'),error=gate.get('error',gate.get('reason')),
                      input_tokens=st['input_tokens']+response.get('input_tokens',0))
            s.save(root/'states'/(k+'.json'),st)
    # All feedback is frozen before the scheduler even opens evaluation inputs.
    evaluation(root,m,states,deadline,inline)
    return states


def evaluation(root,m,states,deadline,inline=False):
    cfg=m['config'];cases={c['id']:c for c in s.load(root/'evaluation/cases.CPU_ONLY.json')}
    semantic=[];cross=[];index={'semantic':{},'cross':{}}
    for cid,c in cases.items():
        initial=s.load(root/'initial'/(s.key(cid)+'.json'))['model']
        contract=s.load(root/'contracts'/(s.key(cid)+'.json')).get('model')
        hk=s.key(cid,'hypothesis')
        semantic.append((hk,{'stage':'semantic','config':cfg,'model':contract,'gold':c['gold']}))
        index['semantic'][hk]={'case_id':cid,'kind':'hypothesis','model_sha256':s.value_digest(contract)}
        for name,model in [('initial',initial),('canonical',c['gold'])]:
            key=s.key(cid,name);job={'stage':'semantic','config':cfg,'model':model,'gold':c['gold']}
            semantic.append((key,job));index['semantic'][key]={'case_id':cid,'kind':name,'model_sha256':s.value_digest(model)}
            for backend in cfg['cross_backends']:
                ck=s.key(cid,name,backend)
                cross.append((ck,{'stage':'cross','config':cfg,'model':model,'backend':backend,'seed':0}))
                index['cross'][ck]={'case_id':cid,'kind':name,'arm':'C0' if name=='initial' else 'GOLD',
                                   'backend':backend,'model_sha256':s.value_digest(model)}
    for tid,st in states.items():
        sk=s.key(tid,'final')
        semantic.append((sk,{'stage':'semantic','config':cfg,'model':st['model'],'gold':cases[st['case_id']]['gold']}))
        index['semantic'][sk]={'case_id':st['case_id'],'trajectory_id':tid,'kind':'final',
                               'model_sha256':s.value_digest(st['model'])}
        for r in st['rounds']:
            candidate=r['guard'].get('candidate')
            key=s.key(tid,'candidate',str(r['round']))
            semantic.append((key,{'stage':'semantic','config':cfg,'model':candidate,'gold':cases[st['case_id']]['gold']}))
            index['semantic'][key]={'case_id':st['case_id'],'trajectory_id':tid,'kind':'candidate','round':r['round'],
                                   'model_sha256':s.value_digest(candidate),'candidate_produced':candidate is not None}
        for backend in cfg['cross_backends']:
            key=s.key(tid,backend)
            cross.append((key,{'stage':'cross','config':cfg,'model':st['model'],'backend':backend,'seed':0}))
            index['cross'][key]={'case_id':st['case_id'],'trajectory_id':tid,'arm':st['arm'],'backend':backend,
                                'model_sha256':s.value_digest(st['model'])}
    immutable(root/'evaluation/index.json',index)
    s.save(root/'status.json',{'stage':'offline_evaluation','time':time.time()})
    print('OFFLINE EVALUATION',len(cross),'cross-solver rows',len(semantic),'semantic rows',flush=True)
    rows=process_jobs(root,cross,cfg['cross_solver_seconds']+cfg['cross_job_overhead_seconds'],deadline,cfg['cpu_workers'],inline)
    for key,row in rows.items():s.save(root/'evaluation/cross'/(key+'.json'),row)
    rows=process_jobs(root,semantic,cfg['semantic_check_seconds']+cfg['semantic_job_overhead_seconds'],deadline,cfg['cpu_workers'],inline)
    for key,row in rows.items():s.save(root/'evaluation/semantic'/(key+'.json'),row)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=BASE/'results/E004-1.5_feedback_witness_6h')
    p.add_argument('--source',type=Path,default=BASE/'results/E004-1.2_feedback_pilot')
    p.add_argument('--model',type=Path,default=BASE/'models/Qwen3-32B');a=p.parse_args();root=a.run.resolve()
    root.mkdir(parents=True,exist_ok=True)
    def interrupted(*_):raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with s.worker_lock(root/'coordinator.lock'):
        cfg=configuration();clock=root/'clock.json'
        if not clock.exists():
            now=time.time();s.save(clock,{'started_at':now,'work_deadline':now+cfg['work_deadline_seconds'],
                                        'hard_deadline':now+cfg['wall_limit_seconds']})
        clock=s.load(clock);gpus=None
        try:
            if time.time()>=clock['work_deadline']:raise TimeoutError('Original work deadline has passed')
            s.save(root/'status.json',{'stage':'preparing','time':time.time(),'hard_deadline':clock['hard_deadline']})
            available=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()
            allocated=int(os.environ.get('SLURM_CPUS_PER_TASK',available))
            if min(available,allocated)<cfg['cpu_workers']:raise ValueError('Need 24 allocated CPU cores')
            # Preparation runs in a child so dependency checks cannot consume the whole job.
            if not (root/'manifest.json').exists():
                prepare_process(root,a.source,a.model,
                                min(cfg['stage_reservations_seconds']['prepare'],max(.01,clock['work_deadline']-time.time())))
            m=validate(root)
            if m['source_run']!=str(a.source.resolve()) or m['model']!=str(a.model.resolve()):
                raise ValueError('Resume source/model arguments differ')
            if m['smoke']:raise ValueError('Production entry refuses smoke manifests')
            validate(root)
            if (root/'stop.json').exists():(root/'stop.json').unlink()
            s.save(root/'status.json',{'stage':'loading_models','time':time.time()})
            feedback_deadline=clock['work_deadline']-cfg['stage_reservations_seconds']['offline_evaluation']
            gpus=GPUWorkers(root,m,feedback_deadline)
            run_pipeline(root,m,gpus,clock['work_deadline'])
        except BaseException as exc:
            s.save(root/'fatal.json',{'error':repr(exc),'time':time.time()})
            raise
        finally:
            if gpus:gpus.close()
            s.save(root/'computation_stopped.json',{'time':time.time()})
            if (root/'manifest.json').exists():
                from .report import report,pack
                summary=report(root)
                archive=root.parent/(root.name+'_analysis.tar.gz')
                s.save(root/'status.json',{'stage':'stopped','status':summary['status'],'time':time.time(),
                                           'rounds_processed_trajectories':summary['rounds_processed_trajectories'],
                                           'archive':str(archive),'time_scope':'through computation and first report; excludes archive export'})
                # Include the persisted stop time in the automatic report/archive, not just a later manual export.
                report(root);archive=pack(root)
                s.save(root/'export_finished.json',{'time':time.time(),'archive':str(archive)})
                print('REPORT',summary['status'],'ARCHIVE',archive,flush=True)


if __name__=='__main__':main()
