"""Durable per-stage state machine; one allocation, independent CPU/GPU queues.

The controller never reads evaluator scores when routing or repairing models.
Atomic responses and stage hashes make replays idempotent after interruption.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import s, validate, default_root, identity, configurations, immutable
from .protocol import build_request, apply_patch, feedback_packet

STOP = False
ENTRY = 'experiments.feedback_study_v1'


def stop(*_):
    global STOP
    STOP = True


def state_path(root, key):
    return root/'states'/(key+'.json')


def save_state(root, state):
    row = dict(state); model = row.pop('model', None)
    if model is not None:
        sha = s.value_digest(model); immutable(root/'mathematics'/(sha+'.json'), model)
        row['model_sha256'] = sha
    else:
        row['model_sha256'] = None
    s.save(state_path(root, state['id']), row)


def load_state(root, key):
    row = s.load(state_path(root, key)); sha = row.get('model_sha256')
    row['model'] = s.load(root/'mathematics'/(sha+'.json')) if sha else None
    if sha and s.value_digest(row['model']) != sha:
        raise ValueError('Mathematics content hash mismatch')
    return row


def enqueue(root, case, state, stage, cfg, packet=None):
    preparation_error=None
    try:
        request = build_request(case, state, stage, '__E004_EVIDENCE__' if packet is not None else None)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,RecursionError) as exc:
        preparation_error=repr(exc)
        request={'prompt':'','stage':stage,'case_id':case['id'],'preparation_error':preparation_error}
    # The JSON-quoted marker must be replaced by an object, not a JSON string.
    request['prompt'] = request['prompt'].replace('"__E004_EVIDENCE__"', '__E004_EVIDENCE__')
    request['_max_new_tokens'] = cfg['initial_output_tokens'] if stage == 'initial' else (
        cfg['review_output_tokens'] if stage == 'review' else cfg['patch_output_tokens'])
    request['input_remaining'] = cfg['trajectory_input_tokens']-state.get('input_tokens', 0)
    if packet is not None:
        request['evidence_payload'] = packet
    key = s.key(case['id'], state.get('id', 'initial'), str(state.get('round', 0)), stage)
    path = root/'requests'/(key+'.json'); immutable(path, request)
    if preparation_error:
        from .gpu import terminal_error
        terminal_error(root,path,'Request preparation failed: '+preparation_error)
    if not (root/'responses'/path.name).exists():
        immutable(root/'pending'/str(int(key,16)%cfg['gpu_replicas'])/path.name, {'request_sha256':s.digest(path)})
    return key


def consume(root, key, state):
    path = root/'responses'/(key+'.json')
    if not path.exists():
        return None
    row = s.load(path)
    if row['request_sha256'] != s.digest(root/'requests'/path.name):
        raise ValueError('Response/request hash mismatch')
    if key not in state.setdefault('consumed_requests', []):
        state['consumed_requests'].append(key)
        for metric in ('input_tokens','output_tokens','generation_attempts','attributed_seconds'):
            state[metric] = state.get(metric, 0)+(row.get(metric) or 0)
        state['cost_incomplete'] = state.get('cost_incomplete', False) or row.get('cost_incomplete', False)
    return row


def cpu_job(root, state, stage, cfg, case=None):
    key = s.key(state['id'], str(state['round']), stage)
    job = {'stage':stage, 'model':state.get('model'), 'config':cfg, 'arm':state['arm'],
           'seed':state['seed']+state['round']*1009, 'trajectory_seed':state['seed']}
    if stage == 'probe':
        job.update(base=state.get('base', {}), review=state.get('review', {}), question=case['question'])
        job['prior_compile_seconds']=state.get('review_compile_seconds',0.)
    if stage=='final':
        job['prior_compile_seconds']=state.get('patch_compile_seconds',0.)
    if stage in ('base','final'):
        job['initial'] = state.get('base', {}).get('final') if stage=='final' else state.get('warm_start')
        candidates = [state['x']] if state.get('x') is not None else []
        if stage == 'final':
            candidates += state.get('base',{}).get('states',[])
            if state.get('base',{}).get('x') is not None:
                candidates.append(state['base']['x'])
            passive = state.get('probe',{}).get('passive',{})
            candidates += passive.get('states',[])
            if passive.get('x') is not None:
                candidates.append(passive['x'])
            for p in state.get('probe',{}).get('probes',[]):
                for w in p.get('windows',[]):
                    candidates += w.get('states',[])
        job['archive'] = candidates
    path = root/'cpu_jobs'/(key+'.json'); immutable(path, job)
    return key, path, root/'cpu_results'/(key+'.json')


def advance(root, case, state, cfg, schedule):
    """At most one durable transition per invocation; returns terminal state."""
    phase = state['stage']; arm = state['arm']
    if phase in ('base','probe','final'):
        if state.get('model') is None:
            row = {'status':'invalid_model', 'x':None, 'seconds':0.}
        else:
            key, job, output = cpu_job(root, state, phase, cfg, case)
            if not output.exists():
                schedule(key, job, output, cfg[{'base':'base_seconds','probe':'probe_seconds','final':'final_seconds'}[phase]])
                return False
            row = s.load(output)
            if row['job_sha256'] != s.digest(job):
                raise ValueError('CPU output hash mismatch')
        state['cpu_seconds'] = state.get('cpu_seconds', 0.)+row.get('wall_seconds', row.get('seconds', 0.))
        if row.get('status') != 'complete':
            state.setdefault('errors', []).append({'round':state['round'], 'stage':phase, 'error':row.get('error',row['status'])})
        else:
            state['searched_windows'] = state.get('searched_windows', 0)+1
        if phase == 'base':
            state['base'] = row
            state['base_model_sha256']=s.value_digest(state['model']) if state.get('model') else None
            state['stage'] = 'probe' if arm == 'C0' else 'review'
        elif phase == 'probe':
            state['probe'] = row
            skip = arm == 'C0'
            if arm == 'event':
                probes = row.get('probes', [])
                score = max((r.get('score', 0.) for r in row.get('routing', [])), default=0.)
                distinct = {tuple(x) for p in probes for w in p.get('windows',[]) for x in w.get('states',[])}
                old = {tuple(x) for x in state.get('base',{}).get('states',[])}
                skip = score < cfg['event_min_score'] or len(distinct-old) < cfg['event_min_new_states']
                if skip:
                    state['event_skips'] = state.get('event_skips',0)+1
            state['stage'] = 'final' if skip else 'patch'
        else:
            # Final-model-only candidate archive. Diagnostic interventions never
            # directly supply final solutions from a different mathematics.
            state['x'] = row.get('x')
            if state['x'] is None and state.get('model') is not None and s.value_digest(state['model'])==state.get('base_model_sha256'):
                state['x']=state.get('base',{}).get('x')
                row['fallback']='retained candidate under unchanged mathematics; final search failed'
            state['warm_start'] = row.get('final') or None
            snapshot = dict(state, final=row)
            save_snapshot(root, snapshot)
            memory = {'round':state['round'], 'model_sha256':s.value_digest(state['model']) if state.get('model') else None,
                      'patch':state.get('patch'), 'witnesses':state.get('base',{}).get('states',[])[:2]}
            if arm == 'C1':
                memory.pop('witnesses')
            elif arm == 'C2':
                memory['witnesses'] = [state['base']['x']] if state.get('base',{}).get('x') is not None else []
            state['memory'] = (state.get('memory',[])+[memory])[-2:] if arm != 'no_memory' else []
            for field in ('base','probe','review','patch','request_id','review_compile_seconds','patch_compile_seconds'):
                state.pop(field, None)
            state['round'] += 1
            state['stage'] = 'done' if state['round'] >= cfg['rounds'] else 'base'
    elif phase in ('review','patch'):
        if state.get('generation_attempts',0) >= cfg['rounds']*2:
            state['stage'] = 'probe' if phase == 'review' else 'final'
        else:
            packet = feedback_packet(arm, state.get('base',{}), state.get('probe',{})) if phase == 'patch' else None
            if 'request_id' not in state:
                state['request_id'] = enqueue(root, case, state, phase, cfg, packet)
                save_state(root, state); return False
            response = consume(root, state['request_id'], state)
            if response is None:
                return False
            compile_started=time.monotonic()
            try:
                if response['status'] != 'generated':
                    raise ValueError(response.get('error', response['status']))
                payload = json.loads(response['response'])
                if state.get('model') is None:
                    from qihc.modeling.typed_sources import compile_math
                    model, reason = compile_math(response['response'], case['question'])
                    if model is None or [v['name'] for v in model['variables']] != case['variable_names']:
                        raise ValueError('Recovered variable mapping mismatch: '+str(reason))
                    state['model'] = model
                elif phase == 'patch':
                    state['model'] = apply_patch(state['model'], payload, case['question'])
                    state['patch'] = payload
                    state['valid_patch_outputs']=state.get('valid_patch_outputs',0)+1
                    state['citation_present'] = bool(payload.get('quote')) and payload['quote'] in case['question']
                else:
                    state['review'] = payload
                    state['valid_review_outputs']=state.get('valid_review_outputs',0)+1
                state.pop('error', None)
            except (ValueError, KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
                state['error'] = str(exc)
                state.setdefault('errors',[]).append({'round':state['round'], 'stage':phase, 'error':str(exc)})
            state[phase+'_compile_seconds']=time.monotonic()-compile_started
            state['cpu_seconds']=state.get('cpu_seconds',0.)+state[phase+'_compile_seconds']
            state.pop('request_id', None)
            state['stage'] = 'probe' if phase == 'review' else 'final'
    save_state(root, state)
    return state['stage'] == 'done'


def save_snapshot(root, state):
    row = dict(state); row.pop('model', None)
    row['model_sha256'] = s.value_digest(state['model']) if state.get('model') else None
    if state.get('model'):
        immutable(root/'mathematics'/(row['model_sha256']+'.json'), state['model'])
    immutable(root/'rounds'/state['id']/(str(state['round'])+'.json'), row)


def initial(root, case, cfg):
    path = root/'initial'/(s.key(case['id'])+'.json')
    if path.exists():
        return s.load(path)
    state = {'id':'initial', 'round':0, 'model':None, 'arm':'initial'}
    key = enqueue(root, case, state, 'initial', cfg)
    response = consume(root, key, state)
    if response is None:
        return None
    row = {'model':None, 'error':None, 'generation_cost':{k:state.get(k,0) for k in
            ('input_tokens','output_tokens','generation_attempts','attributed_seconds')}, 'response_sha256':s.value_digest(response)}
    try:
        from qihc.modeling.typed_sources import compile_math
        if response['status'] != 'generated':
            raise ValueError(response.get('error',response['status']))
        row['model'], reason = compile_math(response['response'], case['question'])
        if row['model'] is None or [v['name'] for v in row['model']['variables']] != case['variable_names']:
            raise ValueError('Initial variable mapping mismatch: '+str(reason))
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        row.update(model=None, error=str(exc))
    immutable(path, row); return row


class Processes:
    def __init__(self, root, cfg):
        self.root, self.cfg, self.active = root, cfg, {}
        self.crashes = 0

    def schedule(self, key, job, output, seconds):
        if key in self.active or output.exists() or len(self.active) >= self.cfg['cpu_workers']:
            return
        logpath = self.root/'logs'/'cpu'/f'{key}.log'; logpath.parent.mkdir(parents=True, exist_ok=True)
        log = logpath.open('ab', buffering=0)
        proc = subprocess.Popen([sys.executable,'-u','-m',ENTRY+'.cpu',str(job),str(output)],
                                stdout=log, stderr=subprocess.STDOUT)
        self.active[key] = (proc,log,job,output,time.monotonic(),seconds)
        s.save(self.root/'cpu_execution_events'/(key+'-'+str(proc.pid)+'.json'),
               {'status':'started','time':time.time(),'job_sha256':s.digest(job),'budget_seconds':seconds})

    def poll(self):
        for key, (proc,log,job,out,start,seconds) in list(self.active.items()):
            code = proc.poll(); timedout = time.monotonic()-start > seconds+self.cfg['cpu_watchdog_grace']
            if code is None and not timedout:
                continue
            if code is None:
                proc.kill(); proc.wait()
            if not out.exists():
                s.save(out, {'status':'hard_timeout' if timedout else 'worker_crash', 'x':None,
                    'error':f'exit={code}; log={log.name}', 'job_sha256':s.digest(job),
                    'wall_seconds':time.monotonic()-start})
                self.crashes += 1
            else:
                self.crashes = 0
            s.save(self.root/'cpu_execution_events'/(key+'-'+str(proc.pid)+'.json'),
                   {'status':'complete','process_seconds':time.monotonic()-start,'job_sha256':s.digest(job),
                    'exit_code':code,'timedout':timedout,'time':time.time(),
                    'scope':'includes process startup, imports, work and file publication'})
            log.close(); del self.active[key]
        if self.crashes >= 20:
            raise RuntimeError('20 consecutive CPU process crashes/timeouts; checkpointed, inspect logs')

    def close(self):
        for proc,log,*_ in self.active.values():
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
            log.close()


def evaluate_tasks(root, state, case, cfg, procs):
    # Only this evaluator scheduling path loads gold; feedback requests/CPU jobs
    # are produced from the separate public case file.
    refpath = root/'evaluation'/'references'/(s.key(case['id'])+'.json')
    if not refpath.exists():
        key = 'ref-'+s.key(case['id']); path = root/'evaluation'/'jobs'/(key+'.json')
        immutable(path, {'stage':'reference','case':case,'config':cfg})
        procs.schedule(key,path,refpath,cfg['reference_seconds']); return False
    out = root/'evaluation'/'scores'/(state['id']+'.json')
    if not out.exists():
        path = root/'evaluation'/'jobs'/(state['id']+'.json')
        job = {'stage':'evaluate','case':case,'config':cfg,'arm':state['arm'],'seed':state['seed'],
               'model':state.get('model'),'x':state.get('x'),'reference':s.load(refpath)}
        rounds=[s.load(p) for p in sorted((root/'rounds'/state['id']).glob('*.json'))]
        job['initial_x']=rounds[0].get('base',{}).get('x') if rounds else None
        job['round_candidates']=[{'round':r['round'],'x':r.get('x')} for r in rounds]
        immutable(path, job); procs.schedule('eval-'+state['id'],path,out,cfg['diagnostic_seconds']+2)
        return False
    return True


def launch(root, cpu_only=False, phase='all'):
    m = validate(root); cfg = m['config']
    if not cpu_only:
        import torch
        if torch.cuda.device_count() != 4:
            raise RuntimeError('One worker with four visible GPUs required; no torchrun wrapper')
        pf = s.load(root/'preflight.json')
        if pf['source_hash'] != m['source_hash'] or pf['status'] != 'passed':
            raise ValueError('Run E004 CPU preparation for this checkout first')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    if not cpu_only and (len(visible) != 4 or len(set(visible)) != 4):
        raise ValueError('Invalid GPU visibility')
    cases = s.load(root/'cases.public.json')
    if s.digest(root/'evaluation'/'cases.CPU_ONLY.json') != m['evaluation_digest']:
        raise ValueError('Evaluator dataset changed')
    gold_cases = {c['id']:c for c in s.load(root/'evaluation'/'cases.CPU_ONLY.json')}
    groups = ['development_debug','development_acceptance','public','extension'] if phase=='all' else [phase]
    if phase in ('public','extension'):
        acceptance = root/'acceptance.json'
        if not acceptance.exists() or not s.load(acceptance).get('passed') or s.load(acceptance).get('source_hash')!=m['source_hash']:
            raise ValueError('Complete real-model development acceptance before formal-only launch')
    replicas = cfg['gpu_replicas']
    if replicas not in (2,4):
        raise ValueError('GPU replicas must be 2 or 4')
    children = {}; restarts = {r:0 for r in range(replicas)}; procs = Processes(root,cfg)
    started = time.monotonic(); last_report = 0.; idle_since = None
    def spawn(rank):
        path = root/'logs'/f'gpu-{rank}-{time.time_ns()}.log'; path.parent.mkdir(parents=True,exist_ok=True)
        log = path.open('ab',buffering=0)
        width=4//replicas
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(visible[rank*width:rank*width+width]),
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        p = subprocess.Popen([sys.executable,'-u','-m',ENTRY+'.gpu','--run',str(root),'--rank',str(rank)],
                             stdout=log,stderr=subprocess.STDOUT,env=env)
        children[rank] = (p,log,time.time()); print('GPU START',rank,str(path),flush=True)
    def check_gpus():
        from .gpu import terminal_error
        for rank,(proc,log,born) in list(children.items()):
            hbpath = root/f'gpu-{rank}.json'; hb = s.load(hbpath) if hbpath.exists() else {}
            timedout = time.time()-max(born,hb.get('time',0)) > cfg['gpu_watchdog_seconds']
            code = proc.poll()
            if code is None and not timedout:
                continue
            if code is None:
                proc.kill(); proc.wait()
            if hb.get('status')=='generating' and hb.get('time',0)>=born:
                for key in hb.get('requests',[]):
                    terminal_error(root,root/'requests'/(key+'.json'),f'GPU exit={code}, watchdog={timedout}; {log.name}',attempted=True)
            log.close(); del children[rank]
            if restarts[rank] >= cfg['max_gpu_restarts']:
                raise RuntimeError('Repeated GPU infrastructure failures; results retained, inspect '+str(log.name))
            restarts[rank] += 1; spawn(rank)
    with s.worker_lock(root/'launch.lock'):
        try:
            # A recycled process may have started inference without publishing a
            # response. Preserve an explicit unknown-cost terminal attempt; never
            # replay it as a free call or consume the same budget twice.
            from .gpu import terminal_error
            for event in (root/'inference_events').glob('*.json'):
                row=s.load(event)
                if row.get('status')=='started':
                    for key in row['requests']:
                        terminal_error(root,root/'requests'/(key+'.json'),'Interrupted inference; tokens unknown, attempt charged',attempted=True)
            s.save(root/'control.json',{'status':'running'})
            if not cpu_only:
                for rank in range(replicas):
                    spawn(rank)
            for group in groups:
                selected = [c for c in cases if c['split']==group]
                # Round-robin by task/scale and index, not exhausting one family.
                selected.sort(key=lambda c:(c.get('generation_seed',0)%1000,c.get('size',0),c['dataset'],c['id']))
                todo = [(c,a,seed,identity(c['id'],a,seed)) for c in selected for a,seed in configurations(c,cfg)]
                states = {}; finished = set(); cursor = 0
                for c,a,seed,key in todo:
                    path = state_path(root,key)
                    if path.exists() and s.load(path).get('stage')=='done' and (root/'evaluation'/'scores'/(key+'.json')).exists():
                        finished.add(key)
                initial_done = len(finished); group_start = time.monotonic()
                while len(finished) < len(todo) and not STOP:
                    procs.poll(); check_gpus()
                    # Prepare bounded lookahead initial requests, keeping model
                    # work available while CPU stages run. New initials shared.
                    checked = 0
                    initials = {}
                    while len(states) < cfg['inflight_trajectories'] and checked < min(len(todo), cfg['inflight_trajectories']*2):
                        c,a,seed,key = todo[cursor%len(todo)]; cursor += 1; checked += 1
                        if key in states or key in finished:
                            continue
                        path = state_path(root,key)
                        if path.exists():
                            state = load_state(root,key)
                        else:
                            if c['id'] not in initials:
                                initials[c['id']] = initial(root,c,cfg)
                            ini = initials[c['id']]
                            if ini is None:
                                continue
                            state = {'id':key,'case_id':c['id'],'arm':a,'seed':seed,'round':0,'stage':'base',
                                     'model':ini['model'],'error':ini.get('error'),'initial_valid':ini['model'] is not None,
                                     'split':group,'dataset':c['dataset'],'size':c.get('size')}
                            save_state(root,state)
                        states[key] = (c,state)
                    for key,(c,state) in list(states.items()):
                        if state['stage'] != 'done':
                            advance(root,c,state,cfg,procs.schedule)
                        if state['stage']=='done' and evaluate_tasks(root,state,gold_cases[c['id']],cfg,procs):
                            finished.add(key); del states[key]
                    pending_count = 0
                    for rank in range(replicas):
                        for marker in (root/'pending'/str(rank)).glob('*.json'):
                            if (root/'responses'/marker.name).exists():
                                marker.unlink(missing_ok=True)
                            else:
                                pending_count += 1
                    idle_since = None if pending_count else (idle_since or time.monotonic())
                    if cpu_only and pending_count and not procs.active:
                        s.save(root/'status.json',{'status':'needs_gpu','phase':group,'pending_requests':pending_count,'time':time.time()})
                        return 0
                    if not cpu_only and idle_since and time.monotonic()-idle_since > cfg['idle_checkpoint_seconds']:
                        s.save(root/'status.json',{'status':'checkpointed_gpu_idle','phase':group,'time':time.time()})
                        print('GPU idle checkpoint; release allocation and resume. No synthetic GPU occupancy.',flush=True)
                        return 0
                    if time.monotonic()-last_report >= cfg['report_interval_seconds']:
                        delta = len(finished)-initial_done; elapsed = time.monotonic()-group_start
                        state = {'status':'running','phase':group,'completed':len(finished),'expected':len(todo),
                                 'active_cpu':len(procs.active),'active_trajectories':len(states),'pending_requests':pending_count,
                                 'eta_seconds':elapsed/delta*(len(todo)-len(finished)) if delta>=10 else None,
                                 'eta_scope':'current phase terminal throughput, includes failures; future phases excluded',
                                 'time':time.time(),'restarts':restarts}
                        s.save(root/'status.json',state); print(json.dumps(state),flush=True)
                        from .report import report
                        report(root, pairs=False); last_report = time.monotonic()
                    time.sleep(.2)
                if STOP:
                    break
                if group == 'development_acceptance':
                    searched = {a:0 for a in cfg['public_arms']}
                    for c,a,seed,key in todo:
                        st = s.load(state_path(root,key))
                        searched[a] += int(st.get('searched_windows',0)>0 and st.get('valid_patch_outputs',0)>0)
                    # C0 intentionally has no repair calls; its initial is real.
                    searched['C0'] = sum(int(s.load(state_path(root,k)).get('searched_windows',0)>0) for c,a,se,k in todo if a=='C0')
                    accepted = {'passed':all(v>0 for v in searched.values()),'searched':searched,
                                'scope':'real generation/search execution only, not semantic quality','source_hash':m['source_hash']}
                    s.save(root/'acceptance.json',accepted)
                    if not accepted['passed']:
                        raise RuntimeError('Real model execution acceptance failed; inspect development records. Formal run not launched.')
            from .report import report
            summary=report(root,pairs=True)
            s.save(root/'status.json',{'status':'stopped_checkpointed' if STOP else
                                      'complete' if summary['status']=='complete' else 'phase_complete',
                                      'requested_phase':phase,'time':time.time(),
                                      'seconds_this_launch':time.monotonic()-started})
        except BaseException as exc:
            s.save(root/'status.json',{'status':'infrastructure_failed','error':repr(exc),'time':time.time()})
            raise
        finally:
            s.save(root/'control.json',{'status':'stopped'})
            for proc,log,born in children.values():
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
                log.close()
            procs.close()
    return 130 if STOP else 0


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run',type=Path,default=default_root())
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--phase',choices=['all','development_debug','development_acceptance','public','extension'],default='all')
    a = p.parse_args(); signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    return launch(a.run,a.cpu_only,a.phase)


if __name__ == '__main__':
    raise SystemExit(main())
