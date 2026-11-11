"""Offline four-GPU queue; failures terminate the process tree; identical command resumes."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from qihc.experiment_runtime.storage import load,save,digest,worker_lock
from qihc.experiment_runtime.identity import model_identity
from .common import CFG,RUN,HISTORY,MODEL,TRAIN_ARMS,cases,check_paths,source_hashes,environment_identity

def validate():
    check_paths(); m=load(RUN/'manifest.json')
    if m['config']!=CFG or m['history']!=str(HISTORY) or m['model']!=str(MODEL): raise ValueError('Preparation mismatch')
    if m['source']!=source_hashes(): raise ValueError('Source changed after preparation')
    if m['environment']!=environment_identity(): raise ValueError('Shared Python environment changed after preparation')
    attrs=lambda fs:[(r['path'],r['bytes'],r['mtime_ns']) for r in fs]
    if attrs(model_identity(MODEL))!=attrs(m['fingerprints']): raise ValueError('Model changed')
    for name,h in m['input_files'].items():
        if digest(RUN/name)!=h: raise ValueError('Input changed '+name)
    for name,h in {**m['donor_files'],**m['data_files']}.items():
        if digest(Path(name))!=h: raise ValueError('Frozen dependency changed '+name)

def gpu_check():
    import torch
    if torch.cuda.device_count()!=4: raise RuntimeError('Allocate exactly four GPUs')
    details=[]
    for i in range(4):
        with torch.cuda.device(i):
            p=torch.cuda.get_device_properties(i)
            if p.total_memory<75*1024**3 or not torch.cuda.is_bf16_supported(): raise RuntimeError('Requires four BF16 GPUs with at least 75 GiB each')
            x=torch.ones((64,64),device=f'cuda:{i}',dtype=torch.bfloat16,requires_grad=True)
            (x@x).float().mean().backward(); torch.cuda.synchronize(i)
            details.append(dict(index=i,name=p.name,total_memory=p.total_memory))
    try:
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=15).strip()
    except (OSError,subprocess.SubprocessError): driver='unavailable; CUDA BF16 runtime check passed'
    save(RUN/'gpu_preflight.json',dict(torch=torch.__version__,cuda=torch.version.cuda,driver=driver,devices=details))

def stop_process(proc):
    if proc.poll() is not None:return
    if os.name=='posix': os.killpg(proc.pid,signal.SIGTERM)
    else: proc.terminate()
    try: proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name=='posix': os.killpg(proc.pid,signal.SIGKILL)
        else: proc.kill()
        proc.wait()

def launch(mode):
    if mode not in ('pilot','formal'): raise ValueError('Use pilot or formal')
    validate(); gpu_check()
    if mode=='formal':
        from .report import report
        report('pilot')
        gate=load(RUN/'pilot/gate.json')
        if not gate['passed']: raise RuntimeError('Pilot mechanism gate failed: '+str(gate['reasons']))
    ids=os.getenv('CUDA_VISIBLE_DEVICES','0,1,2,3').split(',')
    if len(ids)!=4 or len(set(ids))!=4: raise ValueError('Ambiguous CUDA assignment')
    folder=RUN/mode; seeds=[17] if mode=='pilot' else CFG['seeds']
    tasks=[(a,s) for s in seeds for a in CFG['arms'] if not (folder/'models'/str(s)/a/'done.json').exists()]
    live={}; watcher=None
    with worker_lock(RUN/'launcher.lock'):
        marker=folder/'generated.json'
        if tasks and marker.exists(): marker.unlink()
        watcher=subprocess.Popen([sys.executable,'-m','experiments.pvcd_cycle_v1.score',mode],start_new_session=True)
        try:
            while tasks or live:
                if watcher.poll() is not None: raise RuntimeError('Scoring exited before GPU tasks finished')
                for gpu,(proc,log) in list(live.items()):
                    if proc.poll() is None: continue
                    log.close(); del live[gpu]
                    if proc.returncode: raise RuntimeError('Worker failed: '+log.name)
                for gpu in ids:
                    if gpu in live or not tasks: continue
                    arm,seed=tasks.pop(0); path=folder/'logs'/f'{seed}-{arm}.log';path.parent.mkdir(parents=True,exist_ok=True)
                    log=path.open('a')
                    proc=subprocess.Popen([sys.executable,'-m','experiments.pvcd_cycle_v1.worker',arm,str(seed),mode],
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    live[gpu]=(proc,log)
                time.sleep(2)
            save(marker,dict(arms=CFG['arms']))
            if watcher.wait(): raise RuntimeError('Scoring/report failed')
        finally:
            for proc,log in live.values(): stop_process(proc);log.close()
            if watcher is not None: stop_process(watcher)

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    launch(sys.argv[1])
