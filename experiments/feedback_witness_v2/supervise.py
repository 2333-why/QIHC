"""Enforce original absolute 6-hour deadline, including resume and packaging."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .common import BASE,HERE,ENTRY,s,configuration


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=BASE/'results/E004-1.5_feedback_witness_6h')
    p.add_argument('--source',type=Path,default=BASE/'results/E004-1.2_feedback_pilot')
    p.add_argument('--model',type=Path,default=BASE/'models/Qwen3-32B');a=p.parse_args();root=a.run.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with s.worker_lock(root/'supervisor.lock'):
        clock=root/'clock.json';cfg=configuration()
        if not clock.exists():
            now=time.time();s.save(clock,{'started_at':now,'work_deadline':now+cfg['work_deadline_seconds'],'hard_deadline':now+cfg['wall_limit_seconds']})
        deadline=s.load(clock)['hard_deadline'];remaining=deadline-time.time()
        if remaining<=30:raise SystemExit('Original six-hour deadline expired; use report command only')
        cmd=[sys.executable,'-m',ENTRY+'.runtime','--run',str(root),'--source',str(a.source),'--model',str(a.model)]
        proc=subprocess.Popen(cmd,start_new_session=os.name!='nt')
        def terminate(*_):
            if proc.poll() is None:
                if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
        signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
        try:code=proc.wait(timeout=max(.01,remaining-30))
        except subprocess.TimeoutExpired:
            terminate()
            try:code=proc.wait(timeout=max(.01,deadline-time.time()))
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
                proc.wait();code=124
            s.save(root/'hard_deadline.json',{'status':'hard_deadline_reached','time':time.time()})
        raise SystemExit(code)


if __name__=='__main__':main()
