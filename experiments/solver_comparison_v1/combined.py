"""One allocated node: E001 GPU modeling + E002 CPU replay, isolated results."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from qihc.experiment_runtime import storage as s

STOP = False


def stop(*_):
    global STOP
    STOP = True


def main():
    from experiments.prompt_efficiency_v3 import study
    from .run import prepare
    source = Path(os.environ['QIHC_PROMPT_V3_RUN']).resolve()
    root = Path(os.environ['QIHC_E002_RUN']).resolve()
    study.validate(source, model=True)
    prepare(root, source)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with s.worker_lock(root/'combined.lock'):
        children = []; streams = []; logs = root/'logs'; logs.mkdir(exist_ok=True)
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+str(os.getpid())
        try:
            for label, module, args in [
                ('prompt', 'experiments.prompt_efficiency_v3.run', ['launch', '--run', str(source)]),
                ('solvers', 'experiments.solver_comparison_v1.run', ['run', '--run', str(root), '--source', str(source), '--follow'])]:
                path = logs/(stamp+'-'+label+'.log'); stream = path.open('ab', buffering=0); streams.append(stream)
                proc = subprocess.Popen([sys.executable, '-u', '-m', module, *args], stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                children.append(proc); print(f'START {label}: {path}', flush=True)
            last = 0
            while not STOP:
                codes = [p.poll() for p in children]
                if any(code is not None and code != 0 for code in codes):
                    for stream in streams:
                        text = Path(stream.name).read_text(encoding='utf-8', errors='replace')
                        print(stream.name+'\n'+text[-12000:], flush=True)
                    raise RuntimeError('Child failed; checkpoints retained; exit codes='+str(codes))
                if all(code is not None for code in codes):
                    return 0
                if time.monotonic()-last >= 30:
                    print('HEARTBEAT '+str({name: s.load(path/'status.json') if (path/'status.json').exists() else 'starting'
                        for name, path in [('E001-1.3', source), ('E002-1.1', root)]}), flush=True)
                    last = time.monotonic()
                time.sleep(2)
            return 130
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
            deadline = time.monotonic()+60
            while any(p.poll() is None for p in children) and time.monotonic() < deadline:
                time.sleep(.2)
            for p in children:
                if p.poll() is None:
                    if os.name != 'nt':
                        os.killpg(p.pid, signal.SIGKILL)
                    else:
                        p.kill()
                p.wait()
            for stream in streams:
                stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
