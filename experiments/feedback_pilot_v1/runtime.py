"""Launch only the explicitly labelled exploratory pilot phase."""
import argparse
import signal
from pathlib import Path
from .adapter import install, default_root


def main():
    install()
    from experiments.feedback_study_v1 import runtime
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=default_root())
    p.add_argument('--cpu-only', action='store_true')
    a = p.parse_args()
    signal.signal(signal.SIGTERM, runtime.stop)
    signal.signal(signal.SIGINT, runtime.stop)
    return runtime.launch(a.run, a.cpu_only, phase='pilot')


if __name__ == '__main__':
    raise SystemExit(main())
