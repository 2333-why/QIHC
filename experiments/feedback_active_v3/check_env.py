"""Offline dependency check; GPU jobs must never wait for package downloads."""
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import sys


def main():
    from packaging.requirements import Requirement
    here=Path(__file__).parent
    files=['requirements-cpu.txt']+(['requirements-gpu.txt'] if '--gpu' in sys.argv else [])
    errors=[]
    print('Interpreter:',sys.executable,flush=True)
    for name in files:
        for line in (here/name).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'): continue
            req=Requirement(line)
            try:
                installed=version(req.name)
                if installed not in req.specifier: errors.append(f'{req.name}: installed {installed}, required {req.specifier}')
            except PackageNotFoundError: errors.append(f'{req.name}: missing, required {req.specifier}')
    if errors:
        print('\n'.join(errors),file=sys.stderr)
        print('Run bash experiments/feedback_active_v3/00_setup.sh in a network-enabled terminal using the SAME QIHC_ENV; then resubmit. No installation attempted.',file=sys.stderr)
        return 1
    print('Offline dependency versions passed; GPU/native runtime checks follow.')
    return 0


if __name__=='__main__': sys.exit(main())
