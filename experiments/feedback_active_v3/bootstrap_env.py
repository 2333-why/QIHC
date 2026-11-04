"""Create/reuse an isolated environment even when system ensurepip is absent."""
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv


def prepare(directory):
    directory=Path(directory).resolve()
    python=directory/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    if not python.is_file():
        venv.EnvBuilder(with_pip=False).create(directory)
    if subprocess.run([str(python),'-m','pip','--version'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        # A conda env supplied explicitly is also supported; never install into
        # the system interpreter through an accidentally supplied path.
        actual=subprocess.check_output([str(python),'-c','import sys; print(sys.prefix)'],text=True).strip()
        if Path(actual).resolve()!=directory: raise ValueError('Environment prefix does not match requested directory')
        url='https://bootstrap.pypa.io/get-pip.py'
        print('Installing pip into',directory,'using',url,flush=True)
        with urllib.request.urlopen(url,timeout=90) as response: payload=response.read()
        script=directory/'get-pip.bootstrap.py';script.write_bytes(payload)
        (directory/'get-pip.bootstrap.sha256').write_text(hashlib.sha256(payload).hexdigest()+'\n')
        subprocess.run([str(python),str(script)],check=True)
    subprocess.run([str(python),'-m','pip','--version'],check=True)


if __name__=='__main__': prepare(sys.argv[1])
