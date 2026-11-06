"""Validate the host C toolchain before expensive model loading, offline."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def check():
    bindir=Path(sys.executable).parent
    candidates=[os.environ['CC']] if os.environ.get('CC') else [
        str(bindir/'gcc'),str(bindir/'clang'),
        *map(str,sorted(bindir.glob('*-linux-gnu-gcc'))),
        *map(str,sorted(bindir.glob('*-conda-linux-gnu-cc'))),'gcc','cc','clang']
    errors=[]
    for candidate in candidates:
        compiler=shutil.which(candidate)
        if not compiler: continue
        with tempfile.TemporaryDirectory(prefix='qihc-compiler-') as directory:
            src=Path(directory)/'probe.c';out=Path(directory)/'probe.so'
            src.write_text('#include <Python.h>\n#include <stdlib.h>\nint qihc_probe(void){return 0;}\n')
            result=subprocess.run([compiler,'-shared','-fPIC',str(src),'-I'+sysconfig.get_path('include'),'-o',str(out)],
                                  capture_output=True,text=True,timeout=30)
            if result.returncode==0:
                os.environ['CC']=compiler
                print('C compiler and Python headers passed:',compiler,flush=True)
                return compiler
            errors.append(compiler+': '+result.stderr[-2000:])
    raise RuntimeError('No working C compiler/Python headers for Triton. Use a GPU container with gcc and development headers, '
                       'or set CC to a complete shared compiler toolchain accessible on the GPU node. '
                       'Installing gcc only in a separate CPU container does not fix the GPU container. '+ '\n'.join(errors))


if __name__=='__main__': check()
