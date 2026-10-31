"""Consistent per-file analysis snapshot; completed atomic records are immutable."""
import argparse
import tarfile
from pathlib import Path
from .common import s, default_root


def pack(root, output):
    with tarfile.open(output,'w:gz') as tar:
        for name in ('manifest.json','preflight.json','status.json','acceptance.json','summary.json','report.md',
                     'paired_public.json','paired_extension.json','cases.public.json'):
            p=root/name
            if p.exists():
                tar.add(p,arcname=name)
        for name in ('states','mathematics','rounds','evaluation','initial','requests','responses','inference_events','cpu_execution_events','logs'):
            for p in sorted((root/name).rglob('*')):
                if p.is_file() and not p.name.startswith('.'):
                    tar.add(p,arcname=p.relative_to(root).as_posix())
    checksum=s.digest(output); output.with_suffix(output.suffix+'.sha256').write_text(checksum+'\n',encoding='utf-8')
    print(checksum)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=default_root());p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();pack(a.run,a.output)
