"""Small deterministic dataset archive, verified before restoring exact files."""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import tarfile

from qihc.experiment_runtime.storage import load, save

HERE = Path(__file__).parent / 'assets'


def package(directory, archive=HERE / 'public_dataset.tar.gz'):
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob('*') if p.is_file()}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open('wb') as dest, gzip.GzipFile(fileobj=dest, mode='wb', mtime=0) as zipped, tarfile.open(fileobj=zipped, mode='w') as tar:
        for name, path in sorted(files.items()):
            payload = path.read_bytes(); info = tarfile.TarInfo(name); info.size = len(payload); info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
    manifest = dict(archive=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    files={n: hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()})
    save(archive.with_suffix('.lock.json'), manifest)
    return manifest


def restore(directory, archive=HERE / 'public_dataset.tar.gz'):
    lock = load(archive.with_suffix('.lock.json'))
    if not archive.exists():
        for name, expected in lock['files'].items():
            path = directory / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Git dataset missing or changed: ' + name)
        return lock
    if hashlib.sha256(archive.read_bytes()).hexdigest() != lock['sha256']: raise ValueError('Dataset archive hash mismatch')
    seen = set(); directory = directory.resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in lock['files'] or member.name in seen:
                raise ValueError('Unexpected archive member')
            target = (directory / member.name).resolve()
            if directory not in target.parents: raise ValueError('Unsafe archive path')
            payload = tar.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != lock['files'][member.name]: raise ValueError('Dataset member hash mismatch')
            if target.exists() and target.read_bytes() != payload: raise ValueError('Existing dataset differs: ' + str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.write_bytes(payload)
            seen.add(member.name)
    if seen != set(lock['files']): raise ValueError('Incomplete dataset archive')
    return lock


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('action', choices=['package','restore']); p.add_argument('--directory',type=Path,default=HERE/'public_dataset')
    a=p.parse_args(); result=package(a.directory) if a.action=='package' else restore(a.directory)
    print(result['sha256'],len(result['files']))
