"""Find already downloaded solver wheels without importing native extensions."""
import argparse
from importlib.metadata import distributions
from pathlib import Path

REQUIRED = {'highspy': '1.15.1', 'pyscipopt': '6.3.0'}


def find_staging(parent):
    parent = Path(parent).resolve()
    candidates = sorted(parent.glob('.e002-solvers-*'), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if not candidate.is_dir() or candidate.is_symlink() or candidate.resolve().parent != parent:
            continue
        versions = {d.metadata['Name'].lower(): d.version for d in distributions(path=[str(candidate)])
                    if d.metadata.get('Name')}
        if all(versions.get(k) == v and (candidate/k/'__init__.py').is_file() for k, v in REQUIRED.items()):
            return candidate
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent', type=Path)
    args = p.parse_args()
    candidate = find_staging(args.parent)
    print(str(candidate) if candidate else '')


if __name__ == '__main__':
    main()
