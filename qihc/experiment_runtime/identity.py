"""Model metadata identity; no weight downloads or modifications."""
from .storage import digest

def model_identity(directory):
    """Local metadata fingerprint; weight content hashes remain the preparation manifest's job."""
    if not directory.is_dir():
        raise ValueError(f'Missing local model/adapter: {directory}')
    files = []
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.jinja'):
            stat = p.stat()
            files.append({'path': p.relative_to(directory).as_posix(), 'bytes': stat.st_size,
                          'mtime_ns': stat.st_mtime_ns,
                          'sha256': digest(p) if p.suffix in ('.json', '.jinja') else None})
    if not files:
        raise ValueError('Empty model/adapter directory')
    return files
