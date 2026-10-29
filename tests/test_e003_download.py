import shutil
import urllib.error

import pytest

from experiments.public_solver_v1 import data
from qihc.experiment_runtime import storage as s


def test_partial_twelve_files_complete_offline_without_rewriting(tmp_path, monkeypatch):
    lock = s.load(data.ASSETS/'nlco.lock.json'); blobs = data.bundled_files(lock)
    cached = {}
    for name, payload in blobs.items():
        if name.split('/')[0] in ('KP','MIS','MVC','MAXCUT'):
            p = tmp_path/name; p.parent.mkdir(exist_ok=True); p.write_bytes(payload)
            cached[name] = (p.stat().st_mtime_ns, s.digest(p))
    assert len(cached) == 12
    def no_network(*a, **kw):
        raise AssertionError('The fixed Git bundle must work without network')
    monkeypatch.setattr(data.urllib.request, 'urlopen', no_network)
    data.download(tmp_path)
    assert all((tmp_path/n).stat().st_mtime_ns == stamp and s.digest(tmp_path/n) == sha
               for n,(stamp,sha) in cached.items())
    cases, exclusions = data.load_cases(tmp_path)
    assert len(cases)==900 and not exclusions
    assert s.load(tmp_path/'download_manifest.json')['files_sha256'] == lock['files_sha256']
    before = {p:p.stat().st_mtime_ns for p in tmp_path.rglob('*.csv')}
    data.download(tmp_path)
    assert before == {p:p.stat().st_mtime_ns for p in tmp_path.rglob('*.csv')}


def test_corrupted_cache_is_not_silently_accepted(tmp_path):
    p = tmp_path/'KP/KP_S_context.csv'; p.parent.mkdir(); p.write_text('wrong bytes')
    with pytest.raises(ValueError, match='checksum'):
        data.download(tmp_path)
    assert p.read_text() == 'wrong bytes'


def test_bundle_corruption_is_detected(tmp_path, monkeypatch):
    lock = s.load(data.ASSETS/'nlco.lock.json')
    (tmp_path/lock['bundle']).write_bytes(b'bad bundle')
    monkeypatch.setattr(data,'ASSETS',tmp_path)
    with pytest.raises(ValueError,match='checksum'):
        data.bundled_files(lock)


def test_source_only_network_fallback_does_not_retry_403(tmp_path, monkeypatch):
    original = data.ASSETS; lock=s.load(original/'nlco.lock.json'); blobs=data.bundled_files(lock)
    assets=tmp_path/'assets'; assets.mkdir(); shutil.copyfile(original/'nlco.lock.json',assets/'nlco.lock.json')
    monkeypatch.setattr(data,'ASSETS',assets)
    monkeypatch.setenv('HF_ENDPOINT','https://hf-mirror.com')
    calls=[]
    import io
    def fetch(url, **kw):
        calls.append(url)
        if url.startswith('https://hf-mirror.com'):
            raise urllib.error.HTTPError(url,403,'Forbidden',None,None)
        return io.BytesIO(blobs[url.split(data.REVISION+'/')[1]])
    monkeypatch.setattr(data.urllib.request,'urlopen',fetch)
    data.download(tmp_path/'cache')
    assert len(calls)==36  # One refusal, then one official request per file.
