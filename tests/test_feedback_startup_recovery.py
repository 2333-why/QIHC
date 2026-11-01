import copy
import pytest
from experiments.feedback_study_v1 import recover_startup as r
from experiments.feedback_study_v1.common import s


def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'fingerprint',lambda:'new-code')
    monkeypatch.setattr(r,'commit',lambda:'new-commit')
    # validate() independently checks fingerprint in its defining module.
    from experiments.feedback_study_v1 import common
    monkeypatch.setattr(common,'fingerprint',lambda:'new-code')
    s.save(tmp_path/'cases.public.json',[{'id':'a'}])
    s.save(tmp_path/'evaluation/cases.CPU_ONLY.json',[{'id':'a','gold':{}}])
    cfg={'gpu_replicas':4}
    m={'config':cfg,'config_digest':s.value_digest(cfg),'source_hash':r.OLD_SOURCE_HASH,'code_commit':'old-commit',
       'smoke':True,'public_digest':s.digest(tmp_path/'cases.public.json'),
       'evaluation_digest':s.digest(tmp_path/'evaluation/cases.CPU_ONLY.json')}
    pf={'status':'passed','source_hash':r.OLD_SOURCE_HASH,'checked':2220}
    s.save(tmp_path/'manifest.json',m);s.save(tmp_path/'preflight.json',pf)
    s.save(tmp_path/'initial/a.json',{'model':None,'source_record_sha256':'frozen'})
    s.save(tmp_path/'requests/a.json',{'stage':'initial','prompt':'old queued prompt'})
    s.save(tmp_path/'pending/0/a.json',{'request_sha256':s.digest(tmp_path/'requests/a.json')})
    s.save(tmp_path/'logs/old.json',{'error':'original startup error'})
    return m,pf


def test_audited_recovery_preserves_data_queue_and_replays(tmp_path,monkeypatch):
    old,pf=prepared(tmp_path,monkeypatch)
    before={p:s.digest(p) for p in tmp_path.rglob('*.json') if p.name not in ('manifest.json','preflight.json')}
    assert r.recover(tmp_path)['status']=='recovered_pre_inference'
    receipt=s.load(tmp_path/'recovery/startup-devicecheck-v1.json')
    assert receipt['before_manifest']==old and receipt['before_preflight']==pf
    assert receipt['after_manifest']['config']==old['config']
    assert all(s.digest(p)==digest for p,digest in before.items())
    assert s.load(tmp_path/'manifest.json')['source_hash']=='new-code'
    # After a successful migration, later resume can have actual trajectories.
    s.save(tmp_path/'states/now_running.json',{'stage':'base'})
    assert r.recover(tmp_path)['status']=='already_recovered'


@pytest.mark.parametrize('folder',['responses','inference_events','states','rounds','cpu_execution_events','evaluation/scores','evaluation/jobs','cpu_jobs','cpu_results'])
def test_never_relabels_a_run_that_attempted_work(tmp_path,monkeypatch,folder):
    old,pf=prepared(tmp_path,monkeypatch);s.save(tmp_path/folder/'work.json',{'status':'failed'})
    with pytest.raises(ValueError,match='pre-inference'):r.recover(tmp_path)
    assert s.load(tmp_path/'manifest.json')==old
    assert not (tmp_path/'recovery/startup-devicecheck-v1.json').exists()


def test_refuses_unknown_source_or_changed_data(tmp_path,monkeypatch):
    old,pf=prepared(tmp_path,monkeypatch)
    bad=copy.deepcopy(old);bad['source_hash']='unknown';s.save(tmp_path/'manifest.json',bad)
    with pytest.raises(ValueError,match='audited'):r.recover(tmp_path)
    s.save(tmp_path/'manifest.json',old);s.save(tmp_path/'cases.public.json',[])
    with pytest.raises(ValueError,match='data changed'):r.recover(tmp_path)


def test_interrupted_two_file_migration_is_recoverable(tmp_path,monkeypatch):
    old,pf=prepared(tmp_path,monkeypatch);save=s.save
    def interrupted(path,row):
        if path==tmp_path/'preflight.json' and row['source_hash']=='new-code':
            raise OSError('simulated write interruption')
        save(path,row)
    monkeypatch.setattr(s,'save',interrupted)
    with pytest.raises(OSError,match='interruption'):r.recover(tmp_path)
    assert s.load(tmp_path/'manifest.json')['source_hash']=='new-code'
    assert s.load(tmp_path/'preflight.json')==pf
    monkeypatch.setattr(s,'save',save)
    assert r.recover(tmp_path)['status']=='recovered_pre_inference'
    assert r.recover(tmp_path)['status']=='already_recovered'


def test_active_launch_lock_prevents_migration(tmp_path,monkeypatch):
    old,pf=prepared(tmp_path,monkeypatch)
    with s.worker_lock(tmp_path/'launch.lock'):
        with pytest.raises(OSError):r.recover(tmp_path)
    assert s.load(tmp_path/'manifest.json')==old


def test_different_future_patch_cannot_reuse_startup_migration(tmp_path,monkeypatch):
    old,pf=prepared(tmp_path,monkeypatch)
    monkeypatch.setattr(r,'patch_fingerprint',lambda:'unreviewed-algorithm-change')
    with pytest.raises(ValueError,match='audited device-check patch'):r.recover(tmp_path)
    assert s.load(tmp_path/'manifest.json')==old
