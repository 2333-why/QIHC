import hashlib
from types import SimpleNamespace
import pytest
from tools.benchmarks import parallel_inference as p
from tools.benchmarks import comparison as c
from tools.benchmarks.co_status import status


def test_shards_disjoint_and_feedback_first(tmp_path):
    for i in range(20):
        for phase in ('model','qihc'):
            c.enqueue(tmp_path, {'id':str(i)}, phase,'test')
    shards = [p.pending(tmp_path,i,2,100) for i in range(2)]
    names = [{path.name for path,r in shard} for shard in shards]
    assert not names[0]&names[1] and len(names[0]|names[1]) == 40
    for shard in shards:
        phases=[r['phase'] for path,r in shard]
        assert phases == sorted(phases,key=lambda s:s!='qihc')
    path,r=shards[0][0]; c.save(tmp_path/'responses'/path.name,{'status':'generated'})
    assert path.name not in {v.name for v,_ in p.pending(tmp_path,0,2,100)}


def test_guarded_migration_preserves_response(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'code_hash',lambda:'NEW')
    monkeypatch.setattr(p,'model_identity',lambda path:[])
    c.save(tmp_path/'cases.CPU_ONLY.json',[])
    manifest={'settings':{'source':p.LEGACY_SOURCE,'model':'model','model_files':[]},
              'cases_sha256':hashlib.sha256((tmp_path/'cases.CPU_ONLY.json').read_bytes()).hexdigest()}
    c.save(tmp_path/'manifest.json',manifest)
    c.enqueue(tmp_path,{'id':'a'},'model','Q')
    req=next((tmp_path/'requests').glob('*.json'))
    resp=tmp_path/'responses'/req.name
    c.save(resp,{'response':'original','request_sha256':hashlib.sha256(req.read_bytes()).hexdigest()})
    original=resp.read_bytes()
    p.migrate(tmp_path)
    assert resp.read_bytes()==original
    assert c.load(tmp_path/'manifest.before_parallel.json')==manifest
    assert c.load(tmp_path/'manifest.json')['settings']['source']=='NEW'
    p.migrate(tmp_path)
    manifest['settings']['source']='UNRECOGNIZED'
    c.save(tmp_path/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Unknown'):p.migrate(tmp_path)


def test_batched_padding_and_usage():
    torch=pytest.importorskip('torch')
    class Model:
        generation_config=SimpleNamespace(eos_token_id=9)
        def generate(self,**kw):
            assert kw['input_ids'].tolist()==[[0,1],[1,2]]
            assert kw['attention_mask'].tolist()==[[0,1],[1,1]]
            return torch.tensor([[0,1,5,9,0],[1,2,6,7,9]])
    g=SimpleNamespace(torch=torch,device='cpu',processor=None,max_input_tokens=10,max_new_tokens=8,
          tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=9,decode=lambda tokens,**kw:str(tokens)),
          model=Model(),_inputs=lambda text:{'input_ids':torch.tensor([[1]] if text=='a' else [[1,2]])})
    rows=p.batch_generate(g,[{'prompt':'a'},{'prompt':'b'}])
    assert [r['input_tokens'] for r in rows]==[1,2]
    assert [r['output_tokens'] for r in rows]==[2,3]


def test_split_retry_and_status(tmp_path,monkeypatch):
    class OOM(RuntimeError):pass
    generator=SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM,empty_cache=lambda:None)))
    def generate(g,reqs):
        if len(reqs)>1:raise OOM('test memory')
        return [{'response':'ok','status':'generated','output_tokens':2}]
    monkeypatch.setattr(p,'batch_generate',generate)
    for i in range(2):c.enqueue(tmp_path,{'id':str(i)},'model','Q')
    items=p.pending(tmp_path,0,1,4)
    p.execute(generator,items,tmp_path,0,'batch')
    s=status(tmp_path)
    assert s['responses']==2 and s['output_tokens']==4
    assert s['pending_by_phase']=={}
    assert len(list((tmp_path/'inference_events').glob('*.json')))==1


def test_bad_response_refuses_migration(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'code_hash',lambda:'NEW')
    monkeypatch.setattr(p,'model_identity',lambda path:[])
    c.save(tmp_path/'cases.CPU_ONLY.json',[])
    c.save(tmp_path/'manifest.json',{'settings':{'source':p.LEGACY_SOURCE,'model':'model','model_files':[]},
        'cases_sha256':hashlib.sha256((tmp_path/'cases.CPU_ONLY.json').read_bytes()).hexdigest()})
    c.enqueue(tmp_path,{'id':'a'},'model','Q')
    req=next((tmp_path/'requests').glob('*.json'))
    c.save(tmp_path/'responses'/req.name,{'request_sha256':'wrong'})
    with pytest.raises(ValueError,match='integrity'):p.migrate(tmp_path)
    assert c.load(tmp_path/'manifest.json')['settings']['source']==p.LEGACY_SOURCE
