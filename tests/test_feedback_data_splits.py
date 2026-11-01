"""Regress the E004 preparation failure reported on 2026-11-01."""
import copy
import pytest
from experiments.feedback_study_v1 import data
from experiments.feedback_study_v1.common import HERE, s
from experiments.feedback_study_v1.prepare import prepare


def test_full_default_split_reproduces_and_repairs_seed_collisions():
    cfg=s.load(HERE/'config.json')
    formal=data.extension(cfg)
    before=s.value_digest(formal)
    formal_hashes={s.value_digest(c['gold']) for c in formal}
    original=data.development(cfg)
    collisions=[c for c in original if s.value_digest(c['gold']) in formal_hashes]
    assert [c['generation_seed'] for c in collisions]==[202611032,202611056]
    audit=[]; dev=data.development(cfg, excluded=formal, audit=audit)
    hashes=[s.value_digest(c['gold']) for c in dev]
    assert len(dev)==120 and len(formal)==1200
    assert len(set(hashes))==120 and not set(hashes)&formal_hashes
    assert sum(c['split']=='development_debug' for c in dev)==60
    assert sum(c['split']=='development_acceptance' for c in dev)==60
    assert len(audit)==2 and s.value_digest(formal)==before
    repeated=[]
    assert data.development(cfg,excluded=list(reversed(formal)),audit=repeated)==dev
    assert repeated==audit
    replaced={row['index'] for row in audit}
    assert all(a==b for k,(a,b) in enumerate(zip(original,dev)) if k not in replaced)


def test_public_collision_uses_mathematics_not_ids_or_seed_metadata():
    cfg=s.load(HERE/'config.json');cfg['development_cases']=2
    frozen=copy.deepcopy(data.development(cfg)[0])
    frozen.update(id='public-with-an-unrelated-id',generation_seed=-1,split='public')
    audit=[];dev=data.development(cfg,excluded=[frozen],audit=audit)
    assert len(dev)==2 and len(audit)==1
    assert audit[0]['rejected'][0]['collides_with']==frozen['id']
    assert s.value_digest(dev[0]['gold'])!=s.value_digest(frozen['gold'])


def test_collision_retry_is_bounded(monkeypatch):
    cfg=s.load(HERE/'config.json');cfg['development_cases']=1
    fixed=data.make_case('KP',8,1,'public')
    monkeypatch.setattr(data,'make_case',lambda *args,**kwargs:copy.deepcopy(fixed))
    with pytest.raises(ValueError,match='after 100 attempts'):
        data.development(cfg,excluded=[fixed])


def test_prepare_persists_split_audit_and_is_replayable(tmp_path):
    cfg=s.load(HERE/'config.json');cfg.update(development_cases=2,extension_per_cell=0)
    case=copy.deepcopy(data.development(cfg)[0]);case.update(id='frozen-e003',split='public')
    source=tmp_path/'source';root=tmp_path/'run'
    s.save(source/'cases.CPU_ONLY.json',[case])
    s.save(source/'manifest.json',{'cases_digest':s.value_digest([case])})
    s.save(source/'models'/(s.key(case['id'],'llm_typed_flat')+'.json'),
           {'model':case['gold'],'model_sha256':s.value_digest(case['gold'])})
    before={p.relative_to(source):s.digest(p) for p in source.rglob('*.json')}
    first=prepare(root,source,tmp_path/'mock-model',cfg,smoke=True)
    audit=s.load(root/'evaluation/split_audit.json')
    assert audit['status']=='passed' and len(audit['replacements'])==1
    assert first==prepare(root,source,tmp_path/'mock-model',cfg,smoke=True)
    assert before=={p.relative_to(source):s.digest(p) for p in source.rglob('*.json')}
    public=s.load(root/'cases.public.json')
    assert len(public)==3 and all('gold' not in c and 'rejected' not in c for c in public)
    frozen=next(c for c in s.load(root/'evaluation/cases.CPU_ONLY.json') if c['id']=='frozen-e003')
    assert frozen['gold']==case['gold']
