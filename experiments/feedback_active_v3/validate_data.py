"""Fast audit: provenance, frozen hashes, split isolation, valid IR."""
import argparse
from collections import Counter
from pathlib import Path
from qihc.experiment_runtime.storage import load, save
from qihc.generic.bounded import validate
from experiments.public_solver_v1.data import load_cases, validate_csv
from .data import digest


def run(data, workers=1):
    manifest=load(data/'manifest.json'); cases=load(data/'evaluation/cases.gold.json')
    public=load(data/'cases.public.json'); subsets=load(data/'subsets.json')
    assert manifest['formal_public_only'] and manifest['synthetic_formal_instances']==0
    assert digest(cases)==manifest['gold_sha256'] and digest(public)==manifest['public_sha256']
    assert digest(subsets)==manifest['subsets_sha256']
    original=data/'public_originals/nlco'
    for name,sha in manifest['nlco_source']['files_sha256'].items():
        task=name.split('/')[0]; tier=name.split('_')[1]
        validate_csv((original/name).read_bytes(),task,tier,sha)
    reproduced,excluded=load_cases(original)
    assert len(reproduced)==900 and not excluded
    assert {c['id']:c['gold'] for c in reproduced}=={c['id']:c['gold'] for c in cases}
    assert len({c['id'] for c in cases})==900
    assert len({c['question'] for c in cases})==900
    assert Counter(c['split'] for c in cases)==dict(development=120,main=480,scale=300)
    main={c['id'] for c in cases if c['split']=='main'}
    for label,count in [('ablation',120),('audit',120),('replay',60)]:
        assert len(set(subsets[label]))==count and set(subsets[label])<=main
    for c in public:
        assert set(c)=={'id','question','variable_names','family','layer','split'}
    external=load(data/'evaluation/cases.public_external.gold.json')
    assert len(external)==10 and all(c['id'].startswith('cpbench_') for c in external)
    for c in cases+external: validate(c['gold'])
    audit=dict(status='passed',nlco_cases=900,cpbench_cases=10,synthetic_formal_cases=0,
               published_nlco_witnesses_checked=900,split_overlap=0,
               scope='data_integrity_only; optimal references computed offline at runtime')
    save(data/'data_audit.json',audit)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,default=Path(__file__).parent/'assets/public_dataset')
    print(run(p.parse_args().data))
