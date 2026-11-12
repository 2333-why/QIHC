"""CPU-only schema/search compatibility audit; never select methods by test quality."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import numpy as np
from qihc.experiment_runtime.pbit_control import ACTIONS, initial, advance
from qihc.experiment_runtime.storage import digest, load, save
from qihc.generic.factor_search import encoding


def audit(rows):
    counts=Counter();max_variables=0;max_bits=0
    for row in rows:
        spec=row['gold'];original=copy.deepcopy(spec)
        state=initial(spec,17,2)
        for action in ACTIONS:
            result=advance(spec,state,action,29,1)
            json.dumps(result,allow_nan=False)
            assert result['steps']==1 and len(result['history'])==1
            x=np.asarray(result['x'])
            for i,v in enumerate(spec['variables']):
                assert np.all(x[:,i]>=v['lower']) and np.all(x[:,i]<=v['upper'])
        assert spec==original
        counts[row['case']['stage']]+=1
        max_variables=max(max_variables,len(spec['variables']))
        max_bits=max(max_bits,len(encoding(spec)[0]))
    return dict(cases=len(rows),stages=dict(counts),actions_per_case=len(ACTIONS),
                max_variables=max_variables,max_bits=max_bits,
                scope='CPU compatibility only: two chains, one step/action; not full-budget performance or GPU validation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('gold_file',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();raw=load(a.gold_file)
    rows=[dict(case=dict(id=r['id'],stage=r['split']),gold=r['gold']) for r in raw if r['split'] in ('main','scale')]
    if len({r['case']['id'] for r in rows})!=780:raise ValueError('Expected 780 distinct main/scale cases')
    result=audit(rows);result['source_sha256']=digest(a.gold_file)
    if result['stages']!={'main':480,'scale':300}:raise ValueError('Wrong stage sizes')
    save(a.output,result);print(json.dumps(result,ensure_ascii=False))
