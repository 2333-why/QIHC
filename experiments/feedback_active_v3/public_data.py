"""Pinned public originals plus reviewed adapters; never execute dataset code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

from qihc.experiment_runtime.storage import load, save
from qihc.generic.bounded import validate
from .data import digest, public_view

SOURCES = {
    'cpbench_verified': 'https://huggingface.co/datasets/kostis-init/CP-Bench/resolve/ba3ef047747410c02fba7065bb86ad8914191191/dataset_verified.jsonl',
    'mamo_complex_lp': 'https://raw.githubusercontent.com/FreedomIntelligence/MAMO/2c91069dee5c1e4db4b899eb176ec987d0251bb3/data/optimization/Complex_LP/mamo_complex_lp.jsonl',
    'retailopt190': 'https://raw.githubusercontent.com/junbolian/ReLoop/0508ac656a98219ebf005cc0433442bc3a465bb8/data/RetailOpt-190.jsonl',
}


def literals(script):
    """Read only literal assignments from the published model; no eval/exec."""
    result = {}
    for node in ast.parse(script).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try: result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    return result


def cp_case(row):
    cid = row['id']; d = literals(row['model']); rows = []; sense = 'min'; note = ''; scale = 1
    aliases = []; domains = []; linear = []
    def setup(names, upper=1, cost=None):
        nonlocal aliases, domains, linear
        aliases = names; domains = [[0, u] for u in upper] if isinstance(upper, list) else [[0, upper] for _ in names]
        linear = [0] * len(names) if cost is None else cost
    def con(a, op, rhs): rows.append(dict(id=f'c{len(rows)}', a=a, op=op, rhs=rhs))
    if cid.endswith('__capital_budget'):
        setup([f'investment_{i+1}' for i in range(4)], cost=d['npv']); sense = 'max'; scale = 1000
        con(d['cash_flow'], 'le', d['budget'])
        note = 'Use thousands of dollars for objective and cash amounts (exact unit conversion).'
    elif cid.endswith('__chess_set'):
        setup(['small_set', 'large_set'], [53, 66], [5, 20]); sense = 'max'
        con([1, 3], 'le', 200); con([3, 2], 'le', 160)
    elif cid.endswith('__covering_opl'):
        setup([f'worker_{i+1}' for i in range(d['nb_workers'])], cost=d['Cost'])
        for group in d['Qualified']: con([int(i+1 in group) for i in range(len(aliases))], 'ge', 1)
    elif cid.endswith('__three_sum'):
        setup([f'item_{i}' for i in range(len(d['nums']))]); con(d['nums'], 'eq', 0); con([1]*len(aliases), 'eq', 3)
    elif cid.endswith('__2_subsets_100'):
        n = len(d['A']); setup([f'S_{i}' for i in range(n)] + [f'T_{i}' for i in range(n)])
        con(d['A'] + [-x for x in d['A']], 'eq', 0)
        con([1]*n+[0]*n, 'ge', 1); con([0]*n+[1]*n, 'ge', 1)
        for i in range(n): con([int(k in (i,n+i)) for k in range(2*n)], 'le', 1)
    elif cid.endswith('__2_subset_sum'):
        values = d['coin_numbers']; target = d['total_coins_lost']
        setup([f'bags_{v}' for v in values], [target//v for v in values]); con(values, 'eq', target)
    elif cid.endswith('__abbots_puzzle'):
        setup(['men','women','children'],100); con([1,1,1],'eq',100); con([6,4,1],'eq',200); con([5,-1,0],'eq',0)
    elif cid.endswith('__fifty_puzzle'):
        setup([f'dummy_{i}' for i in range(len(d['values']))]); con(d['values'],'eq',d['target_sum'])
    elif cid.endswith('__1_guards_and_apples'):
        setup([f'apples_{i}' for i in range(6)],100); con([0]*5+[1],'eq',1)
        for i in range(5):
            a=[0]*6; a[i]=1; a[i+1]=-2; con(a,'eq',2)
    elif cid.endswith('__1_money_change'):
        setup([f'coins_{v}' for v in d['types_of_coins']],d['available_coins'],[1]*6)
        con(d['types_of_coins'],'eq',d['amount'])
    else: return None
    n=len(aliases); names=[f'x{i}' for i in range(n)]
    gold=validate(dict(schema='bounded-integer-v2',variables=[dict(name=name,lower=b[0],upper=b[1]) for name,b in zip(names,domains)],
        objective=dict(sense=sense,linear=linear,quadratic=[]),constraints=rows,logic=[]))
    # Only source data literals and a naming convention are visible at inference.
    question=row['description']+'\nSOURCE DATA: '+row.get('input_data','')+'\nDecision convention: '+json.dumps(dict(zip(names,aliases)))+'. '+note
    if not any(linear): question+=' This is a satisfaction problem; use a constant-zero objective.'
    return dict(id='cpbench_'+cid,question=question,variable_names=names,gold=gold,split='public',family='CPBench',layer='published',
        objective_value_scale=scale,objective_offset=0,provenance=dict(source=SOURCES['cpbench_verified'],original_id=cid,
        adapter='explicit_reviewed_linear_adapter_v1',oracle_code_executed=False,coverage='supported_subset_not_all_CPBench'))


def prepare(output):
    rawdir=output/'public_originals'; rawdir.mkdir(parents=True,exist_ok=True); sources={}; corpus={}
    for name,url in SOURCES.items():
        path=rawdir/(name+'.jsonl')
        if not path.exists():
            with urllib.request.urlopen(url,timeout=90) as response: payload=response.read()
            path.write_bytes(payload)
        payload=path.read_bytes(); corpus[name]=[json.loads(x) for x in payload.decode('utf-8-sig').splitlines() if x.strip()]
        sources[name]=dict(url=url,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),rows=len(corpus[name]))
    accepted=[]; audit=[]
    for row in corpus['cpbench_verified']:
        case=cp_case(row)
        if case is not None: accepted.append(case); status='supported_reviewed_adapter'
        elif row['id'].endswith('__2_maximal_independent_sets'): status='excluded_source_model_mismatch_maximal_vs_maximum'
        elif row['id'].endswith('__twelve_pack'): status='excluded_arbitrary_reference_bounds_not_full_feasible_domain'
        elif row['id'].endswith('__assignment_costs'): status='pending_source_ambiguity_person_capacity'
        else: status='pending_adapter_and_semantic_review'
        audit.append(dict(dataset='CPBench_verified',id=row['id'],status=status))
    # MAMO and RetailOpt are preserved in full but not automatically rounded or
    # assigned guessed finite bounds to manufacture p-bit-compatible instances.
    for name in ('mamo_complex_lp','retailopt190'):
        for i,row in enumerate(corpus[name]):
            audit.append(dict(dataset=name,id=str(row.get('id',i)),status='pending_discrete_scope_and_reference_adapter'))
    save(output/'cases.public_external.json',[dict(public_view(c),family=c['family'],layer=c['layer'],split='public') for c in accepted])
    save(output/'evaluation/cases.public_external.gold.json',accepted)
    manifest=dict(sources=sources,original_rows=sum(v['rows'] for v in sources.values()),target_cases=120,
        supported_cases=len(accepted),public_sha256=digest([public_view(c) for c in accepted]),gold_sha256=digest(accepted),
        status='limited_external_pilot_not_120_case_paper_panel',audit=audit)
    save(output/'public.manifest.json',manifest)
    return {k:v for k,v in manifest.items() if k not in ('audit','sources')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).parent/'assets/public_dataset')
    a=p.parse_args();print(json.dumps(prepare(a.output),indent=2))
