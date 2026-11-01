import copy
import pytest
from experiments.feedback_comparison_v1.report import build,metrics,compare
from qihc.experiment_runtime import resilient_storage as s


def fixture(tmp_path):
    a,b,out=tmp_path/'e003',tmp_path/'e004',tmp_path/'comparison'
    gold={'schema':'bounded-integer-v2','variables':[{'name':'x0','lower':0,'upper':1}],
          'objective':{'sense':'max','linear':[1],'quadratic':[]},'constraints':[],'logic':[]}
    cases=[{'id':str(i),'dataset':'KP','size':1,'question':str(i),'gold':gold} for i in range(2)]
    s.save(a/'cases.CPU_ONLY.json',cases)
    s.save(a/'manifest.json',{'cases_digest':s.value_digest(cases),
           'config':{'seeds':[0,1,2,3,4],'seconds':[1,5,20],'backends':['cp_sat','pbit']}})
    cfg={'public_arms':['C0','C1','U2','B1','B2'],'seeds':[0,1,2],
         'rounds':3,'base_seconds':4,'probe_seconds':10,'final_seconds':6}
    s.save(b/'manifest.json',{'source_manifest_sha256':s.digest(a/'manifest.json'),'config':cfg})
    s.save(b/'evaluation/cases.CPU_ONLY.json',[dict(c,split='public') for c in cases])
    for c in cases:
        cid=c['id'];ref={'gold_sha256':s.value_digest(gold),'result':{'objective':1,'feasible':True,'optimality_proven':True}}
        s.save(a/'references'/(s.key(cid)+'.json'),ref)
        for track in ('llm_typed_flat','canonical'):
            mp=a/'models'/(s.key(cid,track)+'.json')
            s.save(mp,{'model':gold,'model_sha256':s.value_digest(gold),'response_sha256':'response'})
            if track=='llm_typed_flat':
                s.save(b/'initial'/(s.key(cid)+'.json'),{'source_record_sha256':s.digest(mp),'generation_cost':{'input_tokens':12}})
            for backend in ('cp_sat','pbit'):
                for seed in range(5):
                    s.save(a/'scores'/s.key(cid,track)/f'{backend}-20-{seed}.json',
                           {'case_id':cid,'source_arm':track,'backend':backend,'budget_seconds':20,'seed':seed,
                            'response_sha256':'response','model_sha256':s.value_digest(gold),
                            'verified_feasible':track=='canonical' or seed>=3,
                            'verified_objective':1,'status':'searched'})
        for arm in cfg['public_arms']:
            for seed in cfg['seeds']:
                key=s.key(cid,arm,str(seed));job=b/'evaluation/jobs'/(key+'.json')
                s.save(job,{'case':c,'arm':arm,'seed':seed})
                s.save(b/'evaluation/scores'/(key+'.json'),{'job_sha256':s.digest(job),
                       'original_feasible':arm=='U2','objective':1,'status':'complete'})
                s.save(b/'states'/(key+'.json'),{'generation_attempts':6,'output_tokens':100})
    return a,b,out


def test_shared_seeds_original_accuracy_and_read_only(tmp_path):
    a,b,out=fixture(tmp_path)
    before={p:s.digest(p) for root in (a,b) for p in root.rglob('*.json')}
    result=build(a,b,out,replicates=100)
    assert result['status']=='complete' and result['seeds']==[0,1,2]
    rows={r['method']:r for r in result['rows']}
    assert rows['E003/llm_typed_flat/cp_sat/20s']['original_feasible_assigned']==0
    assert rows['E003/canonical/cp_sat/20s']['optimal_hit_assigned']==1
    assert rows['E004/U2']['quality5_assigned']==1
    assert all(r['completed']==r['expected']==6 for r in result['rows'])
    assert result['budget']['E004_non_llm_cap_seconds']==60
    pair=next(p for p in result['paired'] if p['right']=='E003/llm_typed_flat/cp_sat/20s')
    assert pair['metrics']['original_feasible']['delta']==1 and pair['repaired_runs']==6
    assert pair['comparison']=='historical_extra_feedback_budget'
    assert result['shared_E003_initial_generation_cost_once']['input_tokens']==24
    assert before=={p:s.digest(p) for root in (a,b) for p in root.rglob('*.json')}


def test_pending_is_not_terminal_failure_or_incomplete_seed_pair(tmp_path):
    a,b,out=fixture(tmp_path)
    (b/'evaluation/scores'/(s.key('0','U2','2')+'.json')).unlink()
    result=build(a,b,out,replicates=100)
    row=next(r for r in result['rows'] if r['method']=='E004/U2')
    assert result['status']=='partial' and row['completed']==5
    assert row['original_feasible_completed']==1 and row['original_feasible_assigned']==5/6
    assert all(p['paired_instances']==1 and not p['complete'] for p in result['paired'])


@pytest.mark.parametrize('kind',['case','initial','score'])
def test_mismatched_evidence_rejected(tmp_path,kind):
    a,b,out=fixture(tmp_path)
    if kind=='case':
        p=b/'evaluation/cases.CPU_ONLY.json';rows=s.load(p);rows[0]['question']='changed';s.save(p,rows)
    elif kind=='initial':
        s.save(b/'initial'/(s.key('0')+'.json'),{'source_record_sha256':'wrong'})
    else:
        p=b/'evaluation/scores'/(s.key('0','U2','0')+'.json');r=s.load(p);r['job_sha256']='wrong';s.save(p,r)
    with pytest.raises(ValueError):build(a,b,out,replicates=10)


def test_unproven_and_infeasible_never_count_as_accuracy():
    case={'gold':{'objective':{'sense':'min'}}}
    ref={'result':{'objective':10,'feasible':True,'optimality_proven':False}}
    assert not metrics(True,10,case,ref)['optimal_hit']
    assert not metrics(True,10,case,ref)['quality5']
    ref['result']['optimality_proven']=True
    assert not metrics(False,10,case,ref)['optimal_hit']
    assert metrics(True,10.1,case,ref)['quality1']
    with pytest.raises(ValueError,match='proven optimum'):metrics(True,9,case,ref)


def test_refuses_output_inside_live_results(tmp_path):
    a,b,out=fixture(tmp_path)
    with pytest.raises(ValueError,match='outside'):build(a,b,b/'report')
