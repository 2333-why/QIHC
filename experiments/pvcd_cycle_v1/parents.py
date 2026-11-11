"""Fill absent training SFT predictions with frozen same-seed adapters on GPU only."""
import time
from qihc.experiment_runtime.storage import load,save,value_digest
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.feedback_active_v3.runtime import seed_for
from experiments.posttraining_v1.learn import parsed
from experiments.posttraining_v1.evaluate import worker_lock_wait

def parent_path(run,seed,cid):
    original=run/'inputs/parents'/str(seed)/(cid+'.json')
    return original if original.exists() else run/'derived/parents'/str(seed)/(cid+'.json')

def ensure_parents(engine,reference,rows,seed,run):
    missing=[r for r in rows if not (run/'inputs/parents'/str(seed)/(r['case']['id']+'.json')).exists()]
    if not missing:return
    manifest=load(run/'manifest.json'); plan=manifest['parent_plan'][str(seed)]
    restored=False
    for row in missing:
        # Only the public question/schema goes to inference; never training gold/answer.
        case={k:row['case'][k] for k in ('id','question','variable_names')}
        cid=case['id']; path=parent_path(run,seed,cid)
        if cid not in plan['missing_case_ids']: raise ValueError('Unplanned missing SFT parent '+cid)
        identity=dict(code_sha=manifest['code_sha'],adapter_sha256=plan['adapter_sha256'],
                      seed=seed,case_sha256=value_digest(case))
        with worker_lock_wait(path):
            if path.exists():
                if load(path).get('parent_identity')!=identity: raise ValueError('Generated SFT parent changed identity '+cid)
                continue
            if not restored: engine.restore(reference);restored=True
            started=time.monotonic()
            response=engine.evaluate(model_prompt(case),seed_for(cid,0,'initial'))
            choice=response['choices'][0]; text=choice['message']['content']
            model,error=parsed(text,case) if choice['finish_reason']=='stop' else (None,'length')
            save(path,dict(case_id=cid,model=model,error=error,raw=text,usage=response.get('usage'),
                           origin='frozen_sft_training_parent',parent_identity=identity,
                           seconds=time.monotonic()-started))
            print('SFT_PARENT_READY',seed,cid,flush=True)
