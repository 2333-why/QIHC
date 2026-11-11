import sys
import torch
from qihc.experiment_runtime.storage import load, save, worker_lock
from qihc.modeling.active_protocol_v2 import model_prompt
from experiments.posttraining_v1.engine import Engine
from experiments.posttraining_v1.learn import parsed
from experiments.feedback_active_v3.runtime import seed_for
from .common import CFG, RUN, MODEL, TRAIN_ARMS, cases, donor
from .training import fit

def work(arm,seed,mode):
    if arm not in CFG['arms'] or mode not in ('pilot','formal'): raise ValueError('Unknown job')
    out=RUN/mode/'models'/str(seed)/arm
    with worker_lock(out/'lock'):
        torch.manual_seed(seed); engine=None
        if arm in TRAIN_ARMS:
            engine=Engine(MODEL)
            reference=torch.load(donor(seed,'sft')/'adapter.pt',map_location='cpu',weights_only=True)
            fit(engine,arm,seed,mode,out,reference)
        frozen=load(RUN/'manifest.json')['donor_predictions']
        for case in cases(seed,mode):
            path=out/'predictions'/(case['id']+'.json')
            if path.exists(): continue
            stored=donor(seed,arm)/'predictions'/path.name
            if arm not in TRAIN_ARMS and str(stored) in frozen:
                result=dict(load(stored),origin='frozen_donor')
            else:
                if engine is None:
                    engine=Engine(MODEL)
                    engine.restore(torch.load(donor(seed,arm)/'adapter.pt',map_location='cpu',weights_only=True))
                response=engine.evaluate(model_prompt(case),seed_for(case['id'],0,'initial'))
                choice=response['choices'][0]; text=choice['message']['content']
                model,error=parsed(text,case) if choice['finish_reason']=='stop' else (None,'length')
                result=dict(model=model,error=error,raw=text,usage=response.get('usage'),origin='new_generation')
            save(path,dict(result,case_id=case['id'],stage=case['stage']))
        save(out/'done.json',dict(arm=arm,seed=seed,count=len(cases(seed,mode))))

if __name__=='__main__': work(sys.argv[1],int(sys.argv[2]),sys.argv[3])
