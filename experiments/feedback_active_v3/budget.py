"""Derive the call budget from the actual scheduler, not handwritten totals."""
import argparse
from pathlib import Path
from qihc.experiment_runtime.storage import load, save
from .runtime import public_cases, schedule


def calculate(data):
    cfg=load(Path(__file__).parent/'config.json'); cases=public_cases(data); subsets=load(data/'subsets.json')
    initialized=set(); stages=[]; probe_seconds=0
    for stage in ('development','main','ablation','scale','replay','public'):
        chosen,rows=schedule(cases,subsets,stage,cfg)
        if stage=='public': initial=len(rows); sources=sum(r['arm']!='C0' for r in rows)
        else:
            fresh={c['id'] for c in chosen}-initialized; initial=sources=len(fresh); initialized.update(fresh)
        patches=sum(r['arm']!='C0' for r in rows)*cfg['rounds']
        probes=0 if stage=='replay' else sum(r['arm'] not in ('C0','S','B') for r in rows)*cfg['rounds']
        probe_seconds+=probes*cfg['probe_seconds']
        stages.append(dict(stage=stage,cases=len(chosen),trajectories=len(rows),initial_calls=initial,
                           source_calls=sources,patch_calls=patches,llm_calls=initial+sources+patches))
    mechanism=load(data/'evaluation/mechanism.audit.json')
    return dict(experiment_id='E004_2.1',revision='public_only_v3',stages=stages,
                llm_call_cap=sum(r['llm_calls'] for r in stages),output_token_cap=sum(r['llm_calls'] for r in stages)*cfg['max_output_tokens'],
                trajectories=sum(r['trajectories'] for r in stages),feedback_cpu_hours_cap=probe_seconds/3600,
                mechanism_models=mechanism['models'],mechanism_jobs=mechanism['models']*9*3,
                mechanism_cpu_hours_cap=mechanism['models']*9*3*20/3600,
                second_model=False,notes=['Caps exclude native startup overhead and offline evaluation.',
                    'C0 main has independent 40s reruns; these are not 20s continuations.',
                    'Wall time must be measured on allocated GPUs; no unmeasured completion guarantee.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).parent/'assets/public_dataset')
    p.add_argument('--output',type=Path);a=p.parse_args();result=calculate(a.data)
    if a.output: save(a.output,result)
    print(result)
