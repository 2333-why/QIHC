import os
from pathlib import Path
from qihc.experiment_runtime.storage import load,save,worker_lock,digest
from experiments.posttraining_v1.data import BASE,MODEL,DATA,STAGE_SEEDS
from experiments.feedback_active_v3.data import public_view
from experiments.feedback_active_v3.runtime import seed_for

HISTORY=Path(os.environ.get("QIHC_HISTORY",BASE/"results/E005-1.7_eight"))
IDS={"feedback":"E004_2.3_selective","train":"E005-1.8_paired"}
ARMS={"feedback":("sft","compile","scip_local","pbit_local"),
      "train":("sft","solver_rl","pair_rl","pair_local","pvcd")}
SEEDS=(17,29,43)
def root(job):
    p=Path(os.environ.get("QIHC_NEW_RESULTS",BASE/"results"))/IDS[job]
    if p.resolve()==HISTORY.resolve():raise ValueError("Overlapping history directory")
    return p

def cases(seed,mode):
    rows=load(HISTORY/"cohort.json")
    if mode=="pilot":return [dict(r["case"],stage="pilot") for r in load(HISTORY/"validation.json")]
    return [r for r in rows if seed in STAGE_SEEDS[r["stage"]]]

def donor(seed,arm):return HISTORY/"formal/models"/str(seed)/arm

def generated(engine,prompt,seed):
    response=engine.evaluate(prompt,seed)
    c=response["choices"][0]
    return c["message"]["content"] if c["finish_reason"]=="stop" else ""
