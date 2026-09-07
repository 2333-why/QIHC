#!/usr/bin/env python3
"""Four-GPU executable p-dit + MFC LNS subproblem ablation."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from qihc.problems.cvrp.instance import load_jsonl
from qihc.problems.cvrp.neighborhood import KNNNeighborhoodSelector
from qihc.problems.cvrp.qubo import build_assignment_qubo, decode_assignment
from qihc.problems.cvrp.verifier import greedy_initial_solution, verify_solution
from qihc.s2e import TorchPDitMFCSampler

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--limit",type=int,default=0); p.add_argument("--steps",type=int,default=200); p.add_argument("--chains",type=int,default=512); args=p.parse_args()
    rank=int(os.environ.get("RANK",0)); world=int(os.environ.get("WORLD_SIZE",1)); local=int(os.environ.get("LOCAL_RANK",rank)); args.output.mkdir(parents=True,exist_ok=True)
    dist=None
    if world>1:
        import torch.distributed as td; td.init_process_group("gloo"); dist=td
    rows=[]
    for idx,instance in enumerate(load_jsonl(args.data)[:args.limit or None]):
        if idx%world!=rank: continue
        incumbent=greedy_initial_solution(instance); before=verify_solution(instance,incumbent); proposal=KNNNeighborhoodSelector(12,4).propose(instance,incumbent,0,idx)
        batch=TorchPDitMFCSampler(num_chains=args.chains,steps=args.steps,device=f"cuda:{local}",seed=idx).solve(instance,incumbent,proposal); problem=build_assignment_qubo(instance,incumbent,proposal)
        best=before; feasible=0
        for assignment in batch.assignments:
            bits=np.zeros(problem.num_variables,dtype=np.int8)
            for customer,route in assignment.items():
                pos=problem.model.index.get(("assign",customer,route))
                if pos is not None: bits[pos]=1
            try: candidate=verify_solution(instance,decode_assignment(instance,problem,bits))
            except ValueError: continue
            if candidate.feasible: feasible+=1; best=candidate if candidate.objective<best.objective else best
        rows.append({"instance":instance.name,"initial_objective":before.objective,"final_objective":best.objective,"feasible_samples":feasible,"elapsed_s":batch.elapsed_s,"sampler":batch.metadata})
    target=args.output/f"hybrid_rank{rank}.jsonl"; target.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    if dist: dist.barrier()
    if rank==0:
        all_rows=[]
        for r in range(world): all_rows += [json.loads(x) for x in (args.output/f"hybrid_rank{r}.jsonl").read_text(encoding="utf-8").splitlines() if x]
        (args.output/"summary.json").write_text(json.dumps({"n":len(all_rows),"feasible_rate":sum(x["feasible_samples"]>0 for x in all_rows)/max(len(all_rows),1),"mean_improvement":sum((x["initial_objective"]-x["final_objective"])/max(x["initial_objective"],1e-9) for x in all_rows)/max(len(all_rows),1)},indent=2),encoding="utf-8")
    if dist: dist.barrier(); dist.destroy_process_group()
    return 0
if __name__=="__main__": raise SystemExit(main())
