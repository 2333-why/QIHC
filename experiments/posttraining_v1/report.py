"""Partial and matched-by-instance evaluation; reads E004 only when present."""
import argparse
import os
from collections import defaultdict
from statistics import mean
from qihc.experiment_runtime.storage import load, save, value_digest
from .data import RUN, DATA, BASE, METHODS
from pathlib import Path


def metrics(row):
    scores=[row["cross"][b]["score"] for b in ("pbit","scip")]
    return [int(row["semantic"]["status"]=="verified_correct"),
            int(row["semantic"]["status"]=="unknown"),
            *[int(v[k]) for v in scores for k in ("original_feasible","optimal_hit")]]


def report(mode):
    from .data import STAGE_SEEDS
    root=RUN/mode
    rows=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for p in root.glob("models/*/*/evaluations/*.json"):
        seed,method=p.parts[-4:-2]
        row=load(p)
        fixed=row.get("fixed",{}).get("score",{})
        rows[row["stage"]][method][row["case_id"]][int(seed)]=metrics(row)+[
            fixed.get("original_feasible"),fixed.get("optimal_hit")]
    e004=Path(os.environ.get("QIHC_E004_RUN",BASE/"results/E004_2.2_active_feedback"))
    comparison="external comparison pending"
    if mode=="formal" and (e004/"manifest.json").exists():
        manifest=load(e004/"manifest.json")
        cfg=manifest["config"]
        if (cfg["experiment_id"]=="E004_2.2" and manifest["data_manifest"]==load(DATA/"manifest.json") and
            manifest["model_identity"]==load(RUN/"prepared.json")["identity"] and
            cfg["solver_seconds"]==20 and cfg["chains"]==32 and cfg["max_output_tokens"]==4096 and
            cfg["protocol_version"]=="v2" and not manifest.get("smoke")):
            for path in (e004/"evaluation/trajectories").glob("*.json"):
                row=load(path)
                if row["stage"] in STAGE_SEEDS and row["arm"] in ("C0","P2"):
                    rows[row["stage"]][row["arm"]][row["case_id"]][row["seed"]]=metrics(row)+[None,None]
            comparison="compatible read-only E004_2.2 snapshot"
        else: comparison="external dataset, model or protocol mismatch"
    lines=["# E005-1.7 "+mode,"",comparison,"","开发阶段含训练题，只用于流程诊断。能量四组共享 SFT 模型。"]
    paired={}
    for stage,seeds in ({"pilot":[17]} if mode=="pilot" else STAGE_SEEDS).items():
        methods=list(METHODS)+([] if mode=="pilot" else ["C0","P2"])
        data=rows[stage]
        def required(m): return [0] if m=="C0" and stage!="public" else seeds
        complete={m:{c for c,values in data[m].items() if set(required(m))<=values.keys()} for m in methods}
        common8=set.intersection(*(complete[m] for m in METHODS))
        common_all=set.intersection(*(complete[m] for m in methods))
        paired[stage]=dict(eight=sorted(common8),external=sorted(common_all))
        lines+=["","## "+stage,"","已评分轨迹："+ "，".join(f"{m} {sum(map(len,data[m].values()))}" for m in methods)]
        tables=[("各组进度",methods,complete),("八组共同题",METHODS,{m:common8 for m in METHODS})]
        if mode=="formal":
            tables.append(("与 E004 同题",methods,{m:common_all for m in methods}))
        for title,arms,covered in tables:
            lines+=["",title,"","|方法|题数|完整正确|unknown|p-bit 20秒可行|p-bit 20秒最优|SCIP可行|SCIP最优|固定更新可行|固定更新最优|",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for m in arms:
                vals=[]
                for cid in covered[m]:
                    columns=zip(*(data[m][cid][s] for s in required(m)))
                    vals.append([None if col[0] is None else mean(col) for col in columns])
                cells=["—" if col[0] is None else f"{100*mean(col):.1f}%" for col in zip(*vals)] if vals else ["—"]*8
                lines.append("| "+" | ".join([m,str(len(vals)),*cells])+" |")
        if mode=="formal" and common8:
            lines.extend(["","配对差值（固定更新最优命中率，百分点，按题平均种子）："])
            for other in ("best_lm","rule_weight","uniform_weight"):
                differences=[mean(data["trajectory_lm"][c][s][7]-data[other][c][s][7]
                                  for s in seeds) for c in common8 if all(data["trajectory_lm"][c][s][7] is not None
                                  and data[other][c][s][7] is not None for s in seeds)]
                if differences:
                    lines.append(f"- trajectory_lm 对 {other}：{100*mean(differences):+.2f}（{len(differences)}题）")
    from .energy import ENERGY_METHODS
    gold_rows=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for path in (root/"gold").glob("models/*/*/evaluations/*.json"):
        seed,method=path.parts[-4:-2]
        row=load(path)
        gold_rows[row["stage"]][method][row["case_id"]][int(seed)]=row["fixed"]["score"]
    lines+=["","## GOLD 标准模型上的四个能量策略","","| 阶段 | 共同题 | 方法 | 固定更新可行 | 固定更新最优 |","|---|---:|---|---:|---:|"]
    for stage,seeds in ({"pilot":[17]} if mode=="pilot" else STAGE_SEEDS).items():
        g=gold_rows[stage]
        common=set.intersection(*({c for c,x in g[m].items() if set(seeds)<=x.keys()} for m in ENERGY_METHODS))
        paired[stage]["gold_energy"]=sorted(common)
        for m in ENERGY_METHODS:
            vals=[f"{100*mean(mean(g[m][c][s][k] for s in seeds) for c in common):.1f}%" if common else "—"
                  for k in ("original_feasible","optimal_hit")]
            lines.append(f"|{stage}|{len(common)}|{m}|{vals[0]}|{vals[1]}|")
    if (RUN/"energy/settings.json").exists():
        settings=load(RUN/"energy/settings.json")
        lines.extend(["",f"能量训练：统一权重 {settings['base']}，独立复测偏好题数 {len(settings['pair_ids'])}。"])
    root.mkdir(parents=True,exist_ok=True)
    (root/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(root/"paired_cases.json",dict(stages=paired,cases=paired.get("main",{}).get("external",[]),comparison=comparison))
    print(root/"report.md",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["pilot","formal"],default="formal",nargs="?")
    report(p.parse_args().mode)
