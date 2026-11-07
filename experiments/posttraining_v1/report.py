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
    root=RUN/mode
    seeds=[17] if mode=="pilot" else [17,29,43]
    rows=defaultdict(lambda:defaultdict(dict))
    for p in root.glob("models/*/*/evaluations/*.json"):
        seed,method=p.parts[-4:-2]
        row=load(p)
        rows[method][row["case_id"]][int(seed)]=metrics(row)
    e004=Path(os.environ.get("QIHC_E004_RUN",BASE/"results/E004_2.2_active_feedback"))
    comparison="pilot uses development validation" if mode=="pilot" else "E004 manifest not available"
    if mode=="formal" and (e004/"manifest.json").exists():
        manifest=load(e004/"manifest.json")
        cfg=manifest["config"]
        expected=load(DATA/"manifest.json")
        if (cfg["experiment_id"]=="E004_2.2" and manifest["data_manifest"]==expected and
            manifest["model_identity"]==load(RUN/"prepared.json")["identity"] and
            cfg["solver_seconds"]==20 and cfg["chains"]==32 and cfg["max_output_tokens"]==4096 and
            cfg["protocol_version"]=="v2" and not manifest.get("smoke")):
            for path in (e004/"evaluation/trajectories").glob("*.json"):
                row=load(path)
                if row["stage"]=="main" and row["arm"] in ("C0","P2"):
                    rows[row["arm"]][row["case_id"]][row["seed"]]=metrics(row)
            comparison="compatible E004_2.2 snapshot"
        else: comparison="external comparison skipped: dataset/model/configuration mismatch"
    methods=list(METHODS)
    if mode=="formal": methods+=["C0","P2"]
    complete={}
    for m in methods:
        complete[m]={cid for cid,values in rows[m].items()
                     if (bool(values) if m=="C0" else set(seeds)<=values.keys())}
    common=set.intersection(*(complete[m] for m in methods))
    n=24 if mode=="pilot" else 480
    labels="| 方法 | 原题数 | 完整正确 | unknown | p-bit可行 | p-bit最优 | SCIP可行 | SCIP最优 |"
    def table(cohorts):
        lines=[labels,"|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m in methods:
            cids=cohorts[m]
            vals=[]
            for cid in cids:
                values=rows[m][cid]
                selected=list(values.values()) if m=="C0" else [values[s] for s in seeds]
                vals.append([mean(col) for col in zip(*selected)])
            cells=[f"{100*mean(col):.1f}%" for col in zip(*vals)] if vals else ["—"]*6
            lines.append("| "+" | ".join([m,str(len(vals)),*cells])+" |")
        return lines
    lines=["# E005-1.4 "+mode,"",comparison,"",f"每组目标 {n} 道原题。表中每题先平均已配齐的种子，保留建模失败。","",
           "已评分轨迹：" + "，".join(f"{m} {sum(map(len,rows[m].values()))}" for m in methods),"","各组已配齐题目（不同覆盖，不作排名）","",*table(complete),"",
           f"严格配对，共 {len(common)} 道题","",*table({m:common for m in methods})]
    lines.extend(["","训练信号（零次已证修复意味着该轮没有修复蒸馏样本）","",
                  "| 方法 | 种子 | 训练组数 | 有奖励梯度的组 | 产生反例的候选 | 已证正确修复 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for fp in sorted(root.glob("models/*/*/training_signals.json")):
        sig=load(fp)
        lines.append(f"| {fp.parent.name} | {fp.parent.parent.name} | {sig['groups']} | {sig['reward_updates']} | {sig['witnessed_candidates']} | {sig['distilled_repairs']} |")
    root.mkdir(parents=True,exist_ok=True)
    report_path=root/"report.md"
    report_path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(root/"paired_cases.json",{"cases":sorted(common),"seeds":seeds,"comparison":comparison})
    print(report_path,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["pilot","formal"],default="formal",nargs="?")
    report(p.parse_args().mode)
