"""Matched coverage and case-level percentages; no partial-seed ranking."""
import collections
import os
import sys
from statistics import mean
from pathlib import Path
from qihc.experiment_runtime.storage import load,save
from .common import root,ARMS,HISTORY,STAGE_SEEDS,DATA

def report(job,mode):
    folder=root(job)/mode
    rows=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(dict)))
    def add(row,arm,seed):
        scores=[row["cross"][b]["score"] for b in ("pbit","scip")]
        rows[row["stage"]][arm][row["case_id"]][int(seed)]=[
            row["semantic"]["status"]=="verified_correct",row["semantic"]["status"]=="unknown",
            *[s[k] for s in scores for k in ("original_feasible","optimal_hit")]]
    shifts=collections.defaultdict(collections.Counter)
    for p in folder.glob("models/*/*/evaluations/*.json"):
        row=load(p);arm=p.parts[-3];add(row,arm,p.parts[-4])
        v=shifts[row["stage"],arm];v["scored"]+=1
        v["recovered"]+=int(row["initial_status"]=="invalid" and row["semantic"]["compiled"])
        v["regressed"]+=int(row["initial_status"]=="verified_correct" and row["semantic"]["status"]=="disproved")
        v["accepted"]+=int(row["decision"]=="accepted")
    external=Path(os.getenv("QIHC_E004_RUN",str(HISTORY.parent/"E004_2.2_active_feedback")))
    ext=[]
    if mode=="formal" and (external/"manifest.json").exists():
        m=load(external/"manifest.json");c=m["config"]
        if (m["model_identity"]==load(root(job)/"manifest.json")["model"] and c["experiment_id"]=="E004_2.2"
            and c["chains"]==32 and c["solver_seconds"]==20 and m["data_manifest"]==load(DATA/"manifest.json") and not m.get("smoke")):
            ext=["C0","P2"]
            for p in (external/"evaluation/trajectories").glob("*.json"):
                row=load(p)
                if row["arm"] in ext and row["stage"] in STAGE_SEEDS:add(row,row["arm"],row["seed"])
    history_arms=set()
    if mode=="formal":
        for p in (HISTORY/"formal/models").glob("*/*/evaluations/*.json"):
            arm="E005/"+p.parts[-3]
            history_arms.add(arm)
            add(load(p),arm,p.parts[-4])
    lines=[f"# {job} {mode}","开发阶段含训练题。",""]
    paired={}
    for stage,seeds in ({"pilot":[17]} if mode=="pilot" else STAGE_SEEDS).items():
        data=rows[stage];arms=list(ARMS[job])
        required=lambda a:[0] if a=="C0" and stage!="public" else seeds
        complete={a:{i for i,v in data[a].items() if set(required(a))<=v.keys()} for a in arms+ext+sorted(history_arms)}
        common=set.intersection(*(complete[a] for a in arms))
        tables=[("本实验",arms,common)]
        if ext:tables.append(("含E004",arms+ext,set.intersection(*(complete[a] for a in arms+ext))))
        for a in sorted(history_arms):tables.append((a,arms+[a],common & complete[a]))
        paired[stage]={}
        for label,methods,ids in tables:
            paired[stage][label]=sorted(ids)
            lines.extend([f"## {stage} {label}：{len(ids)}道共同题","",
                "|组|已评分轨迹|完整正确|unknown|p-bit可行|p-bit最优|SCIP可行|SCIP最优|","|---|---:|---:|---:|---:|---:|---:|---:|"])
            for a in methods:
                cells=[f"{100*mean(mean(data[a][cid][s][k] for s in required(a)) for cid in ids):.1f}%" if ids else "—" for k in range(6)]
                lines.append("| "+" | ".join([a,str(sum(map(len,data[a].values()))),*cells])+" |")
            lines.append("")
    lines+=["","## 初稿变化（按已评价轨迹）","","|阶段|组|已评价|无效恢复|正确退化|局部补丁接受|","|---|---|---:|---:|---:|---:|"]
    for (stage,arm),v in sorted(shifts.items()):
        lines.append(f"|{stage}|{arm}|{v['scored']}|{v['recovered']}|{v['regressed']}|{v['accepted']}|")
    save(folder/"paired.json",paired)
    (folder/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))

if __name__=="__main__":report(*sys.argv[1:])
