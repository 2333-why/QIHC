import collections
import sys
import numpy as np
from qihc.experiment_runtime.storage import load,save
from .data import RUN,HISTORY,SEEDS,ARMS

def report(mode):
    root=RUN/mode
    rows=collections.defaultdict(lambda:collections.defaultdict(dict))
    for p in root.glob("models/*/*/evaluations/*.json"):
        r=load(p);rows[r["stage"]][r["arm"]][r["case_id"],r["seed"]]=r
    historic=collections.defaultdict(dict)
    for p in (HISTORY/"evaluation/trajectories").glob("*.json"):
        r=load(p)
        if r["arm"]=="P2":historic[r["stage"]][r["case_id"],r["seed"]]=r
    lines=["# E004_2.5 "+mode,"","主指标为原题已证最优命中率。所有题保留在计划分母，以下百分比按配齐种子的共同题计算。"]
    stats={}
    for stage,seeds in ({"development":[17]} if mode=="pilot" else SEEDS).items():
        arms=rows[stage]
        keys=set.intersection(*(set(arms[a]) for a in ARMS))
        ids=sorted({cid for cid,_ in keys if all((cid,s) in keys for s in seeds)})
        pairs=[(cid,s) for cid in ids for s in seeds]
        lines.extend(["",f"## {stage}，{len(ids)}道共同题","",
        "|组|已评分轨迹|完整正确|p-bit可行|p-bit最优|SCIP最优|新增/丢失p-bit命中对A|40秒p-bit最优|","|---|---:|---:|---:|---:|---:|---:|---:|"])
        def pct(xs):return f"{100*np.mean(xs):.2f}%" if xs else "—"
        def hit(a,k):return int(arms[a][k]["scores"]["20"]["pbit"]["optimal_hit"])
        for a in ARMS:
            rr=[arms[a][k] for k in pairs];cross=[r["scores"]["20"] for r in rr]
            diff=[hit(a,k)-hit("A",k) for k in pairs]
            cells=[pct([r["semantic"]=="verified_correct" for r in rr]),pct([c["pbit"]["original_feasible"] for c in cross]),
                   pct([c["pbit"]["optimal_hit"] for c in cross]),pct([c["scip"]["optimal_hit"] for c in cross]),
                   f"{diff.count(1)}/{diff.count(-1)}",pct([r["scores"]["40"]["pbit"]["optimal_hit"] for r in rr if "40" in r["scores"]])]
            lines.append("| "+" | ".join([a,str(len(arms[a])),*cells])+" |")
        h=historic[stage]
        subset=[k for k in pairs if all((k[0],s) in h for s in seeds)]
        if subset:
            lines.append(f"\n历史P2同题同种子 {len(subset)//len(seeds)}题：P2最优 {pct([h[k]['cross']['pbit']['score']['optimal_hit'] for k in subset])}，D最优 {pct([hit('D',k) for k in subset])}。")
        stage_stats={}
        for left,other in (("D","B"),("D","C"),("E","D"),("F","E"),("G","F"),("F","H")):
            diff=np.array([np.mean([hit(left,(cid,s))-hit(other,(cid,s)) for s in seeds]) for cid in ids])
            if len(diff):
                rng=np.random.default_rng(17)
                boot=rng.choice(diff,(5000,len(diff))).mean(axis=1)
                perm=(rng.choice([-1,1],(5000,len(diff)))*diff).mean(axis=1)
                stage_stats[left+"-"+other]=dict(delta=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                                        p=float((1+(np.abs(perm)>=abs(diff.mean())-1e-12).sum())/5001))
        if stage_stats:
            ordered=sorted(stage_stats,key=lambda a:stage_stats[a]["p"]); last=0.
            for i,a in enumerate(ordered):
                d=stage_stats[a];last=max(last,min(1.,(len(ordered)-i)*d["p"]));d["holm"]=last
                lines.append(f"\n{a} 最优差值 {100*d['delta']:+.2f} 个百分点，配对95%区间 [{100*d['ci95'][0]:.2f}, {100*d['ci95'][1]:.2f}]，Holm p={last:.4f}。")
        stats[stage]=dict(cases=ids,comparisons=stage_stats,decisions={a:dict(collections.Counter(r["action"] for r in arms[a].values())) for a in ARMS})
        lines.append("\n决策计数："+str(stats[stage]["decisions"]))
    save(root/"summary.json",stats)
    text="\n".join(lines)+"\n";(root/"report.md").write_text(text,encoding="utf-8");print(text)

if __name__=="__main__":report(sys.argv[1] if len(sys.argv)>1 else "formal")
