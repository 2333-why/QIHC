# -*- coding: utf-8 -*-
"""Aggregate NSFC evidence JSON artifacts into a single report + figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_table1(report: dict, out_path: Path) -> None:
    rows = report.get("table1_rows", [])
    if not rows:
        return
    modes = [r["mode"] for r in rows]
    x = range(len(modes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    ax.bar([i - w for i in x], [r.get("feasible_rate", 0) for r in rows], w, label="Feasible", color="#72b7b2")
    ax.bar(x, [r.get("exact_match_rate", r.get("accuracy", 0)) for r in rows], w, label="Exact/Acc", color="#4c78a8")
    ax.bar([i + w for i in x], [r.get("mean_pbit_steps", 0) / max(max(r.get("mean_pbit_steps", 1) for r in rows), 1) for r in rows], w, label="p-bit (norm)", color="#e45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_title("NSFC Evidence Table 1 (aggregated)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def build_report(run_dir: Path) -> dict:
    def find_artifact(glob_pattern: str) -> dict | None:
        for p in run_dir.rglob(glob_pattern):
            return load_json(p)
        return None

    dual = find_artifact("dual_axis.json")
    cr = find_artifact("cr_protocol.json")
    handoff = find_artifact("handoff.json")
    pareto = find_artifact("pareto.json")
    scaling = find_artifact("scaling.json")
    f_traj = find_artifact("f_trajectories.json")
    sampler = find_artifact("sampler_ablation.json")

    table1_rows = []
    if dual and "summary" in dual:
        for mode, s in dual["summary"].items():
            table1_rows.append({"arm": "bundled_constraint", "mode": mode, **s})

    cr_rows = []
    if cr and "summary" in cr:
        for mode, s in cr["summary"].items():
            cr_rows.append({"mode": mode, **s})

    report = {
        "run_dir": str(run_dir),
        "table1_rows": table1_rows,
        "cr_protocol": cr_rows,
        "handoff_highlights": None,
        "pareto": pareto.get("frontier") if pareto else None,
        "scaling": scaling.get("scaling") if scaling else None,
        "f_descent_mean": f_traj.get("mean_F_trace_vci2") if f_traj else None,
        "sampler_ablation": sampler.get("summary") if sampler else None,
        "llm_stats": cr.get("llm_stats") if cr else None,
    }

    if handoff and handoff.get("rows"):
        best = max(handoff["rows"], key=lambda r: r.get("feasible_gain", -999))
        report["handoff_highlights"] = best

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot NSFC evidence report")
    parser.add_argument("--run-dir", required=True, help="experiments/outputs/nsfc_evidence/run_YYYYMMDD_HHMMSS")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    report = build_report(run_dir)
    out_json = run_dir / "evidence_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_table1(report, run_dir / "evidence_table1.png")
    print(f"Report: {out_json}")
    print(f"Figure: {run_dir / 'evidence_table1.png'}")
    if report.get("handoff_highlights"):
        h = report["handoff_highlights"]
        print(f"Handoff best gain: σ={h.get('noise_scale')} → {h.get('feasible_gain'):+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
