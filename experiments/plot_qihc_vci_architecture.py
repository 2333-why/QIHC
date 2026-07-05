# -*- coding: utf-8 -*-
"""Generate QIHC + VCI architecture diagram."""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def draw_architecture(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), dpi=140)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef2ff", ec="#4c78a8"):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)

    def arrow(x1, y1, x2, y2, color="#333", style="-|>"):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=12,
            linewidth=1.2,
            color=color,
        )
        ax.add_patch(arr)

    ax.text(6, 7.6, "QIHC Platform + VCI Co-Inference Loop", ha="center", fontsize=14, fontweight="bold")

    # Upper layer
    box(0.5, 5.6, 11, 1.2, "L1  Semantic phase q  ·  LLM / Frontend encode  ·  refine(q,s,feedback)", fc="#dbeafe")
    box(0.5, 4.2, 5.2, 1.1, "L2  Energy layer  ·  E(s|q) → h(q), J(q)\nIF constraints + cardinality", fc="#e0f2fe")
    box(6.3, 4.2, 5.2, 1.1, "L5  VCI scheduler  ·  F(q,s)  ·  converge?", fc="#fef3c7", ec="#d97706")

    # Lower layer
    box(0.5, 2.5, 5.2, 1.2, "L3  Discrete phase s  ·  p-bit PT/SQA/Gibbs\nequilibrate on E(s|q)", fc="#dcfce7", ec="#16a34a")
    box(6.3, 2.5, 5.2, 1.2, "L4  Decode  ·  mask / routing / reasons\nfeasible projection (IF)", fc="#f3e8ff", ec="#9333ea")

    box(0.5, 0.8, 11, 1.2, "L0  Task  ·  Case A reasoning subset  ·  Case B Max-Cut  ·  Case C MoE (aux)", fc="#f8fafc", ec="#64748b")

    # Forward arrows
    arrow(3.1, 5.6, 3.1, 5.35)
    arrow(3.1, 4.2, 3.1, 3.75)
    arrow(3.1, 2.5, 3.1, 2.05)
    arrow(8.9, 2.5, 8.9, 4.2)
    arrow(8.9, 4.2, 8.9, 5.6)

    # VCI loop
    arrow(6.2, 5.0, 6.3, 5.0, color="#d97706")
    arrow(6.2, 3.1, 6.2, 4.9, color="#d97706", style="-|>")
    ax.text(6.55, 4.05, "q-step\nrefine", fontsize=8, color="#d97706")

    ax.text(
        0.6,
        6.95,
        "CR ≈ VCI-1: freeze q, single s-step",
        fontsize=8,
        color="#64748b",
        style="italic",
    )

    legend = [
        mpatches.Patch(facecolor="#dbeafe", edgecolor="#4c78a8", label="GPU / LLM layer"),
        mpatches.Patch(facecolor="#dcfce7", edgecolor="#16a34a", label="p-bit probability layer"),
        mpatches.Patch(facecolor="#fef3c7", edgecolor="#d97706", label="VCI co-inference"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> int:
    out = os.path.join(REPO_ROOT, "docs", "QIHC_VCI_architecture.png")
    draw_architecture(out)
    alt = os.path.join(REPO_ROOT, "experiments", "outputs", "QIHC_VCI_architecture.png")
    draw_architecture(alt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
