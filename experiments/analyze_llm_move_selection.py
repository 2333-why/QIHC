#!/usr/bin/env python3
"""Diagnose how well LLM proposals use objective-aware CVRP move features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=0)
    args = parser.parse_args()

    reports = []
    for path in sorted(args.results.glob("llm_audit_rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("ok") or row.get("iteration") != args.iteration:
                continue
            prompt = row.get("prompt", "")
            marker_start = "Objective-aware move features: "
            marker_end = "\nP-bit logit feedback"
            if marker_start not in prompt or marker_end not in prompt:
                continue
            feature_text = prompt.split(marker_start, 1)[1].split(marker_end, 1)[0]
            features = json.loads(feature_text)
            proposal = row.get("proposal", {})
            feature_map = {int(item["customer"]): item for item in features}
            selected = [int(value) for value in proposal.get("destroy_customers", [])]
            selected_valid = [value for value in selected if value in feature_map]
            ranked = sorted(
                features,
                key=lambda item: (
                    float(item["best_net_delta"]),
                    int(item["customer"]),
                ),
            )
            top = [int(item["customer"]) for item in ranked[: len(selected)]]
            selected_deltas = [
                float(feature_map[value]["best_net_delta"])
                for value in selected_valid
            ]
            top_deltas = [float(feature_map[value]["best_net_delta"]) for value in top]
            route_hits = 0
            route_checked = 0
            logits = proposal.get("candidate_route_logits", {})
            for customer in selected_valid:
                moves = feature_map[customer].get("moves", [])
                customer_logits = logits.get(str(customer), logits.get(customer, {}))
                if not moves or not customer_logits:
                    continue
                best_route = min(moves, key=lambda item: float(item["net_delta"]))["route"]
                preferred = max(
                    customer_logits,
                    key=lambda route: float(customer_logits[route]),
                )
                route_checked += 1
                route_hits += int(int(preferred) == int(best_route))
            report = {
                "instance": row.get("instance"),
                "selected": selected,
                "top_objective_customers": top,
                "customer_overlap": len(set(selected) & set(top)),
                "selected_count": len(selected),
                "mean_selected_delta": (
                    sum(selected_deltas) / len(selected_deltas)
                    if selected_deltas else None
                ),
                "mean_top_delta": sum(top_deltas) / len(top_deltas) if top_deltas else None,
                "negative_selected": sum(value < 0 for value in selected_deltas),
                "route_hits": route_hits,
                "route_checked": route_checked,
            }
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False))

    selected_total = sum(item["selected_count"] for item in reports)
    aggregate = {
        "instances": len(reports),
        "customer_overlap_rate": sum(item["customer_overlap"] for item in reports)
        / max(selected_total, 1),
        "negative_selected_rate": sum(item["negative_selected"] for item in reports)
        / max(selected_total, 1),
        "route_preference_accuracy": sum(item["route_hits"] for item in reports)
        / max(sum(item["route_checked"] for item in reports), 1),
        "mean_selected_delta": sum(
            item["mean_selected_delta"] for item in reports
            if item["mean_selected_delta"] is not None
        ) / max(sum(item["mean_selected_delta"] is not None for item in reports), 1),
        "mean_top_delta": sum(
            item["mean_top_delta"] for item in reports
            if item["mean_top_delta"] is not None
        ) / max(sum(item["mean_top_delta"] is not None for item in reports), 1),
    }
    print(json.dumps({"aggregate": aggregate}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
