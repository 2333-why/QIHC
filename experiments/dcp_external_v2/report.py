"""Coverage, per-instance and per-template results and paired cluster intervals."""

from collections import Counter, defaultdict
import numpy as np
from .common import RUN, ARMS, PRED, SEARCH_SEEDS, cases, repeats, stage_name
from qihc.experiment_runtime.storage import load, save

CONTRASTS = {
    "weighted_minus_parent": [("weighted__schedule", 1), ("parent__schedule", -1)],
    "pvcd_minus_solver_rl": [("pvcd__schedule", 1), ("solver_rl__schedule", -1)],
    "pvcd_minus_sdpo": [("pvcd__schedule", 1), ("sdpo__schedule", -1)],
    "feedback_on_pvcd": [("pvcd_C__schedule", 1), ("pvcd__schedule", -1)],
    "active_on_pvcd": [("pvcd_Q__schedule", 1), ("pvcd__schedule", -1)],
    "control_only": [("parent__rank", 1), ("parent__schedule", -1)],
    "joint_minus_parent": [("weighted__rank", 1), ("parent__schedule", -1)],
    "model_control_interaction": [
        ("weighted__rank", 1),
        ("parent__rank", -1),
        ("weighted__schedule", -1),
        ("parent__schedule", 1),
    ],
    "pvcd_minus_sft": [("pvcd__schedule", 1), ("sft__schedule", -1)],
    "double_review_minus_repair": [("base_C__schedule", 1), ("base_B__schedule", -1)],
    "scip_feedback_minus_repair": [("base_S__schedule", 1), ("base_B__schedule", -1)],
    "active_minus_double_review": [("base_Q__schedule", 1), ("base_C__schedule", -1)],
    "active_minus_plain_pbit": [("base_Q__schedule", 1), ("base_P__schedule", -1)],
    "training_feedback_interaction": [
        ("pvcd_C__schedule", 1),
        ("pvcd__schedule", -1),
        ("base_C__schedule", -1),
        ("base__schedule", 1),
    ],
}


def summarize(by_id, subset):
    completed = [(r, by_id[r["id"]]) for r in subset if r["id"] in by_id]
    rows = [s for _, s in completed]
    families = defaultdict(list)
    for r, s in completed:
        families[r["template"]].append(s["hit"])
    n = len(subset)
    hits = sum(r["hit"] for r in rows)
    gaps = [r["gap"] for r in rows if r["gap"] is not None]
    supported = sum(str(r["status"]).lower() != "unsupported" for r in rows)
    complete = bool(n) and len(rows) == n
    actions = Counter(a for r in rows for a in r.get("actions", []))
    return dict(
        expected=n,
        completed=len(rows),
        missing=n - len(rows),
        supported=supported,
        supported_hit_rate=hits / supported if complete and supported else None,
        valid=(
            sum(r["valid"] for r in rows if r["valid"] is not None)
            if any(r["valid"] is not None for r in rows)
            else None
        ),
        feasible=sum(r["feasible"] for r in rows),
        hits=hits,
        hit_rate=hits / n if complete else None,
        template_macro_hit=(
            float(np.mean([np.mean(xs) for xs in families.values()]))
            if complete
            else None
        ),
        mean_relative_gap=float(np.mean(gaps)) if gaps else None,
        gap_cases=len(gaps),
        by_template={
            t: dict(evaluated=len(values), hits=sum(values))
            for t, values in families.items()
        },
        statuses=dict(Counter(r["status"] for r in rows)),
        actions=dict(actions),
        costs={
            k: sum(r.get(k) or 0 for r in rows)
            for k in (
                "calls",
                "prompt_tokens",
                "completion_tokens",
                "seconds",
                "generation_seconds",
                "probe_seconds",
                "attempts",
                "decision_seconds",
                "intervention_writes",
            )
        },
    )


def contrast(data, subset, terms):
    if not subset or any(
        any(r["id"] not in data[arm] for arm, _ in terms) for r in subset
    ):
        return None
    by_family = defaultdict(list)
    gained, lost = [], []
    for r in subset:
        d = sum(sign * int(data[arm][r["id"]]["hit"]) for arm, sign in terms)
        by_family[r["template"]].append(d)
        if d > 0:
            gained.append(r["id"])
        if d < 0:
            lost.append(r["id"])
    sums = np.array([sum(v) for v in by_family.values()])
    sizes = np.array([len(v) for v in by_family.values()])
    draw = np.random.default_rng(17).integers(0, len(sizes), (5000, len(sizes)))
    rates = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return dict(
        delta=float(sums.sum() / sizes.sum()),
        interval95=np.quantile(rates, [0.025, 0.975]).tolist(),
        positive=gained,
        negative=lost,
        templates=len(sizes),
    )


def report():
    public = cases()
    data = {}
    seed_rows = {}
    for model, method in ARMS:
        name = model + "__" + method
        seed_rows[name] = {}
        for seed in repeats(model, method):
            folder = RUN / stage_name("results", seed) / name
            seed_rows[name][seed] = {p.stem: load(p) for p in folder.glob("*.json")}
        data[name] = seed_rows[name][17]
    result = {}
    for title, subset in [
        ("all235", public),
        ("main159", [r for r in public if r["main"]]),
    ]:
        arms = {name: summarize(byid, subset) for name, byid in data.items()}
        result[title] = dict(
            arms=arms,
            completed=sum(v["completed"] for v in arms.values()),
            expected=len(arms) * len(subset),
            paired={k: contrast(data, subset, v) for k, v in CONTRASTS.items()},
        )
        sensitivity = {}
        for arm, per_seed in seed_rows.items():
            complete_cases = [
                r
                for r in subset
                if all(r["id"] in items for items in per_seed.values())
            ]
            rates = [
                float(np.mean([rows[r["id"]]["hit"] for rows in per_seed.values()]))
                for r in complete_cases
            ]
            feasible = [
                float(
                    np.mean([rows[r["id"]]["feasible"] for rows in per_seed.values()])
                )
                for r in complete_cases
            ]
            done = sum(
                sum(r["id"] in rows for r in subset) for rows in per_seed.values()
            )
            sensitivity[arm] = dict(
                seeds=list(per_seed),
                expected=len(per_seed) * len(subset),
                completed=done,
                complete_cases=len(complete_cases),
                hit_rate=(
                    float(np.mean(rates))
                    if rates and len(rates) == len(subset)
                    else None
                ),
                feasible_rate=(
                    float(np.mean(feasible))
                    if rates and len(rates) == len(subset)
                    else None
                ),
            )
        result[title]["across_search_seeds"] = sensitivity
        result[title]["all_seed_completed"] = sum(
            v["completed"] for v in sensitivity.values()
        )
        result[title]["all_seed_expected"] = sum(
            v["expected"] for v in sensitivity.values()
        )
        print(
            title,
            "including search repeats",
            result[title]["all_seed_completed"],
            "/",
            result[title]["all_seed_expected"],
        )
        print(
            title,
            result[title]["completed"],
            "/",
            result[title]["expected"],
            flush=True,
        )
    predicted = sum(
        sum(1 for _ in (RUN / "predictions" / m).glob("*.json")) for m in PRED
    )
    controlled = sum(
        sum(
            1 for _ in (RUN / stage_name("episodes", s) / (m + "__" + a)).glob("*.json")
        )
        for m in ("parent", "weighted")
        for a in ("rank", "balanced")
        for s in SEARCH_SEEDS
    )
    print(
        "GPU model/feedback",
        predicted,
        "/",
        len(public) * len(PRED),
        "controller episodes",
        controlled,
        "/",
        len(public) * 4 * len(SEARCH_SEEDS),
        flush=True,
    )
    save(
        RUN / "progress.json",
        dict(
            predictions=predicted,
            expected_predictions=len(public) * len(PRED),
            controls=controlled,
            expected_controls=len(public) * 4 * len(SEARCH_SEEDS),
        ),
    )
    save(RUN / "report.json", result)
    lines = ["# E005-1.17 DCP-Bench", ""]
    for title, block in result.items():
        lines.extend(
            [
                f"## {title} ({block['completed']}/{block['expected']})",
                "",
                "| Model / solver | Completed | Valid | Feasible | Optimal | Template macro hit | Mean gap (n) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm, v in block["arms"].items():
            pct = lambda x: "—" if x is None else f"{x:.2%}"
            lines.append(
                f"| {arm} | {v['completed']}/{v['expected']} | {v['valid'] if v['valid'] is not None else '—'} | {v['feasible']} | {v['hits']} | {pct(v['template_macro_hit'])} | {pct(v['mean_relative_gap'])} ({v['gap_cases']}) |"
            )
        lines += [
            "",
            "| Search-seed sensitivity | Evaluations | Within-case mean hit | Within-case mean feasible |",
            "|---|---:|---:|---:|",
        ]
        for arm, v in block["across_search_seeds"].items():
            if len(v["seeds"]) > 1:
                lines.append(
                    f"| {arm} | {v['completed']}/{v['expected']} | {pct(v['hit_rate'])} | {pct(v['feasible_rate'])} |"
                )
        lines.append("")
    import os, tempfile

    fd, tmp = tempfile.mkstemp(dir=RUN)
    with os.fdopen(fd, "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, RUN / "report.md")
    return result


if __name__ == "__main__":
    report()
