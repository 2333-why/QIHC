"""Reuse completed evaluations and prepare supervision from frozen train only."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import subprocess

from experiments.pbit_dual_v1 import common as prior
from experiments.pbit_dual_v1.cpu import nlco_score
from qihc.experiment_runtime.storage import (
    digest,
    key,
    load,
    save,
    value_digest,
    worker_lock,
)
from qihc.generic.feedback.verification import normalized
from .common import (
    CONVENTIONAL,
    E003,
    HISTORY,
    OLD,
    ROOT,
    RUN,
    TRADITIONAL,
    old_model,
    path,
)
from .feedback import objective_example, repair_example
from .penalty import labels

E003_INDEX = Path(__file__).with_name("nlco_e003_seed0_20s.json.gz")


@lru_cache(maxsize=1)
def archive_index():
    if (
        digest(E003_INDEX)
        != "f4385be31fc5a5a27ffb04b07938d64e90354b9eca39bef5edf01ec4ca106110"
    ):
        raise ValueError("Archived E003 score index changed")
    return json.loads(gzip.decompress(E003_INDEX.read_bytes()))["cases"]


def archived_score(item, reference):
    status, feasible, objective, valid, seconds = item
    optimum = reference.get("objective")
    gap = (
        abs(objective - optimum) / max(1, abs(optimum))
        if feasible and optimum is not None
        else None
    )
    return dict(
        status=status,
        feasible=bool(feasible),
        gap=gap,
        valid=valid,
        hit=bool(
            feasible
            and reference.get("optimality_proven")
            and optimum is not None
            and abs(objective - optimum) < 1e-8
        ),
        reference_proven=bool(reference.get("optimality_proven")),
        original_seed=0,
        source="E003-1.1",
        seconds=seconds,
    )


def prior_scores(case):
    """Keep the leading archived model's scores at matched terminal and seeds."""
    old = old_model(case)
    if case["dataset"] == "nlco":
        src = prior.original_artifact(case)
        return {
            "incumbent__schedule__17": nlco_score(
                load(src)["legacy"]["weighted__schedule"]
            )
        }
    return {
        f"incumbent__{term}__{s}": load(
            prior.artifact("scores", case, f"{old}__{term}", s)
        )
        for term in ("schedule", "scip")
        for s in prior.seeds(case, term)
    }


def matched_scores(case):
    rows = {}
    for model in prior.MODELS:
        spec = prior.get_prediction(case, model)["model"]
        identity = value_digest(spec)
        for term in ("schedule", "scip"):
            for s in prior.seeds(case, term):
                file = prior.artifact("scores", case, f"{model}__{term}", s)
                if file.exists():
                    rows.setdefault(f"{identity}__{term}__{s}", load(file))
    return rows


def archive_baselines(case, gold, reference):
    """Reuse E003 nominal 20-second seed 0. Fill only missing compatible entries."""
    rows = {}
    cid = case["id"]
    packed = archive_index().get(cid)
    if packed and packed["gold"] == value_digest(normalized(gold)):
        return {
            f"{arm}__17": archived_score(row, reference)
            for arm, row in packed["rows"].items()
        }
    canonical = E003 / "models" / (key(cid, "canonical") + ".json")
    if not canonical.exists() or normalized(load(canonical)["model"]) != normalized(
        gold
    ):
        return rows
    for track, label in (("canonical", "gold20"), ("llm_typed_flat", "base20")):
        identity = key(cid, track)
        model_file = E003 / "models" / (identity + ".json")
        if not model_file.exists():
            continue
        rec = load(model_file)
        # Historical LLM inputs are comparable only on the same original gold.
        gold_path = E003 / "references" / (key(cid) + ".json")
        if track != "canonical" and (
            not gold_path.exists()
            or not (E003 / "models" / (key(cid, "canonical") + ".json")).exists()
        ):
            continue
        for backend in TRADITIONAL:
            file = E003 / "scores" / identity / f"{backend}-20-0.json"
            if not file.exists():
                continue
            row = load(file)
            if (
                row["case_id"],
                row["backend"],
                row["budget_seconds"],
                row["seed"],
                row["model_sha256"],
            ) != (cid, backend, 20, 0, rec["model_sha256"]):
                raise ValueError(f"Mismatched E003 score: {file}")
            rows[f"{label}__{backend}__17"] = archived_score(
                (
                    row["status"],
                    row["verified_feasible"],
                    row.get("verified_objective"),
                    None if track == "canonical" else rec["model"] is not None,
                    row.get("raw_metrics", {}).get("wall_seconds", 0),
                ),
                reference,
            )
    return rows


def corpus_item(file):
    source = load(Path(file))
    cid = source["case"]["id"]
    dest = RUN / "training" / (cid + ".json")
    if not dest.exists():
        modeling = [
            dict((k, source[k]) for k in ("case", "gold", "candidate", "cards")),
            *source["augmented"],
        ]
        expanded = objective_example(source["case"], source["gold"])
        if expanded is not None:
            modeling.append(expanded)
        repairs = [r for row in modeling if (r := repair_example(row)) is not None]
        models = [source["gold"]]
        if source["candidate"] is not None and source["candidate"] != source["gold"]:
            models.append(source["candidate"])
        controls = [
            r for i, spec in enumerate(models) for r in labels(spec, f"{cid}/{i}")
        ]
        save(dest, dict(modeling=modeling, repairs=repairs, controls=controls))
    row = load(dest)
    return (
        str(dest),
        len(row["modeling"]),
        len(row["repairs"]),
        len(row["controls"]),
        dict(Counter(r["action"] for r in row["controls"] if r["target"] > 0.05)),
    )


def prepare_current():
    if (
        RUN.resolve() == OLD.resolve()
        or OLD.resolve() in RUN.resolve().parents
        or RUN.resolve() in OLD.resolve().parents
    ):
        raise ValueError("Use an independent result directory")
    with worker_lock(RUN / "prepare.lock"):
        if (RUN / "manifest.json").exists():
            from .common import state

            previous = state(verify=True)
            print(
                "CURRENT_READY",
                len(load(RUN / "cases.json")),
                previous["archived"],
                flush=True,
            )
            return
        old = prior.verify()
        hist = load(OLD / "report.json")
        if any(
            hist["scores"][g]["new_complete"] != n
            for g, n in (("nlco780", 5460), ("dcp235", 5875), ("dcp159", 3975))
        ):
            raise ValueError("E005-1.18 results are incomplete")
        cohort = prior.cases()
        prior.freeze(RUN / "cases.json", cohort)
        prior.freeze(RUN / "historical18.json", hist)
        files = {
            str(p): digest(p)
            for p in (
                OLD / "manifest.json",
                OLD / "report.json",
                RUN / "cases.json",
                RUN / "historical18.json",
                *Path(__file__).parent.glob("*.py"),
                Path(__file__).parent / "run.sh",
                E003_INDEX,
            )
        }
        old14 = load(prior.OLD14 / "manifest.json")
        admissions = Counter()
        for case in cohort:
            incumbent = prior.get_prediction(case, old_model(case))
            prior_feedback = prior.get_prediction(case, old_model(case) + "_feedback")
            prior.freeze(path("predictions", case, "incumbent"), incumbent)
            prior.freeze(path("predictions", case, "incumbent_pbit"), prior_feedback)
            archived = prior_scores(case)
            if case["dataset"] == "nlco":
                item = load(prior.original_artifact(case))
                extra = archive_baselines(case, item["gold"], item["reference"])
                archived.update(extra)
                admissions.update(k.split("__", 1)[0] for k in extra)
                for name in CONVENTIONAL:
                    if name == "base":
                        continue
                    cid = case["id"]
                    p = (
                        Path(old14["predictions"][cid])
                        if name == "sft"
                        else HISTORY
                        / "formal/models/17"
                        / name
                        / "predictions"
                        / (cid + ".json")
                    )
                    if p.exists():
                        prior.freeze(path("predictions", case, name), load(p))
                if path("predictions", case, "sft").exists():
                    archived["sft__schedule__17"] = nlco_score(
                        item["legacy14"]["base__schedule"]
                    )
            prior.freeze(path("archived", case, "scores"), archived)
            prior.freeze(path("archived", case, "matched"), matched_scores(case))
            fingerprints = [
                path("archived", case, "scores"),
                path("archived", case, "matched"),
                path("predictions", case, "incumbent"),
                path("predictions", case, "incumbent_pbit"),
            ]
            fingerprints += [
                path("predictions", case, a)
                for a in CONVENTIONAL
                if a != "base"
                if path("predictions", case, a).exists()
            ]
            files.update({str(p): digest(p) for p in fingerprints})
        checkpoints = dict(
            nlco=old["paths"]["weighted"],
            dcp=str(OLD / "trained/model/adapter.pt"),
            parent=old["paths"]["parent"],
        )
        checkpoints.update(
            {
                a: str(HISTORY / "formal/models/17" / a / "adapter.pt")
                for a in CONVENTIONAL
            }
        )
        for name in ("nlco", "dcp", "parent"):
            p = Path(checkpoints[name])
            files[str(p)] = digest(p)
        for name in CONVENTIONAL:
            if name == "base":
                continue
            if any(
                c["dataset"] == "nlco" and not path("predictions", c, name).exists()
                for c in cohort
            ):
                p = Path(checkpoints[name])
                files[str(p)] = digest(p)
        manifest = dict(
            experiment="E005-1.19",
            files=files,
            checkpoints=checkpoints,
            model=old["model"],
            model_files=old["model_files"],
            archived=dict(admissions),
            code=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        )
        prior.freeze(RUN / "manifest.json", manifest)
    print("CURRENT_READY", len(cohort), dict(admissions), flush=True)


def prepare_improved():
    from .common import state

    if not (RUN / "current.json").exists():
        raise ValueError("Finish current-method scoring first")
    state(verify=True)
    sources = prior.verify()["training"]
    if len(sources) != 90:
        raise ValueError("Expected 90 training parents")
    with (
        worker_lock(RUN / "training.lock"),
        ProcessPoolExecutor(
            max_workers=int(os.getenv("QIHC_CPU_WORKERS", "6"))
        ) as pool,
    ):
        files, support, totals = {}, Counter(), Counter()
        for i, (path_name, models, repairs, controls, good) in enumerate(
            pool.map(corpus_item, sources), 1
        ):
            files[path_name] = digest(Path(path_name))
            totals.update(dict(modeling=models, repairs=repairs, controls=controls))
            support.update(good)
            if i % 10 == 0:
                print("new evidence", i, "/ 90", flush=True)
        if not totals["repairs"]:
            raise ValueError("No verified repair targets")
        prior.freeze(
            RUN / "training.json",
            dict(files=files, totals=dict(totals), support=dict(support)),
        )
    print("IMPROVED_READY", dict(totals), dict(support), flush=True)
