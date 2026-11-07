"""Prepare immutable E004-aligned public inputs and training-only labels."""
import os
from pathlib import Path
from collections import defaultdict
import hashlib
import json

from qihc.experiment_runtime.storage import digest, load, save, key
from qihc.generic.feedback.verification import normalized
from qihc.modeling.active_protocol_v2 import compact, parse_model
from experiments.feedback_active_v3.assets import restore
from experiments.feedback_active_v3.data import public_view

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "feedback_active_v3/assets/public_dataset"
BASE = Path("/inspire/hdd/global_user/czxs25210022/why")
RUN = Path(os.environ.get("QIHC_POST_RUN", BASE / "results/E005-1.7_eight"))
MODEL = Path(os.environ.get("QIHC_MODEL", BASE / "models/Qwen3-32B"))
MODEL_METHODS = ("sft", "solver_rl", "sdpo", "pvcd")
from .energy import ENERGY_METHODS
METHODS = MODEL_METHODS + ENERGY_METHODS
STAGE_SEEDS = {"development":(17,), "main":(17,29,43), "scale":(17,29), "public":(17,29)}

def cohort(mode,seed):
    if mode=="pilot":
        return [dict(r["case"],stage="pilot") for r in load(RUN/"validation.json")]
    return [r for r in load(RUN/"cohort.json") if seed in STAGE_SEEDS[r["stage"]]]



def target(model):
    return json.dumps({"model": compact(model)}, ensure_ascii=False, separators=(",", ":"))


def structure(row):
    # Canonical verifier ignores constraint order, ids and positive rescaling.
    return hashlib.sha256(repr(normalized(row["gold"])).encode()).hexdigest()


def prepare():
    if (RUN/"manifest.json").exists() or RUN.name.startswith("E004"):
        raise ValueError("Select a dedicated E005 result directory")
    if (RUN/"split.json").exists() and load(RUN/"split.json").get("experiment")!="E005-1.7":
        raise ValueError("Use the new eight-arm result directory")
    restore(DATA)
    public = {c["id"]: c for c in load(DATA / "cases.public.json")}
    gold = load(DATA / "evaluation/cases.gold.json")
    held = [r for r in gold if r["split"] != "development"]
    extra_gold=load(DATA/"evaluation/cases.public_external.gold.json")
    held.extend(extra_gold)
    used = {structure(r) for r in held}
    questions = {" ".join(r["question"].split()) for r in held}
    grouped = defaultdict(list)
    dropped = []
    for r in sorted((r for r in gold if r["split"] == "development"), key=lambda r: key(r["id"], "17")):
        h = structure(r)
        question = " ".join(r["question"].split())
        if h in used or question in questions:
            dropped.append(r["id"])
            continue
        used.add(h)
        questions.add(question)
        case = public_view(public[r["id"]])
        answer = target(r["gold"])
        if normalized(parse_model(answer, case)) != normalized(r["gold"]):
            raise ValueError("Gold serialization changed " + r["id"])
        grouped[r["family"]].append(dict(case=case, gold=r["gold"], answer=answer))
    train, validation = [], []
    for rows in grouped.values():
        if len(rows) < 5:
            raise ValueError("Too few distinct development cases")
        validation.extend(rows[:4])
        train.extend(rows[4:])
    main = [public_view(r) for r in public.values() if r["split"] == "main"]
    if len(main) != 480:
        raise ValueError("Unexpected E004 main cohort")
    external=load(DATA/"cases.public_external.json")
    full=[dict(public_view(c),stage=c["split"]) for c in public.values()]
    full += [dict(public_view(c),stage="public") for c in external]
    sizes={stage:sum(c["stage"]==stage for c in full) for stage in STAGE_SEEDS}
    if sizes!={"development":120,"main":480,"scale":300,"public":10} or len({c["id"] for c in full})!=910:
        raise ValueError(f"E004 cohort changed: {sizes}")
    files = {"train": train, "validation": validation, "test_public": main, "cohort":full}
    manifest = dict(experiment="E005-1.7", files={}, excluded=dropped,
                    source=digest(DATA / "public.manifest.json"),
                    code={p.name:digest(p) for p in sorted(HERE.glob("*.py"))})
    for name, rows in files.items():
        manifest["files"][name] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    frozen = RUN / "split.json"
    if frozen.exists() and load(frozen) != manifest:
        raise ValueError("Run already bound to different data or code")
    for name, rows in files.items():
        save(RUN / (name + ".json"), rows)
    save(frozen, manifest)
    print({k: len(v) for k,v in files.items()}, "excluded", len(dropped), flush=True)


if __name__ == "__main__":
    prepare()
