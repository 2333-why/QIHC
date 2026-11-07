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
RUN = Path(os.environ.get("QIHC_POST_RUN", BASE / "results/E005-1.4_four"))
MODEL = Path(os.environ.get("QIHC_MODEL", BASE / "models/Qwen3-32B"))
METHODS = ("sft", "solver_rl", "sdpo", "pvcd")


def target(model):
    return json.dumps({"model": compact(model)}, ensure_ascii=False, separators=(",", ":"))


def structure(row):
    # Canonical verifier ignores constraint order, ids and positive rescaling.
    return hashlib.sha256(repr(normalized(row["gold"])).encode()).hexdigest()


def prepare():
    if (RUN/"manifest.json").exists() or RUN.name.startswith("E004"):
        raise ValueError("Select a dedicated E005 result directory")
    restore(DATA)
    public = {c["id"]: c for c in load(DATA / "cases.public.json")}
    gold = load(DATA / "evaluation/cases.gold.json")
    held = [r for r in gold if r["split"] != "development"]
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
    files = {"train": train, "validation": validation, "test_public": main}
    manifest = dict(experiment="E005-1.4", files={}, excluded=dropped,
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
