"""Training-only: independently certify assignments sampled by the p-bit kernel."""
import json
import time
import numpy as np
from qihc.generic.bounded import violations, objective
from qihc.generic.feedback.search import sample


def evidence(model, gold, states):
    xs = np.unique(np.asarray(states, dtype=int), axis=0)
    gm = violations(gold, xs).sum(axis=1) == 0
    cm = violations(model, xs).sum(axis=1) == 0
    rows = [dict(kind="feasibility", x=xs[i].tolist(), correct_feasible=bool(gm[i]),
                 candidate_feasible=bool(cm[i])) for i in np.flatnonzero(gm != cm)]
    both = xs[gm & cm]
    if len(both) > 1:
        def values(spec):
            return objective(spec, both) * (1 if spec["objective"]["sense"] == "min" else -1)
        gv, mv = values(gold), values(model)
        for i in range(len(both)):
            bad = np.flatnonzero(np.sign(gv[i]-gv) != np.sign(mv[i]-mv))
            if len(bad):
                j = bad[0]
                rows.append(dict(kind="objective", x=both[i].tolist(), y=both[j].tolist(),
                                 gold_order=int(np.sign(gv[i]-gv[j])), model_order=int(np.sign(mv[i]-mv[j]))))
                break
    return rows


def probe(model, gold, seed, seconds=20):
    rng = np.random.default_rng(seed)
    # Unclamped and single-variable interventions at both domain endpoints.
    variables = model["variables"]
    actions = [{}] + [{int(i):variables[i][bound]} for i in rng.permutation(len(variables))[:3]
                      for bound in ("lower","upper")]
    totals = np.zeros(len(actions))
    visits = np.zeros(len(actions))
    cards, trace = {}, []
    deadline = time.monotonic() + seconds
    for step in range(18):
        left = deadline - time.monotonic()
        if left <= 0: break
        if step < len(actions): i = step
        elif rng.random() < .2: i = int(rng.integers(len(actions)))
        else: i = int(np.argmax(totals / np.maximum(1, visits)))
        started = time.monotonic()
        row = sample(model, left/(18-step), seed+step, chains=32, temperature=(.25,.5,1.)[step%3],
                     clamps=actions[i], read_q=False)
        before = len(cards)
        for card in evidence(model, gold, row["final"]):
            cards.setdefault(json.dumps(card, sort_keys=True), card)
        duration = max(.001, time.monotonic()-started)
        visits[i] += 1
        totals[i] += (len(cards)-before)/duration
        trace.append(dict(action=actions[i], new=len(cards)-before, seconds=duration))
    return list(cards.values())[:4], trace
