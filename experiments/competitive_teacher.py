"""Same-state teacher competition with held-out rollout-seed confirmation."""
from __future__ import annotations

import copy
import json
import math
import re
from qihc.problems.cvrp.training_reward import rollout_reward


def select_teacher(prompt, teacher, model, context, *, chains=64, steps=48, margin=0.001):
    # Import lazily to avoid a module-level cycle with the data builder.
    from experiments.prepare_joint_training_data import canonical_proposal, align_teacher_domain
    match = re.search(r"KNN reference proposal: ([^\n]+)", prompt)
    if not match:
        raise ValueError("Competitive teachers require an exact KNN reference in the inference prompt")
    reference = json.loads(match.group(1))
    candidates = [("knn", reference), ("pbit", teacher), ("llm", model)]
    unique = {}
    for source, payload in candidates:
        aligned = align_teacher_domain(payload, context["allowed_routes"])
        if aligned is None:
            continue
        current_routes = {c: r for r, route in enumerate(context["routes"]) for c in route}
        limit = int(context.get("solver_config", {}).get("routes_per_customer", 4))
        for customer in aligned["destroy_customers"]:
            key = str(customer)
            choices = aligned["candidate_routes"][key][:limit]
            current = current_routes[int(customer)]
            if current not in choices:
                if len(choices) >= limit:
                    choices[-1] = current
                else:
                    choices.append(current)
            aligned["candidate_routes"][key] = choices
        try:
            answer = canonical_proposal(aligned)
        except (ValueError, KeyError, TypeError):
            continue
        unique.setdefault(answer, source)
    scored = []
    for answer, source in unique.items():
        rewards = []
        for offset in (0, 104729, 209759):
            replay = copy.deepcopy(context)
            replay["seed"] = int(context.get("seed", 0)) + offset
            reward = rollout_reward(json.loads(answer), replay, chains=chains, steps=steps)
            if not math.isfinite(reward):
                raise ValueError("Nonfinite competitive teacher reward")
            rewards.append(reward)
        scored.append({"source": source, "answer": answer, "rewards": rewards,
                       "selection_reward": sum(rewards[:2]) / 2})
    if not scored:
        return None
    # The third seed never participates in ranking; it independently confirms the pair.
    scored.sort(key=lambda x: x["selection_reward"], reverse=True)
    winner = scored[0]
    if min(winner["rewards"]) < 1.0 or winner["selection_reward"] <= 1.0 + margin:
        return {"winner": None, "pair": None, "candidates": scored}
    loser = next((row for row in reversed(scored[1:])
                  if all(a - b >= margin for a, b in zip(winner["rewards"], row["rewards"]))), None)
    return {"winner": winner, "pair": loser, "candidates": scored}
