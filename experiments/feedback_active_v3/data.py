"""Deterministic ten-family data. Only the evaluator opens gold records."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from qihc.experiment_runtime.storage import save, load
from qihc.generic.bounded import validate

FAMILIES = ('KP', 'MIS', 'MVC', 'MAXCUT', 'MCP', 'MDS', 'WSC', 'MAX2SAT', 'GAP', 'SCHEDULE')
LAYERS = ('direct', 'entity_index', 'quantifier', 'business')
ASSETS = Path(__file__).parent / 'assets'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def make_case(family, n, seed, layer=0, split='main'):
    """Generate mathematics and NL independently of any candidate/solver result."""
    rng = np.random.default_rng(seed)
    linear = rng.integers(1, 20, n).tolist()
    rows, quad, paragraphs = [], {}, []
    sense = 'max'

    def con(indices, op, rhs, weights=None):
        a = [0] * n
        for k, i in enumerate(indices):
            a[int(i)] += int(weights[k] if weights is not None else 1)
        rows.append(dict(id=f'c{len(rows)}', a=a, op=op, rhs=int(rhs)))

    def term(i, j, value):
        key = tuple(sorted((int(i), int(j))))
        quad[key] = quad.get(key, 0) + int(value)

    def field(name, value):
        paragraphs.append(name + ' = ' + json.dumps(value, separators=(',', ':')))

    labels = rng.choice(np.arange(1000, 100000), n, replace=False).tolist()
    # Nontrivial remapping is public; never infer variable order from the gold.
    identifiers = labels if layer == 1 else list(range(n))
    field('Decision identifiers in x0,x1,... order', identifiers)
    if family == 'KP':
        w = rng.integers(1, 15, n).tolist(); cap = max(1, sum(w) // 3)
        con(range(n), 'le', cap, w)
        text = 'Choose items to maximize total selected value. Total selected weight may not exceed the capacity.'
        field('values', linear); field('weights', w); field('capacity', cap)
    elif family in ('MIS', 'MVC', 'MAXCUT', 'MCP', 'MDS'):
        # O(n) relations keep all ten families inside the shared 512-row IR limit.
        pairs = {(i, (i + 1) % n) for i in range(n)}
        pairs = {tuple(sorted(e)) for e in pairs}
        while len(pairs) < min(2 * n, n * (n - 1) // 2):
            u, v = rng.choice(n, 2, replace=False); pairs.add(tuple(sorted((int(u), int(v)))))
        edges = sorted(pairs)
        field('weights', linear)
        field('incompatible_pairs' if family == 'MCP' else 'edges', [[identifiers[i], identifiers[j]] for i, j in edges])
        if family in ('MIS', 'MCP'):
            text = ('Maximize total selected vertex weight. For every listed edge, its two endpoints cannot both be selected.' if family == 'MIS' else
                    'Select a maximum-weight clique. All distinct vertices are compatible except the listed incompatible pairs. Each listed pair cannot both be selected.')
            for i, j in edges: con([i, j], 'le', 1)
        elif family == 'MVC':
            sense = 'min'; text = 'Minimize total selected vertex cost. Every listed edge must have at least one selected endpoint.'
            for i, j in edges: con([i, j], 'ge', 1)
        elif family == 'MDS':
            sense = 'min'; text = 'Minimize total selected vertex cost. Each vertex must itself be selected or have at least one selected neighbor; apply this to every vertex.'
            neighborhoods = [[i] + [v if u == i else u for u, v in edges if i in (u, v)] for i in range(n)]
            for group in neighborhoods: con(group, 'ge', 1)
            field('Neighborhoods including self, expressed as x-variable indices', neighborhoods)
        else:
            linear = [0] * n
            text = 'Partition vertices into two groups; x_i is its group (0 or 1). Maximize the number of listed edges with endpoints in different groups. There is no group-size requirement; listed vertex weights do not affect this objective.'
            for i, j in edges:
                linear[i] += 1; linear[j] += 1; term(i, j, -2)
    elif family == 'WSC':
        sense = 'min'; memberships = []
        for k in range(max(4, n // 2)):
            subset = sorted(rng.choice(n, min(4, n), replace=False).tolist())
            memberships.append(subset); con(subset, 'ge', 1)
        text = 'Select sets of minimum total cost. Every universe element must belong to at least one selected set; element_sets[e] lists all set indices containing element e.'
        field('costs', linear); field('element_sets', memberships)
    elif family == 'MAX2SAT':
        # Weighted two-literal SAT uses an exact quadratic loss; constants are
        # reported separately, never inserted into a schema without constants.
        sense = 'min'; linear = [0] * n; clauses = []; offset = 0
        for _ in range(n):
            i, j = sorted(rng.choice(n, 2, replace=False).tolist())
            pi, pj = rng.integers(0, 2, 2).tolist(); w = int(rng.integers(1, 10))
            # Unsatisfied literal: (1-x) for positive, x for negative.
            ai, bi = (1, -1) if pi else (0, 1)
            aj, bj = (1, -1) if pj else (0, 1)
            offset += w * ai * aj; linear[i] += w * bi * aj; linear[j] += w * bj * ai; term(i, j, w * bi * bj)
            clauses.append(dict(i=i, j=j, positive_i=pi, positive_j=pj, weight=w))
        text = 'Assign Boolean variables to minimize total weight of unsatisfied two-literal clauses. positive=1 denotes x, positive=0 denotes NOT x. A clause is satisfied if either literal is true. Additive objective constants may be omitted.'
        field('clauses', clauses)
    elif family == 'GAP':
        agents = 2 if n < 32 else 4
        assert n % agents == 0
        jobs = n // agents; sense = 'min'
        usage = rng.integers(1, 10, n).tolist(); capacities = []
        for j in range(jobs): con(range(j * agents, (j + 1) * agents), 'eq', 1)
        for a in range(agents):
            ids = list(range(a, n, agents))
            cap = sum(usage[j * agents + a] for j in range(jobs) if j % agents == a) + 5
            capacities.append(cap); con(ids, 'le', cap, [usage[i] for i in ids])
        text = f'Assign each of {jobs} jobs to exactly one of {agents} agents, minimizing total cost. Variable x_(j*{agents}+a) selects assignment (job j, agent a). Each agent has its own total resource capacity; costs and resource usage are row-major by job then agent.'
        field('costs', linear); field('usage', usage); field('capacities', capacities)
    elif family == 'SCHEDULE':
        jobs = 2 if n < 32 else 4; slots = n // jobs; sense = 'min'
        assert n % jobs == 0 and slots >= jobs
        text = f'Schedule {jobs} unit-duration jobs into {slots} discrete slots on one machine, minimizing total assignment cost. Every job occupies exactly one slot. Each slot contains at most one job. x_(j*{slots}+t) means job j occupies slot t; costs are row-major. There are no other precedence constraints.'
        for j in range(jobs): con(range(j * slots, (j + 1) * slots), 'eq', 1)
        for t in range(slots): con(range(t, n, slots), 'le', 1)
        field('costs', linear)
    else:
        raise ValueError(family)
    if layer == 2:
        text += ' The rule applies to every listed relation, including boundary indices. An upper limit allows equality; a lower limit also allows equality. Unlisted restrictions must not be invented.'
    if layer == 3:
        text = 'A planning team needs an implementable decision. ' + text + ' The analyst supplies the data below; the operations team requires all stated rules to hold simultaneously. Optimize only the stated objective.'
    names = [f'x{i}' for i in range(n)]
    question = text + '\nAll decision variables are binary, 0 or 1. ' + '\n'.join(paragraphs)
    gold = validate(dict(schema='bounded-integer-v2', variables=[dict(name=name, lower=0, upper=1) for name in names],
                         objective=dict(sense=sense, linear=linear, quadratic=[[i, j, c] for (i, j), c in sorted(quad.items()) if c]),
                         constraints=rows, logic=[]))
    identity = f'{split}_{family}_{seed}'
    return dict(id=identity, family=family, split=split, layer=LAYERS[layer], question=question,
                variable_names=names, gold=gold, objective_offset=offset if family == 'MAX2SAT' else 0,
                provenance=dict(generator='feedback_active_v3', seed=int(seed), independent_instance=True))


def public_view(case):
    return {k: case[k] for k in ('id', 'question', 'variable_names')}


def build():
    cases = []; rng = np.random.default_rng(20261107)
    for split in ('development', 'main', 'scale'):
        for f in FAMILIES:
            count = (20 if f in FAMILIES[:6] else 0) if split == 'development' else (120 if f in FAMILIES[:6] else 60) if split == 'main' else 24
            for k in range(count):
                sizes = [8, 12, 16] if split != 'scale' else ([32, 48, 64] if k < 12 else [128, 192, 256])
                cases.append(make_case(f, sizes[k % len(sizes)], int(rng.integers(1, 2**31 - 1)), k % 4, split))
    assert len(cases) == 1320
    assert len({digest(c['gold']) for c in cases}) == len(cases)
    assert len({digest(c['question']) for c in cases}) == len(cases)
    subsets = {}
    main = [c for c in cases if c['split'] == 'main']
    # Select 1/4 or 1/8 in each family/wording stratum; new families may have
    # floor/ceil allocation, then deterministic remainder completes exact N.
    for label, count in [('ablation', 240), ('audit', 120), ('second', 120), ('replay', 120)]:
        groups = collections.defaultdict(list)
        for c in main: groups[(c['family'], c['layer'])].append(c)
        ordered = {k: sorted(v, key=lambda c: digest([20261108, label, c['id']])) for k, v in groups.items()}
        chosen = []
        for key, values in sorted(ordered.items()): chosen += values[:int(len(values) * count / 960)]
        remaining = sorted((c for c in main if c not in chosen), key=lambda c: digest([label, c['id']]))
        chosen += remaining[:count - len(chosen)]
        subsets[label] = sorted(c['id'] for c in chosen)
    return cases, subsets


def prepare(destination):
    cases, subsets = build(); destination.mkdir(parents=True, exist_ok=True)
    public = [dict(public_view(c), split=c['split'], family=c['family'], layer=c['layer']) for c in cases]
    save(destination / 'cases.public.json', public)
    save(destination / 'evaluation' / 'cases.gold.json', cases)
    save(destination / 'subsets.json', subsets)
    manifest = dict(version='synthetic-test-fixtures-only', formal_public_only=False, cases=len(cases), counts=dict(collections.Counter(c['split'] for c in cases)),
                    public_sha256=digest(public), gold_sha256=digest(cases), subsets_sha256=digest(subsets),
                    public_external_status='separate audited import; not replaced by synthetic data')
    save(destination / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=ASSETS / 'synthetic_test_fixtures')
    args = parser.parse_args(); print(json.dumps(prepare(args.output), indent=2))
