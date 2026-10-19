"""Versioned, family-independent wire format; no gold-dependent repair."""
from __future__ import annotations

from .energy import integer, validate
from .benchmark import parse_object

VERSION = "sparse-binary-v1"


def prompt(description):
    return (
        'Translate the request into a binary optimization model. Return JSON only: '
        '{"schema":"sparse-binary-v1","n":N,"objective":{"sense":"min|max",'
        '"linear":[[index,integer_coefficient],...],"quadratic":[[i,j,integer_coefficient],...]},'
        '"constraints":[{"id":"unique","terms":[[index,integer_coefficient],...],'
        '"op":"le|ge|eq","rhs":integer},...]}. Indices start at zero. '
        'Omitted terms mean zero; indices must be unique, quadratic pairs unique with i<j. '
        'Translate every rule, including negation, implication and interactions. '
        'Do not replace the objective by penalties or delete difficult constraints. '
        'The compiler generates slack variables and penalties. Supported scope: at most 96 '
        'binary variables, integer quadratic objective and at most 128 integer linear constraints. '
        'If the request cannot be represented, return {"unsupported":"reason"}, never silently approximate.\n'
        + description
    )


def encode(spec):
    validate(spec)
    pairs = lambda a: [[i, c] for i, c in enumerate(a) if c]
    return {"schema": VERSION, "n": spec["n"],
            "objective": {**spec["objective"], "linear": pairs(spec["objective"]["linear"])},
            "constraints": [{"id": c["id"], "terms": pairs(c["a"]), "op": c["op"],
                             "rhs": c["rhs"]} for c in spec["constraints"]]}


def decode(answer):
    if not isinstance(answer, dict) or set(answer) != {"schema", "n", "objective", "constraints"}:
        raise ValueError("Expected sparse model, not unsupported or extra fields")
    if answer["schema"] != VERSION:
        raise ValueError("Unknown schema version")
    n = integer(answer["n"], 96)
    if n < 1:
        raise ValueError("n must be positive")

    def dense(terms):
        if not isinstance(terms, list) or len(terms) > n:
            raise ValueError("Invalid sparse terms")
        out, seen = [0] * n, set()
        for term in terms:
            if not isinstance(term, list) or len(term) != 2:
                raise ValueError("Terms must be [index, coefficient]")
            i, c = map(integer, term)
            if not 0 <= i < n or i in seen:
                raise ValueError("Duplicate or out-of-range index")
            seen.add(i)
            out[i] = c
        return out

    obj = answer["objective"]
    if not isinstance(obj, dict) or set(obj) - {"sense", "linear", "quadratic"} or "sense" not in obj:
        raise ValueError("Invalid objective")
    # Same deterministic zero defaults for every model; no semantic inference.
    objective = {"sense": obj["sense"], "linear": dense(obj.get("linear", [])),
                 "quadratic": obj.get("quadratic", [])}
    if not isinstance(answer["constraints"], list):
        raise ValueError("constraints must be a list")
    constraints = []
    for c in answer["constraints"]:
        if not isinstance(c, dict) or set(c) != {"id", "terms", "op", "rhs"}:
            raise ValueError("Invalid sparse constraint")
        constraints.append({"id": c["id"], "a": dense(c["terms"]), "op": c["op"], "rhs": c["rhs"]})
    return validate({"n": n, "objective": objective, "constraints": constraints})


def parse(text):
    return decode(parse_object(text))
