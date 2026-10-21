"""Opt-in bounded integer / Boolean IR and exact QUBO compiler (v2).

No generated code is executed. Dense software compiler, not an int8 FPGA driver.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from .energy import Energy, bounded_weights, integer

VERSION = 'bounded-integer-v2'


def validate(spec):
    if not isinstance(spec, dict) or set(spec) != {'schema', 'variables', 'objective', 'constraints', 'logic'} or spec['schema'] != VERSION:
        raise ValueError('Expected bounded-integer-v2 schema')
    variables = spec['variables']
    if not isinstance(variables, list) or not 1 <= len(variables) <= 256:
        raise ValueError('Expected 1..256 variables')
    names = set()
    for v in variables:
        if not isinstance(v, dict) or set(v) != {'name', 'lower', 'upper'}:
            raise ValueError('Variable requires name/lower/upper')
        if not isinstance(v['name'], str) or not 1 <= len(v['name']) <= 64 or v['name'] in names:
            raise ValueError('Unique bounded variable names required')
        names.add(v['name'])
        if integer(v['lower']) > integer(v['upper']):
            raise ValueError('Invalid variable bounds')
    n = len(variables)
    obj = spec['objective']
    if not isinstance(obj, dict) or set(obj) != {'sense', 'linear', 'quadratic'} or obj['sense'] not in ('min', 'max'):
        raise ValueError('Invalid objective')
    def linear(a):
        if not isinstance(a, list) or len(a) != n:
            raise ValueError('Invalid coefficient dimension')
        for c in a:
            integer(c)
    linear(obj['linear'])
    if not isinstance(obj['quadratic'], list) or len(obj['quadratic']) > n * (n + 1) // 2:
        raise ValueError('Invalid quadratic terms')
    seen = set()
    for term in obj['quadratic']:
        if not isinstance(term, list) or len(term) != 3:
            raise ValueError('Expected [i,j,c]')
        i, j, c = map(integer, term)
        if not 0 <= i <= j < n or (i, j) in seen:
            raise ValueError('Quadratic indices must be unique with i<=j')
        seen.add((i, j))
    if any(not isinstance(spec[k], list) for k in ('constraints', 'logic')) or len(spec['constraints']) + len(spec['logic']) > 512:
        raise ValueError('At most 512 source constraints')
    ids, nodes = set(), [0]
    def expression(expr, depth=0):
        nodes[0] += 1
        if depth > 16 or nodes[0] > 512 or not isinstance(expr, dict):
            raise ValueError('Invalid/oversized logic expression')
        if set(expr) == {'var'}:
            i = integer(expr['var'])
            if not 0 <= i < n or (variables[i]['lower'], variables[i]['upper']) != (0, 1):
                raise ValueError('Logic leaves require binary variables')
            return
        if set(expr) != {'op', 'args'} or expr['op'] not in ('not', 'and', 'or', 'implies', 'iff') or not isinstance(expr['args'], list):
            raise ValueError('Unsupported logic operator')
        arity = len(expr['args'])
        if (expr['op'] == 'not' and arity != 1) or (expr['op'] in ('implies', 'iff') and arity != 2) or not 1 <= arity <= 32:
            raise ValueError('Invalid logic arity')
        for arg in expr['args']:
            expression(arg, depth + 1)
    for con in spec['constraints'] + spec['logic']:
        if not isinstance(con, dict):
            raise ValueError('Invalid constraint')
        key = con.get('id')
        if not isinstance(key, str) or not 1 <= len(key) <= 64 or key in ids:
            raise ValueError('Unique bounded constraint IDs required')
        ids.add(key)
    for con in spec['constraints']:
        if set(con) != {'id', 'a', 'op', 'rhs'} or con['op'] not in ('le', 'ge', 'eq'):
            raise ValueError('Invalid linear constraint')
        linear(con['a'])
        integer(con['rhs'])
    for con in spec['logic']:
        if set(con) != {'id', 'expr'}:
            raise ValueError('Invalid logical constraint')
        expression(con['expr'])
    return spec


def logic_value(expr, x):
    if 'var' in expr:
        return x[..., expr['var']] == 1
    args = [logic_value(a, x) for a in expr['args']]
    op = expr['op']
    if op == 'not':
        return ~args[0]
    if op == 'implies':
        return ~args[0] | args[1]
    if op == 'iff':
        return args[0] == args[1]
    return np.logical_and.reduce(args) if op == 'and' else np.logical_or.reduce(args)


def objective(spec, x):
    x = np.asarray(x)
    out = x @ np.asarray(spec['objective']['linear'], dtype=float)
    for i, j, c in spec['objective']['quadratic']:
        out = out + c * x[..., i] * x[..., j]
    return out


def violations(spec, x):
    """Original variable domains and constraints, independent of auxiliary bits."""
    x = np.asarray(x, dtype=float)
    if x.shape[-1] != len(spec['variables']) or not np.all(np.isfinite(x)):
        raise ValueError('Invalid candidate vector')
    lo = np.array([v['lower'] for v in spec['variables']])
    hi = np.array([v['upper'] for v in spec['variables']])
    domain = (np.maximum(lo-x, 0) + np.maximum(x-hi, 0) + np.abs(x-np.rint(x))).sum(axis=-1)
    out = [domain]
    for con in spec['constraints']:
        r = x @ np.asarray(con['a']) - con['rhs']
        out.append(abs(r) if con['op'] == 'eq' else np.maximum(0, r if con['op'] == 'le' else -r))
    out.extend((~logic_value(c['expr'], x)).astype(float) for c in spec['logic'])
    return np.stack(out, axis=-1)


@dataclass
class Compiled:
    spec: dict
    energy: Energy
    mapping: np.ndarray
    lower: np.ndarray
    offset: float
    row_sources: list
    decision_bits: int
    gate_bits: int

    @property
    def bits(self):
        return self.energy.bits

    def decode(self, bits):
        bits = np.asarray(bits)
        if bits.shape[-1] != self.bits or not np.all((bits == 0) | (bits == 1)):
            raise ValueError('Expected complete binary state')
        return self.lower + bits[..., :self.decision_bits] @ self.mapping.T

    def qubo(self, weights):
        q, constant = self.energy.qubo(weights)
        return q, constant + self.offset


def compile_energy(spec, *, max_bits=2048):
    spec = copy.deepcopy(validate(spec))
    if type(max_bits) is not int or not 1 <= max_bits <= 4096:
        raise ValueError('Dense compiler max_bits must be 1..4096')
    vs = spec['variables']
    lower = np.array([v['lower'] for v in vs], dtype=float)
    encodings = [bounded_weights(v['upper']-v['lower']) for v in vs]
    decisions = sum(map(len, encodings))
    if decisions > max_bits:
        raise ValueError('Decision encoding exceeds bit budget')
    mapping = np.zeros((len(vs), decisions))
    pos = 0
    for i, ws in enumerate(encodings):
        mapping[i, pos:pos+len(ws)] = ws
        pos += len(ws)
    rows, sources = [], []
    total = decisions
    def row(terms, op, rhs, source):
        coefficients = {}
        for index, value in terms:
            coefficients[index] = coefficients.get(index, 0) + value
        rows.append((coefficients, op, rhs))
        sources.append(source)
    for source, c in enumerate(spec['constraints']):
        a = np.array(c['a'])
        row(enumerate(a @ mapping), c['op'], c['rhs'] - a @ lower, source)
    def gate(op, args, source):
        nonlocal total
        z = total
        total += 1
        if total > max_bits:
            raise ValueError('Logic encoding exceeds bit budget')
        if op == 'not':
            row([(z, 1), (args[0], 1)], 'eq', 1, source)
        elif op == 'and':
            for a in args:
                row([(z, 1), (a, -1)], 'le', 0, source)
            row([(z, 1)] + [(a, -1) for a in args], 'ge', 1-len(args), source)
        elif op == 'or':
            for a in args:
                row([(z, 1), (a, -1)], 'ge', 0, source)
            row([(z, 1)] + [(a, -1) for a in args], 'le', 0, source)
        return z
    def expression(expr, source):
        if 'var' in expr:
            return int(np.flatnonzero(mapping[expr['var']])[0])
        args = [expression(a, source) for a in expr['args']]
        op = expr['op']
        if op == 'implies':
            return gate('or', [gate('not', [args[0]], source), args[1]], source)
        if op == 'iff':
            left = gate('or', [gate('not', [args[0]], source), args[1]], source)
            right = gate('or', [gate('not', [args[1]], source), args[0]], source)
            return gate('and', [left, right], source)
        return gate(op, args, source)
    for source, c in enumerate(spec['logic'], len(spec['constraints'])):
        row([(expression(c['expr'], source), 1)], 'eq', 1, source)
    gates = total-decisions
    # Integer slack has exactly the required finite range (no invalid overflow codes).
    expanded = []
    for coefficients, op, rhs in rows:
        lo = sum(min(0, v) for v in coefficients.values())
        hi = sum(max(0, v) for v in coefficients.values())
        upper = max(0, rhs-lo) if op == 'le' else max(0, hi-rhs)
        ws = [] if op == 'eq' else bounded_weights(int(upper))
        for w in ws:
            coefficients[total] = w if op == 'le' else -w
            total += 1
        if total > max_bits:
            raise ValueError('Compiled model including auxiliary/slack bits exceeds bit budget')
        expanded.append((coefficients, rhs))
    # Fixed variables can produce a zero-bit, constant feasibility problem.
    total = max(1, total)
    obj = spec['objective']
    original_q = np.zeros((len(vs), len(vs)))
    for i, j, c in obj['quadratic']:
        original_q[i, j] += c if i == j else c/2
        if i != j:
            original_q[j, i] += c/2
    transformed = mapping.T @ original_q @ mapping
    transformed[np.diag_indices(decisions)] += mapping.T @ (np.array(obj['linear']) + 2*original_q @ lower)
    bound = max(1., float(np.abs(transformed).sum()))
    sign = 1 if obj['sense'] == 'min' else -1
    q = np.zeros((total, total))
    q[:decisions, :decisions] = sign*transformed/bound
    a = np.zeros((len(rows), total))
    rhs = []
    for k, (coefficients, value) in enumerate(expanded):
        for i, c in coefficients.items():
            a[k, i] = c
        rhs.append(value)
    energy = Energy(spec, q, a, np.array(rhs), bound)
    offset = sign * (lower @ original_q @ lower + np.array(obj['linear']) @ lower)/bound
    return Compiled(spec, energy, mapping, lower, float(offset), sources, decisions, gates)


def prompt(description):
    return ('Translate into JSON only, schema bounded-integer-v2. Fields: '
            'variables:[{name:string,lower:integer,upper:integer}], '
            'objective:{sense:min|max,linear:[integer per variable],quadratic:[[i,j,c],...]}, '
            'constraints:[{id:string,a:[integer per variable],op:le|ge|eq,rhs:integer}], '
            'logic:[{id:string,expr:EXPR}]. EXPR is {var:index} for a binary variable, or '
            '{op:not|and|or|implies|iff,args:[EXPR,...]}. Every logic root must be true. '
            'Indices are zero-based. Quadratic pairs unique with i<=j. No penalties or auxiliary '
            'variables: the compiler creates them. All variable bounds must follow from the request; '
            'never invent finite bounds. Max 256 variables, 512 source constraints, integer '
            'bounds/coefficients of magnitude <=10000. If continuous, unbounded or unsupported, '
            'return {"unsupported":"reason"}; never silently discretize.\n' + description)


def parse(text):
    """Parse a model response with the existing JSON extractor, then validate v2."""
    from .benchmark import parse_object
    return validate(parse_object(text))
