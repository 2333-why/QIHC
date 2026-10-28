"""Source-independent mathematical warnings, never automatic formula repair."""
from collections import defaultdict


def diagnostics(spec):
    groups = defaultdict(list)
    linear = spec['objective']['linear']
    reused = []
    for c in spec['constraints']:
        groups[(tuple(c['a']), c['op'], c['rhs'])].append(c['id'])
        if any(linear) and c['a'] == linear:
            reused.append(c['id'])
    binary = all((v['lower'], v['upper']) == (0, 1) for v in spec['variables'])
    return {
        'constant_false_constraints': [c['id'] for c in spec['constraints'] if not any(c['a']) and
            not (0 <= c['rhs'] if c['op'] == 'le' else 0 >= c['rhs'] if c['op'] == 'ge' else c['rhs'] == 0)],
        'duplicate_constraint_groups': [ids for ids in groups.values() if len(ids) > 1][:8],
        'objective_vector_reused_as_constraint': reused[:8],
        'uniform_linear_binary_objective': binary and len(set(linear)) == 1 and not spec['objective']['quadratic'],
        'constraint_support_sizes': [sum(a != 0 for a in c['a']) for c in spec['constraints'][:16]],
        'quadratic_terms': len(spec['objective']['quadratic']),
        'scope': 'Warnings only. Repeated constraints and uniform objectives may be correct. Check against text; never delete or replace automatically.'}
