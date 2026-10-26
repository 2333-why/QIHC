"""Independent scoring; never imported by GPU generation workers."""
from qihc.generic.bounded import objective, violations
from qihc.generic.public_modeling import answer_agreement

def evaluate(case, spec, result):
    out = dict(result)
    out['predicted_model_feasible'] = bool(result.get('feasible', False))
    if case['gold']:
        try:
            if result.get('x') is None:
                raise ValueError('No candidate')
            if len(result['x']) != len(spec['variables']):
                raise ValueError('Candidate dimension differs from model')
            values = {v['name']: x for v, x in zip(spec['variables'], result['x'])}
            if set(values) != {v['name'] for v in case['gold']['variables']}:
                raise ValueError('Original variable mapping differs')
            x = [values[v['name']] for v in case['gold']['variables']]
            out['verified_feasible'] = bool(violations(case['gold'], x).sum() == 0)
            out['verified_objective'] = float(objective(case['gold'], x))
        except (ValueError, TypeError, KeyError) as exc:
            out.update(verified_feasible=False, verification_error=str(exc))
    else:
        out['verified_feasible'] = None
        out['reference_answer_agreement'] = answer_agreement(result, case['reference_answer']) if result.get('objective') is not None else False
    return out
