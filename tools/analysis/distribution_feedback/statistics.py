"""Predeclared paired comparisons; instances (not seeds) are sampling units."""
from __future__ import annotations

import hashlib
import math

import numpy as np


PAIRS = [
    ('search_feedback', 'pbit_best', 'pbit_self_review'),
    ('distribution_information', 'pbit_distribution', 'pbit_best'),
    ('intervention_search', 'pbit_probe_blind', 'pbit_distribution'),
    ('intervention_information', 'pbit_intervention', 'pbit_probe_blind'),
    ('energy_selection', 'pbit_energy', 'pbit_intervention'),
    ('update_selection', 'pbit_block', 'pbit_intervention'),
    ('joint_vs_energy', 'pbit_selected', 'pbit_energy'),
    ('joint_vs_update', 'pbit_selected', 'pbit_block'),
    ('selection_vs_random', 'pbit_selected', 'pbit_random_policy'),
    ('framework_on_metropolis', 'metropolis_selected', 'metropolis_best'),
    ('pbit_vs_metropolis', 'pbit_selected', 'metropolis_selected'),
]


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def sign_pvalue(wins, losses):
    """Exact two-sided paired sign test; ties excluded; tests directions, not mean."""
    n = wins+losses
    if n == 0:
        return 1.
    k = min(wins, losses)
    logs = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
    largest = max(logs)
    return min(1., 2*math.exp(largest)*sum(math.exp(v-largest) for v in logs))


def paired_summary(values, *, higher_better, seed, bootstrap):
    differences = np.array(values, dtype=float)
    n = len(differences)
    if not n:
        return {'instances': 0, 'mean_delta_treatment_minus_control': None,
                'bootstrap_95pct': None, 'wins': 0, 'ties': 0, 'losses': 0,
                'sign_test_p_two_sided': None, 'higher_is_better': higher_better}
    improved = differences if higher_better else -differences
    wins, losses = int((improved > 1e-12).sum()), int((improved < -1e-12).sum())
    interval = None
    if n >= 2:
        rng = np.random.default_rng(seed)
        means = []
        for start in range(0, bootstrap, 64):
            means.extend(rng.choice(differences, (min(64, bootstrap-start), n)).mean(axis=1).tolist())
        interval = np.quantile(means, [.025, .975]).tolist()
    return {'instances': n, 'mean_delta_treatment_minus_control': float(differences.mean()),
            'bootstrap_95pct': interval, 'wins': wins, 'ties': n-wins-losses, 'losses': losses,
            'sign_test_p_two_sided': sign_pvalue(wins, losses) if n >= 2 else None,
            'higher_is_better': higher_better,
            'ci_scope': 'Exploratory instance bootstrap; not a simultaneous interval; null for n<2.'}


def holm(results):
    """One family across all emitted group x pair x stage x metric tests."""
    ordered = sorted([r for r in results if r['sign_test_p_two_sided'] is not None],
                     key=lambda r: r['sign_test_p_two_sided'])
    previous = 0.
    for i, result in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered)-i)*result['sign_test_p_two_sided']))
        result['sign_test_p_holm'] = previous
    for result in results:
        result.setdefault('sign_test_p_holm', None)
        result['holm_family_tests'] = len(ordered)


def success(records, metric):
    # An invalid/unsupported model is a completed failure, never silently dropped.
    return sum(bool(record['runs'] and record['runs'][0].get(metric) is True) for record in records)/len(records)


def common_gap(case, a, b):
    """Compare SAME seed slots feasible in both arms; then average within instance."""
    if not case.get('gold') or not case.get('reference_status'):
        return None, 0
    try:
        ref = float(case['reference_answer'])
    except (TypeError, ValueError):
        return None, 0
    if not math.isfinite(ref):
        return None, 0
    sign = 1 if case['gold']['objective']['sense'] == 'min' else -1
    differences = []
    for ra, rb in zip(a, b):
        if not ra['runs'] or not rb['runs']:
            continue
        va, vb = ra['runs'][0], rb['runs'][0]
        if (va.get('verified_feasible') is True and vb.get('verified_feasible') is True
                and finite(va.get('verified_objective')) and finite(vb.get('verified_objective'))):
            differences.append(sign*(va['verified_objective']-vb['verified_objective'])/max(abs(ref), 1e-9))
    return (sum(differences)/len(differences) if differences else None), len(differences)


def recorded_seconds(records):
    values = [r['runs'][0].get('seconds') for r in records if r['runs']]
    return sum(values) if all(finite(v) and v >= 0 for v in values) else None


def compare(snapshot, *, bootstrap=2000):
    if not 200 <= bootstrap <= 20000:
        raise ValueError('bootstrap must be 200..20000')
    cases, arms = snapshot['cases'], snapshot['settings']['arms']
    group_names = sorted({c['dataset'] for c in cases})
    groups = [(name, [c for c in cases if c['dataset'] == name], 'dataset') for name in group_names]
    groups += [('ALL_gold_verified', [c for c in cases if c.get('gold')], 'pooled_equal_instance'),
               ('ALL_answer_only', [c for c in cases if not c.get('gold')], 'pooled_equal_instance')]
    outputs, tests = [], []
    for name, selected, group_type in groups:
        if not selected:
            continue
        # Do not mix independently verified feasibility with numerical agreement.
        if len({bool(c.get('gold')) for c in selected}) != 1:
            selected_groups = [('verified', [c for c in selected if c.get('gold')]),
                               ('answer_only', [c for c in selected if not c.get('gold')])]
        else:
            selected_groups = [('', selected)]
        for suffix, subset in selected_groups:
            group = name+(':'+suffix if suffix else '')
            metric = 'verified_feasible' if subset[0].get('gold') else 'reference_answer_agreement'
            for label, treatment, control in PAIRS:
                if treatment not in arms or control not in arms:
                    continue
                for stage in ('initial', 'revised'):
                    differences = {metric: [], 'common_feasible_signed_gap': [], 'recorded_search_seconds': [], 'output_tokens': []}
                    paired_ids, common_seeds, a_total, b_total, invalid_a, invalid_b = [], 0, 0., 0., 0, 0
                    possible_seed_pairs = 0
                    for case in subset:
                        a = snapshot['complete_stages'].get((case['id'], treatment, stage))
                        b = snapshot['complete_stages'].get((case['id'], control, stage))
                        if a is None or b is None:
                            continue
                        paired_ids.append(case['id'])
                        va, vb = success(a, metric), success(b, metric)
                        a_total += va; b_total += vb
                        invalid_a += int(all(not r['runs'] for r in a))
                        invalid_b += int(all(not r['runs'] for r in b))
                        differences[metric].append(va-vb)
                        delta, count = common_gap(case, a, b)
                        possible_seed_pairs += min(len(a), len(b))
                        common_seeds += count
                        if delta is not None:
                            differences['common_feasible_signed_gap'].append(delta)
                        ta, tb = recorded_seconds(a), recorded_seconds(b)
                        if ta is not None and tb is not None:
                            differences['recorded_search_seconds'].append(ta-tb)
                        phases = ['model', 'model'] if stage == 'initial' else ['review_'+treatment, 'review_'+control]
                        ga = snapshot['responses'].get((case['id'], phases[0]), {}).get('output_tokens')
                        gb = snapshot['responses'].get((case['id'], phases[1]), {}).get('output_tokens')
                        if finite(ga) and finite(gb):
                            differences['output_tokens'].append(ga-gb)
                    summary = {}
                    for key, values in differences.items():
                        seed = int(hashlib.sha256((group+label+stage+key).encode()).hexdigest()[:8], 16)
                        r = paired_summary(values, higher_better=key == metric, seed=seed, bootstrap=bootstrap)
                        summary[key] = r
                        tests.append(r)
                    n = len(paired_ids)
                    outputs.append({'group': group, 'group_type': group_type, 'comparison': label,
                        'treatment': treatment, 'control': control, 'stage': stage, 'expected_instances': len(subset),
                        'paired_instances': n, 'pending_instances': len(subset)-n, 'paired_case_ids': paired_ids,
                        'coverage': n/len(subset), 'status': 'complete' if n == len(subset) else 'partial',
                        'primary_metric': metric, 'treatment_rate': a_total/n if n else None,
                        'control_rate': b_total/n if n else None,
                        'invalid_or_unsupported_instances': {'treatment': invalid_a, 'control': invalid_b},
                        'common_feasible_seed_pairs': common_seeds, 'possible_seed_pairs': possible_seed_pairs,
                        'metrics': summary})
    holm(tests)
    return {'comparisons': outputs, 'bootstrap_resamples': bootstrap,
            'holm_family': 'All emitted dataset/pooled-group x comparison x stage x metric sign tests in this snapshot.',
            'warnings': ['Partial coverage is potentially selection-biased; do not draw final conclusions from early completions.',
                         'Repeated watching is exploratory, not a sequential hypothesis test; Holm does not adjust repeated looks.',
                         'Sign tests assess paired win/loss directions, not equality of mean effects.',
                         'Gap conditions on same-seed feasibility in both arms; always report feasibility alongside it.',
                         'Initial-stage policy comparisons share the initial model; revised stages measure the whole loop.',
                         'Feasible returned solutions do not prove semantic equivalence of generated models.']}
