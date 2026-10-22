import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.benchmarks import comparison as c
from tools.benchmarks.distribution import experiment as e
from tools.analysis.distribution_feedback import eta, report, statistics as s


def make_run(root, arms=('pbit_intervention', 'pbit_probe_blind', 'pbit_selected', 'pbit_random_policy'), count=3, seeds=2):
    cases = [{'id': str(i), 'dataset': 'fixture', 'gold': {'objective': {'sense': 'min'}},
              'reference_answer': 100, 'reference_status': 'published_reference'} for i in range(count)]
    c.save(root/'cases.CPU_ONLY.json', cases)
    cfg = {'arms': list(arms), 'seeds': seeds, 'solver_seconds': 5, 'source': e.source_hash()}
    c.save(root/'manifest.json', {'experiment': 'distribution_feedback_v1', 'settings': cfg, 'cases': count,
           'cases_sha256': hashlib.sha256((root/'cases.CPU_ONLY.json').read_bytes()).hexdigest()})
    for case in cases:
        for phase in ['model']+['review_'+arm for arm in arms]:
            c.enqueue(root, case, phase, 'Question only')
            request = root/'requests'/(c.key(case['id'], phase)+'.json')
            c.save(root/'responses'/request.name, {'response': 'fixture', 'status': 'generated', 'output_tokens': 20,
                    'request_sha256': hashlib.sha256(request.read_bytes()).hexdigest()})
    return cases, cfg


def checkpoint(root, cid, arm, stage, seed, *, feasible=True, objective=110, seconds=3, status='searched'):
    row = {'arm': arm, 'seed': seed, 'status': status,
           'input_response_sha256': hashlib.sha256(b'fixture').hexdigest(),
           'runs': [{'verified_feasible': feasible, 'verified_objective': objective, 'seconds': seconds}]
                   if status == 'searched' else []}
    c.save(root/'progress'/c.key(cid, 'case')/f'{arm}.{stage}.{seed}.json', row)


def fill(root, cases, cfg, *, stage='revised'):
    for case in cases:
        for arm in cfg['arms']:
            for seed in range(cfg['seeds']):
                checkpoint(root, case['id'], arm, stage, seed, objective=110 if arm in ('pbit_intervention', 'pbit_selected') else 120)


def comparison(stats, name, stage='revised', group='fixture'):
    return next(r for r in stats['comparisons'] if (r['comparison'], r['stage'], r['group']) == (name, stage, group))


def test_key_pairs_common_seed_gaps_and_coverage(tmp_path):
    cases, cfg = make_run(tmp_path)
    fill(tmp_path, cases, cfg)
    stats = s.compare(report.load_snapshot(tmp_path), bootstrap=200)
    for name in ('intervention_information', 'selection_vs_random'):
        row = comparison(stats, name)
        assert row['paired_instances'] == row['expected_instances'] == 3
        assert row['common_feasible_seed_pairs'] == 6
        assert row['metrics']['verified_feasible']['mean_delta_treatment_minus_control'] == 0
        assert row['metrics']['common_feasible_signed_gap']['mean_delta_treatment_minus_control'] == pytest.approx(-.1)
        assert row['metrics']['common_feasible_signed_gap']['wins'] == 3
        assert comparison(stats, name, 'initial')['paired_instances'] == 0


def test_partial_seeds_not_pseudoreplicated_and_failures_kept(tmp_path):
    cases, cfg = make_run(tmp_path, arms=('pbit_intervention', 'pbit_probe_blind'), count=3, seeds=3)
    for arm in cfg['arms']:
        for seed in range(3):
            checkpoint(tmp_path, '0', arm, 'revised', seed,
                       status='invalid_model' if arm == 'pbit_probe_blind' else 'searched')
        for seed in range(2):
            checkpoint(tmp_path, '1', arm, 'revised', seed)
    row = comparison(s.compare(report.load_snapshot(tmp_path), bootstrap=200), 'intervention_information')
    assert row['paired_instances'] == 1 and row['pending_instances'] == 2
    assert row['metrics']['verified_feasible']['instances'] == 1  # Not 3 seeds.
    assert row['metrics']['verified_feasible']['mean_delta_treatment_minus_control'] == 1
    assert row['metrics']['verified_feasible']['bootstrap_95pct'] is None
    assert row['invalid_or_unsupported_instances']['control'] == 1


def test_gap_only_uses_same_seed_feasible_on_both_sides(tmp_path):
    cases, cfg = make_run(tmp_path, arms=('pbit_intervention', 'pbit_probe_blind'), count=1)
    for seed in range(2):
        checkpoint(tmp_path, '0', 'pbit_intervention', 'revised', seed, feasible=seed == 0)
        checkpoint(tmp_path, '0', 'pbit_probe_blind', 'revised', seed, feasible=seed == 1)
    row = comparison(s.compare(report.load_snapshot(tmp_path), bootstrap=200), 'intervention_information')
    assert row['treatment_rate'] == row['control_rate'] == .5
    assert row['common_feasible_seed_pairs'] == 0
    assert row['metrics']['common_feasible_signed_gap']['instances'] == 0


def test_max_gap_sign_and_missing_reference():
    case = {'gold': {'objective': {'sense': 'max'}}, 'reference_answer': 100, 'reference_status': 'feasible_reference'}
    a = [{'runs': [{'verified_feasible': True, 'verified_objective': 95}]}]
    b = [{'runs': [{'verified_feasible': True, 'verified_objective': 90}]}]
    assert s.common_gap(case, a, b) == (-.05, 1)
    case.pop('reference_status')
    assert s.common_gap(case, a, b) == (None, 0)


def test_bootstrap_reproducible_sign_test_and_holm():
    a = s.paired_summary([1, 0, 1, -1], higher_better=True, seed=4, bootstrap=200)
    assert a == s.paired_summary([1, 0, 1, -1], higher_better=True, seed=4, bootstrap=200)
    assert (a['wins'], a['ties'], a['losses']) == (2, 1, 1)
    assert s.sign_pvalue(3, 0) == pytest.approx(.25)
    assert s.sign_pvalue(0, 0) == 1
    rows = [{'sign_test_p_two_sided': p} for p in (.01, .04, .03, None)]
    s.holm(rows)
    assert [r['sign_test_p_holm'] for r in rows] == pytest.approx([.03, .06, .06, None], nan_ok=True)


def speed_snapshot():
    now = 10000.
    arms = ['pbit_best', 'pbit_selected']
    cases = [{'id': str(i), 'dataset': 'D'} for i in range(20)]
    tasks = []
    for case in cases:
        for arm in arms:
            for stage in ('initial', 'revised'):
                tasks.append({'case_id': case['id'], 'dataset': 'D', 'arm': arm, 'stage': stage,
                              'backend': 'pbit', 'record': None})
    for task in tasks[:10]:
        task['record'] = {'runs': [{'seconds': 8.}], '_mtime': now-60}
    responses = {(case['id'], 'model'): {'reused_from': 'legacy', '_mtime': now} for case in cases}
    for i in range(12):
        batch = i//4
        responses[(str(i), 'review_pbit_best')] = {
            'status': 'generated', 'batch_id': str(batch), 'replica': batch % 2,
            'batch_size': 4, 'batch_seconds': 40., '_mtime': now-180+batch*60}
    requests = dict(responses)
    for i in range(3):
        requests[(str(i), 'review_pbit_selected')] = {}
    return {'cases': cases, 'settings': {'arms': arms, 'seeds': 1, 'solver_seconds': 5}, 'tasks': tasks,
            'responses': responses, 'requests': requests, 'final_cases': 0,
            'cpu_state': {'status': 'running', 'time': now},
            'gpu_states': [{'status': 'generating', 'time': now}]*2, 'events': [{'seconds': 7}]}, now


def test_eta_counts_future_work_deduplicates_batches_and_excludes_reuse():
    snapshot, now = speed_snapshot()
    result = eta.estimate(snapshot, now=now)
    assert result['state'] == 'conditional_estimate'
    assert result['cpu']['tasks_remaining'] == 70
    assert result['cpu']['remaining_wall_seconds_conditional'] == 70
    gpu = result['gpu']
    assert gpu['requests_expected'] == 60 and gpu['requests_remaining'] == 28
    assert gpu['future_not_yet_enqueued'] == 25 and gpu['queued_pending'] == 3
    assert gpu['reused_responses_excluded_from_speed'] == 20
    assert gpu['measured_batch_seconds_sum_once'] == 120  # NOT 4*120.
    assert gpu['observed_responses_per_second'] == pytest.approx(8/120)
    assert gpu['remaining_seconds_at_observed_queue_throughput'] == 420
    assert sum(gpu['estimated_remaining_replica_seconds']) == 280
    assert result['retry_overhead_seconds_observed'] == 7


def test_eta_warmup_and_unobserved_review_phase_are_unknown_not_zero():
    snapshot, now = speed_snapshot()
    snapshot['responses'] = {key: row for key, row in snapshot['responses'].items() if key[1] == 'model'}
    result = eta.estimate(snapshot, now=now)
    assert result['state'] == 'warming_up'
    assert result['gpu']['remaining_busy_wall_seconds_conditional'] is None
    assert result['gpu']['uncalibrated_requests'] == 40
    assert result['overlap_scenarios'] is None


def test_eta_stopped_and_stale_are_not_promised_finish_times():
    snapshot, now = speed_snapshot()
    snapshot['cpu_state'] = {'status': 'stopped_checkpointed', 'time': now-1000}
    snapshot['gpu_states'] = [{'status': 'stopped_checkpointed'}]*2
    result = eta.estimate(snapshot, now=now)
    assert result['state'] == 'resume_required'
    assert len(result['resume_required']) == 2
    assert any('stale' in w for w in result['warnings'])


def test_eta_partial_batch_is_not_used_as_full_batch_speed():
    snapshot, now = speed_snapshot()
    del snapshot['responses'][('0', 'review_pbit_best')]
    result = eta.estimate(snapshot, now=now)
    assert result['gpu']['complete_calibration_batches'] == 2
    assert result['gpu']['remaining_busy_wall_seconds_conditional'] is None


def test_eta_completion_vs_finalization():
    snapshot, now = speed_snapshot()
    for task in snapshot['tasks']:
        task['record'] = {'runs': [], '_mtime': now}
    for case in snapshot['cases']:
        for phase in ['model']+['review_'+a for a in snapshot['settings']['arms']]:
            snapshot['responses'].setdefault((case['id'], phase), {})
    result = eta.estimate(snapshot, now=now)
    assert result['state'] == 'awaiting_finalization'
    assert result['cpu']['remaining_wall_seconds_conditional'] == result['gpu']['remaining_busy_wall_seconds_conditional'] == 0
    snapshot['final_cases'] = 20
    assert eta.estimate(snapshot, now=now)['state'] == 'complete'


def test_unknown_cpu_timing_not_hidden_by_fast_gpu():
    snapshot, now = speed_snapshot()
    for task in snapshot['tasks']:
        task['record'] = None
    result = eta.estimate(snapshot, now=now)
    assert result['cpu']['uncalibrated_tasks'] == 80
    assert result['cpu']['remaining_wall_seconds_conditional'] is None


def file_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_report_generation_leaves_live_run_byte_identical_and_hashes_unchanged(tmp_path):
    run, output = tmp_path/'live', tmp_path/'reports'
    cases, cfg = make_run(run)
    fill(run, cases, cfg)
    before = file_bytes(run)
    report.generate(run, output, bootstrap=200)
    assert file_bytes(run) == before
    assert {p.name for p in output.iterdir()} == {'paired_statistics.json', 'eta.json', 'snapshot.json', 'report.md'}
    text = (output/'report.md').read_text(encoding='utf-8')
    assert 'pbit_intervention − pbit_probe_blind' in text and 'pbit_selected − pbit_random_policy' in text
    assert c.code_hash() == '2559c9609a22b94258d18e296c37cc6873fb0a385b52b8a75db453f507bdf787'
    assert e.source_hash() == '8f8040684e337b2794590463a5e8d8c711990874a1644c81cdf44e6e31d87c1c'


def test_refuse_output_inside_run_and_invalid_checksums(tmp_path):
    run = tmp_path/'live'
    make_run(run)
    before = file_bytes(run)
    with pytest.raises(ValueError, match='OUTSIDE'):
        report.generate(run, run/'reports')
    assert file_bytes(run) == before
    path = next((run/'responses').glob('*.json'))
    row = json.loads(path.read_text()); row['request_sha256'] = 'tampered'
    c.save(path, row)
    with pytest.raises(ValueError, match='checksum'):
        report.load_snapshot(run)


def test_changed_model_answer_detected_even_when_request_still_matches(tmp_path):
    cases, cfg = make_run(tmp_path)
    fill(tmp_path, cases, cfg)
    path = tmp_path/'responses'/(c.key('0', 'review_pbit_intervention')+'.json')
    row = json.loads(path.read_text()); row['response'] = 'changed'
    c.save(path, row)
    with pytest.raises(ValueError, match='Checkpoint input'):
        report.load_snapshot(tmp_path)


def test_cached_reader_observes_new_atomic_checkpoints(tmp_path):
    cases, cfg = make_run(tmp_path)
    reader = report.Reader()
    before = report.load_snapshot(tmp_path, reader)
    assert not before['complete_stages']
    fill(tmp_path, cases, cfg)
    after = report.load_snapshot(tmp_path, reader)
    assert len(after['complete_stages']) == len(cases)*len(cfg['arms'])


def test_mixed_gold_and_answer_only_never_share_one_primary_metric():
    snapshot = {'cases': [{'id': 'a', 'dataset': 'mixed', 'gold': {'objective': {'sense': 'min'}}},
                          {'id': 'b', 'dataset': 'mixed', 'gold': None}],
                'settings': {'arms': ['pbit_best', 'pbit_distribution']}, 'complete_stages': {}, 'responses': {}}
    output = s.compare(snapshot, bootstrap=200)
    assert {r['group'] for r in output['comparisons']} == {'mixed:verified', 'mixed:answer_only', 'ALL_gold_verified', 'ALL_answer_only'}


@pytest.mark.parametrize('workers,replicas,window', [(0, 2, 3600), (8, 0, 3600), (8, 2, 1)])
def test_eta_validation(workers, replicas, window):
    snapshot, now = speed_snapshot()
    with pytest.raises(ValueError):
        eta.estimate(snapshot, now=now, cpu_workers=workers, replicas=replicas, window_seconds=window)


def test_real_pipeline_records_and_cli_report(tmp_path):
    run, output = tmp_path/'run', tmp_path/'reports'
    arms = ('pbit_probe_blind', 'pbit_intervention', 'pbit_selected', 'pbit_random_policy')
    cases, cfg = make_run(run, arms=arms, count=1, seeds=1)
    from qihc.generic.bounded import VERSION
    model = {'schema': VERSION, 'variables': [{'name': 'x0', 'lower': 0, 'upper': 1}],
             'objective': {'sense': 'max', 'linear': [1], 'quadratic': []}, 'constraints': [], 'logic': []}
    case = {'id': '0', 'dataset': 'real_cpu_fixture', 'question': 'Maximize binary x0.', 'gold': model,
            'reference_answer': 1, 'reference_status': 'exhaustive_fixture'}
    text = json.dumps({'model': model, 'grounding': [{'target': target, 'quote': case['question'],
                       'explanation': 'Fixture only.'} for target in ('objective', 'variable:x0')]})
    cfg.update(budget=256, chains=2)
    c.save(run/'cases.CPU_ONLY.json', [case])
    m = c.load(run/'manifest.json'); m['settings'] = cfg
    m['cases_sha256'] = hashlib.sha256((run/'cases.CPU_ONLY.json').read_bytes()).hexdigest()
    c.save(run/'manifest.json', m)
    # Fresh protocol requests use the actual experiment's prompt builder.
    for phase in ['model']+['review_'+a for a in arms]:
        name = c.key('0', phase)+'.json'
        if phase == 'model':
            c.save(run/'requests'/name, {'id': c.key('0', phase), 'case_id': '0', 'phase': phase,
                                        'prompt': c.modeling_prompt(case)})
        else:
            (run/'requests'/name).unlink()
            (run/'responses'/name).unlink()
    for _ in range(30):
        for path in (run/'requests').glob('*.json'):
            c.save(run/'responses'/path.name, {'response': text, 'status': 'generated', 'output_tokens': 10,
                   'request_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        if e.process_case(run, case, cfg) == 'complete':
            break
    before = file_bytes(run)
    result = subprocess.run([sys.executable, str(Path(report.__file__)), '--run', str(run),
                             '--output', str(output), '--bootstrap', '200'], capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    assert json.loads((output/'eta.json').read_text())['state'] == 'complete'
    assert file_bytes(run) == before


def test_cpu_cp_sat_fallback_explicit_and_invalid_stage_needs_no_solver_work():
    snapshot, now = speed_snapshot()
    for task in snapshot['tasks']:
        task['record'] = None
        task['backend'] = 'cp_sat'
    result = eta.estimate(snapshot, now=now)
    assert result['cpu']['remaining_wall_seconds_conditional'] == 80*5/8
    assert result['cpu']['calibration_task_counts']['configured_cp_sat_limit_not_measurement'] == 80
    snapshot['tasks'][0]['record'] = {'runs': [], '_mtime': now}
    duplicate = dict(snapshot['tasks'][0], record=None)
    snapshot['tasks'].append(duplicate)
    result = eta.estimate(snapshot, now=now)
    assert result['cpu']['calibration_task_counts']['known_invalid_model_no_solver_work'] == 1


def test_failed_generation_batches_do_not_make_eta_optimistic():
    snapshot, now = speed_snapshot()
    snapshot['responses'][('0', 'review_pbit_best')]['status'] = 'generation_error'
    result = eta.estimate(snapshot, now=now)
    assert result['gpu']['complete_calibration_batches'] == 2
    assert result['gpu']['remaining_busy_wall_seconds_conditional'] is None


def test_cpu_sequential_last_case_not_divided_by_eight_workers():
    snapshot, now = speed_snapshot()
    for task in snapshot['tasks']:
        if task['case_id'] != '19':
            task['record'] = {'runs': [{'seconds': 8.}], '_mtime': now-60}
    result = eta.estimate(snapshot, now=now)
    assert result['cpu']['tasks_remaining'] == 4
    assert result['cpu']['estimated_remaining_process_seconds_calibrated_only'] == 32
    assert result['cpu']['remaining_wall_seconds_conditional'] == 32  # Not 4s.
