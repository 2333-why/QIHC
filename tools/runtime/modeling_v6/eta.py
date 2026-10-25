"""Conditional ETA from actual checkpoint durations and de-duplicated batches."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from .statistics import finite
from .experiment import initial_phases, ARMS


def request_kind(phase):
    if phase.startswith('model_'):return phase
    arm=phase.removeprefix('review_')
    return 'review_'+ARMS[arm].get('review_mode',ARMS[arm]['frontend'])


def request_key(case_id, phase):
    return hashlib.sha256((case_id+'\0'+phase).encode()).hexdigest()


def estimate(snapshot, *, now, cpu_workers=8, replicas=2, window_seconds=3600):
    if not 1 <= cpu_workers <= 64 or not 1 <= replicas <= 8 or window_seconds < 60:
        raise ValueError('Invalid ETA worker/replica/window settings')
    cutoff = now-window_seconds
    cfg = snapshot['settings']
    warnings = ['Conditional estimates exclude scheduler queueing, future interruptions, model loading and unobserved failures.',
                'V6 logical-request times sum allocated substep batch durations; they are not measured end-to-end latency. Flat/staged and review types calibrated separately.',
                'CPU worker and GPU replica counts are supplied assumptions, not inferred from utilization.',
                'Future, not-yet-enqueued review and scoring tasks ARE included.',
                'No credit is taken for unfinished in-flight work; estimates may drop after checkpoints arrive.',
                'Early completed tasks may be easier than future tasks; pooled timing fallback is an approximation, not a deadline.']
    pools, recent_pools = defaultdict(list), defaultdict(list)
    invalid_stages = set()
    completed_cpu = 0
    for task in snapshot['tasks']:
        record = task['record']
        if record is None:
            continue
        completed_cpu += 1
        if not record['runs']:
            invalid_stages.add((task['case_id'], task['arm'], task['stage']))
            continue
        value = record['runs'][0].get('seconds')
        if not finite(value) or value <= 0:
            continue
        groups = [('exact', task['dataset'], task['arm'], task['stage']),
                  ('dataset_backend', task['dataset'], task['backend']), ('backend', task['backend'])]
        if task['backend'] != 'cp_sat':
            groups.append(('all_sampling',))
        for group in groups:
            pools[group].append(value)
            if cutoff <= record['_mtime'] <= now+5:
                recent_pools[group].append(value)
    cpu_work, uncalibrated_cpu, fallbacks, recent_used = 0., 0, Counter(), 0
    cpu_case_work = defaultdict(float)
    for task in snapshot['tasks']:
        if task['record'] is not None:
            continue
        if (task['case_id'], task['arm'], task['stage']) in invalid_stages:
            fallbacks['known_invalid_model_no_solver_work'] += 1
            continue
        groups = [('exact', task['dataset'], task['arm'], task['stage']),
                  ('dataset_backend', task['dataset'], task['backend']), ('backend', task['backend'])]
        if task['backend'] != 'cp_sat':
            groups.append(('all_sampling',))
        chosen, source = None, None
        for pool, age in ((recent_pools, 'recent'), (pools, 'historical')):
            for group in groups:
                if len(pool[group]) >= 3:
                    chosen, source = pool[group], age+':'+group[0]
                    break
            if chosen is not None:
                break
        if chosen is not None:
            duration = sum(chosen)/len(chosen)
            cpu_work += duration
            cpu_case_work[task['case_id']] += duration
            fallbacks[source] += 1
            recent_used += int(source.startswith('recent:'))
        elif task['backend'] == 'cp_sat':
            cpu_work += cfg['solver_seconds']
            cpu_case_work[task['case_id']] += cfg['solver_seconds']
            fallbacks['configured_cp_sat_limit_not_measurement'] += 1
        else:
            uncalibrated_cpu += 1
    pending_cpu = len(snapshot['tasks'])-completed_cpu
    # The producer never dispatches concurrent seeds of the SAME case. Account
    # for the sequential tail instead of dividing one remaining case by 8 cores.
    cpu_tail = max(cpu_case_work.values(), default=0.)
    cpu_eta = max(cpu_work/cpu_workers, cpu_tail) if not uncalibrated_cpu else None

    expected = [(case['id'], phase) for case in snapshot['cases']
                for phase in initial_phases(cfg['arms'])+['review_'+arm for arm in cfg['arms']]]
    remaining = [key for key in expected if key not in snapshot['responses']]
    pending_by_phase = Counter(key[1] for key in remaining)
    batches = defaultdict(list)
    reused = 0
    for (case_id, phase), row in snapshot['responses'].items():
        if row.get('reused_from'):
            reused += 1
            continue
        if not row.get('batch_id') or type(row.get('replica')) is not int:
            continue
        batches[(row['replica'], row['batch_id'])].append((case_id, phase, row))
    valid_batches = []
    for (rank, batch_id), rows in batches.items():
        sample = rows[0][2]
        size, seconds = sample.get('batch_size'), sample.get('batch_seconds')
        if type(size) is not int or size < 1 or len(rows) != size or not finite(seconds) or seconds <= 0:
            continue  # Partial batch writes / no timing: do not overestimate speed.
        if any(r.get('batch_size') != size or r.get('batch_seconds') != seconds or r.get('status') != 'generated'
               for _, _, r in rows):
            continue
        valid_batches.append({'rank': rank, 'id': batch_id, 'rows': rows, 'size': size, 'seconds': seconds,
                              'end': max(r['_mtime'] for _, _, r in rows)})
    valid_batches.sort(key=lambda b: b['end'])
    recent_batches = [b for b in valid_batches if cutoff <= b['end'] <= now+5]
    calibration = recent_batches if len(recent_batches) >= 3 else valid_batches
    gpu_age = 'recent' if len(recent_batches) >= 3 else 'historical_or_insufficient'
    duration_pools = defaultdict(list)
    for batch in calibration:
        for _, phase, _ in batch['rows']:
            kind = request_kind(phase)
            duration_pools[(batch['rank'], kind)].append(batch['seconds']/batch['size'])
            duration_pools[('pooled', kind)].append(batch['seconds']/batch['size'])
    rank_work, uncalibrated_gpu = [0.]*replicas, 0
    for case_id, phase in remaining:
        rank = int(request_key(case_id, phase), 16) % replicas
        kind = request_kind(phase)
        values = duration_pools[(rank, kind)]
        if len(values) < 3:
            values = duration_pools[('pooled', kind)]
        if len(calibration) < 3 or len(values) < 3:
            uncalibrated_gpu += 1
        else:
            rank_work[rank] += sum(values)/len(values)
    gpu_eta = max(rank_work) if not uncalibrated_gpu else None
    throughput, observed_eta, observed_span = None, None, None
    if len(recent_batches) >= 3:
        first, last = recent_batches[0]['end'], recent_batches[-1]['end']
        observed_span = last-first
        if observed_span >= 60:
            # Exclude every batch at the left boundary: its work predates the window.
            observed_count = sum(b['size'] for b in recent_batches if b['end'] > first)
            throughput = observed_count/observed_span
            observed_eta = len(remaining)/throughput if throughput > 0 else None
    if not remaining:
        gpu_eta, observed_eta = 0., 0.
    if not pending_cpu:
        cpu_eta = 0.
    current_cpu = snapshot.get('cpu_state') or {}
    gpu_states = snapshot.get('gpu_states', [])
    suspended = []
    if pending_cpu and current_cpu.get('status', '').startswith('stopped'):
        suspended.append('CPU reports stopped; resume CPU before interpreting a finish time.')
    if remaining and gpu_states and all(row.get('status', '').startswith('stopped') for row in gpu_states):
        suspended.append('All reported GPU replicas stopped; resume GPU allocation.')
    cpu_age = max(0., now-current_cpu['time']) if finite(current_cpu.get('time')) else None
    if cpu_age is None or cpu_age > 120:
        warnings.append('CPU status is absent/stale; process liveness is not established.')
    if uncalibrated_cpu or uncalibrated_gpu:
        warnings.append('Insufficient matching timing samples: unavailable ETA components remain null, not zero.')
    if any(key.startswith('historical') for key in fallbacks) or gpu_age != 'recent':
        warnings.append('Some estimates use historical durations, which may not reflect current hardware or contention.')
    scenarios = None
    if cpu_eta is not None and gpu_eta is not None:
        scenarios = {'full_overlap_seconds': max(cpu_eta, gpu_eta),
                     'no_overlap_seconds': cpu_eta+gpu_eta,
                     'meaning': 'Pipeline overlap scenarios under calibrated speeds, NOT confidence bounds or guaranteed deadlines.'}
    requests_ready = sum(key in snapshot['requests'] for key in remaining)
    all_done = pending_cpu == 0 and not remaining and snapshot['final_cases'] == len(snapshot['cases'])
    if all_done:
        state = 'complete'
    elif pending_cpu == 0 and not remaining:
        state = 'awaiting_finalization'
    elif suspended:
        state = 'resume_required'
    elif uncalibrated_cpu or uncalibrated_gpu:
        state = 'warming_up'
    else:
        state = 'conditional_estimate'
    return {'state': state, 'assumed_cpu_workers': cpu_workers, 'assumed_gpu_replicas': replicas,
            'window_seconds': window_seconds, 'cpu': {
                'tasks_expected': len(snapshot['tasks']), 'tasks_completed': completed_cpu, 'tasks_remaining': pending_cpu,
                'uncalibrated_tasks': uncalibrated_cpu, 'estimated_remaining_process_seconds_calibrated_only': cpu_work,
                'longest_remaining_case_seconds_calibrated_only': cpu_tail,
                'remaining_wall_seconds_conditional': cpu_eta, 'calibration_task_counts': dict(fallbacks),
                'recently_calibrated_tasks': recent_used, 'status_age_seconds': cpu_age},
            'gpu': {'requests_expected': len(expected), 'responses_completed': len(snapshot['responses']),
                'reused_responses_excluded_from_speed': reused, 'requests_remaining': len(remaining),
                'queued_pending': requests_ready, 'future_not_yet_enqueued': len(remaining)-requests_ready,
                'pending_by_phase': dict(pending_by_phase), 'uncalibrated_requests': uncalibrated_gpu,
                'complete_calibration_batches': len(calibration), 'calibration_age': gpu_age,
                'measured_batch_seconds_sum_once': sum(b['seconds'] for b in valid_batches),
                'estimated_remaining_replica_seconds': rank_work, 'remaining_busy_wall_seconds_conditional': gpu_eta,
                'observed_window_seconds': observed_span, 'observed_responses_per_second': throughput,
                'remaining_seconds_at_observed_queue_throughput': observed_eta,
                'last_batch_age_seconds': max(0., now-valid_batches[-1]['end']) if valid_batches else None},
            'overlap_scenarios': scenarios, 'resume_required': suspended,
            'retry_overhead_seconds_observed': sum(row.get('seconds', 0) for row in snapshot.get('events', [])
                                                  if finite(row.get('seconds')) and row['seconds'] >= 0),
            'warnings': warnings}
