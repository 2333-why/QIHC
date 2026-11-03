"""Conservative phase bounds without any cache-speedup assumption."""
import json
import math
from .common import HERE, ROOT, s, configuration


def assess(c, selection):
    n = c['case_count']; a = len(c['arms']); w = c['cpu_workers']; g = c['gpu_replicas']
    quick=c['experiment_id']=='E004-1.6'
    expected=6 if quick else 18
    if n != expected or len(selection['cases']) != n or len({r['dataset'] for r in selection['cases']}) != expected:
        raise ValueError('Freeze the existing 18-stratum development selection')
    if set(c['arms']) != {'S', 'R', 'P', 'X', 'A', 'M'} or a != 6:
        raise ValueError('All six prespecified arms required')
    if c['rounds'] != (1 if quick else 2) or c['seeds'] != [0] or c['cross_seeds'] != [0]:
        raise ValueError('Only the fixed two-round, single-seed design is budgeted')
    if c['gpu_batch_size'] != 1 or c['max_generation_attempts'] != 1 or c['max_gpu_restarts'] != 0:
        raise ValueError('No retries, restarts or batching budgeted')
    if not c['retain_invalid_initial_models'] or c['allow_outcome_adaptive_case_or_arm_drop']:
        raise ValueError('Keep failures and the prespecified cohort')
    count = n*a; patches = count*c['rounds']; cross = n*(a+2)*len(c['cross_backends'])
    gpu = math.ceil(n/g)*(c['contract_timeout_seconds']+c['gpu_request_cleanup_seconds'])
    gpu += math.ceil(count/g)*c['rounds']*(c['patch_timeout_seconds']+c['gpu_request_cleanup_seconds'])
    feedback = math.ceil(count/w)*c['rounds']*(sum(c[k] for k in ('base_seconds','probe_seconds','final_seconds','guard_seconds'))+4*c['cpu_stage_cleanup_seconds'])
    feedback += math.ceil(n/w)*(c['guard_seconds']+c['cpu_stage_cleanup_seconds'])
    # Initial, canonical, every candidate AND every final model (even a retained one).
    semantic = 3*n+patches+count
    offline = math.ceil(cross/w)*(c['cross_solver_seconds']+c['cross_job_overhead_seconds']+c['score_job_seconds'])
    offline += math.ceil(semantic/w)*(c['semantic_check_seconds']+c['semantic_job_overhead_seconds'])
    reserves = c['stage_reservations_seconds']; total = gpu+sum(reserves.values())
    checks = {'feedback_fits': feedback <= reserves['feedback_cpu'],
              'offline_fits': offline <= reserves['offline_evaluation'],
              'work_fits': total-reserves['report_and_pack'] <= c['work_deadline_seconds'],
              'pack_fits': c['work_deadline_seconds']+reserves['report_and_pack'] <= c['wall_limit_seconds'],
              'reserve_fits': c['wall_limit_seconds']-total >= c['minimum_reserve_seconds'],
              'wall_cap': c['wall_limit_seconds'] <= (1800 if quick else 21600)}
    return {'experiment_id': c['experiment_id'], 'status': 'capacity_check_passed' if all(checks.values()) else 'capacity_check_failed',
            'checks': checks, 'counts': {'trajectories': count, 'contract_calls': n, 'patch_calls': patches,
                                      'llm_calls': n+patches, 'cross_solver_jobs_max': cross, 'semantic_jobs_max': semantic},
            'output_token_cap': n*c['contract_output_tokens']+patches*c['patch_output_tokens'],
            'seconds': {'gpu_request_envelope': gpu, 'feedback_cpu_calculated': feedback, 'offline_calculated': offline,
                        'reserved_stages': reserves, 'planned_total': total, 'spare': c['wall_limit_seconds']-total,
                        'wall_limit': c['wall_limit_seconds']},
            'assumptions': ['4 dedicated single-GPU replicas; 24 allocated single-thread CPU workers',
                            'bounded I/O and cleanup; no retries; cache speedups not assumed',
                            'deadline enforces termination, not successful results under hardware/model failures']}


if __name__ == '__main__':
    c = configuration(); result = assess(c, s.load(ROOT/c['selection_file']))
    print(json.dumps(result, indent=2))
    if result['status'] != 'capacity_check_passed':
        raise SystemExit(2)
