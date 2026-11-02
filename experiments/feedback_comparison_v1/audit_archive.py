"""Read-only E004 archive audit; never imports or executes archived code."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import tarfile


def audit(archive):
    states, scores, requests, responses = {}, {}, {}, {}
    roots, errors, cpu, initial_errors, rounds, response_status = (Counter() for _ in range(6))
    examples, metadata, gpu_logs, failures = {}, {}, {}, []
    round_rows = []
    members = 0
    unpacked_bytes = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            root = name.split('/')[0]
            members += 1
            unpacked_bytes += member.size
            roots[root] += 1
            if root == 'logs':
                if Path(name).name.startswith('gpu-'):
                    value = tar.extractfile(member).read().decode('utf-8', errors='replace')
                    gpu_logs[name] = value[-10000:]
                continue
            if not name.endswith('.json'):
                continue
            if root not in ('states', 'evaluation', 'requests', 'responses', 'initial',
                            'rounds', 'cpu_execution_events', 'gpu_failures') and '/' in name:
                continue
            if root == 'evaluation' and not name.startswith('evaluation/scores/'):
                continue
            value = json.load(tar.extractfile(member))
            key = Path(name).stem
            if '/' not in name:
                metadata[name] = value
            elif root == 'states':
                keep = ('id', 'case_id', 'arm', 'seed', 'split', 'dataset', 'size', 'stage', 'round',
                        'initial_valid', 'errors', 'valid_patch_outputs', 'valid_review_outputs',
                        'generation_attempts', 'input_tokens', 'output_tokens', 'attributed_seconds',
                        'searched_windows', 'request_id', 'model_sha256', 'cost_incomplete')
                states[key] = {k: value[k] for k in keep if k in value}
                for error in value.get('errors', []):
                    text = str(error.get('error'))
                    errors[text] += 1
                    examples.setdefault(text, {'path': name, 'case_id': value['case_id'], 'event': error})
            elif root == 'evaluation':
                scores[key] = value
            elif root == 'requests':
                requests[key] = {k: value.get(k) for k in ('stage', 'case_id')}
            elif root == 'responses':
                responses[key] = {k: value.get(k) for k in ('status', 'error', 'input_tokens', 'output_tokens', 'generation_attempts', 'attributed_seconds', 'time')}
                response_status[value.get('status')] += 1
                if value.get('status') != 'generated':
                    examples.setdefault('response: ' + str(value.get('error')), {'path': name, 'record': value})
            elif root == 'initial':
                if value.get('error'):
                    initial_errors[str(value['error'])] += 1
            elif root == 'cpu_execution_events':
                cpu[str((value.get('status'), value.get('exit_code'), value.get('timedout')))] += 1
            elif root == 'gpu_failures':
                failures.append(value)
            elif root == 'rounds':
                split, arm = value['split'], value['arm']
                rounds[f'{split}/{arm}'] += 1
                round_rows.append({'path': name, 'id': value['id'], 'split': split, 'arm': arm,
                                   'round': value['round'], 'patch': value.get('patch'),
                                   'base_model_sha256': value.get('base_model_sha256'),
                                   'model_sha256': value.get('model_sha256'),
                                   'final_status': value.get('final', {}).get('status')})
    grouped = defaultdict(list)
    for key, score in scores.items():
        grouped[(score['split'], score['arm'])].append((key, score, states[key]))
    measured = []
    mismatch = []
    for (split, arm), rows in sorted(grouped.items()):
        err = Counter(str(e.get('error')) for _, _, st in rows for e in st.get('errors', []))
        rec = {'split': split, 'arm': arm, 'completed': len(rows),
               'instances': len({s['case_id'] for _, s, _ in rows}),
               'feasible': sum(bool(s.get('original_feasible')) for _, s, _ in rows),
               'optimal': sum(bool(s.get('optimal_hit')) for _, s, _ in rows),
               'quality5': sum(bool(s.get('original_feasible') and s.get('reference_proven') and s.get('gap_percent') is not None and s['gap_percent'] <= 5 + 1e-8) for _, s, _ in rows),
               'compiled': sum(bool(s.get('compiled')) for _, s, _ in rows),
               'trajectories_with_errors': sum(bool(st.get('errors')) for _, _, st in rows),
               'valid_patch_outputs': sum(st.get('valid_patch_outputs', 0) for _, _, st in rows),
               'trajectories_with_valid_patch': sum(st.get('valid_patch_outputs', 0) > 0 for _, _, st in rows),
               'errors': dict(err), 'execution_errors': sum(err.values()),
               'datasets': dict(Counter(s['dataset'] for _, s, _ in rows))}
        measured.append(rec)
        expected = next(r for r in metadata['summary.json']['rows'] if r['split'] == split and r['arm'] == arm)
        for field in ('completed', 'execution_errors'):
            if rec[field] != expected[field]:
                mismatch.append([split, arm, field, rec[field], expected[field]])
        for count, rate in [('feasible', 'feasible_all_assigned'), ('optimal', 'optimal_all_assigned'), ('quality5', 'quality5_all_assigned')]:
            if abs(rec[count] / expected['expected'] - expected[rate]) > 1e-10:
                mismatch.append([split, arm, rate, rec[count], expected[rate]])
    phases = Counter(f"{s['split']}/{s['round']}/{s['stage']}" for s in states.values())
    response_groups = Counter(f"{requests.get(k, {}).get('stage')}/{r['status']}/{r.get('error')}" for k, r in responses.items())
    tz = timezone(timedelta(hours=8))
    report = {'archive': str(archive.resolve()), 'sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
              'archive_bytes': archive.stat().st_size, 'members': members, 'unpacked_bytes': unpacked_bytes,
              'member_roots': dict(roots),
              'summary_time_shanghai': datetime.fromtimestamp(metadata['summary.json']['time'], tz).isoformat(),
              'status': metadata['status.json'], 'manifest': metadata['manifest.json'],
              'acceptance': metadata['acceptance.json'], 'measured': measured, 'summary_mismatches': mismatch,
              'state_phases': dict(phases), 'all_state_error_events': dict(errors), 'error_examples': examples,
              'cpu_events': dict(cpu), 'initial_errors': dict(initial_errors),
              'response_status': dict(response_status), 'response_groups': dict(response_groups),
              'gpu_failures': failures, 'gpu_log_tails': gpu_logs,
              'round_counts': dict(rounds), 'rounds_with_patch': sum(r['patch'] is not None for r in round_rows),
              'rounds_model_changed': sum(r['base_model_sha256'] != r['model_sha256'] for r in round_rows),
              'public_instances_started': len({s['case_id'] for s in states.values() if s['split'] == 'public'})}
    return report, {'states': states, 'scores': scores, 'requests': requests, 'responses': responses, 'rounds': round_rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    result, cache = audit(args.archive)
    for path, obj in ((args.output, result), (args.cache, cache)):
        with path.open('x', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write('\n')
    print(json.dumps({k: result[k] for k in ('summary_time_shanghai', 'sha256', 'summary_mismatches', 'state_phases', 'all_state_error_events', 'cpu_events', 'response_status', 'rounds_with_patch', 'rounds_model_changed')}, ensure_ascii=False, indent=2))
