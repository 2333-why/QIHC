import json
from experiments.boundary_feedback_v1.diagnose import diagnose


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_shared_failures_and_calls_are_not_counted_per_arm(tmp_path):
    counts = {'candidate_parse_error': 1, 'judge_parse_error': 1}
    put(tmp_path, 'pilot/summary.json', dict(complete=True, expected=144,
        stages={'development': {'arms': {a: {'actions': counts} for a in ('S', 'P', 'Q')}}}))
    put(tmp_path, 'pilot/shared/17/candidate.json', dict(action='candidate_parse_error',
        error='invalid JSON', text='', usage=[dict(finish_reason='length', text='{"model":',
        usage={'completion_tokens': 4096})]))
    for arm in ('S', 'P', 'Q'):
        put(tmp_path, f'pilot/models/17/{arm}/predictions/candidate.json',
            dict(action='candidate_parse_error', usage=[]))
        put(tmp_path, f'pilot/models/17/{arm}/predictions/judge.json',
            dict(action='judge_parse_error', usage=[dict(finish_reason='stop', text='{}')],
                 evidence=[dict(phase='discover', assignments=[[0]], pairs=[],
                                verdict=dict(status='judge_parse_error', error='judge envelope', text='{}'))]))
    report = diagnose(tmp_path)
    assert report['unique_saved_calls'] == {'total': 4, 'length': 1}
    assert len(report['failures']) == 4  # one shared candidate, three actual judge calls
    assert report['failures'][0]['calls'][0]['text'] == '{"model":'
    assert report['summary_mismatches'] == []
    assert report['read_errors'] == []
    assert report['reported_expected'] == 144 and report['observed_predictions'] == 6


def test_missing_or_corrupt_evidence_is_visible(tmp_path):
    put(tmp_path, 'pilot/summary.json', dict(complete=True, stages={'development': {
        'arms': {'S': {'actions': {'candidate_parse_error': 2}}}}}))
    put(tmp_path, 'pilot/models/17/S/predictions/a.json', dict(action='candidate_parse_error'))
    bad = tmp_path / 'pilot/shared/17/bad.json'
    bad.parent.mkdir(parents=True)
    bad.write_text('{', encoding='utf-8')
    report = diagnose(tmp_path)
    assert len(report['read_errors']) == 2
    assert len(report['summary_mismatches']) == 1
    assert not (tmp_path / 'pilot/gate.json').exists()
