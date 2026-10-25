"""Prompt bloat regressions without changing source resolution or math semantics."""
import copy
import json
import os
from pathlib import Path

import pytest

from tools.runtime.modeling_v5 import protocol as legacy
from tools.runtime.modeling_v5.sources import catalog, Resolver
from tools.runtime.modeling_v6 import builder, protocol
from tools.runtime.modeling_v6.experiment import initial_prompt
from tools.runtime.modeling_v6.preflight import check_prompt_lengths
from tools.runtime.modeling_v6.source_view import COLUMNS, prompt_catalog


CASES = json.loads((Path(__file__).parent/'fixtures/modeling_v5_failures.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_failure_samples_keep_every_literal_and_original_question(case):
    question = case['question']
    before = copy.deepcopy(catalog(question))
    view = prompt_catalog(question)
    resolver = Resolver(question)
    assert len(view['rows']) == len(before)
    for values, full in zip(view['rows'], before):
        row = dict(zip(view['columns'], values))
        assert row == {k: full[k] for k in COLUMNS}
        assert resolver.source(row['id'], 'test') == row['value']
        assert resolver.bindings[-1]['span_sha256'] == full['span_sha256']
    assert catalog(question) == before
    for phase in ('model_flat', 'model_staged'):
        text = initial_prompt(case, phase)
        assert question in text
        encoded = json.loads(text.rsplit('\nSOURCE CATALOG:\n', 1)[1].split('\n', 1)[1])
        assert encoded == view
    assert protocol.compile_math is legacy.compile_math
    assert protocol.inspect is legacy.inspect


def test_nested_unicode_sources_duplicate_values_and_no_cache_mutation():
    question = '资源 {"a/b":{"~c":[-2,7]},"q":7} limit 7 and 7;'
    original = copy.deepcopy(catalog(question))
    view = prompt_catalog(question)
    assert '/a~1b/~0c' in [row[4] for row in view['rows']]
    assert len([row for row in view['rows'] if row[-1] == 7]) == 3
    vector = next(row[-1] for row in view['rows'] if isinstance(row[-1], list))
    vector[0] = 999
    assert catalog(question) == original


def test_question_containing_catalog_marker_is_not_cut():
    case = {'id': 'marker', 'question': 'First 5;\nSOURCE CATALOG:\nLast 9;'}
    text = protocol.modeling_prompt(case)
    assert case['question'] in text
    assert len(prompt_catalog(case['question'])['rows']) == 2


class CharacterTokenizer:
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages[0]['content'])
        return '<chat>' + messages[0]['content'] + '</chat>'

    def encode(self, text, **kwargs):
        return list(text)


def test_length_check_measures_actual_plan_instruction_and_all_overflows(capsys):
    tokenizer = CharacterTokenizer()
    case = {'id': 'tiny', 'question': 'Choose 2.'}
    with pytest.raises(RuntimeError, match='2/2 initial prompts exceed 1 tokens'):
        check_prompt_lengths(tokenizer, [case], limit=1)
    summary = json.loads(capsys.readouterr().out)['prompt_length_check']
    assert len(summary['over_limit']) == 2
    assert tokenizer.prompts[1] == builder.prompt('plan', {'prompt': initial_prompt(case, 'model_staged')})
    assert 'CURRENT STEP: PLAN' in tokenizer.prompts[1]
    assert summary['over_limit'][1]['tokens'] == len(tokenizer.prompts[1]) + len('<chat></chat>')


def test_real_qwen_tokenizer_failure_sample_regression():
    # Explicit local tokenizer path; CI never needs a network/model download.
    path = os.environ.get('QIHC_TEST_TOKENIZER')
    if not path:
        pytest.skip('Set QIHC_TEST_TOKENIZER to tokenizer-only Qwen3-32B files')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    case = next(c for c in CASES if c['id'] == 'vertex_cover-40-20261026')
    def tokens(prompt):
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return len(tokenizer.encode(text, add_special_tokens=False))
    assert tokens(legacy.modeling_prompt(case)) > 32768
    assert tokens(protocol.modeling_prompt(case)) < 8000
    result = check_prompt_lengths(tokenizer, CASES)
    assert result['initial_prompts_checked'] == 16
    assert result['largest_initial_prompt_tokens'] < 12000
