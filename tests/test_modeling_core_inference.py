"""Real offline Transformers loading + constrained generation, with a tiny CPU model."""
from pathlib import Path
import pytest


@pytest.mark.parametrize('typed', [False, True])
def test_real_saved_model_backend_and_batch_decoder(tmp_path, monkeypatch, typed):
    from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
    from qihc.modeling.backend import LocalModel
    from qihc.modeling import generation as g, formatting as f
    vocab = {'<eos>': 0, '<unk>': 1, '<pad>': 2}
    vocab.update({chr(i): i-29 for i in range(32, 127)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex('.'), behavior='isolated')
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', unk_token='<unk>', pad_token='<pad>')
    tokenizer.save_pretrained(tmp_path)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=256, n_embd=16, n_layer=1, n_head=1,
                                     eos_token_id=0, pad_token_id=2, bos_token_id=0))
    model.save_pretrained(tmp_path)
    gen = LocalModel(str(tmp_path), device='cpu', temperature=0, max_input_tokens=200)
    expected = ['{"unsupported":"fixture"}', '{"unsupported":"second fixture"}']
    ids = [gen.tokenizer.encode(text, add_special_tokens=False) for text in expected]
    calls = []
    def forced(generator, prompt_width=0, kind='flat', schema_override=None):
        real = f.prefix_function(generator, prompt_width, kind, schema_override)
        row_index = len(calls)
        calls.append(schema_override)
        def fn(batch, sent):
            offset = len(sent)-prompt_width
            selected = row_index if typed else batch
            token = ids[selected][offset] if offset < len(ids[selected]) else 0
            assert token in real(batch, sent)
            return [token]
        return fn
    monkeypatch.setattr(g, 'prefix_function', forced)
    items = [(Path('a'), {'prompt': 'Q', '_kind': 'flat', '_max_new_tokens': 128}),
             (Path('b'), {'prompt': 'QQ', '_kind': 'flat', '_max_new_tokens': 128})]
    if typed:
        import json
        for (_, task), text in zip(items, expected):
            task['_schema'] = {'type': 'object', 'required': ['unsupported'], 'additionalProperties': False,
                               'properties': {'unsupported': {'const': json.loads(text)['unsupported']}}}
    rows = g.generate(gen, items)
    assert [r['response'] for r in rows] == expected
    assert all(r['status'] == 'generated' for r in rows)
    assert len(calls) == (2 if typed else 1)
