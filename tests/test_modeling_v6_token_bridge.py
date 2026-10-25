"""Actual Transformers tokenizer/generate integration; no Hub access or downloads."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.runtime.modeling_v6 import formatting as f
from tools.runtime.modeling_v6.token_bridge import build_tokenizer_data


@pytest.fixture(scope='module')
def tokenizer():
    from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    vocabulary={'<eos>':0,'<unk>':1,'<pad>':2}
    vocabulary.update({chr(i):i-29 for i in range(32,127)})
    backend=Tokenizer(models.WordLevel(vocabulary,unk_token='<unk>'))
    backend.pre_tokenizer=pre_tokenizers.Split(Regex('.'),behavior='isolated')
    backend.decoder=decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<eos>',
                                  unk_token='<unk>',pad_token='<pad>')


def probe(fn,tokenizer,text,prompt_width=3):
    import torch
    sequence=[tokenizer.pad_token_id]*prompt_width
    assert tokenizer.eos_token_id not in fn(0,torch.tensor(sequence))
    for token in tokenizer.encode(text,add_special_tokens=False):
        assert token in fn(0,torch.tensor(sequence)),text
        sequence.append(token)
    assert tokenizer.eos_token_id in fn(0,torch.tensor(sequence))


def test_all_schemas_with_real_tokenizer_and_old_import_blocked(tokenizer,monkeypatch):
    original_import=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='transformers.tokenization_utils' or name=='lmformatenforcer.integrations.transformers':
            raise AssertionError('Legacy Transformers integration must never be imported')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    model={'schema':'bounded-integer-v2','variables':{'count':32,'prefix':'x','lower':0,'upper':1},
           'objective':{'sense':'max','linear':[1]*32,'quadratic':[]},'constraints':[],'logic':[]}
    values={'flat':{'model':model},'plan':{'variables':model['variables'],'requirements':[{'id':'d'},{'id':'o'}]},
            'objective':{'objective':model['objective']},'rules':{'items':[{'constraints':[1]*32}]},
            'patch':{'notes':[],'rule_replacements':[]}}
    gen=SimpleNamespace(tokenizer=tokenizer)
    for kind,value in values.items():
        fn=f.prefix_function(gen,prompt_width=3,kind=kind)
        assert fn.token_enforcer.root_parser.config.max_json_array_length==16384
        probe(fn,tokenizer,json.dumps(value,separators=(',',':')))
    data=gen._v6_tokenizer_data
    f.prefix_function(gen,kind='patch')
    assert gen._v6_tokenizer_data is data  # expensive vocabulary build is shared


def test_bridge_matches_upstream_tokenizer_convention(tokenizer):
    # Independent reference of the pinned adapter's vocabulary conversion; no legacy import.
    anchor=tokenizer.encode('0')[-1]
    expected=[]
    for i in range(len(tokenizer)):
        if i in tokenizer.all_special_ids:continue
        contextual=tokenizer.decode([anchor,i])[1:]
        expected.append((i,contextual,len(contextual)>len(tokenizer.decode([i]))))
    data=build_tokenizer_data(tokenizer)
    assert data.regular_tokens==expected
    assert data.eos_token_id==tokenizer.eos_token_id
    assert not set(tokenizer.all_special_ids)&{t[0] for t in data.regular_tokens}


def test_word_start_context_and_partial_unicode_decoder():
    class ContextTokenizer:
        eos_token_id=0
        all_special_ids=[0]
        def __len__(self):return 4
        def encode(self,text,add_special_tokens=False):return [1]
        def decode(self,ids,**kwargs):
            mapping={(1,):'0',(2,):'word',(1,2):'0 word',(1,1):'00',
                     (3,):'\ufffd',(1,3):'0\ufffd',(3,3):'\u4e2d'}
            return mapping[tuple(ids)]
    data=build_tokenizer_data(ContextTokenizer())
    assert (2,' word',True) in data.regular_tokens
    assert data.decoder([3])=='' and data.decoder([3,3])=='\u4e2d'


def test_missing_eos_fails_instead_of_disabling_filter():
    class Missing:
        eos_token_id=None
        def __len__(self):return 10
    with pytest.raises(ValueError,match='EOS'):build_tokenizer_data(Missing())


def test_actual_batched_transformers_generate_uses_bridge(tokenizer,monkeypatch):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    from tools.runtime.modeling_v6 import generation as g
    torch.manual_seed(1)
    torch.set_num_threads(1)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=256,n_embd=16,n_layer=1,
                                   n_head=2,eos_token_id=tokenizer.eos_token_id,
                                   pad_token_id=tokenizer.pad_token_id,bos_token_id=tokenizer.eos_token_id)).eval()
    gen=SimpleNamespace(tokenizer=tokenizer,model=model,device=torch.device('cpu'),max_input_tokens=128,
                        _inputs=lambda text:tokenizer(text,return_tensors='pt',add_special_tokens=False))
    expected=['{"unsupported":"first"}','{"unsupported":"second"}']
    outputs=[tokenizer.encode(text,add_special_tokens=False) for text in expected]
    actual_prefix=f.prefix_function
    calls=[]
    def fixed_prefix(generator,prompt_width=0,kind='flat'):
        real_filter=actual_prefix(generator,prompt_width,kind)
        def callback(batch_id,sent):
            allowed=real_filter(batch_id,sent)
            offset=len(sent)-prompt_width
            token=outputs[batch_id][offset] if offset<len(outputs[batch_id]) else tokenizer.eos_token_id
            assert token in allowed
            calls.append(batch_id)
            # Force a known legal path: the tiny random model is not a modeling benchmark.
            return [token]
        return callback
    monkeypatch.setattr(g,'prefix_function',fixed_prefix)
    rows=g.generate(gen,[(Path('a'),{'prompt':'Q','_kind':'flat','_max_new_tokens':128}),
                         (Path('b'),{'prompt':'QQ','_kind':'flat','_max_new_tokens':128})])
    assert {0,1}==set(calls)
    assert [r['response'] for r in rows]==expected
    assert all(r['status']=='generated' and r['format_enforcement']=='lm-format-enforcer-0.11.3-json-schema' for r in rows)


def test_preparation_uses_project_python_and_compatible_bridge():
    script=(Path(__file__).parents[1]/'tools/runtime/modeling_v6/00_prepare_format.sh').read_text()
    assert '"$PY" -m pip' in script and 'check_runtime' in script
    assert 'lmformatenforcer.integrations.transformers' not in script


def test_public_runtime_imports():
    result=f.check_runtime()
    assert result['weights_loaded'] is False and result['transformers']
