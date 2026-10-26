"""LMFE core -> public Transformers prefix_allowed_tokens_fn protocol.

The upstream 0.11.3 integration imports PreTrainedTokenizerBase through a removed
Transformers module and also imports unused model/logit-warper internals. We need
neither: tokenizer encode/decode plus TokenEnforcer are sufficient. No monkeypatch,
site-packages edit, dependency downgrade, or unconstrained-generation fallback.

Tokenizer context/word-start handling follows the LMFE integration convention:
https://github.com/noamgat/lm-format-enforcer/blob/v0.11.3/lmformatenforcer/integrations/transformers.py
"""
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData


def build_tokenizer_data(tokenizer):
    size = len(tokenizer)
    eos = tokenizer.eos_token_id
    if type(eos) is not int or not 0 <= eos < size:
        raise ValueError('Format filtering requires an in-vocabulary tokenizer EOS id')
    special = set(tokenizer.all_special_ids)
    anchor = tokenizer.encode('0', add_special_tokens=False)
    if not anchor:
        raise ValueError('Tokenizer cannot encode the format-filter context anchor')

    def decode(ids):
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    prefix = decode(anchor)
    if not prefix:
        raise ValueError('Empty format-filter context anchor')
    regular = []
    for token in range(size):
        if token in special:
            continue
        contextual = decode(anchor + [token])
        if not contextual.startswith(prefix):
            raise ValueError(f'Tokenizer changes context anchor at token {token}; refusing unsafe token filtering')
        piece = contextual[len(prefix):]
        single = decode([token])
        regular.append((token, piece, len(piece) > len(single)))
    # Byte tokenizers may emit a partial UTF-8 code point; LMFE waits for its suffix.
    return TokenEnforcerTokenizerData(regular, lambda ids: decode(ids).rstrip('\ufffd'),
                                     eos, False, size)


class PrefixFilter:
    def __init__(self, tokenizer_data, parser, prompt_width=0):
        self.token_enforcer = TokenEnforcer(tokenizer_data, parser)
        self.prompt_width = prompt_width

    def __call__(self, batch_id, sent):
        suffix = sent[self.prompt_width:]
        tokens = suffix.tolist() if hasattr(suffix, 'tolist') else list(suffix)
        return self.token_enforcer.get_allowed_tokens(tokens).allowed_tokens
