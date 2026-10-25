"""Pinned, mandatory JSON-schema token filtering. Math validation remains separate."""
from importlib.metadata import version

VERSIONS = {'lm-format-enforcer':'0.11.3','interegular':'0.3.3'}


def check_dependencies():
    for name, expected in VERSIONS.items():
        if version(name) != expected:
            raise RuntimeError(f'{name} must be {expected}; run 00_prepare_format.sh on online CPU')
    import pydantic, yaml, packaging  # noqa: F401


def schema():
    array = {'type':'array','items':{},'maxItems':16384}
    object_ = {'type':'object'}
    flexible = {'anyOf':[array,object_]}
    model = {'type':'object','required':['schema','variables','objective','constraints','logic'],
             'additionalProperties':False,'properties':{
                 'schema':{'const':'bounded-integer-v2'},'variables':flexible,
                 'objective':{'type':'object','required':['sense','linear','quadratic'],
                              'additionalProperties':False,'properties':{
                                  'sense':{'enum':['min','max']},'linear':flexible,'quadratic':flexible}},
                 'constraints':array,'logic':array}}
    envelope = {'type':'object','required':['model'],'additionalProperties':False,
                'properties':{'model':model,'requirements':array,'grounding':array,'revision_notes':array}}
    unsupported = {'type':'object','required':['unsupported'],'additionalProperties':False,
                   'properties':{'unsupported':{'type':'string'}}}
    return {'anyOf':[envelope,unsupported]}


def parser():
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    # The library default (20 array items) must never silently shrink CO models.
    return JsonSchemaParser(schema(),config=CharacterLevelParserConfig(max_json_array_length=16384))


def restore_parser_limits(enforcer):
    enforcer.root_parser.config.max_json_array_length = 16384


def prefix_function(generator, prompt_width=0):
    check_dependencies()
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
    if not hasattr(generator,'_v5_tokenizer_data'):
        generator._v5_tokenizer_data = build_token_enforcer_tokenizer_data(generator.tokenizer)
    fn = build_transformers_prefix_allowed_tokens_fn(generator._v5_tokenizer_data,parser())
    # TokenEnforcer rebuilds the parser config to install the tokenizer alphabet.
    # Restore the explicit nested-array limit after that reset, without losing alphabet.
    restore_parser_limits(fn.token_enforcer)
    # Grammar state depends only on generated JSON, not the long source prompt.
    # All batch rows use the same schema; equal output prefixes can share states.
    return (lambda batch_id,sent: fn(batch_id,sent[prompt_width:])) if prompt_width else fn
