"""Pinned, mandatory JSON-schema token filtering. Math validation remains separate."""
from importlib.metadata import version

VERSIONS = {'lm-format-enforcer':'0.11.3','interegular':'0.3.3'}


def check_dependencies():
    for name, expected in VERSIONS.items():
        if version(name) != expected:
            raise RuntimeError(f'{name} must be {expected}; run 00_prepare_format.sh on online CPU')
    import pydantic, yaml, packaging  # noqa: F401


def check_runtime():
    """Check the exact public API used by this runtime; load no model weights."""
    check_dependencies()
    import sys
    import torch
    import transformers
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList  # noqa: F401
    from .token_bridge import PrefixFilter, build_tokenizer_data  # noqa: F401
    parser()
    return {'python':sys.executable,'transformers':transformers.__version__,
            'torch':torch.__version__,'format_bridge':'qihc-lmfe-core-public-prefix-api',
            'weights_loaded':False}


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


def stage_schema(kind):
    if kind == 'flat':
        return schema()
    array = {'type':'array','maxItems':16384}
    flexible = {'anyOf':[array,{'type':'object'}]}
    props = {
        'plan': {'variables':flexible,'requirements':{'type':'array','minItems':2,'maxItems':64}},
        'objective': {'objective':schema()['anyOf'][0]['properties']['model']['properties']['objective']},
        'rules': {'items':{'type':'array','minItems':1,'maxItems':64}},
        'patch': {'notes':array,'variables':flexible,'objective':{'type':'object'},
                  'rule_replacements':array,'new_requirements':array,'requirement_updates':array},
    }[kind]
    required = ['notes'] if kind=='patch' else list(props)
    return {'anyOf':[{'type':'object','properties':props,'required':required,'additionalProperties':False},
                     {'type':'object','required':['unsupported'],'additionalProperties':False,
                      'properties':{'unsupported':{'type':'string'}}}]}


def parser(kind='flat', schema_override=None):
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    # The library default (20 array items) must never silently shrink CO models.
    return JsonSchemaParser(token_schema(schema_override if schema_override is not None else stage_schema(kind)),config=CharacterLevelParserConfig(max_json_array_length=16384))


def token_schema(value):
    """LMFE 0.11.3 sends non-string const values to a string parser.

    Keep string consts exact. Numeric consts use numeric enums; bool consts
    use JSON boolean syntax (not Python True/False). The compiler still
    enforces true-only row_index/flatten, so false never becomes valid math.
    This transforms a copy for token filtering, not the execution schema.
    """
    if isinstance(value, list):
        return [token_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {k: token_schema(v) for k, v in value.items()}
    if 'const' in out and type(out['const']) is not str:
        literal = out.pop('const')
        if type(literal) is bool:
            out.pop('enum', None)
            out['type'] = 'boolean'
        elif literal is None:
            out['type'] = 'null'
        elif type(literal) in (int, float):
            out['enum'] = [literal]
        else:
            raise ValueError('Unsupported non-scalar schema const')
    return out


def restore_parser_limits(enforcer):
    enforcer.root_parser.config.max_json_array_length = 16384


def prefix_function(generator, prompt_width=0, kind='flat', schema_override=None):
    check_dependencies()
    from .token_bridge import build_tokenizer_data, PrefixFilter
    if not hasattr(generator,'_v6_tokenizer_data'):
        generator._v6_tokenizer_data = build_tokenizer_data(generator.tokenizer)
    fn = PrefixFilter(generator._v6_tokenizer_data,parser(kind, schema_override),prompt_width)
    # TokenEnforcer rebuilds the parser config to install the tokenizer alphabet.
    # Restore the explicit nested-array limit after that reset, without losing alphabet.
    restore_parser_limits(fn.token_enforcer)
    # Grammar state depends only on generated JSON, not the long source prompt.
    # All batch rows use the same schema; equal output prefixes can share states.
    return fn
