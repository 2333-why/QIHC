"""Validate the explicit compact-protocol subset without an extra dependency.

Not a general JSON Schema engine. Unknown schema keywords fail closed.
"""
def validate(value, schema):
    allowed={'anyOf','const','enum','type','properties','required','additionalProperties',
             'items','minItems','maxItems','minimum','maximum','maxLength'}
    if set(schema)-allowed:
        raise ValueError('Unsupported schema keyword')
    if 'anyOf' in schema:
        for variant in schema['anyOf']:
            try:
                validate(value,variant)
                return
            except ValueError:
                pass
        raise ValueError('Value does not match any allowed variant')
    if 'const' in schema and value!=schema['const']:raise ValueError('Wrong constant')
    if 'enum' in schema and value not in schema['enum']:raise ValueError('Unknown enum/source reference')
    kind=schema.get('type')
    types={'object':dict,'array':list,'integer':int,'string':str}
    if kind and (kind not in types or type(value) is not types[kind]):raise ValueError('Wrong value type')
    if kind=='object':
        props=schema.get('properties',{})
        if set(schema.get('required',[]))-set(value):raise ValueError('Missing required fields')
        if schema.get('additionalProperties') is False and set(value)-set(props):raise ValueError('Unexpected fields')
        for key,item in value.items():
            if key in props:validate(item,props[key])
    elif kind=='array':
        if not schema.get('minItems',0)<=len(value)<=schema.get('maxItems',float('inf')):raise ValueError('Wrong array length')
        for item in value:validate(item,schema['items'])
    elif kind=='integer':
        if not schema.get('minimum',float('-inf'))<=value<=schema.get('maximum',float('inf')):raise ValueError('Integer outside bounds')
    elif kind=='string' and len(value)>schema.get('maxLength',float('inf')):
        raise ValueError('String too long')
