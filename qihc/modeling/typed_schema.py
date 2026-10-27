"""Nested source-aware JSON syntax; mathematics and semantic audits stay independent."""
from .sources import catalog


def obj(props, required=None):
    return {'type': 'object', 'properties': props, 'required': list(props) if required is None else required,
            'additionalProperties': False}


def array(items, maximum=512, minimum=0):
    return {'type': 'array', 'items': items, 'minItems': minimum, 'maxItems': maximum}


def union(*variants):
    return variants[0] if len(variants) == 1 else {'anyOf': list(variants)}


INT = {'type': 'integer', 'minimum': -10000, 'maximum': 10000}
TEXT = {'type': 'string'}


def schema(kind, question, n, repair_paths=None):
    data = catalog(question)
    index = {'type': 'integer', 'minimum': 0, 'maximum': n-1}
    scalars = [r['id'] for r in data if r['kind'] == 'scalar']
    arrays = [r['id'] for r in data if isinstance(r['value'], list)]
    scalar_variants = [INT]
    if scalars:
        scalar_variants.append(obj({'source': {'enum': scalars}, 'scale': INT}, ['source']))
    if arrays:
        scalar_variants.append(obj({'length_of': {'enum': arrays}}))
    scalar = union(*scalar_variants)
    sparse = obj({str(i): INT for i in range(n)}, [])
    vector_variants = [array(INT, n, n), sparse]
    quadratic_variants = [array(array(INT, 3, 3), 131072)]
    constraints = []
    term_variants = []

    def columns(entry):
        rows = entry['value']
        if not isinstance(rows, list) or not rows:
            return []
        if all(isinstance(r, dict) for r in rows):
            return sorted(set.intersection(*(set(r) for r in rows)))
        if all(isinstance(r, list) for r in rows):
            return list(range(min(map(len, rows))))
        return []

    for entry in data:
        sid = {'const': entry['id']}
        values = entry['value']
        cols = columns(entry)
        if entry['kind'] == 'vector' and len(values) == n:
            vector_variants.append(obj({'source': sid, 'scale': INT}, ['source']))
        if not cols:
            continue
        column = {'enum': cols}
        row_scalar = union(scalar, obj({'column': column, 'scale': INT}, ['column']))
        selectors = [index, obj({'row_index': {'const': True}}),
                     obj({'column': column}), obj({'column': column, 'offset': INT})]
        for mapping in data:
            unique = [c for c in columns(mapping) if len({r[c] for r in mapping['value']}) == len(mapping['value'])]
            if unique:
                selectors.append(obj({'index_of': obj({'source': {'const': mapping['id']},
                    'column': {'enum': unique}, 'value': row_scalar})}))
        selector = union(*selectors)
        terms = array(obj({'index': selector, 'coefficient': row_scalar}), 256, 1)
        term_variants.append(terms)
        vector_variants.append(obj({'sum_rows': obj({'source': sid, 'terms': terms})}))
        if len(values) == n:
            vector_variants.append(obj({'source': sid, 'column': column, 'scale': INT}, ['source', 'column']))
        if all(isinstance(r, list) for r in values) and sum(map(len, values)) == n:
            vector_variants.append(obj({'source': sid, 'flatten': {'const': True}, 'scale': INT}, ['source', 'flatten']))
        quadratic_variants.append(obj({'sum_rows': obj({'source': sid,
            'terms': array(obj({'i': selector, 'j': selector, 'c': row_scalar}), 256, 1)})}))
        constraints.append(obj({'forall_rows': obj({'source': sid, 'id_prefix': TEXT, 'terms': terms,
                            'op': {'enum': ['le', 'ge', 'eq']}, 'rhs': row_scalar})}))
    vector = union(*vector_variants)
    objective = obj({'sense': {'enum': ['min', 'max']}, 'linear': vector, 'quadratic': union(*quadratic_variants)})
    constraints.insert(0, obj({'id': TEXT, 'a': vector, 'op': {'enum': ['le', 'ge', 'eq']}, 'rhs': scalar}))
    expr = union(obj({'var': index}), obj({'op': {'enum': ['not', 'and', 'or', 'implies', 'iff']},
                                        'args': array({'$ref': '#/$defs/expr'}, 256, 1)}))
    logic = array(obj({'id': TEXT, 'expr': {'$ref': '#/$defs/expr'}}))
    variables = union(obj({'count': scalar, 'prefix': TEXT, 'lower': scalar, 'upper': scalar}),
                      array(obj({'name': TEXT, 'lower': INT, 'upper': INT}), n, n))
    requirement = obj({'id': TEXT, 'kind': {'enum': ['objective', 'domain', 'constraint']},
                       'quote': TEXT, 'explanation': TEXT, 'expected_count': scalar, 'targets': array(TEXT)}, ['id', 'kind'])
    model = obj({'schema': {'const': 'bounded-integer-v2'}, 'variables': variables,
                 'objective': objective, 'constraints': array(union(*constraints)), 'logic': logic})
    variants = {
        'flat': obj({'model': model, 'requirements': array(requirement, 64), 'grounding': array({'type': 'object'})}, ['model']),
        'plan': obj({'variables': variables, 'requirements': array(requirement, 64)}),
        'objective': obj({'objective': objective}),
        'rules': obj({'items': array(obj({'requirement_id': TEXT, 'constraints': array(union(*constraints)), 'logic': logic}), 64, 1)}),
    }
    if kind == 'local_repair':
        if not repair_paths:
            raise ValueError('Local patch requires explicit allowed paths')
        value_types = {'linear': vector, 'a': vector, 'quadratic': union(*quadratic_variants),
                       'sense': {'enum': ['min', 'max']}, 'rhs': scalar, 'lower': scalar, 'upper': scalar,
                       'op': {'enum': ['le', 'ge', 'eq', 'not', 'and', 'or', 'implies', 'iff']},
                       'expr': {'$ref': '#/$defs/expr'}, 'terms': union(*term_variants) if term_variants else array(INT)}
        edits = [obj({'path': {'const': path}, 'value': value_types[path.rsplit('/', 1)[-1]]}) for path in repair_paths]
        body = obj({'edits': array(union(*edits), 8, 1), 'reason': TEXT})
    else:
        body = union(variants[kind], obj({'unsupported': TEXT}))
    body['$defs'] = {'expr': expr}
    return body
