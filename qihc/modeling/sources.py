"""Literal-only source catalog and bounded declarative expansion. No task classifiers."""
import copy
import hashlib
import json
import re
from functools import lru_cache
from qihc.generic.energy import integer


def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate source key: '+key)
        out[key] = value
    return out


@lru_cache(maxsize=256)
def catalog(question):
    """Parse only literal JSON spans or standalone decimal integers from original text."""
    if len(question) > 250000:
        raise ValueError('Source text too large')
    rows, spans = [], []
    decoder = json.JSONDecoder(object_pairs_hook=unique)
    def visit(value, start, end, path):
        kind = None
        if type(value) is int:
            kind = 'scalar'
        elif isinstance(value, list) and value and len(value) <= 4096:
            if all(type(v) is int for v in value):
                kind = 'vector'
            elif all(isinstance(v, dict) and v and all(type(x) is int for x in v.values()) for v in value):
                kind = 'table'
            elif all(isinstance(v, list) and v and all(type(x) is int for x in v) for v in value):
                kind = 'table'
        if kind:
            rows.append({'kind':kind, 'start':start, 'end':end, 'path':path, 'value':value,
                         'context':question[max(0,start-65):start]})
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child,start,end,path+'/'+key.replace('~','~0').replace('/','~1'))
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child,start,end,path+'/'+str(i))
    i = 0
    while i < len(question):
        if question[i] in '[{':
            try:
                value, length = decoder.raw_decode(question[i:])
            except (ValueError, RecursionError):
                i += 1
                continue
            end = i+length
            spans.append((i,end)); visit(value,i,end,''); i = end
        else:
            i += 1
    for match in re.finditer(r'(?<![\w.])[+-]?\d+(?![\w.])',question):
        if not any(a <= match.start() < b for a,b in spans):
            visit(int(match.group()),match.start(),match.end(),'')
    if len(rows) > 2048:
        raise ValueError('Too many source literals')
    rows.sort(key=lambda r:(r['start'],r['path']))
    for i,row in enumerate(rows):
        row['id'] = 'd'+str(i)
        row['span_sha256'] = hashlib.sha256(question[row['start']:row['end']].encode()).hexdigest()
    return rows


class Resolver:
    def __init__(self, question):
        self.data = {r['id']:r for r in catalog(question)}
        self.bindings = []

    def source(self, key, target):
        if not isinstance(key,str) or key not in self.data:
            raise ValueError(f'{target}: unknown source reference {key!r}')
        row = self.data[key]
        self.bindings.append({'target':target,'source':key,'path':row['path'],'span_sha256':row['span_sha256']})
        return copy.deepcopy(row['value'])

    def scalar(self, value, target, row=None):
        if isinstance(value,dict):
            if 'scale' in value:
                selector = {k:v for k,v in value.items() if k!='scale'}
                return integer(self.scalar(selector,target,row)*integer(value['scale']))
            if set(value) == {'column'} and row is not None:
                try:
                    column = value['column']
                    if isinstance(row,list) and (type(column) is not int or not 0 <= column < len(row)):
                        raise IndexError()
                    if isinstance(row,dict) and not isinstance(column,str):
                        raise KeyError()
                    value = row[column]
                except (KeyError,IndexError,TypeError):
                    raise ValueError(f'{target}: missing/invalid column') from None
            elif set(value) == {'source'}:
                value = self.source(value['source'],target)
            elif set(value) == {'length_of'}:
                data = self.source(value['length_of'],target)
                if not isinstance(data,list):
                    raise ValueError(f'{target}: length_of requires an array')
                value = len(data)
            else:
                raise ValueError(f'{target}: unsupported scalar reference')
        return integer(value)

    def index(self, value, row, row_index, n, target):
        use_row = isinstance(value,dict) and set(value)=={'row_index'} and value['row_index'] is True
        index = row_index if use_row else self.scalar(value,target,row)
        if not 0 <= index < n:
            raise ValueError(f'{target}: index {index} outside 0..{n-1}')
        return index

    def rows(self, source, target):
        rows = self.source(source,target)
        if not isinstance(rows,list) or not 1 <= len(rows) <= 512:
            raise ValueError(f'{target}: row expansion needs 1..512 rows')
        return rows

    def linear_terms(self, terms, row, ri, n, target):
        if not isinstance(terms,list) or not 1 <= len(terms) <= 256:
            raise ValueError(f'{target}: expected 1..256 linear terms')
        a = [0]*n
        for term in terms:
            if not isinstance(term,dict) or set(term) != {'index','coefficient'}:
                raise ValueError(f'{target}: linear term needs index,coefficient')
            index = self.index(term['index'],row,ri,n,target)
            a[index] += self.scalar(term['coefficient'],target,row)
        return a

    def vector(self, value, n, target):
        if isinstance(value,dict) and 'source' in value:
            if set(value)-{'source','scale','column','flatten'} or ('column' in value and 'flatten' in value):
                raise ValueError(f'{target}: vector supports source, scale, and either column or flatten')
            data = self.source(value['source'],target)
            if 'column' in value:
                if not isinstance(data,list):
                    raise ValueError(f'{target}: column requires rows')
                data = [self.scalar({'column':value['column']},target,row) for row in data]
            if 'flatten' in value:
                if value['flatten'] is not True or not isinstance(data,list) or not all(isinstance(row,list) for row in data):
                    raise ValueError(f'{target}: flatten requires an array of arrays and true flag')
                data = [x for row in data for x in row]
            if not isinstance(data,list) or len(data) != n or not all(type(x) is int for x in data):
                raise ValueError(f'{target}: source must be an integer vector of length {n}')
            scale = self.scalar(value.get('scale',1),target)
            return [integer(scale*x) for x in data]
        if isinstance(value,dict) and set(value) == {'sum_rows'}:
            spec = value['sum_rows']
            if not isinstance(spec,dict) or set(spec) != {'source','terms'}:
                raise ValueError(f'{target}: sum_rows requires source,terms')
            a = [0]*n
            for ri,row in enumerate(self.rows(spec['source'],target)):
                a = [x+y for x,y in zip(a,self.linear_terms(spec['terms'],row,ri,n,target))]
            return [integer(x) for x in a]
        return value

    def quadratic(self, value, n):
        target = 'objective.quadratic'
        if isinstance(value,dict) and set(value) == {'sum_rows'}:
            spec = value['sum_rows']
            if not isinstance(spec,dict) or set(spec) != {'source','terms'} or not isinstance(spec['terms'],list) or not 1 <= len(spec['terms']) <= 256:
                raise ValueError('quadratic sum_rows requires source, 1..256 terms')
            value = []
            for ri,row in enumerate(self.rows(spec['source'],target)):
                for term in spec['terms']:
                    if not isinstance(term,dict) or set(term) != {'i','j','c'}:
                        raise ValueError('quadratic row term requires i,j,c')
                    value.append([self.index(term['i'],row,ri,n,target),self.index(term['j'],row,ri,n,target),self.scalar(term['c'],target,row)])
        if not isinstance(value,list) or len(value) > 131072:
            raise ValueError('Expected bounded quadratic term list')
        combined = {}
        for term in value:
            if isinstance(term,dict) and set(term) == {'i','j','c'}:
                term = [term['i'],term['j'],term['c']]
            if not isinstance(term,list) or len(term) != 3:
                raise ValueError('Expected [i,j,c] or {i,j,c}')
            i,j,c = map(integer,term)
            if not 0 <= i < n or not 0 <= j < n:
                raise ValueError('Quadratic index out of range')
            key = tuple(sorted((i,j)))
            combined[key] = combined.get(key,0)+c
        # Preserve the polynomial exactly: do not change signs or discard duplicates.
        return [[i,j,integer(c)] for (i,j),c in sorted(combined.items()) if c]


def expand(obj, question, *, resolver_type=Resolver):
    obj = copy.deepcopy(obj)
    resolver = resolver_type(question)
    model = obj.get('model')
    if not isinstance(model,dict):
        return obj, []
    variables = model.get('variables')
    if isinstance(variables,dict) and set(variables) == {'count','prefix','lower','upper'}:
        count = resolver.scalar(variables['count'],'variables.count')
        if not 1 <= count <= 256 or not isinstance(variables['prefix'],str):
            raise ValueError('Invalid compact variables')
        model['variables'] = [[variables['prefix']+str(i),resolver.scalar(variables['lower'],'variables.lower'),resolver.scalar(variables['upper'],'variables.upper')] for i in range(count)]
    if not isinstance(model.get('variables'),list):
        return obj, resolver.bindings
    n = len(model['variables'])
    if not 1 <= n <= 256:
        raise ValueError('1..256 variables required')
    objective = model.get('objective')
    if isinstance(objective,dict):
        objective['linear'] = resolver.vector(objective.get('linear'),n,'objective.linear')
        objective['quadratic'] = resolver.quadratic(objective.get('quadratic'),n)
    constraints = model.get('constraints')
    if isinstance(constraints,list):
        out = []
        for con in constraints:
            if isinstance(con,dict) and set(con) == {'forall_rows'}:
                spec = con['forall_rows']
                if not isinstance(spec,dict) or set(spec) != {'source','id_prefix','terms','op','rhs'} or not isinstance(spec['id_prefix'],str):
                    raise ValueError('forall_rows needs source,id_prefix,terms,op,rhs')
                rows = resolver.rows(spec['source'],'constraints.forall_rows')
                for ri,row in enumerate(rows):
                    cid = spec['id_prefix']+str(ri)
                    out.append({'id':cid,'a':resolver.linear_terms(spec['terms'],row,ri,n,'constraint:'+cid),'op':spec['op'],'rhs':resolver.scalar(spec['rhs'],'constraint:'+cid,row)})
            else:
                if isinstance(con,list) and len(con)==4:
                    con = dict(zip(('id','a','op','rhs'),con))
                if isinstance(con,dict):
                    con['a'] = resolver.vector(con.get('a'),n,'constraint:'+str(con.get('id')))
                    con['rhs'] = resolver.scalar(con.get('rhs'),'constraint:'+str(con.get('id'))+'.rhs')
                out.append(con)
            if len(out) > 512:
                raise ValueError('Too many expanded constraints')
        model['constraints'] = out
    return obj, resolver.bindings
