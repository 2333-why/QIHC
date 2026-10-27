"""Explicit, source-grounded index transforms. Never infer an offset or a formula."""
import copy
from . import sources, protocol
from qihc.generic.energy import integer


class Resolver(sources.Resolver):
    def index(self, value, row, row_index, n, target):
        if isinstance(value, dict) and set(value) == {'column', 'offset'}:
            index = self.scalar({'column': value['column']}, target, row)+integer(value['offset'])
        elif isinstance(value, dict) and set(value) == {'index_of'}:
            mapping = value['index_of']
            if not isinstance(mapping, dict) or set(mapping) != {'source', 'column', 'value'}:
                raise ValueError(f'{target}: index_of requires source,column,value')
            rows = self.rows(mapping['source'], target)
            labels = [self.scalar({'column': mapping['column']}, target, r) for r in rows]
            if len(set(labels)) != len(labels):
                raise ValueError(f'{target}: index_of labels must be unique')
            label = self.scalar(mapping['value'], target, row)
            if label not in labels:
                raise ValueError(f'{target}: index_of label {label} absent from declared source')
            index = labels.index(label)
        else:
            return super().index(value, row, row_index, n, target)
        if not 0 <= index < n:
            raise ValueError(f'{target}: explicit mapped index {index} outside 0..{n-1}')
        return index


def expand(obj, question):
    return sources.expand(obj, question, resolver_type=Resolver)


def compile_math(text, question=''):
    obj, _ = expand(protocol.load_object(text), question)
    return protocol.compile_object(obj)


def inspect(text, question):
    obj = protocol.load_object(text)
    expanded, bindings = expand(obj, question)
    # Use the existing independent audit on explicit mathematics, preserving the ledger.
    import json
    model, reason, audits = protocol.inspect(json.dumps(expanded), question)
    if model is not None:
        audits = copy.deepcopy(audits)
        audits['source_bindings'] = bindings
        audits['semantic_diagnostics'] = protocol.diagnostics(model, question, bindings)
    return model, reason, audits


def describe(question):
    """Only source types/available selectors; no task names or gold mathematics."""
    result = []
    for entry in sources.catalog(question):
        value = entry['value']
        row = {'id': entry['id'], 'kind': entry['kind']}
        if isinstance(value, list):
            row['length'] = len(value)
            if entry['kind'] == 'vector':
                row['vector_reference'] = {'source': entry['id']}
                row['columns'] = []
                row['warning'] = 'Scalar array: no named columns, no flatten.'
            elif all(isinstance(r, dict) for r in value):
                row['columns'] = sorted(set.intersection(*(set(r) for r in value)))
                row['unique_columns'] = [c for c in row['columns'] if len({r[c] for r in value}) == len(value)]
                row['column_ranges'] = {c: [min(r[c] for r in value), max(r[c] for r in value)] for c in row['columns']}
            elif all(isinstance(r, list) for r in value):
                row['columns'] = list(range(min(map(len, value))))
                row['row_lengths'] = sorted(set(map(len, value)))
                row['flatten_reference'] = {'source': entry['id'], 'flatten': True}
        result.append(row)
    return result
