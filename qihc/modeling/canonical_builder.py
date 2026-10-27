"""Opt-in deterministic staged IDs; names carry provenance, never mathematics."""
import copy
from . import builder as legacy
from .typed_builder import Builder as TypedBuilder


class Builder(TypedBuilder):
    def canonical_state(self, state):
        result = copy.deepcopy(state)
        mapping = []
        positions = {r['id']: i for i, r in enumerate(state['requirements'])}
        for rule in result['rules']:
            group = positions[rule['requirement_id']]
            for field, tag in [('constraints', 'c'), ('logic', 'l')]:
                for i, item in enumerate(rule[field]):
                    key = f'r{group}_{tag}{i}'
                    if 'forall_rows' in item:
                        old = item['forall_rows']['id_prefix']
                        new = key+'_row_'
                        item['forall_rows']['id_prefix'] = new
                        kind = 'prefix'
                    else:
                        old, new = item['id'], key
                        item['id'] = new
                        kind = 'id'
                    mapping.append({'requirement_id': rule['requirement_id'], 'field': field,
                                    'position': i, 'kind': kind, 'original': old, 'canonical': new})
        return result, mapping

    def assemble(self, state, question, complete=True):
        # Keep raw state/checkpoints untouched. Repeated aliases are one-to-many,
        # disambiguated by requirement, field and position, never silently merged.
        canonical, _ = self.canonical_state(state)
        obj = super().assemble(canonical, question, complete)
        for req in obj['requirements']:
            if req['kind'] == 'domain':
                model, _ = self.compile_math(self.dump(obj), question)
                req['targets'] = ['variable:'+v['name'] for v in model['variables']]
        return obj

    def finish(self, state, question):
        text = legacy.dump(self.assemble(state, question))
        self.inspect(text, question)
        return text
