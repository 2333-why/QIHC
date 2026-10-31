"""Source-only review and typed local patches, shared by all E004 arms."""
import copy
import json
from qihc.generic.bounded import validate
from qihc.modeling import typed_schema, typed_sources, typed_builder


def compact_model(spec):
    if spec is None:
        return None
    m = copy.deepcopy(spec)
    m['objective']['linear'] = {str(i): v for i, v in enumerate(m['objective']['linear']) if v}
    for c in m['constraints']:
        c['a'] = {str(i): v for i, v in enumerate(c['a']) if v}
    return m


def schemas(question, names):
    flat = typed_schema.schema('flat', question, len(names))
    body = flat.get('anyOf', [flat])[0]
    model = body['properties']['model']['properties']
    props = model['objective']['properties']
    obj, arr = typed_schema.obj, typed_schema.array
    text = {'type': 'string', 'maxLength': 1200}
    variants = [obj({'op': {'const': 'sense'}, 'value': props['sense']}),
                obj({'op': {'const': 'domain'}, 'index':{'type':'integer','minimum':0,'maximum':len(names)-1},
                     'lower':typed_schema.INT, 'upper':typed_schema.INT}),
                obj({'op': {'const': 'linear'}, 'value': props['linear']}),
                obj({'op': {'const': 'quadratic'}, 'value': props['quadratic']}),
                obj({'op': {'const': 'constraint'}, 'value': model['constraints']['items']}),
                obj({'op': {'const': 'logic'}, 'value': model['logic']['items']}),
                obj({'op': {'const': 'remove_constraint'}, 'id': text}),
                obj({'op': {'const': 'remove_logic'}, 'id': text})]
    patch = obj({'edits': arr({'anyOf': variants}, 8), 'quote': text, 'reason': text})
    patch['$defs'] = flat['$defs']
    review = obj({'hypotheses': arr({k: v for k,v in patch.items() if k != '$defs'}, 2), 'notes': text})
    review['$defs'] = flat['$defs']
    return flat, review, patch


def apply_patch(spec, patch, question):
    if not isinstance(patch, dict) or not isinstance(patch.get('edits'), list) or len(patch['edits']) > 8:
        raise ValueError('Expected <=8 typed edits')
    m = compact_model(spec)
    for edit in patch['edits']:
        op = edit['op']
        if op in ('sense', 'linear', 'quadratic'):
            m['objective'][op] = edit['value']
        elif op=='domain':
            i=edit['index']
            if type(i) is not int or not 0<=i<len(m['variables']):
                raise ValueError('Invalid domain edit index')
            m['variables'][i].update(lower=edit['lower'],upper=edit['upper'])
        elif op in ('constraint', 'logic'):
            field = 'constraints' if op == 'constraint' else 'logic'
            value = edit['value']
            # Expand source bindings in a temporary envelope, then enforce unique
            # IDs across BOTH constraint kinds. Replacement never guesses an ID.
            temp = copy.deepcopy(m); temp[field] = [value]
            temp['logic' if field == 'constraints' else 'constraints'] = []
            expanded, _ = typed_sources.expand({'model': temp}, question)
            additions = expanded['model'][field]
            ids = {v['id'] for v in additions}
            m[field] = [v for v in m[field] if v['id'] not in ids]+additions
        elif op in ('remove_constraint', 'remove_logic'):
            field = 'constraints' if op == 'remove_constraint' else 'logic'
            if not any(c['id'] == edit['id'] for c in m[field]):
                raise ValueError('Cannot remove unknown constraint ID')
            m[field] = [c for c in m[field] if c['id'] != edit['id']]
        else:
            raise ValueError('Unknown edit operation')
    out, reason = typed_sources.compile_math(json.dumps({'model': m}), question)
    if out is None or [v['name'] for v in out['variables']] != [v['name'] for v in spec['variables']]:
        raise ValueError('Patch changed variable mapping or unsupported model: '+str(reason))
    return validate(out)


def alternatives(spec, response, question):
    valid, errors = [], []
    if spec is None:
        return valid, errors
    for candidate in response.get('hypotheses', [])[:2]:
        try:
            alt = apply_patch(spec, candidate, question)
            if alt['variables'] != spec['variables']:
                errors.append('Domain hypothesis retained for source review but not compared on incompatible conditional state space')
            elif alt != spec and alt not in valid:
                valid.append(alt)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(str(exc))
    return valid, errors


def build_request(case, state, stage, evidence=None):
    """Only whitelisted public fields enter prompts; never case gold or labels."""
    question, names = case['question'], case['variable_names']
    flat, review, patch = schemas(question, names)
    model = state.get('model')
    if stage == 'initial' or model is None:
        prompt = typed_builder.prompt(question, names, 'flat')
        if stage != 'initial':
            prompt += '\nPrevious model is not executable. Rebuild from the full source; do not omit requirements. Previous error: '+str(state.get('error', 'invalid initial model'))[:1500]
        syntax = flat
    else:
        public = {'source_question': question, 'current_model': compact_model(model),
                  'variable_mapping': names, 'round': state['round'],
                  'previous_execution_error': state.get('error')}
        if stage == 'review':
            prompt = ('Read the original problem independently before judging the current model. '
                      'Propose at most two LOCAL alternative interpretations with typed edits; retaining the current model is always an implicit candidate. '
                      'Check objective direction, quantifiers, data references and variable relationships. '
                      'Quote the source supporting each alternative. Do not specialize by benchmark name. '
                      'Return hypotheses and notes, not a solution or executable code.\n')
            syntax = review
        else:
            public['evidence'] = evidence
            public['review'] = state.get('review', {})
            public['memory'] = [] if state['arm'] == 'no_memory' else state.get('memory', [])[-2:]
            prompt = ('Compare the source with the current model and finite-time evidence. '
                      'Sampling describes the generated model, NOT ground truth; a lower energy or easier feasibility is not proof of correct semantics. '
                      'Return at most 8 typed local edits plus quote/reason. Empty edits retain the model. '
                      'Only change source-supported mathematics; never delete requirements just to obtain feasibility. '
                      'Memory is fallible; explicitly reconsider earlier interpretations if necessary.\n')
            syntax = patch
        prompt += json.dumps(public, ensure_ascii=False, separators=(',', ':'))
        prompt += ('\nEdits use op=sense/linear/quadratic with value, op=constraint/logic with a complete item in value, '
                   'op=domain with index/lower/upper, or op=remove_constraint/remove_logic with id. Sparse vectors map zero-based indices to coefficients. '
                   'Source references and forall_rows follow the public source catalog. No variable renaming.\nPublic source catalog:\n'
                   +json.dumps(typed_sources.describe(question), ensure_ascii=False, separators=(',', ':')))
    return {'prompt': prompt, '_kind': 'flat', '_schema': syntax,
            'stage': stage, 'case_id': case['id'], 'trajectory_id': state.get('id'),
            'question_sha256': __import__('hashlib').sha256(question.encode()).hexdigest()}


def feedback_packet(arm, base, probe_result):
    if arm == 'C1':
        return {'scope': 'source-only self-review; no search observations'}
    if arm == 'C2':
        options=[r for r in (base,probe_result.get('passive',{})) if r.get('best_key') is not None]
        if options:
            base=min(options,key=lambda r:tuple(r['best_key']))
        return {'best_candidate': base.get('x'), 'predicted_objective':base.get('best_objective'),
                'predicted_residuals':base.get('best_residuals'), 'scope': 'single candidate under predicted model'}
    from qihc.generic.feedback import compact
    return {'base': compact(base, arm != 'no_q'), 'responses': probe_result,
            'scope': 'predicted model observations, not semantic truth'}


def bounded_evidence(packet, tokenizer, limit):
    """Reduce diagnostic payload only; never silently truncate original source."""
    value = copy.deepcopy(packet)
    def size():
        return len(tokenizer.encode(json.dumps(value, ensure_ascii=False, separators=(',', ':')), add_special_tokens=False))
    original = size()
    # Keep response pairing intact by removing whole windows/diagnostics, not
    # cutting JSON strings or losing model/condition labels.
    while size() > limit-64:
        responses = value.get('responses', {})
        reports = responses.get('probes', [])
        if len(reports) > 1:
            reports.pop(); continue
        if reports and len(reports[0].get('windows', [])) > 2:
            reports[0]['windows'] = reports[0]['windows'][:2]; continue
        if 'base' in value:
            base = value['base']
            for field in ('q', 'states', 'mean', 'variance', 'residual_mean', 'violation_rates'):
                if len(base.get(field, [])) > 2:
                    base[field] = base[field][:2]; break
            else:
                value.pop('responses', None)
                if size() > limit:
                    value = {'best_candidate': base.get('x'), 'evidence_omitted': 'diagnostic token cap'}
        else:
            value = {'evidence_omitted': 'diagnostic token cap'}
    value['compression'] = {'original_tokens': original, 'cap': limit}
    return value
