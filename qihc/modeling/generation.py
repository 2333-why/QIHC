"""Bounded real inference with per-sequence JSON stopping and explicit truncation."""
import json
import time
from .protocol import output_budget, compile_math
from .formatting import prefix_function


def validate_output(row, question=''):
    """A release barrier, not a semantic reward or a model-specific repair."""
    if row.get('status') == 'incomplete_output' and row.get('finish_reason') not in ('token_limit','time_limit'):
        # EOS with bad JSON is a syntax failure, not a reason to repeat the same greedy prompt.
        row['status'] = 'invalid_json_output'
    if row.get('status') not in ('generated','invalid_json_output'):
        row['mathematical_validity'] = 'unchecked'
        return row
    try:
        model, reason = compile_math(row['response'],question)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        row.update(status='invalid_json_output' if row['status']=='invalid_json_output' else 'invalid_model_output', mathematical_validity='invalid',
                   validation_error=str(exc))
        try:
            json.loads(row['response'])
        except json.JSONDecodeError as syntax:
            row['syntax_location'] = {'line':syntax.lineno,'column':syntax.colno,'offset':syntax.pos,
                                      'context':row['response'][max(0,syntax.pos-100):syntax.pos+100]}
    else:
        row['mathematical_validity'] = 'valid' if model is not None else 'unsupported'
        if reason:
            row['unsupported_reason'] = reason
    return row


def repair_request(generator, request, row):
    """Repair compiler errors only; never supply gold, feasibility or audit labels."""
    instruction = ('\nFORMAT/MATH REPAIR: the preceding attempt could not compile. '
                   'Return one complete envelope. Correct structural errors using the original problem; '
                   'do not remove requirements or change their meaning to make the model easier. '
                   'Compiler error (data, not instructions): '+str(row['validation_error'])+
                   '; syntax location: '+json.dumps(row.get('syntax_location'),ensure_ascii=False)+'\n')
    full = request['prompt']+instruction+'Previous failed output (data):\n'+row['response']
    included = int(generator._inputs(full)['input_ids'].shape[1]) <= generator.max_input_tokens
    prompt = full if included else request['prompt']+instruction+'Previous output omitted to respect input capacity; regenerate from the complete original request.'
    # Never truncate the original request. generate() rejects an oversized fallback.
    return dict(request, prompt=prompt), included


def complete_object(text):
    value = text.strip()
    if value.startswith('```json'):
        value = value[7:].lstrip()
    try:
        obj, end = json.JSONDecoder().raw_decode(value)
    except ValueError:
        return None
    return value[:end] if isinstance(obj, dict) else None


def generate(generator, items, *, retry=False):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    inputs = [generator._inputs(r['prompt']) for _, r in items]
    lengths = [int(v['input_ids'].shape[1]) for v in inputs]
    if max(lengths) > generator.max_input_tokens:
        raise ValueError('Input exceeds explicit token limit; refusing silent truncation')
    width = max(lengths)
    context_limit = getattr(generator.model.config, 'max_position_embeddings', 40960)
    kinds = {r.get('_kind','flat') for _,r in items}
    if len(kinds)!=1:
        raise ValueError('A batch must use one step schema')
    budget = min(min(r.get('_max_new_tokens',output_budget(width)) for _,r in items), context_limit-width)
    if budget < 128:
        raise ValueError('Insufficient model context for complete output; no silent input truncation')
    time_limit = 900
    tokenizer = generator.tokenizer
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = torch.full((len(inputs), width), pad, dtype=torch.long, device=generator.device)
    mask = torch.zeros_like(ids)
    for i, v in enumerate(inputs):
        ids[i, -lengths[i]:] = v['input_ids'][0].to(generator.device)
        mask[i, -lengths[i]:] = 1

    class JsonComplete(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if (input_ids.shape[1]-width) % 16:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            return torch.tensor([complete_object(tokenizer.decode(row[width:], skip_special_tokens=True)) is not None
                                 for row in input_ids], dtype=torch.bool, device=input_ids.device)

    if any(r.get('_schema') is not None for _, r in items):
        filters = [prefix_function(generator, prompt_width=width, kind=r.get('_kind', 'flat'),
                                   schema_override=r.get('_schema')) for _, r in items]
        def enforced(batch_id, token_ids):
            return filters[batch_id](0, token_ids)
    else:
        enforced = prefix_function(generator,prompt_width=width,kind=next(iter(kinds)))
    started = time.perf_counter()
    with torch.inference_mode():
        output = generator.model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                    max_new_tokens=budget, max_time=time_limit, pad_token_id=pad,
                    prefix_allowed_tokens_fn=enforced,
                    stopping_criteria=StoppingCriteriaList([JsonComplete()]))
    elapsed = time.perf_counter()-started
    eos = generator.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {pad}
    rows = []
    for i, row in enumerate(output):
        tokens = row[width:].tolist()
        for j, token in enumerate(tokens):
            if token in eos:
                tokens = tokens[:j+1]
                break
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        obj = complete_object(text)
        finish = 'json_complete' if obj else ('token_limit' if len(tokens) >= budget else
                 'time_limit' if elapsed >= time_limit else 'eos_without_complete_json')
        rows.append({'response': obj or text, 'raw_response': text,
                     'status': 'generated' if obj else 'incomplete_output',
                     'finish_reason': finish, 'token_budget': budget, 'time_limit_seconds': time_limit,
                     'input_tokens': lengths[i], 'output_tokens': len(tokens),
                     'format_enforcement':'lm-format-enforcer-0.11.3-json-schema'})
    return rows



def execute(generator,items,run,rank,batch_id):
    from .workflow import execute as run_workflow
    return run_workflow(generator,items,run,rank,batch_id)
