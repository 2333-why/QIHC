"""Bounded real inference with per-sequence JSON stopping and explicit truncation."""
import hashlib
import json
import time
from tools.benchmarks import comparison as c
from tools.runtime.modeling_v3.protocol import output_budget


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
    budget = min(16384 if retry else output_budget(width), context_limit-width)
    if budget < 1024:
        raise ValueError('Insufficient model context for complete output; no silent input truncation')
    time_limit = 1800 if retry else 1200
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

    started = time.perf_counter()
    with torch.inference_mode():
        output = generator.model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                    max_new_tokens=budget, max_time=time_limit, pad_token_id=pad,
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
                     'input_tokens': lengths[i], 'output_tokens': len(tokens)})
    return rows


def execute(generator, items, run, rank, batch_id, retry=False):
    start = time.perf_counter()
    try:
        rows = generate(generator, items, retry=retry)
    except (generator.torch.cuda.OutOfMemoryError, ValueError) as exc:
        exc.__traceback__ = None
        oom = isinstance(exc, generator.torch.cuda.OutOfMemoryError)
        if oom:
            generator.torch.cuda.empty_cache()
        if len(items) > 1:
            c.save(run/'inference_events'/f'{batch_id}.json', {'event': 'batch_retry', 'seconds': time.perf_counter()-start,
                   'error': str(exc), 'batch_size': len(items), 'rank': rank})
            mid = len(items)//2
            execute(generator, items[:mid], run, rank, batch_id+'a', retry=retry)
            execute(generator, items[mid:], run, rank, batch_id+'b', retry=retry)
            return
        if oom:
            raise
        rows = [{'response': '', 'status': 'generation_error', 'error': str(exc)}]
    for i, row in enumerate(rows):
        row['generation_attempts'] = 1
        if row.get('status') == 'incomplete_output' and not retry:
            # Persist raw failed output separately, never expose it as a completed response.
            c.save(run/'generation_attempts'/f'{batch_id}-{i}-first.json', row)
            try:
                replacement = generate(generator, [items[i]], retry=True)[0]
            except generator.torch.cuda.OutOfMemoryError:
                generator.torch.cuda.empty_cache()
                c.save(run/'inference_events'/f'{batch_id}-retry-oom.json',
                       {'event': 'unpublished_batch_retry_oom', 'seconds': time.perf_counter()-start,
                        'rank': rank, 'batch_size': len(items)})
                raise  # Request remains pending; no false quality failure.
            except ValueError as exc:
                replacement = {'response': row['response'], 'status': 'incomplete_output',
                               'finish_reason': 'retry_error', 'error': str(exc),
                               'input_tokens': 0, 'output_tokens': 0}
            replacement['previous_finish_reason'] = row.get('finish_reason')
            replacement['input_tokens'] = replacement.get('input_tokens', 0)+row.get('input_tokens', 0)
            replacement['output_tokens'] = replacement.get('output_tokens', 0)+row.get('output_tokens', 0)
            replacement['generation_attempts'] = 2
            rows[i] = replacement
    elapsed = time.perf_counter()-start
    for (path, request), row in zip(items, rows):
        row.update(seconds=elapsed/len(items), batch_seconds=elapsed, batch_size=len(items),
                   batch_id=batch_id, replica=rank, execution='audited_modeling_v3',
                   request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        c.save(run/'responses'/path.name, row)
    print(f'REPLICA {rank}: saved {len(items)} in {elapsed:.1f}s; '+str([r['status'] for r in rows]), flush=True)
