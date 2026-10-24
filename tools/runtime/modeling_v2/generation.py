"""Bounded real inference with per-sequence JSON stopping and explicit truncation."""
import hashlib
import json
import time
from tools.benchmarks import comparison as c


def complete_object(text):
    value = text.strip()
    if value.startswith('```json'):
        value = value[7:].lstrip()
    try:
        obj, end = json.JSONDecoder().raw_decode(value)
    except ValueError:
        return None
    return value[:end] if isinstance(obj, dict) else None


def generate(generator, items):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    inputs = [generator._inputs(r['prompt']) for _, r in items]
    lengths = [int(v['input_ids'].shape[1]) for v in inputs]
    if max(lengths) > generator.max_input_tokens:
        raise ValueError('Input exceeds explicit token limit; refusing silent truncation')
    width = max(lengths)
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

    with torch.inference_mode():
        output = generator.model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                    max_new_tokens=4096, max_time=600, pad_token_id=pad,
                    stopping_criteria=StoppingCriteriaList([JsonComplete()]))
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
        rows.append({'response': obj or text, 'raw_response': text,
                     'status': 'generated' if obj else 'incomplete_output',
                     'finish_reason': 'json_complete' if obj else 'eos_or_token_or_time_limit',
                     'input_tokens': lengths[i], 'output_tokens': len(tokens)})
    return rows


def execute(generator, items, run, rank, batch_id):
    start = time.perf_counter()
    try:
        rows = generate(generator, items)
    except (generator.torch.cuda.OutOfMemoryError, ValueError) as exc:
        exc.__traceback__ = None
        oom = isinstance(exc, generator.torch.cuda.OutOfMemoryError)
        if oom:
            generator.torch.cuda.empty_cache()
        if len(items) > 1:
            c.save(run/'inference_events'/f'{batch_id}.json', {'event': 'batch_retry', 'seconds': time.perf_counter()-start,
                   'error': str(exc), 'batch_size': len(items), 'rank': rank})
            mid = len(items)//2
            execute(generator, items[:mid], run, rank, batch_id+'a')
            execute(generator, items[mid:], run, rank, batch_id+'b')
            return
        if oom:
            raise
        rows = [{'response': '', 'status': 'generation_error', 'error': str(exc)}]
    elapsed = time.perf_counter()-start
    for (path, request), row in zip(items, rows):
        row.update(seconds=elapsed/len(items), batch_seconds=elapsed, batch_size=len(items),
                   batch_id=batch_id, replica=rank, execution='validated_compact_v2',
                   request_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        c.save(run/'responses'/path.name, row)
    print(f'REPLICA {rank}: saved {len(items)} in {elapsed:.1f}s; '+str([r['status'] for r in rows]), flush=True)
