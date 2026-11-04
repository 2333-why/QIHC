"""Bounded, least-loaded local vLLM endpoints with durable attempt accounting."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx

from .storage import load, save, value_digest


class InferencePool:
    def __init__(self, endpoints, directory, config, model='qihc', tokenizer=None):
        self.endpoints = endpoints; self.directory = Path(directory); self.config = config
        self.model = model; self.tokenizer = tokenizer
        self.slots = asyncio.Queue()
        for _ in range(config['requests_per_gpu']):
            for e in endpoints: self.slots.put_nowait(e.rstrip('/'))
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(config['request_timeout'], connect=10),
                                       limits=httpx.Limits(max_connections=len(endpoints) * config['requests_per_gpu']))
        self.locks = {}

    async def close(self):
        await self.client.aclose()

    async def generate(self, key, prompt, seed, temperature=None):
        async with self.locks.setdefault(key, asyncio.Lock()):
            return await self._generate(key, prompt, seed, temperature)

    async def _generate(self, key, prompt, seed, temperature):
        cfg = self.config; body = dict(model=self.model, messages=[dict(role='user', content=prompt)],
            temperature=cfg['temperature'] if temperature is None else temperature, seed=int(seed),
            max_tokens=cfg['max_output_tokens'], chat_template_kwargs={'enable_thinking': False},
            response_format={'type': 'json_object'})
        fingerprint = value_digest(body); path = self.directory / (key + '.json')
        if path.exists():
            row = load(path)
            if row['request_hash'] != fingerprint: raise ValueError('Immutable request mismatch: ' + key)
            if row['status'] == 'in_flight':
                # Unknown outcome is spent, not silently retried or counted free.
                row.update(status='interrupted_unknown', error='Previous attempt interrupted; no automatic extra call')
                save(path, row)
            return row
        started = time.monotonic(); endpoint = await self.slots.get()
        row = dict(request_hash=fingerprint, seed=seed, endpoint=endpoint, status='in_flight',
                   started_at=time.time(), queue_seconds=time.monotonic() - started, attempts=0,
                   input_tokens=0, output_tokens=0, text='')
        try:
            if self.tokenizer is not None:
                tokens = self.tokenizer.apply_chat_template(body['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
                row['input_tokens'] = len(tokens)
                if len(tokens) + cfg['max_output_tokens'] > cfg['max_model_len']:
                    row.update(status='context_overflow', error='Input not truncated; exceeds declared context')
                    save(path, row); return row
            row['attempts'] = 1; save(path, row)
            service = time.monotonic()
            response = await self.client.post(endpoint + '/v1/chat/completions', json=body)
            response.raise_for_status(); data = response.json(); choice = data['choices'][0]
            usage = data.get('usage', {})
            row.update(status='complete' if choice['finish_reason'] == 'stop' else 'truncated',
                       text=choice['message'].get('content') or '', finish_reason=choice['finish_reason'],
                       input_tokens=usage.get('prompt_tokens', row['input_tokens']), output_tokens=usage.get('completion_tokens', 0),
                       service_seconds=time.monotonic() - service)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            row.update(status='request_failed', error=str(exc)[:1600])
        finally:
            row['total_seconds'] = time.monotonic() - started
            save(path, row); self.slots.put_nowait(endpoint)
        return row
