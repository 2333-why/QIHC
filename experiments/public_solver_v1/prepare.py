"""Prepare independent E003; oversize/invalid individual prompts become skips."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from qihc.experiment_runtime import resilient_storage as s
from .common import prepare, default_root
from .modeling import failure


def main():
    base = Path(os.environ.get('QIHC_BASE', '/inspire/hdd/global_user/czxs25210022/why'))
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, default=default_root())
    p.add_argument('--data', type=Path, default=base/'data/E003-NLCO')
    p.add_argument('--model', type=Path, default=Path(os.environ.get('QIHC_MODEL', str(base/'models/Qwen3-32B'))))
    a = p.parse_args(); m = prepare(a.run, a.data, a.model)
    from qihc.modeling.formatting import check_runtime, parser, prefix_function
    from qihc.modeling.typed_schema import schema
    from qihc.modeling.typed_workflow import Job
    from transformers import AutoTokenizer
    print(check_runtime(), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    lengths, failures, probed = [], [], set()
    generator = SimpleNamespace(tokenizer=tokenizer)
    for path in sorted((a.run/'requests').glob('*.json')):
        r = s.load(path)
        try:
            nvars = len(r['variable_names'])
            syntax = schema('flat', r['question'], nvars)
            parser('flat', syntax)
            if r['queue_group'] not in probed:
                fn = prefix_function(generator, prompt_width=0, kind='flat', schema_override=syntax)
                probe = {'model': {'schema':'bounded-integer-v2',
                    'variables':{'count':nvars,'prefix':'x','lower':0,'upper':1},
                    'objective':{'sense':'min','linear':{},'quadratic':[]}, 'constraints':[], 'logic':[]}}
                sequence = []
                for token in tokenizer.encode(json.dumps(probe, separators=(',', ':')), add_special_tokens=False):
                    if token not in fn(0, sequence):
                        raise RuntimeError('Tokenizer/schema integration failed; fix environment before GPU launch')
                    sequence.append(token)
                if tokenizer.eos_token_id not in fn(0, sequence):
                    raise RuntimeError('Tokenizer/schema EOS rejected')
                probed.add(r['queue_group'])
            first_prompt = Job(path, r).task()['prompt']
            text = tokenizer.apply_chat_template([{'role': 'user', 'content': first_prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
            n = len(tokenizer(text, add_special_tokens=False)['input_ids']); lengths.append(n)
            if n > 32768:
                raise ValueError(f'Input has {n} tokens; exceeds 32768, no silent truncation')
        except (ValueError, TypeError, KeyError) as exc:
            failure(a.run, path, 'CPU preflight: '+str(exc)); failures.append({'case_id': r['case_id'], 'error': str(exc)})
    s.save(a.run/'preflight.json', {'status': 'complete', 'source_hash': m['source_hash'],
        'requests': len(m['requests']), 'max_input_tokens': max(lengths, default=0),
        'skips': failures, 'tokenizer_schema_groups_probed': sorted(probed), 'weights_loaded': False,
        'scope': 'No semantic/gold gate; real generation remains for GPU.'})
    print('E003 CPU READY. Canonical solver track can start now; GPU modeling is independent.', flush=True)


if __name__ == '__main__':
    main()
