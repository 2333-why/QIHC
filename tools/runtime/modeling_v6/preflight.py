"""Online CPU: verify installed grammar against the actual local tokenizer, without loading weights."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from .formatting import check_runtime, prefix_function
from .experiment import initial_prompt, initial_phases, ARMS
from .builder import prompt as step_prompt


def check_prompt_lengths(tokenizer, cases, limit=32768):
    """Measure actual first calls for both frontends; never truncate source text."""
    largest = {}
    exceeded = []
    checked = 0
    for case in cases:
        for phase in initial_phases(ARMS):
            kind = 'flat' if phase == 'model_flat' else 'plan'
            prompt = step_prompt(kind, {'prompt': initial_prompt(case, phase)})
            messages = [{'role': 'user', 'content': prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
            length = len(tokenizer.encode(text, add_special_tokens=False))
            row = {'case_id': case['id'], 'phase': phase, 'tokens': length}
            if phase not in largest or length > largest[phase]['tokens']:
                largest[phase] = row
            if length > limit:
                exceeded.append(row)
            checked += 1
    summary = {'initial_prompts_checked': checked, 'input_token_limit': limit,
               'largest_initial_prompt_tokens': max((r['tokens'] for r in largest.values()), default=0),
               'largest_by_frontend': largest, 'over_limit': exceeded}
    print(json.dumps({'prompt_length_check': summary}), flush=True)
    if exceeded:
        raise RuntimeError(f'{len(exceeded)}/{checked} initial prompts exceed {limit} tokens; '
                           'see prompt_length_check for case IDs, frontends and actual lengths; no silent truncation')
    return summary


def main():
    base=Path(os.environ.get('QIHC_BASE','/inspire/hdd/global_user/czxs25210022/why'))
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path(os.environ.get('QIHC_DISTRIBUTION_SOURCE',str(base/'results/co_comparison_v1'))))
    args=cli.parse_args()
    print(json.dumps(check_runtime()),flush=True)
    manifest=json.loads((args.source/'manifest.json').read_text(encoding='utf-8'))
    model=Path(manifest['settings']['model'])
    from transformers import AutoTokenizer
    import torch
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    print('Building format tokenizer index (no weights loaded)...',flush=True)
    example={'model':{'schema':'bounded-integer-v2','variables':{'count':32,'prefix':'x','lower':0,'upper':1},
             'objective':{'sense':'max','linear':[1]*32,'quadratic':[]},'constraints':[],'logic':[]},
             'requirements':[{'id':'domain','kind':'domain','expected_count':32,
                              'targets':['variable:x'+str(i) for i in range(32)],'quote':'test'}]}
    # Mimic a padded prompt prefix followed by actual tokenizer-produced output tokens.
    examples={'flat':example,'plan':{'variables':example['model']['variables'],
              'requirements':[{'id':'d','kind':'domain'},{'id':'o','kind':'objective'}]},
              'objective':{'objective':example['model']['objective']},
              'rules':{'items':[{'requirement_id':'r','constraints':[['c',{},'le',1]],'logic':[]}]},
              'patch':{'notes':[],'rule_replacements':[]}}
    generator=SimpleNamespace(tokenizer=tokenizer)
    for kind,value in examples.items():
        fn=prefix_function(generator,prompt_width=3,kind=kind)
        sequence=[tokenizer.eos_token_id]*3
        for token in tokenizer.encode(json.dumps(value,separators=(',',':')),add_special_tokens=False):
            allowed=fn(0,torch.tensor(sequence,dtype=torch.long))
            if token not in allowed:
                raise RuntimeError('Actual tokenizer/grammar compatibility probe failed: '+kind)
            sequence.append(token)
        if tokenizer.eos_token_id not in fn(0,torch.tensor(sequence,dtype=torch.long)):
            raise RuntimeError('Grammar did not allow EOS after '+kind)
    cases=json.loads((args.source/'cases.CPU_ONLY.json').read_text(encoding='utf-8'))
    lengths=check_prompt_lengths(tokenizer,cases)
    print(json.dumps({'format_tokenizer_probe':'passed','schemas_checked':list(examples),**lengths,'model':str(model),
                      'scope':'Tokenizer/schema compatibility only; real GPU generation and search gate still required'}),flush=True)


if __name__=='__main__':
    main()
