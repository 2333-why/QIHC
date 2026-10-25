"""Online CPU: verify installed grammar against the actual local tokenizer, without loading weights."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from .formatting import check_dependencies, prefix_function
from .protocol import modeling_prompt


def main():
    base=Path(os.environ.get('QIHC_BASE','/inspire/hdd/global_user/czxs25210022/why'))
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source',type=Path,default=Path(os.environ.get('QIHC_DISTRIBUTION_SOURCE',str(base/'results/co_comparison_v1'))))
    args=cli.parse_args()
    check_dependencies()
    manifest=json.loads((args.source/'manifest.json').read_text())
    model=Path(manifest['settings']['model'])
    from transformers import AutoTokenizer
    import torch
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    print('Building format tokenizer index (no weights loaded)...',flush=True)
    fn=prefix_function(SimpleNamespace(tokenizer=tokenizer),prompt_width=3)
    example={'model':{'schema':'bounded-integer-v2','variables':{'count':32,'prefix':'x','lower':0,'upper':1},
             'objective':{'sense':'max','linear':[1]*32,'quadratic':[]},'constraints':[],'logic':[]},
             'requirements':[{'id':'domain','kind':'domain','expected_count':32,
                              'targets':['variable:x'+str(i) for i in range(32)],'quote':'test'}]}
    # Mimic a padded prompt prefix followed by actual tokenizer-produced output tokens.
    sequence=[tokenizer.eos_token_id]*3
    for token in tokenizer.encode(json.dumps(example,separators=(',',':')),add_special_tokens=False):
        allowed=fn(0,torch.tensor(sequence,dtype=torch.long))
        if token not in allowed:
            raise RuntimeError('Actual tokenizer/grammar compatibility probe failed; do not launch GPU')
        sequence.append(token)
    if tokenizer.eos_token_id not in fn(0,torch.tensor(sequence,dtype=torch.long)):
        raise RuntimeError('Grammar did not allow EOS after complete model')
    cases=json.loads((args.source/'cases.CPU_ONLY.json').read_text())
    largest=0
    for case in cases:
        messages=[{'role':'user','content':modeling_prompt(case)}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        length=len(tokenizer.encode(text,add_special_tokens=False))
        if length>32768:
            raise RuntimeError(f'Initial prompt {case["id"]} exceeds 32768 tokens; no silent truncation')
        largest=max(largest,length)
    print(json.dumps({'format_tokenizer_probe':'passed','initial_prompts_checked':len(cases),
                      'largest_initial_prompt_tokens':largest,'model':str(model),
                      'scope':'Tokenizer/schema compatibility only; real GPU generation and search gate still required'}),flush=True)


if __name__=='__main__':
    main()
