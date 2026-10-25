"""Online CPU: verify installed grammar against the actual local tokenizer, without loading weights."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from .formatting import check_dependencies, prefix_function
from .experiment import initial_prompt, initial_phases, ARMS


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
    cases=json.loads((args.source/'cases.CPU_ONLY.json').read_text())
    largest=0
    for case,phase in ((case,phase) for case in cases for phase in initial_phases(ARMS)):
        messages=[{'role':'user','content':initial_prompt(case,phase)}]
        text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        length=len(tokenizer.encode(text,add_special_tokens=False))
        if length>32768:
            raise RuntimeError(f'Initial prompt {case["id"]} exceeds 32768 tokens; no silent truncation')
        largest=max(largest,length)
    print(json.dumps({'format_tokenizer_probe':'passed','schemas_checked':list(examples),'initial_prompts_checked':len(cases)*2,
                      'largest_initial_prompt_tokens':largest,'model':str(model),
                      'scope':'Tokenizer/schema compatibility only; real GPU generation and search gate still required'}),flush=True)


if __name__=='__main__':
    main()
