"""CPU-only tokenizer/schema and complete first-call input checks."""
import json
from types import SimpleNamespace
from qihc.experiment_runtime import storage as s
from qihc.modeling import builder
from qihc.modeling.formatting import check_runtime, prefix_function
from . import study


def check_tokenizer(tokenizer, requests):
    import torch
    examples = {
        'flat': {'model': {'schema': 'bounded-integer-v2', 'variables': {'count': 32, 'prefix': 'x', 'lower': 0, 'upper': 1},
                          'objective': {'sense': 'max', 'linear': [1]*32, 'quadratic': []}, 'constraints': [], 'logic': []}},
        'plan': {'variables': {'count': 32, 'prefix': 'x', 'lower': 0, 'upper': 1},
                 'requirements': [{'id': 'o', 'kind': 'objective'}, {'id': 'd', 'kind': 'domain'}]},
        'objective': {'objective': {'sense': 'min', 'linear': [1]*32, 'quadratic': []}},
        'rules': {'items': [{'requirement_id': 'r', 'constraints': [{'id': 'c', 'a': [1]*32, 'op': 'le', 'rhs': 1}], 'logic': []}]},
        'patch': {'notes': [], 'rule_replacements': []},
    }
    generator = SimpleNamespace(tokenizer=tokenizer)
    for kind, example in examples.items():
        fn = prefix_function(generator, prompt_width=2, kind=kind)
        sequence = [tokenizer.eos_token_id]*2
        for token in tokenizer.encode(json.dumps(example, separators=(',', ':')), add_special_tokens=False):
            if token not in fn(0, torch.tensor(sequence, dtype=torch.long)):
                raise RuntimeError('Real tokenizer/schema probe failed: '+kind)
            sequence.append(token)
        if tokenizer.eos_token_id not in fn(0, torch.tensor(sequence, dtype=torch.long)):
            raise RuntimeError('Schema did not allow EOS: '+kind)
    lengths = {}; exceeded = []
    for request in requests:
        kind = 'flat' if request['frontend'] == 'flat' else 'plan'
        prompt = builder.prompt(kind, request)
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
        size = len(tokenizer.encode(text, add_special_tokens=False))
        lengths.setdefault(request['phase'], []).append(size)
        if size > 32768:
            exceeded.append({'case': request['case_id'], 'arm': request['phase'], 'tokens': size})
    if exceeded:
        raise ValueError('No input truncation; oversized prompts: '+json.dumps(exceeded))
    return {'schemas_checked': list(examples), 'first_prompts_checked': sum(map(len, lengths.values())),
            'by_arm': {k: {'count': len(v), 'max_tokens': max(v), 'mean_tokens': sum(v)/len(v)} for k, v in lengths.items()}}


def prepare(root, source):
    with s.worker_lock(root/'launch.lock'), s.worker_lock(root/'prepare.lock'):
        print(json.dumps(check_runtime()), flush=True)
        manifest = study.prepare(root, source)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(manifest['model'], local_files_only=True, trust_remote_code=False)
        print('Checking actual tokenizer/schema and all five prompt arms; weights not loaded', flush=True)
        requests = (s.load(p) for p in sorted((root/'formal/requests').glob('*.json')))
        result = check_tokenizer(tok, requests)
        examples = root/'prompt_examples'; examples.mkdir(exist_ok=True)
        for arm in manifest['config']['arms']:
            case = study.gate_cases(s.load(root/'formal/cases.CPU_ONLY.json'))[0]
            request = study.request(case, arm)
            kind = 'flat' if request['frontend'] == 'flat' else 'plan'
            (examples/(arm+'.txt')).write_text(builder.prompt(kind, request), encoding='utf-8')
        result.update(status='passed', source_hash=manifest['source_hash'],
                      scope='CPU preflight only; real GPU generation/search acceptance remains required')
        s.save(root/'preflight.json', result)
        if not (root/'record.md').exists():
            (root/'record.md').write_text('# Prompt efficiency experiment\n\nInitial status: prepared; current status is in status.json.\n\n'
              +'Code fingerprint: '+manifest['source_hash']+'\n\n'
              +'Git commit: '+manifest['code_commit']+'\n\n'
              +'Model: '+manifest['model']+'\n\n'
              +'Configuration: manifest.json. Prompt checks: preflight.json.\n'
              +'Runtime status: status.json; logs: logs/; gate and formal reports: gate/report.md and formal/report.md.\n'
              +'GPU requests contain original text and shared variable layout, never gold objectives/constraints/answers.\n', encoding='utf-8')
        print(json.dumps(result), flush=True)
        print('CPU READY. Launch 01_run_4gpu.sh in a single worker with four GPUs; CPU host can stop.', flush=True)
