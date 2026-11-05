"""CPU-side local model completeness check; never downloads model files."""
import json
from pathlib import Path
import sys


def check(path):
    path=Path(path).resolve()
    json.loads((path/'config.json').read_text())
    indexes=list(path.glob('*.index.json'))
    weights=list(path.glob('*.safetensors'))+list(path.glob('pytorch_model*.bin'))
    if not weights: raise ValueError('No local model weights: '+str(path))
    for index in indexes:
        for name in set(json.loads(index.read_text()).get('weight_map',{}).values()):
            shard=(path/name).resolve()
            if path not in shard.parents or not shard.is_file() or shard.stat().st_size==0:
                raise ValueError('Missing/invalid model shard: '+name)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    tokenizer.apply_chat_template([dict(role='user',content='test')],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    print('Local model shards and chat tokenizer ready:',path)


if __name__=='__main__': check(sys.argv[1])
