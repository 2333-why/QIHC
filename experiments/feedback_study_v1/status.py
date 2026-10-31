"""Cheap live status from CPU login node; no model loading or CUDA needed."""
import argparse
import json
from pathlib import Path
import time
from .common import s, default_root


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=default_root());a=p.parse_args()
    for name in ('status.json','acceptance.json','gpu-0.json','gpu-1.json','gpu-2.json','gpu-3.json'):
        f=a.run/name
        if f.exists():
            row=s.load(f)
            print(name,json.dumps(row,ensure_ascii=False))
            print('  file_age_seconds=',round(time.time()-f.stat().st_mtime,1))
    for rank in range(4):
        print('pending_gpu',rank,len(list((a.run/'pending'/str(rank)).glob('*.json'))))
    print('request_count',len(list((a.run/'requests').glob('*.json'))))
    print('response_count',len(list((a.run/'responses').glob('*.json'))))
    print('completed_scores',len(list((a.run/'evaluation'/'scores').glob('*.json'))))
    print('File age is a liveness clue, not proof that a platform job still exists.')


if __name__=='__main__':
    main()
