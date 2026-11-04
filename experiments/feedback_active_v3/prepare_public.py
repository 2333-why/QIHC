"""Formal PUBLIC-ONLY cohort. Synthetic generators are test fixtures only."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil

from qihc.experiment_runtime.storage import load, save
from experiments.public_solver_v1.data import download, load_cases, REPO, REVISION
from .data import digest, public_view
from .public_data import prepare as prepare_external

HERE = Path(__file__).parent


def select(cases, label, count):
    groups = defaultdict(list)
    for case in cases: groups[(case['family'], case['layer'])].append(case)
    selected = []
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda c: digest([20261108, label, c['id']]))
        selected += ordered[:len(ordered) * count // len(cases)]
    chosen = {c['id'] for c in selected}
    remaining = sorted((c for c in cases if c['id'] not in chosen), key=lambda c: digest([20261108, label, c['id']]))
    return sorted(c['id'] for c in selected + remaining[:count-len(selected)])


def prepare(output):
    original = output / 'public_originals/nlco'
    download(original)
    cases, excluded = load_cases(original)
    if len(cases) != 900 or excluded: raise ValueError('Pinned six-task NLCO cohort incomplete')
    groups = defaultdict(list)
    for c in cases:
        _, family, tier = c['dataset'].split('_')
        c.update(family=family, tier=tier, layer=c['surface_format'],
                 variable_names=[v['name'] for v in c['gold']['variables']],
                 provenance=dict(source='https://huggingface.co/datasets/'+REPO, revision=REVISION,
                                 original_id=c['original_id'], public_dataset=True,
                                 prior_project_exposure='E003; not an unseen benchmark'))
        groups[(family, tier)].append(c)
    for (_, tier), values in sorted(groups.items()):
        ordered = sorted(values, key=lambda c: digest([20261108, 'split', c['id']]))
        dev = 12 if tier == 'S' else 8 if tier == 'M' else 0
        for i, c in enumerate(ordered): c['split'] = 'scale' if tier == 'L' else 'development' if i < dev else 'main'
    main = [c for c in cases if c['split'] == 'main']
    subsets = {name: select(main, name, count) for name,count in [('ablation',120),('audit',120),('replay',60)]}
    subsets['second'] = []
    public = [dict(public_view(c), family=c['family'], layer=c['layer'], split=c['split']) for c in cases]
    save(output/'cases.public.json', public)
    save(output/'evaluation/cases.gold.json', cases)
    save(output/'subsets.json', subsets)
    # Reuse downloaded, pinned original files; no need to redownload during setup.
    cached = HERE/'assets/dataset/public_originals'
    for path in cached.glob('*.jsonl'):
        target = output/'public_originals'/path.name
        if not target.exists(): shutil.copyfile(path, target)
    external = prepare_external(output)
    manifest = dict(version='E004_2.1-public-only-v3', formal_public_only=True,
        cases=len(cases), counts=dict(Counter(c['split'] for c in cases)),
        public_sha256=digest(public), gold_sha256=digest(cases), subsets_sha256=digest(subsets),
        nlco_source=load(original/'download_manifest.json'), external_cases=external['supported_cases'],
        scope='NLCO six-task 900-row subset plus CP-Bench 10 adapted cases; not full NLCO leaderboard',
        synthetic_formal_instances=0, second_model_enabled=False,
        historical_exposure='NLCO used in E003; dev/main disjoint here, not previously unseen benchmark')
    save(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=HERE/'assets/public_dataset')
    print(prepare(p.parse_args().output))
