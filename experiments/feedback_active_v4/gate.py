"""Source-only development health gate, never selects by test accuracy."""
from qihc.experiment_runtime.storage import load,save


def check(run,data,cfg):
    ids={c['id'] for c in load(data/'cases.public.json') if c['split']=='development'}
    rows=[load(p) for p in (run/'initial').glob('*.json')]
    rows=[r for r in rows if r['case_id'] in ids]
    compiled=sum(r.get('model') is not None for r in rows)
    checks=sum(bool(r.get('checks',{}).get('checks')) for r in rows)
    limits=cfg['development_gate'];n=len(ids)
    passed=len(rows)==n and compiled/n>=limits['compiled_min'] and checks/n>=limits['checks_min']
    save(run/'development_gate.json',dict(passed=passed,expected=n,compiled=compiled,with_checks=checks,thresholds=limits,
         scope='interface_health_only_not_semantic_accuracy'))
    if not passed:raise RuntimeError('Development interface gate failed; main experiment stopped. See development_gate.json')
