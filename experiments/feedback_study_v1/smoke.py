"""CPU deployment acceptance: real backend imports, source patches, and probes."""
import json
from .common import s, HERE
from .data import make_case
from .protocol import apply_patch
from qihc.generic.feedback import sample, scip_pool


def main():
    from qihc.generic.solver_baselines.isolation import BackendPools
    c=make_case('MAXCUT',5,1,'smoke'); spec=c['gold']
    assert apply_patch(spec,{'edits':[],'quote':'','reason':'retain'},c['question'])==spec
    for method in ('pbit','metropolis'):
        row=sample(spec,.1,1,chains=4,method=method,clamps={0:1})
        assert all(x[0]==1 for x in row['states'])
    # Separate OS processes avoid CP-SAT/SCIP/HiGHS library symbol collisions.
    import subprocess,sys
    for backend in ('cp_sat','scip'):
        code=('from qihc.generic.solver_baselines.registry import solve; '
              'from experiments.feedback_study_v1.data import make_case; '
              f'r=solve(make_case("MAXCUT",5,1,"smoke")["gold"],"{backend}",2,0); '
              'assert r["feasible"],r; print(r["status"])')
        subprocess.run([sys.executable,'-c',code],check=True)
    row=scip_pool(spec,1,0,clamps={0:1})
    assert row['states'] and all(x[0]==1 for x in row['states'])
    print(json.dumps({'status':'passed','scope':'CPU actual solvers and conditional kernels; not real LLM acceptance'}))


if __name__=='__main__':
    main()
