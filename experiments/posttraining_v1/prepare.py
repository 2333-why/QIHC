"""CPU preparation reads the shared environment without installing into it."""
import hashlib
import sys
from importlib.metadata import version
from qihc.experiment_runtime.identity import model_identity
from qihc.experiment_runtime.storage import digest, load, save, worker_lock, value_digest
from experiments.feedback_active_v3.check_local_model import check
from .data import RUN, MODEL, prepare


def main():
    import torch, transformers, accelerate, pyscipopt
    from .engine import Engine
    from .learn import parsed
    from experiments.feedback_active_v3.cpu import execute
    with worker_lock(RUN/"prepare.lock"):
        check(MODEL)
        prepare()
        files=model_identity(MODEL)
        for row in files:
            if row["sha256"] is None:
                with (MODEL/row["path"]).open("rb") as f:
                    row["sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
        identity=dict(model=str(MODEL.resolve()),files=files,identity=value_digest(files),
                      env=sys.executable,packages={k:version(k) for k in ("torch","transformers","accelerate","pyscipopt","numpy")})
        ready=RUN/"prepared.json"
        if ready.exists() and load(ready)!=identity:
            raise RuntimeError("Different environment or model; use a separate run")
        row=load(RUN/"train.json")[0]
        result=execute(dict(stage="solve",model=row["gold"],backend="scip",seconds=2,chains=32,seed=17))
        if result.get("x") is None: raise RuntimeError("SCIP runtime preflight failed: "+str(result))
        save(ready,identity)
        from .energy_data import prepare_energy
        prepare_energy()
        settings=load(RUN/"energy/settings.json")
        inputs=[RUN/"energy/settings.json"]+[RUN/"energy/preferences"/(c+".json") for c in settings["pair_ids"]]
        save(RUN/"energy/inputs.json",{p.relative_to(RUN).as_posix():digest(p) for p in inputs})
        print("E005_EIGHT_CPU_READY",RUN)


if __name__=="__main__": main()
