# Direct-LLM NL-CVRP baseline

This branch is a standalone study of whether one open model can solve a
constrained CVRP directly from a complete numerical instance and natural-
language requirements. It intentionally excludes QIHC/p-bit sampling, QUBO,
OR-Tools, HGS, greedy initial solutions, solution completion and repair.

The model must return `{"routes":[[customer_id,...],...]}`. The shared strict
verifier then checks every customer exactly once, fleet size, capacity and every
hard natural-language constraint. Malformed JSON or infeasible routes are
failures, not repaired outputs.

## Recommended model

Default: `Qwen/Qwen3.5-35B-A3B`.

It is a recent Apache-2.0 open-weight MoE model with 35B total / 3B active
parameters and native 262K context. It is text-capable but uses a multimodal
Transformers API, which this branch supports in text-only mode. Its BF16 files
are about 72 GB, so the default launcher uses eight 80 GB A100 GPUs with one
independent model replica per GPU. The formal launcher covers 30 cases spread
over 100--400 customers; use a short pilot first when moving to a new model
revision or runtime, then report strict feasibility over the complete set.


## 1. Checkout and setup

On a completely new server, first create the shared QIHC environment (this
also installs Miniforge, PyTorch, HGS and the default Qwen3.5 model):

```bash
cd /hdd/wl2/QIHC
bash scripts/s2e/pro6000_online_setup.sh
```

On a server where `/hdd/wl2/conda-envs/qihc` already exists, skip that step
and run the direct-solve setup below:

```bash
cd /hdd/wl2/QIHC
git fetch origin
git switch codex/llm-direct-cvrp
git pull --ff-only origin codex/llm-direct-cvrp

MODEL_ID=Qwen/Qwen3.5-35B-A3B \
bash scripts/direct_llm/setup.sh
```

The model is placed at `/hdd/wl2/models/Qwen--Qwen3.5-35B-A3B`. Downloads are
resumable and create `qihc_model_manifest.json`, which records the model identity,
requested revision, resolved revision when available and the checked shard list.

To fetch updates incrementally, use a **new experiment result directory** after
the model changes:

```bash
UPDATE_MODEL=1 MODEL_ID=Qwen/Qwen3.5-35B-A3B \
bash scripts/s2e/prepare_model.sh
```

For a formal run, inspect the manifest and pin its `resolved_revision` SHA with
`MODEL_REVISION=<SHA>` and a new `MODEL_DIR`.

## 2. Optional real-model smoke test

Ensure GPU 0 is idle, then:

```bash
source scripts/s2e/activate_qihc.sh
MODEL_DIR=/hdd/wl2/models/Qwen--Qwen3.5-35B-A3B \
python experiments/smoke_local_llm.py --model-path "$MODEL_DIR" --device cuda:0
```

## 3. Run the experiment

```bash
cd /hdd/wl2/QIHC
source scripts/s2e/activate_qihc.sh

export MODEL_ID=Qwen/Qwen3.5-35B-A3B
export MODEL_DIR=/hdd/wl2/models/Qwen--Qwen3.5-35B-A3B
export RUN_ROOT=/hdd/wl2/results/direct_llm_qwen35_cvrp_v1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

mkdir -p "$RUN_ROOT"
nohup setsid bash scripts/direct_llm/run_direct_cvrp.sh \
  > "$RUN_ROOT/launcher.log" 2>&1 < /dev/null &
echo $! | tee "$RUN_ROOT/launcher.pid"
```

Defaults: 30 official CVRPLIB X instances spread over 100--400 customers, twelve
controlled natural-language clauses per instance, and seeds `0 1 2`. The public
CVRPLIB solution is used only offline to create satisfiable clauses and is never
exported to the model or used as an incumbent.

Progress:

```bash
tail -n 40 "$RUN_ROOT/launcher.log"
tail -n 40 "$RUN_ROOT/logs/direct_llm.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Primary report:

```bash
python -m json.tool "$RUN_ROOT/direct_llm_report.json"
```

## 4. Package results for transfer

After completion:

```bash
bash scripts/direct_llm/package_results.sh "$RUN_ROOT"
```

This writes `<RUN_ROOT>.tar.gz` plus a `.sha256` checksum alongside the result
directory. The archive contains experiment data, model revision manifest, logs,
records, failure reports and summaries; it never contains model weights or cache.
