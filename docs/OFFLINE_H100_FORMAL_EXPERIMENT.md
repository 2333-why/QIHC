# QIHC-LNS：4×H100 离线正式实验操作手册

## 1. 目标环境

- 离线计算节点：80 核 CPU、约 900 GB 内存、4×NVIDIA H100 80GB；
- 联网资源端：Linux x86_64，建议与离线端使用相同 Python 次版本；
- 代码交互：GitHub ↔ 联网资源端，联网端再生成 `git bundle` 供离线端读取；
- 大文件交互：模型、wheelhouse、外部数据和结果通过获准的共享盘或文件摆渡区传输，不放入普通 Git 历史。

正式实验采用一卡一进程。`torchrun` 产生四个 rank，每个 rank：

1. 在本地 H100 上加载一份离线 LLM；
2. 负责互不重复的 `(instance, search_seed, method)` 子任务；
3. 使用同一张 H100 执行 batched p-bit；
4. 每完成一个任务立即追加 `results_rankN.jsonl`，支持故障后保留已完成结果。

## 2. 第一次部署：本地开发端

确认改动后创建分支并推送：

```bash
cd /path/to/QIHC
git status --short
python -m pytest -q

git switch -c feature/nl-cvrp-formal
git add \
  .gitignore pyproject.toml requirements-formal.txt \
  qihc/problems qihc/ising/batched.py qihc/ising/__init__.py \
  experiments/run_cvrp_lns.py \
  experiments/analyze_cvrp_results.py \
  experiments/merge_cvrp_results.py \
  experiments/evaluate_constraint_ir.py \
  experiments/prepare_nlcvrp_jsonl.py \
  scripts/offline docs
git commit -m "feat: add offline multi-GPU QIHC-LNS experiments"
git push -u origin feature/nl-cvrp-formal
```

不要执行 `git add -A`，避免把历史实验输出或模型文件误加入版本库。

## 3. 第一次部署：联网 CPU 资源端

以下命令下载代码、Linux wheels、CUDA PyTorch、模型、HGS 和 RoutBench，并生成离线包。

```bash
export WORK_ROOT=/data/$USER/qihc-transfer
export REPO_DIR=$WORK_ROOT/QIHC
export BUNDLE_DIR=$WORK_ROOT/offline_bundle
mkdir -p "$WORK_ROOT"

git clone --branch feature/nl-cvrp-formal \
  https://github.com/2333-why/QIHC.git "$REPO_DIR"
cd "$REPO_DIR"

python3 --version
uname -m

export PYTHON_BIN=python3
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124

bash scripts/offline/prepare_connected.sh "$BUNDLE_DIR"

sha256sum --check "$BUNDLE_DIR.tar.gz.sha256"
ls -lh "$BUNDLE_DIR.tar.gz" "$BUNDLE_DIR.tar.gz.sha256"
```

注意：

- 联网端和离线端应同为 Linux x86_64，并使用相同 Python 次版本，例如均为 Python 3.11；
- `cu124` PyTorch wheel 要求离线端 NVIDIA 驱动兼容 CUDA 12.4；若驱动条件不同，在联网端修改 `TORCH_INDEX_URL`；
- 模型路径完全下载后才能打包，不能只复制 Hugging Face 缓存中的符号链接；
- 首次包可能达到数十 GB，不要通过普通 Git 上传。

将以下两个文件通过获准的摆渡区送往离线服务器：

```text
offline_bundle.tar.gz
offline_bundle.tar.gz.sha256
```

## 4. 第一次部署：离线 H100 服务器

假设摆渡文件位于 `/transfer/incoming`：

```bash
export TRANSFER_DIR=/transfer/incoming
export WORK_ROOT=/data/$USER/qihc
mkdir -p "$WORK_ROOT"

cd "$TRANSFER_DIR"
sha256sum --check offline_bundle.tar.gz.sha256
tar -xzf offline_bundle.tar.gz -C "$WORK_ROOT"

git clone "$WORK_ROOT/offline_bundle/qihc.git.bundle" "$WORK_ROOT/QIHC"
cd "$WORK_ROOT/QIHC"
git log -1 --oneline
cat "$WORK_ROOT/offline_bundle/QIHC_COMMIT.txt"

export PYTHON_BIN=python3
bash scripts/offline/install_offline.sh \
  "$WORK_ROOT/offline_bundle" \
  "$WORK_ROOT/QIHC/.venv-formal"
```

安装脚本会：

- 校验离线包内所有文件；
- 从 wheelhouse 安装 CUDA PyTorch 和其余依赖；
- 以 editable 模式安装当前 QIHC；
- 使用 80 核 CPU 编译 HGS-CVRP；
- 检查 CUDA 和四张 GPU；
- 运行全部单元测试。

## 5. 正式运行前的手工检查

```bash
cd /data/$USER/qihc/QIHC
source .venv-formal/bin/activate

nvidia-smi
python - <<'PY'
import os
import torch

print("cpu_count", os.cpu_count())
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu_count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, torch.cuda.get_device_name(i), p.total_memory / 2**30, "GiB")
assert torch.cuda.device_count() == 4
PY

MODEL_DIR=/data/$USER/qihc/offline_bundle/models/Qwen--Qwen2.5-7B-Instruct
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/tokenizer_config.json"
```

## 6. 小规模 GPU pilot

先用一张 GPU 跑一个短实验，确认模型加载、CUDA p-bit、结果落盘和统计脚本都正常：

```bash
cd /data/$USER/qihc/QIHC
source .venv-formal/bin/activate

export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR=/data/$USER/qihc/offline_bundle/models/Qwen--Qwen2.5-7B-Instruct
export PILOT_OUT=/data/$USER/qihc/formal_outputs/pilot_$(date +%Y%m%d_%H%M%S)

python experiments/run_cvrp_lns.py \
  --dataset synthetic \
  --output "$PILOT_OUT" \
  --sizes 25 \
  --instance-seeds 0 1 \
  --search-seeds 0 \
  --methods greedy random knn llm \
  --sampler torch \
  --iterations 5 --patience 5 \
  --destroy-size 6 --routes-per-customer 3 \
  --sampling-steps 100 --num-chains 256 --top-samples 16 \
  --model-path "$MODEL_DIR"

python experiments/analyze_cvrp_results.py "$PILOT_OUT" \
  --reference knn --bootstrap-samples 1000

cat "$PILOT_OUT/failures.json"
cat "$PILOT_OUT/summary.json"
cat "$PILOT_OUT/statistical_report.json"
```

通过条件：

- `failures.json` 为空；
- 四种方法均有结果；
- 所有最终解 `feasible=true`；
- `llm_audit_rank0.jsonl` 中结构化解析成功率可接受；
- H100 显存稳定，没有 CUDA OOM。

## 7. 四卡正式实验

完整脚本运行 Track A 和 Track B：

```bash
cd /data/$USER/qihc/QIHC
source .venv-formal/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/data/$USER/qihc/hf-cache
export CPU_WORKERS=40

export BUNDLE_ROOT=/data/$USER/qihc/offline_bundle
export OUTPUT_ROOT=/data/$USER/qihc/formal_outputs
export MODEL_DIR=$BUNDLE_ROOT/models/Qwen--Qwen2.5-7B-Instruct

mkdir -p "$OUTPUT_ROOT"
nohup bash scripts/offline/run_formal_4xh100.sh \
  "$BUNDLE_ROOT" "$OUTPUT_ROOT" "$MODEL_DIR" \
  > "$OUTPUT_ROOT/formal_launcher.log" 2>&1 &

echo $! > "$OUTPUT_ROOT/formal_launcher.pid"
tail -f "$OUTPUT_ROOT/formal_launcher.log"
```

正式配置包含：

- 自然语言约束到 Constraint IR 的 schema 合法率、exact match 和 micro-F1；
- HGS/OR-Tools 默认使用 40 个 CPU 进程并行，GPU 方法使用 4 个进程；
- 最多 100 个 CVRPLIB 实例；
- 每实例 5 个搜索随机种子；
- HGS、OR-Tools、greedy、random+p-bit、kNN+p-bit、LLM+p-bit；
- 每个 QUBO 2,048 条并行 p-bit 链、1,000 个采样步；
- 100 轮 LNS、30 轮无改进早停；
- LLM 每 5 轮刷新一次，其余轮次复用候选；
- Track B 使用已知可行解构造同车、互斥、先后约束。

预计时间取决于实例规模和模型生成速度。先根据 pilot 计算单任务耗时，再决定是否将 `--limit`、迭代数或随机种子数分批运行。正式统计必须保持同一批实例和种子。

## 8. 监控与故障恢复

资源监控：

```bash
watch -n 2 nvidia-smi
htop
tail -f /data/$USER/qihc/formal_outputs/formal_launcher.log
```

定位失败任务：

```bash
find /data/$USER/qihc/formal_outputs -name failures.json -print -exec cat {} \;
find /data/$USER/qihc/formal_outputs -name 'results_rank*.jsonl' -print
```

当前 runner 每次启动会重写 rank shard。若要补跑失败任务，应新建输出目录并缩小 `--data/--limit/--search-seeds/--methods`，不要覆盖原始正式输出。最后再在联网端按 `(instance, seed, method)` 去重合并。

## 9. 后续代码更新：联网端到离线端

联网端先从 GitHub 获取新提交，再创建完整分支 bundle：

```bash
export REPO_DIR=/data/$USER/qihc-transfer/QIHC
export UPDATE_DIR=/data/$USER/qihc-transfer/updates
mkdir -p "$UPDATE_DIR"

cd "$REPO_DIR"
git fetch origin
git switch feature/nl-cvrp-formal
git pull --ff-only

STAMP=$(date +%Y%m%d_%H%M%S)
git bundle create "$UPDATE_DIR/qihc-$STAMP.bundle" feature/nl-cvrp-formal
sha256sum "$UPDATE_DIR/qihc-$STAMP.bundle" \
  > "$UPDATE_DIR/qihc-$STAMP.bundle.sha256"
```

摆渡 bundle 和校验文件后，在离线服务器执行：

```bash
export UPDATE_BUNDLE=/transfer/incoming/qihc-YYYYMMDD_HHMMSS.bundle
cd /data/$USER/qihc/QIHC

sha256sum --check "$UPDATE_BUNDLE.sha256"
git status --short
git fetch "$UPDATE_BUNDLE" feature/nl-cvrp-formal:refs/remotes/offline-update/main
git merge --ff-only refs/remotes/offline-update/main

source .venv-formal/bin/activate
python -m pip install --no-deps --editable .
python -m pytest -q
```

如果 `git status --short` 出现代码文件修改，先停止更新并保存差异；不要执行 `git reset --hard`。

## 10. 结果返回离线端到联网端

正式脚本结束时会生成：

```text
formal_YYYYMMDD_HHMMSS.tar.gz
formal_YYYYMMDD_HHMMSS.tar.gz.sha256
```

在离线服务器确认：

```bash
cd /data/$USER/qihc/formal_outputs
sha256sum --check formal_YYYYMMDD_HHMMSS.tar.gz.sha256
```

摆渡至联网端后：

```bash
export RESULT_ARCHIVE=/transfer/incoming/formal_YYYYMMDD_HHMMSS.tar.gz
sha256sum --check "$RESULT_ARCHIVE.sha256"
mkdir -p /data/$USER/qihc-results
tar -xzf "$RESULT_ARCHIVE" -C /data/$USER/qihc-results

find /data/$USER/qihc-results -name failures.json -print -exec cat {} \;
find /data/$USER/qihc-results -name statistical_report.json -print -exec cat {} \;
```

原始结果压缩包不建议提交 Git。Git 中只提交经过审计的小型汇总表、绘图脚本和实验 manifest，并在提交信息中记录原始压缩包的 SHA-256。

## 11. 正式实验判定标准

在下结论前至少满足：

- 所有方法使用相同实例、种子和 wall-clock/评估预算；
- 最终解由确定性验证器判定可行；
- 报告 BKS gap、可行率、TTT、候选覆盖率、端到端时间和 p-bit 时间；
- LLM 结构化解析失败和 kNN fallback 单独计数；
- 采用实例级配对比较和 bootstrap 95% 置信区间；
- 分开报告标准 CVRP 和自然语言约束 CVRP；
- 负结果保留：如果低 Ising 能量没有转化为更低真实路线成本，不得只报告能量或可行率。
