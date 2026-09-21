# QIHC 八卡 A100 实验交接

这份文件是新执行人员的唯一入口。默认工作区为 `/hdd/wl2`，模型权重、缓存、环境和结果都放在该工作区，不写入系统盘。

## 1. 获取固定代码

```bash
mkdir -p /hdd/wl2
cd /hdd/wl2
git clone --branch feature/nl-cvrp-formal \
  https://github.com/2333-why/QIHC.git QIHC
cd QIHC
git log -1 --oneline
```

已有仓库则执行：

```bash
cd /hdd/wl2/QIHC
git pull --ff-only origin feature/nl-cvrp-formal
```

## 2. 首次准备环境

```bash
cd /hdd/wl2/QIHC
bash scripts/s2e/pro6000_online_setup.sh
```

脚本创建 `/hdd/wl2/conda-envs/qihc`，安装正式依赖、编译 HGS、运行测试，并默认准备 `Qwen/Qwen3.5-35B-A3B`。该模型是 35B 总参数、3B 激活参数的开放权重 MoE；项目以文本方式使用它的通用语言模型能力，每张 80 GB A100 各加载一份模型并行处理不同实例。

## 3. 单独下载、续传或更新模型

模型保存在 `/hdd/wl2/models`，不会提交进 Git。

首次下载或检查已有分片：

```bash
cd /hdd/wl2/QIHC
MODEL_ID=Qwen/Qwen3.5-35B-A3B \
bash scripts/s2e/prepare_model.sh
```

检查远端 `main` 并增量更新；完整分片会从缓存复用：

```bash
UPDATE_MODEL=1 \
MODEL_ID=Qwen/Qwen3.5-35B-A3B \
bash /hdd/wl2/QIHC/scripts/s2e/prepare_model.sh
```

正式论文实验应固定不可变提交。先查看
`/hdd/wl2/models/Qwen--Qwen3.5-35B-A3B/qihc_model_manifest.json`
中的 `resolved_revision`，之后用该 SHA 重新准备一个固定目录：

```bash
export MODEL_REVISION=<resolved_revision_SHA>
export MODEL_DIR=/hdd/wl2/models/Qwen--Qwen3.5-35B-A3B-pinned
bash /hdd/wl2/QIHC/scripts/s2e/prepare_model.sh
```

旧的 `Qwen2.5-32B-Instruct` 不删除，可通过显式设置 `MODEL_DIR` 重现实验。

## 4. 每次重新登录

```bash
source /hdd/wl2/QIHC/scripts/s2e/activate_qihc.sh
```

首次交接检查（GPU 0 空闲时包含真实模型加载）：

```bash
RUN_MODEL_SMOKE=1 SMOKE_GPU=0 \
bash /hdd/wl2/QIHC/scripts/s2e/verify_handoff.sh
```

若 GPU 忙，只做非侵入式检查：

```bash
bash /hdd/wl2/QIHC/scripts/s2e/verify_handoff.sh
```

## 5. 正式困难实例对照实验

此实验使用官方 CVRPLIB X 系列 400--1000 客户实例，受控加入自然语言约束；从空路线开始，不向求解器暴露公开 witness 路线。默认使用 2048 条 p-bit 并行链并比较完整 QIHC、去反馈、KNN+p-bit、随机+p-bit、Oracle、Greedy、OR-Tools、HGS 和不修复的 LLM 直接求解。

每次改变模型、p-bit 链数或实验参数都必须使用新的 `RUN_ROOT`：

```bash
source /hdd/wl2/QIHC/scripts/s2e/activate_qihc.sh
export MODEL_DIR=/hdd/wl2/models/Qwen--Qwen3.5-35B-A3B-pinned
export RUN_ROOT=/hdd/wl2/results/qihc_x400_1000_qwen3coder_v1
export NUM_CHAINS=2048
export TOP_SAMPLES=128
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8

mkdir -p "$RUN_ROOT"
nohup setsid bash /hdd/wl2/QIHC/scripts/s2e/run_hard_cvrp_comparison.sh \
  > "$RUN_ROOT/launcher.log" 2>&1 < /dev/null &
echo $! | tee "$RUN_ROOT/launcher.pid"
```

检查状态：

```bash
export RUN_ROOT=/hdd/wl2/results/qihc_x400_1000_qwen3coder_v1
ps -fp "$(cat "$RUN_ROOT/launcher.pid")" || true
tail -n 40 "$RUN_ROOT/launcher.log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader
```

最终主报告：

```bash
python -m json.tool "$RUN_ROOT/method_comparison.json"
```

关键日志在 `$RUN_ROOT/logs/`；每个方法目录均包含 `manifest_rank*.json`、`results.json`、`failures.json` 和 `summary.json`。失败任务也计入最终可行率，不得只汇报成功样本。

## 6. 仓库边界

- `qihc/`：算法与问题实现。
- `experiments/`：数据准备、实验入口和汇总器。
- `scripts/s2e/`：环境、模型、交接检查和正式流水线。
- `tests/`：无需 GPU 的回归测试。
- `docs/HARD_DUAL_TRACK_EXPERIMENT.md`：实验定义、数据泄漏边界和指标解释。
- 模型、缓存、CVRPLIB 数据和实验结果均位于 `/hdd/wl2`，不进入 Git。

遇到异常时先保存：当前 Git commit、`qihc_model_manifest.json`、对应的 `manifest_rank0.json`、`failures.json` 和日志末尾 200 行。不要直接覆盖已有结果目录。
