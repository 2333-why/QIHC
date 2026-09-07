# QIHC-S²E 正式实验：联网 CPU 端与离线 4×H100 端

## 固定目录

```bash
export WORK_ROOT=/inspire/hdd/project/wuliqifa/yanjunchi-24040/why
export GLOBAL_ROOT=/inspire/hdd/global_user/yanjunchi-24040/qihc
export CONDA_ROOT=$WORK_ROOT/miniforge3
export REPO_DIR=$WORK_ROOT/QIHC
export BUNDLE_ROOT=$GLOBAL_ROOT/offline_bundle_s2e
export MODEL_DIR=$GLOBAL_ROOT/models/Qwen--Qwen2.5-32B-Instruct
```

模型、Hugging Face 缓存、pip 缓存和 wheelhouse 放在个人全局目录；代码、conda 环境和实验结果放在项目个人目录。

## 一、联网 CPU 端

首次执行或代码更新：

```bash
export WORK_ROOT=/inspire/hdd/project/wuliqifa/yanjunchi-24040/why
export REPO_DIR=$WORK_ROOT/QIHC
mkdir -p "$WORK_ROOT"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch feature/nl-cvrp-formal https://github.com/2333-why/QIHC.git "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin
  git -C "$REPO_DIR" switch feature/nl-cvrp-formal
  git -C "$REPO_DIR" pull --ff-only
fi

cd "$REPO_DIR"
bash scripts/s2e/cpu_connected_prepare.sh
```

该脚本会补齐 qihc conda 环境、CUDA PyTorch、TRL/PEFT、离线 wheelhouse、HGS-CVRP 源码和 Qwen2.5-32B-Instruct。下载中断时直接重新执行，Hugging Face 会续传。

每次重新登录联网 CPU 端：

```bash
export WORK_ROOT=/inspire/hdd/project/wuliqifa/yanjunchi-24040/why
export GLOBAL_ROOT=/inspire/hdd/global_user/yanjunchi-24040/qihc
source "$WORK_ROOT/QIHC/scripts/s2e/activate_qihc.sh"
```

提交并上传代码：

```bash
cd /inspire/hdd/project/wuliqifa/yanjunchi-24040/why/QIHC
git status --short
git add qihc/s2e experiments scripts/s2e requirements-training.txt docs/QIHC_S2E_CPU_GPU_RUNBOOK.md
git commit -m "feat: implement verified QIHC-S2E pipeline"
git push origin feature/nl-cvrp-formal
```

## 二、离线 4×H100 端

若代码目录由共享存储直接可见，只需拉取动作在联网 CPU 端完成；GPU 端不执行任何网络命令。

首次安装与验证：

```bash
export WORK_ROOT=/inspire/hdd/project/wuliqifa/yanjunchi-24040/why
export GLOBAL_ROOT=/inspire/hdd/global_user/yanjunchi-24040/qihc
cd "$WORK_ROOT/QIHC"
bash scripts/s2e/gpu_offline_install.sh
```

每次重新登录离线 GPU 端：

```bash
export WORK_ROOT=/inspire/hdd/project/wuliqifa/yanjunchi-24040/why
export GLOBAL_ROOT=/inspire/hdd/global_user/yanjunchi-24040/qihc
source "$WORK_ROOT/QIHC/scripts/s2e/activate_qihc.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi
```

先执行不训练的 smoke：

```bash
cd "$REPO_DIR"
INSTANCE_LIMIT=8 LNS_ITERATIONS=5 RUN_TRAINING=0 \
RUN_ID=s2e_smoke bash scripts/s2e/gpu_run_formal.sh
```

正式实验：

```bash
cd "$REPO_DIR"
INSTANCE_LIMIT=100 \
LNS_ITERATIONS=100 \
SFT_STEPS=500 \
DPO_STEPS=300 \
GRPO_STEPS=300 \
RUN_TRAINING=1 \
RUN_ID=s2e_formal_$(date +%Y%m%d_%H%M%S) \
nohup bash scripts/s2e/gpu_run_formal.sh \
  > "$WORK_ROOT/s2e_formal_launcher.log" 2>&1 &

echo $! > "$WORK_ROOT/s2e_formal.pid"
tail -f "$WORK_ROOT/s2e_formal_launcher.log"
```

监控：

```bash
watch -n 2 nvidia-smi
ps -fp "$(cat "$WORK_ROOT/s2e_formal.pid")"
du -sh "$WORK_ROOT/results"/*
```

结果默认写入 `$WORK_ROOT/results/$RUN_ID`，结束后同时生成 `.tar.gz` 和 `.sha256`。把结果带回联网 CPU 端后，可提交小型 JSON/图表；不要把模型权重、大型日志和原始 checkpoint 推入 Git。

## 实验阶段

1. 生成 CPP 并运行 A0–A4 验证。
2. 编译 QUBO、p-dit 或 MFC 表示选择计划。
3. 运行 random、KNN、LLM 选邻域的 p-bit-LNS 对照。
4. 由 gold CPP、错误输出和验证结果构造 SFT/DPO/GRPO 数据。
5. 三种后训练从相同基座模型独立开始，保证消融比较公平。
6. 归档全部配置、分片记录、summary、日志与 checksum。
