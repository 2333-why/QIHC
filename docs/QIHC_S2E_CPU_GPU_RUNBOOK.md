# QIHC-S²E：联网双 RTX PRO 6000 正式实验手册

当前部署为单台可联网服务器，不再区分 CPU 联网端和 GPU 离线端。默认资源：

- 工作区：`/hdd/wl2`
- 仓库：`/hdd/wl2/QIHC`
- Miniforge：`/hdd/wl2/miniforge3`
- Conda 环境：`/hdd/wl2/conda-envs/qihc`
- 模型：`/hdd/wl2/models/Qwen--Qwen2.5-32B-Instruct`
- HGS-CVRP：`/hdd/wl2/runtime/src/HGS-CVRP`
- 结果：`/hdd/wl2/results/<RUN_ID>`
- GPU：`CUDA_VISIBLE_DEVICES=0,1`，`torchrun --nproc_per_node=2`

## 1. 首次克隆

```bash
export WORK_ROOT=/hdd/wl2
mkdir -p "$WORK_ROOT"

if [ ! -d "$WORK_ROOT/QIHC/.git" ]; then
  git clone --branch feature/nl-cvrp-formal \
    https://github.com/2333-why/QIHC.git \
    "$WORK_ROOT/QIHC"
else
  git -C "$WORK_ROOT/QIHC" fetch origin
  git -C "$WORK_ROOT/QIHC" switch feature/nl-cvrp-formal
  git -C "$WORK_ROOT/QIHC" pull --ff-only
fi

cd "$WORK_ROOT/QIHC"
```

## 2. 一次性联网安装

安装脚本会完成 Miniforge、Python 3.11 环境、CUDA 12.8 PyTorch、训练依赖、32B 模型、HGS-CVRP 编译、双卡 NCCL/BF16 自检和 pytest：

脚本通过仓库内的 `configs/condarc-pro6000.yaml` 强制只使用 `conda-forge`，不会继承服务器用户目录中可能指向 `repo.anaconda.com` 的 Conda channel。

```bash
cd /hdd/wl2/QIHC
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/s2e/pro6000_online_setup.sh
```

若只希望先用较小模型验证全流程：

```bash
cd /hdd/wl2/QIHC
MODEL_ID=Qwen/Qwen2.5-14B-Instruct \
MODEL_DIR=/hdd/wl2/models/Qwen--Qwen2.5-14B-Instruct \
bash scripts/s2e/pro6000_online_setup.sh
```

## 3. 每次重新登录

```bash
export WORK_ROOT=/hdd/wl2
source "$WORK_ROOT/QIHC/scripts/s2e/activate_qihc.sh"
export CUDA_VISIBLE_DEVICES=0,1
```

## 4. 小规模端到端验证

先关闭后训练并缩小实例与采样规模：

```bash
cd /hdd/wl2/QIHC

RUN_ID=pro6000_smoke \
INSTANCE_LIMIT=4 \
LNS_ITERATIONS=3 \
SAMPLING_STEPS=50 \
NUM_CHAINS=32 \
HYBRID_STEPS=20 \
HYBRID_CHAINS=32 \
RUN_FEEDBACK_ABLATION=0 \
RUN_TRAINING=0 \
bash scripts/s2e/gpu_run_formal.sh
```

## 5. 正式实验

```bash
cd /hdd/wl2/QIHC

export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export INSTANCE_LIMIT=100
export LNS_ITERATIONS=100
export SAMPLING_STEPS=1000
export NUM_CHAINS=2048
export RUN_FEEDBACK_ABLATION=1
export RUN_TRAINING=1

nohup bash scripts/s2e/gpu_run_formal.sh \
  > /hdd/wl2/s2e_pro6000_launcher.log 2>&1 &

echo $! > /hdd/wl2/s2e_pro6000.pid
tail -f /hdd/wl2/s2e_pro6000_launcher.log
```

监控：

```bash
nvidia-smi
ps -fp "$(cat /hdd/wl2/s2e_pro6000.pid)"
du -sh /hdd/wl2/results/*
```

## 6. 可调参数

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `MODEL_ID` | `Qwen/Qwen2.5-32B-Instruct` | 首次安装时下载的模型 |
| `MODEL_DIR` | `/hdd/wl2/models/Qwen--Qwen2.5-32B-Instruct` | 推理和训练使用的本地权重 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | 两张 PRO 6000 |
| `NPROC_PER_NODE` | `2` | 每卡一个分布式进程 |
| `INSTANCE_LIMIT` | `100` | 正式实例数量上限 |
| `RUN_FEEDBACK_ABLATION` | `1` | 是否运行无 p-bit→LLM logits 反馈消融 |
| `RUN_TRAINING` | `1` | 是否执行 SFT、DPO、GRPO |
| `HF_WORKERS` | `8` | 联网模型下载并发数 |

RTX PRO 6000 Blackwell 是 compute capability 12.0，因此安装脚本固定使用支持 Blackwell 的 PyTorch 2.8.0 CUDA 12.8 wheel。环境自检会验证两张 GPU、BF16 矩阵核和 NCCL all-reduce，而不是只检查 `nvidia-smi`。
