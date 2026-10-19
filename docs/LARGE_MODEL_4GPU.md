# 32B、四卡独立实验入口

默认模型为 `Qwen/Qwen3-32B`，稠密、BF16、LoRA，而非全参数训练。不同于现有 Qwen3.5-9B；跨模型结果不能解释为纯参数规模效应。官方模型说明：https://huggingface.co/Qwen/Qwen3-32B

## 申请资源

| 项目 | 配置 |
|---|---|
| GPU | 同一节点 4 × H100 80GB；1 个 worker，非 4 个独立 worker |
| 显存检查 | 每卡至少 70 GiB 空闲；不依赖 nvidia-smi |
| CPU/内存 | 建议至少 16 核、256GB 主机内存；这是配置建议，不是已测峰值 |
| 存储 | 建议额外预留 300–500GB，包含约 65GB 权重、缓存和训练检查点；检查实际配额 |
| 互联 | 建议节点内 NVLink；必须通过 NCCL collective 测试 |
| 作业时间 | 程序每次启动上限 96 小时，不保证这个时限完成；支持原目录续跑 |

训练使用 4 进程 ZeRO-3，微批次 1，梯度累积 1，全局 batch=4；上下文 8192。推理单进程 `device_map=auto` 跨四卡分层，输出上限 4096 token，不是四副本并行或 tensor parallel。

## 两条命令

前提：这些新代码已复制到服务器仓库；之前准备成功的 Miniconda、`envs/qihc-cu128` 继续使用。新脚本不是全新机器的环境安装器，不更新现有依赖或下载 9B。

联网 CPU 端：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash scripts/prepare_large_cpu.sh
```

默认 `HF_ENDPOINT=https://hf-mirror.com`，使用 `hf download`；默认不向第三方镜像发送 HF token。失败保留已下载分片；完成时固定实际 commit、核对远端可用的 SHA256、文件尺寸及 safetensors 索引。权重保存在 `models/Qwen3-32B`，准备清单在 `artifacts/large32b/prepared.json`；旧 prepared.json 不变。

离线 GPU 作业的唯一启动命令：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash scripts/run_large_4gpu.sh
```

**不要再包一层 torchrun。** 平台配置 1 个 worker、4 张卡，内部自行启动四个训练进程。自动激活 conda，禁用在线下载，验证代码、数据、环境版本和模型文件。

若中断，以日志输出的实际目录续跑：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && QIHC_LARGE_RESUME=/inspire/hdd/global_user/czxs25210022/why/results/实际运行目录 bash scripts/run_large_4gpu.sh
```

完成阶段跳过；未完成训练从可用 checkpoint 续跑，未保存 checkpoint 时重跑该阶段；推理按实例续跑、搜索按完整实例续跑。不得在旧目录修改协议重跑。部分损坏的预测 JSONL 会显式报错，不自动删数据。

## 本入口包含什么

- 720 合成实例：训练/验证/测试各 240。训练单规则；测试未见三规则组合，含较大尺寸。
- 两臂逐实例匹配 SFT 行数和 DPO 对数；训练各 192 SFT 更新、128 DPO 更新。实际数据量及筛选在 teacher_audit/budget 中记录。
- 直接解题 Base/SFT/DPO 与稀疏建模 Base/SFT/DPO，共六组预测。
- 建模预测由 p-bit 有反馈、p-bit 无反馈、Metropolis、CP-SAT 求解；采样器 4096 步×64链×3种子，CP-SAT 每例最多10秒。不是等墙钟预算。
- 真正训练前运行 NCCL/ZeRO-3 测试，以及数据中最长样本的一步全模型 SFT/DPO 显存预检。
- 推理记录输入/输出 token 和耗时；各阶段记录耗时；原始求解输出保存 objective、feasible、semantic_match。summary.json 汇总按实例平均的可行率，不能对不同实例目标直接求平均后比较优劣。

日志：`logs/prepare-large-*.log`、`logs/large-four-gpu-*.log`；实验目录内有阶段日志、`run.json`、`summary.json`、预测与求解 JSONL、适配器。

## 不应过度解释

本轮仍是语义教师后训练 + 求解后端比较，不是 p-bit 造数后训练；尚不含 LLM 自修正反馈、等 token 训练对照、直接 LLM 等时间多采样或真实 FPGA 能耗。匹配行数与更新数不等于匹配 FLOPs。现有模型/协议不声称覆盖任意自然语言优化问题。CPU 准备成功不代表四卡正式训练已验证；自动 GPU 预检不通过会停止并保留日志。
