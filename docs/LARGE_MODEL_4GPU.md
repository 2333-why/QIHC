# 32B、四卡独立实验入口

## 2026-10-21：GPU / CPU 分离（优先遵循本节）

平台已确认按资源利用率自动回收。旧流水线在 GPU 作业内串行执行 CPU 搜索，可能使 GPU 持续空闲。本版 GPU 入口不再执行任何评分阶段，不靠空转维持利用率；不能保证平台不会因其他低利用率窗口回收，仍需确认管理员规则。

原实验无需重新准备模型和数据。GPU 续跑命令不变：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && QIHC_LARGE_RESUME=/inspire/hdd/global_user/czxs25210022/why/results/large32b_20261019T163631Z-20 bash scripts/run_large_4gpu.sh
```

在**获准使用的 CPU 计算节点/CPU 作业**运行评分（不要在限制计算的登录节点长跑）：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash scripts/score_large_cpu.sh /inspire/hdd/global_user/czxs25210022/why/results/large32b_20261019T163631Z-20
```

CPU 与 GPU 可以同时运行。CPU 每次取已完成推理阶段的快照，复用已有完整实例评分；尚未完成的推理标记 waiting_for_gpu 并退出，不无限等待。GPU 完成后再执行一次相同 CPU 命令，补齐其余组。

- `run.json` 由 GPU 独占写入；最终 `gpu_complete_cpu_scoring_pending` 表示 GPU 工作已完成，不是实验全部完成。
- `cpu_run.json` 和 `summary.json` 由 CPU 评分器写入；cpu_run 中 complete 才表示当前协议所有组评分完成。
- CPU 不加载模型权重、不调用 CUDA；评分日志每实例输出进度。两个作业各自加锁，避免重复启动同类任务。
- 原数据和训练设置保持不变，原本 running 的评分阶段在 GPU 状态中改为 deferred_to_cpu，旧结果不删除。

### 可选：扩大验证（不重新训练、不更改原测试集）

CPU 生成独立评估集，输出目录必须不存在：

```bash
/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128/bin/python scripts/large_model_pipeline.py prepare-extra --output /inspire/hdd/global_user/czxs25210022/why/data/large32b_extended_v1
```

新增 240 例：20/32/48/64 变量，每个尺寸包含5种单规则、10种三规则组合，每种4个实例。新随机实例不进入训练。这是合成组合/尺寸测试，不等价于真实业务泛化。单规则与三规则按尺寸分组汇总，避免只看混合平均。

在 GPU 的原续跑命令中再加环境变量：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && QIHC_EXTRA_DATA=/inspire/hdd/global_user/czxs25210022/why/data/large32b_extended_v1 QIHC_LARGE_RESUME=/inspire/hdd/global_user/czxs25210022/why/results/large32b_20261019T163631Z-20 bash scripts/run_large_4gpu.sh
```

原480例结果与新增240例文件分开（新增以 extra_ 开头），复用各组对应适配器；评估集哈希固定后不得修改。CPU 评分命令不变。若 CPU 评分早于 GPU 注册扩展集启动，其状态仅反映当时协议，请在 GPU 完成后再评分一次。

独立训练种子复验：可设置 `QIHC_TRAIN_SEED=43` 或44，**不设置旧 QIHC_LARGE_RESUME**，创建全新实验目录。旧实验种子42不能修改。此版本不宣称训练方法已改善，也未新增 p-bit 造数；先获得可信对照结果再决定算法修改。

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

完成阶段跳过；未完成训练从可用 checkpoint 续跑，未保存 checkpoint 时重跑该阶段；推理按实例续跑、搜索按完整实例续跑。不得在旧目录修改协议重跑。预测使用原子整文件保存；旧版预测文件若只有末尾未写完整，保留原始文件的 `.bak` 副本后恢复完整行；中间损坏或完整行非法仍报错。

### 2026-10-20 续跑加固

已兼容 `92056b8` 版本准备的清单，仅允许运行编排改动，模型/训练/数据/求解依赖必须与旧版保持一致。无需重下载权重、重新造数或重跑 CPU 准备。运行清单记录实际新源码指纹，但不修改旧 prepared.json。

阶段运行时每60秒向平台日志输出 HEARTBEAT，并写入 `heartbeat.json`（最后观测时间不是存活保证）；子进程失败记录阶段、退出码及日志末尾。可捕获 SIGTERM，但 SIGKILL、节点断电可能仍留下 running。增加单任务锁，避免同一准备目录上重复启动并发作业。此次未改变采样、训练、评估预算，也没有证据能确认此前终止的根因。

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
