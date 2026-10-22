# 组合优化对照：安全可运行子集 v1

## 当前状态（不是全部论文基线复现完成）

| 项目 | 状态 |
|---|---|
| 基础 Qwen3-32B 直接解、LLM+CP-SAT、等轮数 CP-SAT 反馈、QIHC p-bit 反馈 | 已实现；不后训练、不运行消融 |
| NLCO | 固定 79438eb01b3d14e23bc945fd6b31854048c1c6b4，KP/MIS/MVC/MAXCUT，各 S/M/L 前 20 题，共 240 |
| 原有合成题 | 背包/顶点覆盖/分配/平衡割，60 题 |
| ReEvo/EoH | 仅登记固定版本，暂缓下载和运行；Docker 接口未实现 |
| LLMCoSolver | 仅登记固定版本，暂缓下载和运行；待核实权重及适配 |
| CO-Bench | 仅登记固定版本，暂缓下载和运行；未完成评测适配 |

NLCO 是允许工具、统一二进制输出的子集评测，不是官方禁用工具排行榜。
并非全 43 类支持；不含连续优化，不声称处理任意问题。
题面保留，附加 x0..xn 与题面标识符的输出约定，各组一致。
专用适配只用于隐藏金标准验证，LLM 建模和 p-bit 搜索仍使用通用接口。
不把参考解、参考目标、隐藏约束传给 LLM；反馈仅取预测模型的搜索信息。
参考 witness 必须通过独立模型约束与目标核对。样例先选择再审计，错误列入 coverage.json，不补选容易样例。

## CPU（联网计算节点，保留运行）

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash tools/benchmarks/01_co_cpu.sh
```

复用 why/envs/qihc-cu128 与 why/models/Qwen3-32B，不重建旧环境。
模型缺失时固定版本从 HF 镜像下载，失败尝试官方源；已有模型先进行权重头与 tokenizer 校验。
NLCO 已发布题目在作者 GitHub 仓库中，不调用 Gurobi 重造数据。
GitHub 源使用固定 commit，没有声称存在已验证的国内 GitHub 镜像；不能访问时需为 Git 配置可用代理。
出现 `CPU ready` 后启动 GPU。CPU 不是仅下载后退出：它持续处理搜索和反馈请求，请放在 CPU 计算配额中，不要在受限登录节点长期运行。

## GPU（单 worker，多张卡可见，一行命令）

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash tools/benchmarks/02_co_gpu.sh
```

自动使用上述环境；离线推理；不用 torchrun，不依赖 nvidia-smi。
四卡默认两副本，每副本两卡均衡放置模型，每批 4 条真实请求（合计最多 8 条）；不是 DDP 训练。
可用 QIHC_INFERENCE_REPLICAS、QIHC_INFERENCE_BATCH 显式调整。OOM 自动拆小批次，单条仍 OOM 则停止并保留请求。
实际 32B/四卡加载仍需服务器验证；不能保证 GPU/显存利用率达到某个阈值。
GPU 无请求超过 45 秒安全退出，状态为 waiting_for_cpu_resubmit_same_command；CPU 继续生成请求后，重新提交同一 GPU 命令。不是自动重提作业。
不以空转矩阵运算规避平台管理规则，也不能保证平台绝不回收。中断后复用原目录续跑已保存请求。
源码或模型身份改变则拒绝混用旧结果；修改配置请用 QIHC_CO_RUN 指定新目录。

## 结果与口径

默认 why/results/co_comparison_v1/：

- summary.json：完成量、独立可行率、可行实例先平均种子的参考 gap、LLM tokens/时间和搜索时间。
- final/ 与 progress/：逐题逐阶段结果；requests/ 与 responses/：可恢复推理队列。
- coverage.json：数据校验与排除；baseline_availability.json：明确论文方法未运行原因。
- gpu_status.json；日志 why/logs/co-cpu-*、comparison-gpu-*。

gap 是相对公开参考值的有符号 gap，不能无条件称“最优 gap”；负值不截断。
金标准 CP-SAT/退火/随机组获得了真实模型，必须与自然语言端到端组分栏。
两轮反馈共享相同首轮 LLM 输出；CP-SAT 5 秒单线程；p-bit 3 种子、16 链、4096 步。
这些是固定预算配置，不声称二者等墙钟计算量；成本完整记录，后续可增加等时间曲线。

## 从 cebfb1b 的中断实验续跑

1. 停止旧 GPU 作业；CPU 终端按 Ctrl+C，等待当前搜索阶段保存后退出（不要直接更新运行中的源码）。
2. `git pull --ff-only origin llm-cvrp`。
3. 原 CPU 命令重新运行。它只允许已审计 cebfb1b 源码指纹迁移，核对模型、题目、响应/request 哈希，备份 manifest.before_parallel.json。
4. 出现 CPU ready 后提交原 GPU 命令；不删除 responses/、progress/ 或 final/。

模型、题面、解码上限、搜索预算不变；批量/设备布局可能产生浮点数值差异。
manifest 的 execution_migration 记录迁移；新响应标记 execution、replica、batch_id。
旧响应耗时保留，新批次墙钟时间均摊到请求；并行副本的时间和不是作业墙钟时间。
OOM 失败批次耗时在 inference_events/ 单独记录，不可在效率结论中忽略。
完整性能比较建议之后另开全量同执行配置的 run；本次保留旧样本的结果用于续跑验证。

每批刷新队列并优先处理反馈，不再等 600 个初始请求全部生成。
每副本分片互斥；GPU 整体锁防止重复启动。终止时未保存的批次会重算，已保存请求跳过。
不能保证一次作业完成：真实请求耗尽时仍主动释放 GPU，CPU 产生后续请求后重提同一命令。

查看真实阶段进度（不需要 GPU，不只统计 final）：

```bash
/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128/bin/python tools/benchmarks/co_status.py --run /inspire/hdd/global_user/czxs25210022/why/results/co_comparison_v1
```

gpu_replica_0.json / gpu_replica_1.json 为最近批次状态；placement 文件记录模型放置。
memory 中 allocated/reserved 是 PyTorch 分配数据，不是平台 GPU 计算利用率；状态文件不代表进程仍存活。
