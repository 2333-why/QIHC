# E002-1.1 独立运行

2026-10-28 按用户要求将两项实验分开。以下为当前推荐入口；后面的联合命令仅保留作历史复现。

| 操作 | 独立入口 |
|---|---|
| 联网 CPU 只安装 E002 求解器依赖 | `bash experiments/solver_comparison_v1/05_prepare_standalone_cpu.sh` |
| CPU 计算节点启动/续跑 E002，建议 32 核，无需 GPU | `bash experiments/solver_comparison_v1/06_run_standalone_cpu.sh` |
| 只更新 E002 报告 | `bash experiments/solver_comparison_v1/07_report_standalone.sh` |

独立结果目录为 `$BASE/results/E002-1.1_solver_standalone`。E002 只读 `$BASE/results/E001-1.3_prompt_fixed` 的 `legacy_flat`、`semantic_staged` 冻结建模输出；两个来源分别统计。完整 300 题对应 600 份模型响应（包含建模失败记录）。响应未齐会在计算前报出缺少数量，不等待、不启动 E001、不占 GPU。生成成本仍从原响应读取，七个求解器不重复调用 LLM。

E001 自己使用固定代码版本的 `03_prepare_fixed_cpu.sh`、`04_run_fixed_4gpu.sh`，报告调用其 `02_report.sh` 并设置 `QIHC_PROMPT_V3_RUN` 到新结果路径。可先准备两个环境，随后运行 E001，建模输出齐全后再启动 E002。

旧联合结果 `E002-1.1_solver_comparison` 保留；若已经在跑，继续原固定工作树，不能用新代码混合旧检查点。新独立入口保留相同求解预算、种子和方法，属于同一实验配置的独立运行方式。

## 历史联合入口（新运行不推荐）

完整设置与公平性：[E002-1.1 记录](../../docs/experiments/E002_solver_comparison/E002-1.1_record.md)。

从固定提交创建独立工作树后，在其根目录顺序执行：

```bash
# 联网 CPU：隔离安装两个开源求解器、真实 CPU 冒烟、格式/分词器预检；完成即退出
bash experiments/solver_comparison_v1/00_prepare_cpu.sh

# 一个 worker、4×H100、80 CPU 核：自动启用项目 Python，启动两份 LLM 副本与 CPU 评分
bash experiments/solver_comparison_v1/01_run_4gpu.sh

# 中断后在相同工作树/环境/结果路径执行同一命令续跑
bash experiments/solver_comparison_v1/01_run_4gpu.sh

# 自动报告每 30 秒更新；另一个 CPU 终端也可手动读取/更新
bash experiments/solver_comparison_v1/02_report.sh
```

默认 BASE 为 `/inspire/hdd/global_user/czxs25210022/why`。默认环境 `envs/qihc-cu128`，复用已准备的 Qwen3-32B 权重与 `results/co_comparison_v1` 数据，不重新下载模型。

- E001-1.3 新结果：`$BASE/results/E001-1.3_prompt_fixed`。
- E002-1.1 新结果：`$BASE/results/E002-1.1_solver_comparison`。
- 求解器依赖：`$BASE/deps/e002-solvers-highs1151-scip630`。
- 可用 `QIHC_BASE`、`QIHC_ENV_PREFIX`、`QIHC_FORMAT_DEPS`、`QIHC_SOLVER_DEPS`、`QIHC_PROMPT_V3_RUN`、`QIHC_E002_RUN` 覆盖对应路径；准备与运行必须一致。

联合命令保持 E001 的真实模型验收：通过后才放行全部建模。E002 逐步读取正式阶段的两个预先指定建模来源，七个求解器都复用其冻结模型，不增加 LLM 调用。新实验不写旧 v1/v2/v3 结果；代码/模型/配置变化触发拒绝混跑。

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
cat "$BASE/results/E001-1.3_prompt_fixed/status.json"
cat "$BASE/results/E001-1.3_prompt_fixed/gate_acceptance.json"
cat "$BASE/results/E002-1.1_solver_comparison/status.json"
cat "$BASE/results/E002-1.1_solver_comparison/report.md"
```

若 E001 已完成而 E002 尚有 CPU 搜索尾项，可在 CPU 计算节点仅续跑以下命令；不需要联网、不需要 GPU。不要同时在两个节点启动同一个 E002 run（锁会拒绝）。

```bash
bash experiments/solver_comparison_v1/03_run_cpu_replay.sh
```

四卡实际用于 E001 模型推理；七个后端本轮都是 CPU 实现。模型生成与 CPU 搜索并行，队列按数据集交错，避免长时间只处理某一类题。不会用无效矩阵计算或显存填充制造利用率，不能承诺平台不回收。收到停止信号保存检查点；同一命令恢复。

打包两个实验，保留模型输出、失败和分种子证据（不含权重）：

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
tar -czf "$BASE/E001-1.3_E002-1.1_analysis_$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S).tar.gz" \
  -C "$BASE/results" E001-1.3_prompt_fixed E002-1.1_solver_comparison
```
