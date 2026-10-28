# E001 历史运行快照 001：v1 验收失败，v2 正式阶段状态陈旧

- 接收日期：2026-10-28（Asia/Shanghai）。同一 E001 大实验内补证据，不新增算法迭代。
- 来源：[用户粘贴原文](E001_20261028_runtime_001_source.txt)；[结构化状态与校验](E001_20261028_runtime_001_metrics.json)。
- 原文 SHA-256：`aaf909fa61269b73ddae49aff4e2d12eda83e5fe11a8b0e3b97300af968a63ab`。
- 只收到 v1/v2 的共享存储状态及 GPU 子日志末尾，没有平台主进程完整日志、容器退出码或 v3 目录信息。

## 分轮判断

| 迭代 | 本次证据 | 判断 |
|---|---|---|
| E001-1.1 / v1 | status=failed，`Execution gate failed: ['legacy_staged']`；两个模型副本加载成功并保存子步骤 | 已知执行验收失败，主程序按门槛主动停止。不是本次新发现的模型加载错误；正式阶段未启动，与此前分析一致 |
| E001-1.2 / v2 | phase=formal，134/1500 responses；134/1500 completed_case_arms；日志有 flat/plan/objective/rules/local_repair 的已发布子步骤 | 确实进入正式评测并保存部分结果；134 是题目×方法组合，不是 134 个不同原题，也不代表这些解全部正确 |
| E001-1.3 / v3 | 输出中没有对应目录标题、status 或日志 | 本次没有 v3 运行证据。可能默认目录不存在、使用自定义 run 或输出不完整；不能直接标记 v3 失败或从未运行 |

两个 GPU 副本均各跨两张卡加载，日志中的 GPU 0/1 是副本编号，placement 里的 0/1 是各副本内部可见卡编号；不能据此认为只用了总共两张卡。`PYTORCH_CUDA_ALLOC_CONF is deprecated` 为告警，当前输出没有显示它导致退出。

## running 不是实时存活证明

从 status 中 Unix 时间换算为 Asia/Shanghai：

| 状态 | 时间 |
|---|---|
| v1 最后更新 | 2026-10-27 02:12:31.516921 +08:00 |
| v2 最后更新 | **2026-10-27 18:35:37.616849 +08:00** |
| 前次平台事件 Worker failed | 2026-10-27 18:41:29，平台显示时区待核实 |

v2 的 running 是历史写盘值。若平台显示时间同为北京时间，失败事件比该次写盘晚约 5 分 51 秒；v2 GPU 日志文件名 `20261027T045424Z` 对应北京时间 12:54:24，也接近该 Job 创建的 12:52:05。时间吻合提示可能为同一次运行，但仍缺启动命令、Job 到 run 的映射，不能当作确认。

父进程正常捕获异常时会写 failed，正常处理停止信号后会写停止状态；若进程被强制终止、节点异常，或尚在信号后的等待阶段，最终状态可能来不及写盘，留下 running。当前资料不能在这些原因中确定一种，也不能从没有 traceback 单独证明外部强杀。

快照里的 generation_eta_seconds≈132679（约 36.86 小时）是**当时**根据已完成请求速度推算的生成剩余时间，不能当作当前 ETA。active_cpu=0 只表明那个时刻没有待执行的 CPU 评分任务，GPU 日志仍在生成；不能据此认定整项实验闲置。

## 当前不支持的判断

- 尚无新的 Python 异常、OOMKilled、退出码或自动回收原因，无法确定 v2 为何最终停止。
- 这份输出没有逐题模型/独立评分明细，不能补算 typed_flat 语义错误类型或质量收益。
- 不应因 v1 的已知 gate 错误再次重跑 v1，或修改验收门槛以放行。
- v2 的 134 个已评分组合及已发布生成子步骤需保留；后续同配置恢复仍使用原固定工作树。v3 是独立 E001-1.3，不能把 v2 检查点混入它。

## 下一步需要的证据

优先取 **job-cfc7af77 的平台主运行/训练日志最后 100 行、退出码及启动命令**。当前发送的是 GPU 子日志和状态快照；平台 Events、主日志、子日志承担不同作用。

CPU 终端可补查共享目录中的运行时间线及最后一批状态：

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
RUN="$BASE/results/prompt_efficiency_v2"
stat -c '%y %s %n' "$RUN/status.json" "$RUN/gpu-0.json" "$RUN/gpu-1.json"
tail -n 35 "$RUN/record.md"
cat "$RUN/gpu-0.json" "$RUN/gpu-1.json"
```

如需确认默认 v3 是否准备过，再查看 `results/prompt_efficiency_v3/manifest.json` 和 `preflight.json` 是否存在；自定义 run 需按实际启动命令定位。文件存在仅说明准备/写盘进度，不是存活证明。
