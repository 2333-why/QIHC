# E001 平台失败事件：job-cfc7af77

- 接收/归档日期：2026-10-28（Asia/Shanghai）。原平台时间戳原样保留，其显示时区尚未核实。
- 大实验：E001（prompt）；具体 E001-1.x 迭代待核实，不能仅凭提交日期或“之前实验”推定对应 v2/v3。
- Job ID：`job-cfc7af77-00e3-401e-b29b-05e6364ea178`，worker-0。
- 来源：用户在聊天粘贴的平台事件；[原文副本](E001_20261028_job_cfc7af77_events.txt)。未收到该任务的启动命令、完整事件列表、容器日志或退出码。
- 状态：平台报告失败；实际执行到哪一阶段及根因待查。此次只补事件档案，不创建新实验迭代、不修改运行代码和既有结果。

## 事件解读

| 平台时间 | 事件 | 可确认的内容 |
|---|---|---|
| 10-27 12:52:05 | 创建 Pod、Service；自定义调度器信息 | 创建了任务对象；不能据此确认训练/推理进程已开始 |
| 10-27 12:52:15、12:52:23 | Unschedulable；等待资源释放并满足 minAvailable | 当时未能满足整体调度条件；提示候选节点 qb-prod-gpu186，不代表已分配该节点 |
| 10-27 18:41:29 | PyTorchJobFailed，1 个 Worker failed | 平台判定任务失败；未给出底层容器退出原因 |
| 之后至 10-28 02:57:47 | 删除 Pod/Service/PodGroup，JobTerminated | 任务清理事件；不表示期间持续计算或多次完成实验 |

两条早期调度失败与最终 Worker 失败之间缺少运行状态证据，不能证明任务一直排队、排队超时，或已经启动后代码报错。也没有提供 `OOMKilled`、具体 exit code、Python traceback 或自动回收原因，因此不能把最终失败归给显存、代码、模型理解能力或利用率回收。

Kubernetes 官方文档说明：调度失败需检查 Pod Events；容器终止应检查 reason、exit code 及起止时间。参考：[Debug Pods](https://kubernetes.io/docs/tasks/debug/debug-application/debug-pods/)、[Pod Lifecycle](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)。这些文档解释通用状态，不证明本平台最终失败的具体原因。

## 待补证据及处理顺序

1. 该平台任务的实际启动命令、容器退出码和最后 100 行运行日志，核实所属迭代。
2. 若容器曾运行，查看退出信息及对应 run 的 `status.json`、GPU 子进程日志、gate/report.md 和 formal/report.md。
3. 若确认未启动且没有容器执行日志，再按平台排队/调度失败处理。此时修改 Python 代码不能解决资源排队。
4. 获取具体错误后，判断同配置原位续跑还是新配置迭代；保留现有检查点和结果，不提前重跑模型下载或覆盖 run。

可在联网 CPU 终端读取共享存储上的历史状态和最后两份 GPU 日志：

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
for V in v1 v2 v3; do
  RUN="$BASE/results/prompt_efficiency_$V"
  [ -d "$RUN" ] || continue
  printf '\n=== %s ===\n' "$RUN"
  if [ -f "$RUN/status.json" ]; then cat "$RUN/status.json"; else printf 'No status.json\n'; fi
  find "$RUN/logs" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null |
    sort -nr | head -n 2 | cut -d ' ' -f 2- |
    while IFS= read -r LOG; do printf '\n--- %s ---\n' "$LOG"; tail -n 60 "$LOG"; done
done
```

这些目录可能有其他任务或更晚续跑的结果；只有与该 Job 的启动命令、时间和固定工作树对应后，才能归因。没有 `status.json` 或新日志只说明共享目录中未观察到对应输出，不能单独证明容器未启动。

## 当前结论

任务失败属实；早期调度受阻属实；最终根因及实验实际完成量待补证据。现有事件不能作为算法效果失败的结论，也不据此修改历史实验完成状态。

## 2026-10-28 后续运行快照

收到 [v1/v2 状态与 GPU 子日志](E001_20261028_runtime_001_analysis.md)：v2 于北京时间 10-27 18:35:37 最后保存 running、formal 134/1500；GPU 日志名对应约 12:54 启动。若平台显示为北京时间，时间与本 Job 接近，倾向关联 E001-1.2，但缺启动命令/Job 映射，仍未确认。新输出无最终 traceback、退出码或 v3 运行信息，最终停止原因仍待查。
