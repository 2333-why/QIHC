> 2026-11-03：服务器缺少新增 jsonschema，导致 E004-1.5 导入阶段退出。当前修复已移除该运行依赖，且启动前实际导入 runtime。新增 E004-1.6 半小时入口 `04_run_30m.sh`，六类小题×六组×一轮；结果导出用 `05_pack_30m.sh`。详见 [E004-1.6 记录](../../docs/experiments/E004_feedback/E004-1.6_record.md)。以下仍为原六小时配置说明。

# E004-1.5：六小时修复版对比

实验记录：[E004-1.5_record.md](../../docs/experiments/E004_feedback/E004-1.5_record.md)。原 E004-1.4 文件与结果不改动。复用历史 GPU worker、探针数值实现、七求解器、独立语义 evaluator 和评分函数；本目录仅版本化协议、调度、身份及报告，不修改共享模块全局变量。

## 直接运行

四张能独立加载 Qwen3-32B 的 GPU、至少 24 个分配给作业的 CPU 核，沿用此前服务器环境。代码由 `codex/e004-1.5-witness-6h` 分支发布；首次建立独立工作树：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc
git fetch origin codex/e004-1.5-witness-6h
git worktree add --detach ../qihc-E004-1.5-git origin/codex/e004-1.5-witness-6h
cd ../qihc-E004-1.5-git
bash experiments/feedback_witness_v2/00_check_cpu.sh
```

在一个四卡、24 核作业内执行：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc-E004-1.5-git
export QIHC_PILOT_SOURCE=/inspire/hdd/global_user/czxs25210022/why/results/E004-1.2_feedback_pilot
bash experiments/feedback_witness_v2/01_run_4gpu.sh
```

必须提供完整 E004-1.2 运行目录，包含 manifest、cases.public.json、initial、evaluation；不要把 E004-1.4 分析包当作输入。新结果默认为 `$QIHC_BASE/results/E004-1.5_feedback_witness_6h`，不会覆盖历史目录。若已经存在同名工作树，进入原目录即可，不重复 `git worktree add`。

`00_check_cpu.sh` 检查容量和语法，不生成结果。完整依赖、七后端和输入哈希预检在 GPU 启动命令的六小时时钟内执行。环境可通过 `QIHC_BASE`、`QIHC_ENV_PREFIX`、`QIHC_FORMAT_DEPS`、`QIHC_SOLVER_DEPS`、`QIHC_MODEL`、`QIHC_WITNESS_RUN` 覆盖；沿用历史默认值，不安装或下载模型。

## 获取结果与续跑

```bash
cat /inspire/hdd/global_user/czxs25210022/why/results/E004-1.5_feedback_witness_6h/status.json
bash experiments/feedback_witness_v2/02_report.sh
bash experiments/feedback_witness_v2/03_pack.sh
```

正常退出自动打包；`03_pack.sh` 可事后再导出已有结果。上传以下文件即可分析：

```text
/inspire/hdd/global_user/czxs25210022/why/results/E004-1.5_feedback_witness_6h_analysis.tar.gz
```

同一启动命令在原截止时间内续跑，`clock.json` 不重置；无自动 GPU 重试。不要同时对同一目录启动两个监督器，也不要在运行中更新其代码。报告命令可读取已落盘的部分结果；不要与运行器同时执行打包命令，以免争用同一输出包。

## 设置与修复

- 仍为原 18 道开发诊断题，S 自审、R 随机选变量对、P 主动 p-bit、X SCIP 探针、A 解析对照、M Metropolis 探针；两轮、seed=0、32 链。每组共享同一初稿和每题一个独立来源假设，不按表现删题。
- 108 条轨迹，18 次假设生成＋216 次补丁，均最多 6144 输出 token；假设 240 秒、补丁 210 秒，超时或截断保留为失败。输入 32768/请求、131072/轨迹，反馈 4096 token。完整假设不会直接提供给补丁模型，各组只得到状态与来源编号，数值证据仅按组设计提供。
- CPU 同任务先按哈希去重，执行时持有跨线程/进程锁，再给逻辑行分发独立结果副本。等待活跃任务不会写入 `interrupted_no_retry`。故障续跑只在拿到排他锁且存在未完成旧 claim 时记中断，不把历史失败改成成功。
- 原文引用使用确定性的片段 ID，保存字符位置与 SHA256。模型不再复制长引文。片段存在不等于语义正确。
- 紧凑代数支持常数填充向量、稀疏向量和逐行完整约束族；约束 ID 自动分配，移除易误用的 `forall_affine`。没有任务名称分支或从 gold 注入公式。仍需 LLM 正确提取完整关系和系数。
- 独立假设为可错对象：有限冲突检查保留为诊断，不能否决一个可能正确的补丁。编译、源片段、变量域和未声明族保护仍是硬检查；完整正确性由冻结后的独立 evaluator 判断。
- 额外比较 C0 冻结初稿、GOLD 标准模型，连同六组最终模型，各给七后端相同 20 秒，总计最多 1008 行；相同数学任务只算一次、所有逻辑行保留。标准模型只在反馈冻结后参与评价，不进入 GPU 提示。
- 端到端全部 18 题为分母，单列初稿无效恢复、已证错误修复、正确模型回归、未知、假设成功/正确率、证据覆盖、求解器失败。固定模型求解能力单独列出，建模失败不能算作求解器失败。重复哈希不当作独立样本。

## 六小时边界

不用缓存节省作假设的容量上界为 18825 秒（5 小时 13 分 45 秒），余量 2775 秒（46 分 15 秒）。4 小时 55 分停止新反馈工作，保留 35 分离线比较；5.5 小时计算截止，六小时监督器终止整个进程组，包含加载、准备和报告打包。预算审计见实验记录。

这是有截止保护的小规模比较，服务器故障、模型输出错误或请求超时仍可能留下部分结果；截止不是成功率保证。硬杀或断电未生成包时，事后运行 `03_pack.sh`。单种子、18 道已观察题不能支持未见测试集上的正式优势结论。

## 本地验证

```bash
python -m experiments.feedback_witness_v2.budget
python -m pytest tests/test_feedback_witness_v2.py -q
```

覆盖同任务实际并发子进程、等待者不得覆盖活任务、真实七后端、来源隔离与格式约束、原子修复/无效恢复、六组两轮与续跑、固定结束时间。18 个标准模型另做纯离线可表示性检查，均能无损编译；它不表示 LLM 已成功生成。真实四卡 Qwen3-32B 效果待服务器返回。

发布隔离目录最终回归：新旧版本合计 16 项通过（42.62 秒），包含监督器实际终止超时子进程的测试。
