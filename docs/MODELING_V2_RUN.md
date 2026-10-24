# 建模接口 v2：真实启动验收后再运行 12 组实验

旧 CO、distribution_feedback_v1 和 distribution_feedback_4gpu80_v1 均原样保留。
所有新增模块在 tools/runtime/modeling_v2；旧源码校验值不变。不复用任何旧模型回答，
也不自动补写 schema、删除约束、使用 gold 修复模型或修改旧结果。

## 一条命令（1 worker / 4 H100 / 80 CPU）

在新的固定版本 checkout 执行：

```bash
bash tools/runtime/modeling_v2/01_run_4gpu.sh
```

环境/模型/数据沿用个人目录已有资源，无下载、无联网 CPU 求解。默认 32 个 CPU 搜索进程，
2 个各使用 2 GPU 的副本，每副本 batch 上限 4（不再默认 8），OOM 自动拆批。
禁止在旧四卡任务还占用同一组 GPU 时叠加启动；应先在平台停止已确认失败的那个任务。
脚本不主动停止任何其他任务。

## 新接口与长输出

完整模板明确 model.schema="bounded-integer-v2"，保留严格整数域、维度、逻辑、引用检查。
linear/a 允许稀疏索引字典，未列项为零；通用编译为旧密集表示，不针对题型写答案。
同一原文证据可以覆盖多个变量/约束，展开后仍要求每个目标恰好一次、引用确实在原文中。
不自动猜测缺失版本、不将非法 JSON 当作有效模型。

仍关闭 thinking，禁止多余解释；每条输出最多 4096 tokens，一次 generate 最长约 600 秒
（Transformers 在生成步之间检查，非硬实时限制）。每 16 token 检查逐序列 JSON 完成，
提前结束完成项。保留 raw_response、token 和 finish_reason；取首个完整 JSON 是预先约定的
输出协议，不是用答案修复。输入超过 16384 token 直接失败，绝不悄悄截断问题。
GPU 整批超过 1800 秒触发外部看门狗；发现模型 offload 到 CPU/disk 直接报错。

## 自动验收与熔断

默认先在 results/distribution_feedback_modeling_v2_gate 运行真实模型。
按数据集族选择问题文字最长的一个实例（NLCO 合并 S/M/L；synthetic 按族），当前 8 族
即 8 个实例，选择独立于模型表现。仅 pbit_selected 一组，512 score budget、4 chains、1 seed。
每个实例必须有真实生成、完整性一致的初始及反馈后响应，并且两阶段均有 candidate_evaluations>0
的 searched 记录。模拟回复、只生成不搜索、任一族解析失败均不能通过。
独立验证可行率记录在验收中，但接口验收不是语义正确或性能优势的证明。

验收成功打印 REAL MODEL GATE PASSED 后才启动完整 12 组；失败退出，完整实验不放行。
验收结果保存在 gate_acceptance.json；若更早异常，查看 gate/job_status.json 和 gate/job_logs。
已经完成且源码/模型身份校验通过的验收可在续跑时复用。验收开销单独保留；不混入正式收益。

正式阶段每 30 秒检查 health.json：最近最多 24 个不同的 (实例,模型回答哈希) 中，
至少 12 个且失败比例 >=80% 时退出。相同回答跨种子/对照重复失败只计一次。
该阈值也可能因合法 unsupported 比例过高触发，必须解释，不允许删失败样本来美化指标。
空队列 900 秒退出、子进程报错退出、原子检查点、进程组清理沿用统一运行器。

## 正式实验是什么

默认完整复制旧审计实例集合（当前 300：240 NLCO +60 synthetic），重新生成全部模型。
预算保持 8192 score budget、16 chains、3 seeds；每组初始和反馈后两阶段，CP-SAT 为 5 秒时限参考。

| 组 | 方法 | 要检验的因素 |
|---|---|---|
| 1 | pbit_self_review | 不提供搜索证据的模型自审 |
| 2 | pbit_best | 只反馈最优候选 |
| 3 | pbit_distribution | 增加分布统计 |
| 4 | pbit_probe_blind | 执行约束干预但隐藏干预诊断 |
| 5 | pbit_intervention | 显示干预诊断；5 对 4 隔离诊断信息作用 |
| 6 | pbit_energy | 选择能量表示 |
| 7 | pbit_block | 选择更新策略 |
| 8 | pbit_selected | 联合选择能量表示和更新策略 |
| 9 | pbit_random_policy | 相同试运行预算，随机选策略；8 对 9 检验选择机制 |
| 10 | metropolis_best | 传统随机搜索＋最优候选反馈 |
| 11 | metropolis_selected | 传统随机搜索＋联合选择 |
| 12 | cp_sat_feedback | 精确求解器反馈参考，预算单位不同 |

本次不训练 SFT/DPO，不做 FPGA 实验；研究建模接口修复后的搜索闭环、反馈和策略选择。
所有组使用同一新协议、模型、生成上限及数据；所有试探开销计入搜索预算。
解析失败仍进入失败分母；不能用通过解析的子集代表完整数据集。

## 查看及续跑

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
cat "$BASE/results/distribution_feedback_modeling_v2_gate/gate_acceptance.json"
cat "$BASE/results/distribution_feedback_modeling_v2/health.json"
cat "$BASE/results/distribution_feedback_modeling_v2_reports/report.md"
```

正式目录还不存在可能表示验收未通过，先看 gate 日志。中断后相同启动命令续跑，
不删除已完成记录；源码/模型变更会拒绝混入。修复协议以后须用新目录重做验收。
报告自动每 60 秒刷新，配对统计沿用原报告器。不能保证平台永不回收或四卡持续满载。
本地测试没有替代远程 Qwen3-32B/H100 真实验收，也不宣称已实测通过。
