# 等量后训练、不使用 p-bit 对照

目标是测量优势，不预设本方法胜出。上一版 sparse 语义 SFT/DPO 的教师是正确模型和反例验证，并未使用 p-bit，不能将其收益归因于 p-bit 造数。

## 必须分开回答的问题

| 对比 | 可支持的结论 |
|---|---|
| Base 直接解题 vs SFT/DPO 后直接解题 | 普通后训练收益 |
| 同量后训练直接解题 vs 建模后 p-bit 搜索 | 整套系统相对直接 LLM 的收益，不单独证明 p-bit 优势 |
| 同一建模模型 + CP-SAT/Metropolis vs + p-bit | 求解后端的增量价值 |
| 同一建模模型 + p-bit，无反馈 vs 有反馈 | 搜索反馈贡献 |
| 同一候选池由传统求解器 vs p-bit 生成偏好后分别训练 | p-bit 造数的贡献；尚未实现 |

## 本次已实现

`controls` 在训练集上以 CP-SAT 生成直接解题教师，单比特扰动生成候选；只保留原约束下不可行或目标严格更差的 rejected。未证明最优的教师不标称最优。全程不调用 p-bit。验证与测试实例不能进入训练。

与稀疏建模训练集逐实例匹配 SFT 条数和 DPO 对数。若直接解题教师没有可行解或有效偏好不足，两臂同时过滤该训练实例，记录 teacher_audit。评估集合不因此缩减。直接基线也包含修正任务，避免只有本方法获得纠错训练。

```bash
python experiments/compositional_energy.py controls --output results/composition_pilot
```

输出 `matched_controls/modeling_sft.jsonl`、`modeling_dpo.jsonl`、`direct_sft.jsonl`、`direct_dpo.jsonl`、`direct_eval_prompts.jsonl`、`teacher_audit.jsonl`、`budget.json`。对应两臂必须使用这些匹配文件，而非原先未匹配的数据。

直接解题模型预测保存为每例一条 `{case_id,response}` 后，可运行：

```bash
python experiments/compositional_energy.py direct-replay --output results/composition_pilot --predictions results/composition_pilot/direct_predictions.jsonl
```

格式错误保留为失败；缺例和重复例拒绝。评分用原始目标和约束，不依据模型自报的可行性。

## 正式训练和评估预算

- 两臂使用相同基础权重、LoRA 配置、优化器、更新步数、有效 batch、随机种子；DPO 均从各自 SFT 出发，不能让一臂多训练一轮。
- 匹配样本数是第一口径，**不等于匹配 token/FLOPs**。正式流水线须再记录实际有效 token、GPU 小时和峰值显存；增加等 token 或等 GPU 时间的次要对照，不用填充零 token 冒充有效训练。
- 教师造数费用与训练费用分别统计；不能把本方法搜索造数当作免费，也不能隐去基线 CP-SAT 教师费用。
- 推理至少报告固定调用预算和相同墙钟预算两种口径。直接 LLM 应允许在等预算下多次采样/自我修正，而非仅比较一次解码与大量搜索。
- 报告全测试集可行率、两臂共同可行实例的配对目标差、失败率、端到端时间和 token；按规则组合/尺寸分别汇总。不可比较各自不同可行子集的平均 gap 后声称胜出。
- 用配对实例及训练随机种子复验。CPU 模拟 p-bit 的时间不是 FPGA 实测能耗，不能由模拟结果推断真机节能。

当前已完成对照数据和直接解评分模块，未接入双卡自动训练编排、真实 token 预算匹配或传统后端全套评估；没有训练结果或优势结论。
