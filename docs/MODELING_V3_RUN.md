# Modeling v3：数学合法性、证据审计、语义覆盖分开

本版是独立入口，旧 CO、modeling_v2 代码和结果不迁移、不覆盖。复用已下载模型/环境，
重新生成新协议下的全部回答。所有方法共享相同协议、生成和重试规则，不使用 gold 修复模型。
问题范围仍是有界整数/布尔组合优化；不支持任意连续或无界模型。

## 四卡一键入口

```bash
bash tools/runtime/modeling_v3/01_run_4gpu.sh
```

1 worker、4 H100、80 CPU。默认 32 CPU 搜索进程，2 个各占两卡的模型副本，
batch 上限改为每副本 2，降低长短请求混批的显存及尾部等待压力；OOM 会拆批。
不是新增 CPU 计算节点；联网 CPU 只拉代码，现有环境/模型无需重新下载。
不要与旧失败任务在同一组 GPU 叠加运行；本脚本不会主动停止其他任务。

默认结果：

- 验收：results/distribution_feedback_modeling_v3_gate
- 正式：results/distribution_feedback_modeling_v3
- 配对统计和 ETA：results/distribution_feedback_modeling_v3_reports/report.md
- 三层审计汇总：正式目录 summary.md / summary.json
- 实时失败统计：health.json；主任务日志也打印 model_health

## 三层状态

1. mathematical_validity：结构、整数域、维度、索引、逻辑合法才搜索。重复 JSON 键直接拒绝，
   不再让 JSON 解码器悄悄覆盖系数。空 dense 数组不能冒充零向量。
2. citation_audit：引用是否真实出现、目标是否存在、有无缺失证据。允许一个目标有多个证据，
   重复证据记警告；引用错误不再直接阻断数学合法的模型。它也不能被记作语义正确。
3. coverage_audit：要求模型先输出逐条 requirements 清单，区分 objective/domain/constraint，
   列出每条量化需求 expected_count 与模型 targets。检查计数、映射、类型、遗漏映射、重复需求 ID。
   独立的原问题可行性仍在 runs.verified_feasible 中单列。

没有针对背包、分配、MaxCut 等类型的修复分支。无 gold 系数/约束进入反馈；gold 只用于
求解之后的独立评价和验收判断。所有组都获得相同的模型审计信息，组间只改变搜索证据。
覆盖清单是模型自报的一致性检查：模型可能同时漏报需求和约束，因此它不是自然语言语义完备性证明。
得到一个独立可行解也不证明完整数学模型等价于原问题。

## 长输出

- 允许所有变量以 [name,lower,upper] 三元组、所有线性约束以 [id,a,op,rhs] 四元组表示；
  通用编译器展开为标准 IR，不推断或补写原题约束。
- 首次输出预算按实际 prompt token 数三倍、1024 向上取整，限制在 6144..12288；
  另受模型上下文容量限制。一次 generate 最长约 1200 秒（生成步间检查）。
- 不完整输出最多重试一次，单条生成，上限 16384 tokens / 1800 秒；不使用截断模型搜索。
- 保留 generation_attempts 下原始截断文本，记录 token_limit/time_limit/eos_without_complete_json。
  最终 response 发布前完成重试，避免 CPU 消费半成品。输出 token、输入 token、耗时包含重试；
  generation_attempts 和 summary 中 attributed_generation_attempts_including_truncation_retries
  区分逻辑阶段数与实际尝试数。OOM/中断的未完成开销另看 inference_events。
- 默认两条批次最坏可能经历初次及两次单条重试，外部硬看门狗设为 90 分钟；
  不保证每个问题都在预算内完成，也不靠无限输出或虚假计算占卡。

## 真实验收和熔断

仍按问题族选择最长问题，当前 8 个验收实例，初始/反馈后均必须真实搜索。
验收提高到与正式相同的 8192 score budget、16 chains，但每阶段一个种子。
同时要求两阶段 requirements 清单各自通过一致性检查、候选经独立验证可行；引用问题单独报告不作数学失败。
验收未通过不放行 300 实例、12 组正式实验。不是放松阈值跳过故障。

正式阶段每 30 秒按不同 (实例,回答哈希) 去重检查：
最近至少 12、最多 24 个模型中，数学/搜索失败率 >=80% 或覆盖清单问题率 >=80%，熔断退出。
仅引用问题不触发该数学熔断。报告不删除失败实例，不将通过解析子集替代全体。
同一版本/同一目录重跑支持续跑；修复协议或预算后使用新目录，不能混入旧结果。

## 验证边界

tests/fixtures/modeling_v3_gate_regression.json 收录本次验收包的 8 个真实初始回答、问题及
标准模型，仅用于离线回归，绝不作为在线生成输入。覆盖重复引用、近似引用、缺失约束、
错误目标维度、重复系数键、两类截断，以及新清单、真实 CPU 搜索、验收和重试计费。
本地回归不能替代 Qwen3-32B/H100 实测。通过验收也不代表算法优于基线。
本轮正式实验仍是分布反馈/约束干预/能量与更新策略选择的 12 组对照，不训练 SFT/DPO。
