# 建模接口与局部纠错实验 v2

修复依据：[v1 真实失败归档](../../docs/experiments/20261026_prompt_efficiency_v1/gate_analysis.md)。
本轮记录：[20261027_prompt_efficiency_v2](../../docs/experiments/20261027_prompt_efficiency_v2/record.md)。

## 比较方法

| 组 | 处理方式 | 主要比较 |
|---|---|---|
| legacy_flat | 原版一次建模提示、编译器、宽格式约束与整步重试 | 保留原基线，提示逐字一致 |
| audit_staged | 原版分步提示；缺少 domain/证据说明改为审计，数学检查不放宽 | 与 typed_staged 比较类型接口组合的收益 |
| typed_flat | 一次建模＋数据结构提示＋嵌套类型约束＋显式编号映射 | 与 legacy_flat 比较完整接口改进，不称为仅提示词消融 |
| typed_staged | 同一类型接口，分变量／目标／约束步骤，错误时整步重试 | 与 typed_flat 比较分步流程 |
| typed_staged_local | typed_staged，错误时优先用局部字段替换修复 | 与 typed_staged 比较局部纠错 |

审计分离的直接证据是同一批失败计划的回归回放；本轮不重跑旧 legacy_staged 作为正式组，不将历史结果当作新的同批配对试验。

各组使用同一题目、模型、变量名称顺序辅助、thinking=false 和确定性生成。每逻辑请求累计输出最多 16384 tokens、最多 8 次调用、最多一次纠错。单步 staged 最多 6144、flat 最多 8192，单次生成时间上限 900 秒。输入、实际输出与调用成本如实报告，不声称等 FLOPs。

局部纠错仅在失败步骤能解析为 JSON 且有允许修改的数学字段时启用；只替换已有字段，不删除规则、修改 ID 或改写需求表。重复 JSON 键／不完整 JSON 等无法可靠解析的输出回退为整步重试，仍共享一次纠错额度。所有补丁应用后重新编译，失败结果照常计分。数学合法不等于原题语义正确。

## 通用性与范围

- 输入仍是完整原文；依据实际数据形状提供数组长度、列名、唯一列、列值范围与合法引用，未按问题名称选择公式。
- `column+offset` 与 `index_of` 必须由模型显式生成。编译器从不猜测一基编号或自动减一；编号映射需要与公开变量顺序一致。
- 约束、目标、变量范围、索引和引用类型仍执行硬检查。缺少 domain 分类、证据引文或计数注释只进入独立审计，不能绕过真正的数学错误。
- 仍限于现有 bounded-integer-v2 范围，最多 256 个变量／512 个展开线性约束；不是任意自然语言优化问题的完整求解器。

## 数据、硬件与预算

复用 `results/co_comparison_v1` 的 300 题（NLCO 240＋合成 60），以及其中配置的已下载模型，预期 Qwen3-32B。题目均有 CPU 独立数学标准模型；标准目标、约束和参考答案不发送给 GPU。变量名称／顺序是所有方法共享的辅助，不声称无辅助建模。

先运行每类原文最长的 8 题 × 5 组验收；每组至少一个可执行模型且两个后端都实际搜索，才进入正式阶段。语义覆盖和低解质量作为测量，不作为放行奖励。旧结果不导入新评分，新验收不直接沿用旧通过状态。

每个生成模型：CP-SAT 单线程 5 秒一次；p-bit 3 种子（0/1/2）、16 链、每链 8192 次候选评分上限。关闭搜索反馈和后训练。正式最多 1500 个逻辑建模请求、4500 次 p-bit 搜索、1500 次 CP-SAT 调用。双方停止预算不同，不能把质量差异解释为等成本优势。

单 worker、4 H100、80 CPU 核：两个常驻双卡模型副本，每副本 batch 2；32 个 CPU 搜索进程。每行使用自己的数据 schema，剩余输出预算不同的请求分开生成，避免互相截短。模型跨 gate/formal 常驻；ETA 每 30 秒更新。完成 GPU 请求后的 CPU 尾部超过 15 分钟会保存状态退出，无法保证平台不回收，也不制造虚假占用。

## 准备、运行和续跑

在本轮独立、固定提交的 worktree 中，联网 CPU 执行：

```bash
bash experiments/prompt_efficiency_v2/00_prepare_cpu.sh
```

复用 `$BASE/envs/qihc-cu128`、`$BASE/deps/modeling-v5-format-0113` 和已下载权重；缺少格式依赖时仅从国内 PyPI 镜像安装两个固定小包，不升级原环境或重新下载模型。CPU 检查真实 tokenizer、类型 schema 与全体首轮输入长度，不载入大模型权重。准备完成后联网 CPU 可退出。

在 1 worker、4 H100、80 核任务中执行一次：

```bash
bash experiments/prompt_efficiency_v2/01_run_4gpu.sh
```

同一固定提交、同一配置重复该命令可续跑；不要套 torchrun 或每卡运行一份。默认结果目录是 `$BASE/results/prompt_efficiency_v2`，与 v1 分离。需要另一个运行目录时，通过 `QIHC_PROMPT_V2_RUN` 指定，CPU/GPU 必须一致。任何源代码／配置改变都须新建结果目录，不能修改旧 manifest 强行续跑。

## 报告与问题定位

```bash
BASE=/inspire/hdd/global_user/czxs25210022/why
PYTHONPATH="$PWD" "$BASE/envs/qihc-cu128/bin/python" -m experiments.prompt_efficiency_v2.run status
bash experiments/prompt_efficiency_v2/02_report.sh
cat "$BASE/results/prompt_efficiency_v2/gate/report.md"
cat "$BASE/results/prompt_efficiency_v2/formal/report.md"
tar -czf "$BASE/prompt_efficiency_v2_analysis.tar.gz" -C "$BASE/results" prompt_efficiency_v2
```

自动报告区分编译数、原题可行率、可行解参考 gap 与其样本数、生成/搜索成本；未完成单列、失败计入分母。summary.json 另有分数据集与同题配对统计。分摊生成耗时不是单请求独占 GPU 延迟。可行率不证明数学等价；只有原题可行解的目标值才用于 gap。参考值不一律视为已证明最优。

`manifest.json` 固定提交、代码指纹、数据、模型元数据和配置；`record.md` 自动追加运行时间线。`builder_steps/` 保存原始生成和局部补丁，`scores/` 保存独立评分和审计。验收失败会在主日志打印组别及失败汇总，不必仅凭退出码猜测。
