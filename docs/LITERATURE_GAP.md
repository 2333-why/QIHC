# 文献竞争边界与增量定位（立项依据附件）

> 对应项目方案 QIHC + VCI v4 · 申请代码 F0214

## 1. 必引文献对照

| 文献 | ID | 做了什么 | 未做什么 | 本项目增量 |
|------|-----|----------|----------|------------|
| **Combinatorial Reasoning (CR)** | arXiv:2407.00071 | LLM→QUBO→Ising 单向求解 | 无统一 F；无 q↔s 交替；弱 IF | **VCI-1 极限**；VCI-2 协推理 |
| **QCR-LLM** | arXiv:2510.24509 | 量子延伸 CR | 无 VCI；无 p-bit 专精 | QIHC 平台 + 经典 p-bit 后端 |
| **LLM-QUBO** | arXiv:2509.00099 | NL→QUBO 编译 | 非协推理循环 | VCI 闭环 + refine |
| **E2E CO Solver** | NeurIPS'25 | LLM 直接解 CO | 非异构协处理 | 组合优化作 LLM **协处理器** |
| **FALCON** | arXiv:2602.01090 | 100% 可行路由 | 无 Ising/p-bit | IF 写入能量 + p-bit equilibrate |
| **IsingFormer/TAPT** | arXiv:2509.23043 | 全局提议+PT | 无语义层 | s-步内部加速（VCI 特例） |
| **Camsari p-bit 栈** | IEEE JXCDC 2023 | 概率比特器件与仿真 | 未接 LLM 协推理 | QIHC 下层后端 |

## 2. 相对 CR 的三条增量（申请书用）

1. **范式**：统一 \(\mathcal{F}(q,s)\)，\(q\leftrightarrow s\) 交替，非 pipeline 拼接  
2. **机制**：IF 约束进能量 + q-步 refine + p-bit 专精 s-步  
3. **验证**：可行率–质量–延迟 trade-off（BBH 子集 + 标度 + MoE 辅证）

## 3. 禁止表述

- ~~国际首创 LLM + QUBO + Ising~~（CR 已存在）  
- ~~MoE 路由为主实验~~（仅 Case C 一句）

## 4. 推荐表述

> 在 CR 等工作的基础上，本项目于 QIHC 异构平台上提出 VCI 变分协推理范式，以 \(\mathcal{F}(q,s)\) 统一语义相与离散相，系统研究 p-bit 专精后端的协推理循环及可行率–质量 trade-off。

## 5. 已有实验对文献叙事的支撑

| 实验 | 支撑论点 |
|------|----------|
| 玩具 Case A：vci-2 可行率 100% vs greedy 78% | VCI 协推理优于单向贪心 |
| BBH 40 题同算力：可行率 100% vs greedy 90% | IF/约束满足（主指标） |
| Handoff：σ≥0.15 时 VCI-2 增益 +5~10% | 语义噪声高时 q-refine 更有价值 |
| MoE +0.91 有效得分 | IF 可迁移至路由（辅证） |
| TTS 标度 PT @ N=12–20 | p-bit s-步标度规律（Q1） |

## 6. 待引用的本项目代码/数据

- 仓库：`https://github.com/2333-why/QIHC`（v0.2.0）
- 复现：`python experiments/run_vci_bbh.py` 等（见 README）
