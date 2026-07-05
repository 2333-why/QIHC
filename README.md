# QIHC

**Quantum-Inspired Intelligence Heterogeneous Computing**  
量子启发智能异构计算 —— p-bit 概率 Ising 求解后端仿真平台

构建「AI 智能前端 + p-bit 概率求解后端」的异构计算体系：上层负责语义理解、QUBO/Ising 编码与解评估；下层负责 Gibbs / Parallel Tempering / SQA 等概率采样，在组合爆炸的离散空间中高效搜索。

---

## 功能特性

- **p-bit / Ising 概率网络仿真**（`qihc.ising`）
  - Gibbs 采样（顺序 / 并行）
  - Parallel Tempering（副本交换）
  - Simulated Quantum Annealing（Suzuki–Trotter 分解）
  - 模拟退火（同步 / 异步）
- **Max-Cut 基准**（图生成、Ising 映射、暴力最优解、可视化）
- **QIHC 异构闭环**（`qihc.orchestrator`：前端 → 编码 → p-bit → 调度）
- **VCI 变分协推理**（`VCIOrchestrator`：\(q \leftrightarrow s\) 交替最小化，Case A 子集选择）
- **随机计算查表库**（`qihc.stochastic`，基于预计算 parquet 查找表）

---

## 目录结构

```
QIHC/
├── qihc/                    # 可安装 Python 包
│   ├── ising/
│   │   ├── model.py         # IsingModel + 采样内核
│   │   └── maxcut.py        # Max-Cut 工具
│   ├── stochastic/
│   │   ├── sc.py            # 随机计算查表
│   │   └── data/            # 逻辑门查找表 (.parquet)
│   └── orchestrator/
│       ├── frontend/            # L1 语义相 q
│       ├── encoder.py           # L2/L4 编码与解码
│       ├── backend.py           # L3 p-bit 后端
│       ├── scheduler.py         # 单向 QIHC 闭环
│       ├── free_energy.py       # VCI 自由能 F(q,s)
│       ├── reasoning.py         # Case A 子集选择任务
│       └── vci_scheduler.py     # VCI 协推理调度
├── experiments/
│   ├── run_sampler_benchmark.py
│   ├── run_sampler_scaling.py   # TTS 标度（Case B）
│   ├── run_moe_poc.py           # Case C MoE 辅助
│   ├── run_vci_reasoning.py     # Case A VCI 对比
│   └── plot_qihc_vci_architecture.py
├── notebooks/
│   ├── demo_Ising.ipynb     # Ising / Max-Cut 演示
│   └── demo_SC.ipynb        # 随机计算演示
├── tests/
│   ├── test_samplers.py
│   ├── test_orchestrator.py
│   └── test_vci.py
├── docs/
│   ├── technical.pdf
│   ├── VCI_GUIDE.md             # VCI 范式与复现说明
│   └── QIHC_VCI_architecture.png
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库后进入根目录
cd QIHC

# 推荐：创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -U pip
pip install -e ".[dev]"
# B 档 QIHC 闭环（小模型前端）额外安装：
# pip install -e ".[llm]"
# 或仅安装依赖
# pip install -r requirements.txt
```

### 2. QIHC MoE 最小闭环 PoC（B 档）

```bash
# B 档：DistilGPT-2 前端 + p-bit 后端（推荐，需 pip install -e ".[llm]"）
python experiments/run_moe_poc.py --tier b

# A 档：无 torch/transformers，mock 前端（CI / 纯 CPU）
python experiments/run_moe_poc.py --tier a

# C 档：gpt2 + 更多专家（升级路径占位）
python experiments/run_moe_poc.py --tier c --steps 600
```

输出：`experiments/outputs/moe_poc/metrics.json` 与 `moe_poc_comparison.png`

### 3. 运行采样器基准实验

```bash
python experiments/run_sampler_benchmark.py
# 自定义参数
python experiments/run_sampler_benchmark.py --nodes 20 --steps 800 --seed 0
```

输出示例：

```
=== Max-Cut benchmark (n=20, steps=800, seed=0) ===
Brute-force optimal cut: 61
  Gibbs                  cut=  43  ratio=0.705
  Parallel Tempering     cut=  54  ratio=0.885
  SQA                    cut=  60  ratio=0.984
  ...
```

结果图保存至：`experiments/outputs/sampler_energy_convergence.png`

### 4. VCI 协推理实验（Case A，A 档 mock，无需 GPU）

```bash
# Greedy vs VCI-1 (≈CR) vs VCI-2 可行率与 F 曲线
python experiments/run_vci_reasoning.py --problems 32 --steps 250 --seed 0
```

输出：

- `experiments/outputs/vci_reasoning/metrics.json`
- `experiments/outputs/vci_reasoning/comparison_bars.png`
- `experiments/outputs/vci_reasoning/free_energy_demo.png`

### 5. TTS 标度实验（Case B）

```bash
python experiments/run_sampler_scaling.py --nodes 12 14 16 18 20 --trials 8 --steps 800
```

输出：`experiments/outputs/sampler_scaling/tts_scaling.png`

### 6. 生成 QIHC+VCI 架构图

```bash
python experiments/plot_qihc_vci_architecture.py
```

### 7. BBH mini-set 实验（Case A）

```bash
python experiments/run_vci_bbh.py --budget-steps 250
```

40 题 BBH 风格子集选择（`qihc/data/bbh_subset.json`），同 p-bit 算力对比。

### 8. Handoff / 温度映射探针

```bash
python experiments/run_vci_handoff.py --budget-steps 200
```

输出：`experiments/outputs/vci_handoff/handoff_curve.png`

### 9. 运行测试

```bash
pytest
```

### 10. Notebook 演示

```bash
jupyter notebook notebooks/demo_Ising.ipynb
jupyter notebook notebooks/demo_SC.ipynb
```

---

## QIHC + VCI 架构

**QIHC** = 异构平台（LLM 智能层 + p-bit 概率层）  
**VCI** = 变分协推理范式：最小化 \(F(q,s)\)，交替更新语义相 \(q\) 与离散相 \(s\)

```
Step 0  任务解析
Step 1  encode → q^(0) → E(s|q) → h, J
Loop    s-step: p-bit equilibrate → decode → F, violations
        q-step: frontend.refine(q, feedback) → 重编码
Step N  回注 LLM → 最终输出
```

| 模式 | 说明 |
|------|------|
| `greedy` / `vci-0` | 贪心 top-k |
| `vci-1` | 单轮 s 步，无 refine（≈ CR 极限） |
| `vci-2` | 最多 2 轮 q↔s（国自然默认） |

代码示例：

```python
from qihc.orchestrator import VCIConfig, VCIOrchestrator, demo_problem

orch = VCIOrchestrator(VCIConfig.tier_a(sampling_steps=250))
result = orch.solve_subset(demo_problem(), mode="vci-2")
print(result.final_feasible, result.final_free_energy)
print([r.free_energy.total for r in result.rounds])
```

详见 [`docs/VCI_GUIDE.md`](docs/VCI_GUIDE.md)。

---

## QIHC 异构架构（可升级）

```
qihc/orchestrator/
├── frontend/          # AI 智能前端（mock → DistilGPT-2 → 更大 LLM）
├── encoder.py         # MoE logits → Ising/QUBO
├── backend.py         # p-bit 概率求解后端
└── scheduler.py       # 下发 / 采样 / 回注闭环
```

升级路径：`QIHCConfig.tier_a()` → `tier_b()` → `tier_c()`，或自定义 `model_name` / `sampler`。

---

## 代码示例

```python
import networkx as nx
from qihc import IsingModel
from qihc.ising import maxcut

G = nx.erdos_renyi_graph(20, 0.5, seed=0)
J = maxcut.max_cut_to_ising(G)
model = IsingModel(size=len(G.nodes()))

# Gibbs 采样
spins, energy, temps = model.gibbs_sampling_Maxcut(J, steps=1000)

# Parallel Tempering
spins, energy, ladder = model.parallel_tempering_Maxcut(J, steps=1000, n_replicas=8)

# 模拟量子退火 (SQA)
spins, energy, gamma = model.simulated_quantum_annealing_Maxcut(
    J, steps=1000, m_slices=8
)

cut = maxcut.calculate_cut_value(G, maxcut.convert_spins_to_cut(spins))
print("Max-Cut value:", cut)
```

---

## 依赖

| 包 | 用途 |
|----|------|
| numpy | 数值计算 |
| matplotlib | 绘图 |
| networkx | 图与 Max-Cut |
| pandas / pyarrow | 随机计算查找表 |
| scikit-learn | Notebook 中 SC 演示指标 |

---

## 引用与许可

本项目采用 [MIT License](LICENSE)。

如在国自然 / 学术工作中使用，请在前期工作或论文中说明基于 QIHC 仿真平台完成实验。

---

## 路线图

- [x] p-bit Ising 采样内核（Gibbs / PT / SQA / SA）
- [x] Max-Cut 基准与采样器对比实验
- [x] QIHC 异构闭环 PoC（MoE 路由 → Ising → p-bit → 回注，B 档小模型）
- [x] VCI 协推理最小闭环（free_energy + refine + VCIOrchestrator）
- [x] Case A 玩具实验（Greedy / VCI-1 / VCI-2 对比）
- [x] TTS 标度 benchmark 脚本与成图
- [x] Case A BBH mini-set（40 题 bundled JSON）
- [x] Handoff / 温度映射探针（`run_vci_handoff.py`）
- [ ] Case A 真实 HuggingFace BBH 子集（可选扩展）
- [ ] 训练路由头 / 接入真实 MoE 层 hidden states
- [ ] FPGA/ASIC 协处理器接口占位
