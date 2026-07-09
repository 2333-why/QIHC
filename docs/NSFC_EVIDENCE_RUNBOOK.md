# NSFC 证据链实验 — 服务器运行手册

## 0. 双机分工（联网 4090 + 离线 4×H200）★推荐

| 机器 | 脚本 | 作用 |
|------|------|------|
| **联网 4090** | `experiments/nsfc_evidence/run_online_4090.sh` | 下载模型/BBH、打包 `offline_bundle/`、跑 7B 轻量 P0 实验 |
| **离线 H200** | `experiments/nsfc_evidence/run_offline_h200.sh` | 读 `manifest.env`，4 卡并行跑全量批量实验 |

### 4090（联网）一步启动

```bash
cd /path/to/QIHC
bash experiments/nsfc_evidence/run_online_4090.sh
```

仅下载不跑实验：`SKIP_EXPERIMENTS=1 bash experiments/nsfc_evidence/run_online_4090.sh`

产出：`offline_bundle/`（含 `models/`、`hf_home/`、`manifest.env`、`wheels/`）+ `online_4090_*` 实验结果。

### 拷到 H200 后（离线）

```bash
# 需要同时拷贝：整个 QIHC/ 目录 + offline_bundle/
cd /path/to/QIHC
bash experiments/nsfc_evidence/run_offline_h200.sh
```

未下载 14B 时：`SKIP_CASE_H=1 bash experiments/nsfc_evidence/run_offline_h200.sh`

结果打包在 `experiments/outputs/nsfc_evidence/offline_h200_*/offline_h200_results.tar.gz`。

---

## 1. 上传代码

将本地 `QIHC/` 整个目录上传到服务器，例如：

```bash
scp -r QIHC user@your-server:/data/projects/QIHC
```

## 2. 一键运行（推荐）

```bash
cd /data/projects/QIHC
chmod +x experiments/nsfc_evidence/run_on_server.sh
bash experiments/nsfc_evidence/run_on_server.sh
```

默认配置：
- **4× H200**：`CUDA_VISIBLE_DEVICES=0,1,2,3`
- **模型**：`Qwen/Qwen2.5-7B-Instruct`
- **Profile**：`server`（完整证据链）

快速冒烟测试（约 2 分钟）：

```bash
PROFILE=smoke bash experiments/nsfc_evidence/run_on_server.sh
```

## 3. 实验内容（server profile）

| 步骤 | 脚本 | 产出 | 立项作用 |
|------|------|------|----------|
| dual_bundled | run_dual_evidence | dual_axis.json/png | **主表**：约束集可行率 VCI-2 vs greedy |
| dual_hf_llm | run_dual_evidence + LLM | 同上 | 真实语义势场下的 HF BBH |
| cr_protocol_llm | run_cr_bbh + 采样 | cr_protocol.json | **对标 CR**：linear/quadratic/vci |
| cr_bundled | run_cr_bbh | cr_protocol.json | CR 协议 + IF 约束 |
| f_trajectories | run_f_trajectories | f_descent_mean.png | **F 下降曲线** |
| sampler_ablation | run_sampler_ablation | sampler_ablation.png | PT vs Gibbs（对标 TAPT） |
| handoff | run_vci_handoff | handoff_curve.png | **handoff 相变** |
| pareto | run_vci_pareto | pareto_frontier.png | 质量–可行率–算力 |
| scaling | run_sampler_scaling | tts_scaling.png | Q1 标度规律 |
| bbh_hf_full | run_vci_bbh | bbh_comparison.png | HF 全量对比 |

## 4. 日志与产物

每次运行在：

```
experiments/outputs/nsfc_evidence/run_YYYYMMDD_HHMMSS/
├── run.log              # 主日志
├── manifest.json        # 步骤清单 + 产物列表
├── console.log          # 终端完整输出
├── evidence_report.json # 汇总报告（自动生成）
├── evidence_table1.png  # 汇总图
├── nsfc_evidence_bundle.tar.gz  # 打包下载
└── <step_name>/         # 各子实验 JSON + PNG
```

## 5. 下载结果给我分析

```bash
# 在服务器上
RUN=experiments/outputs/nsfc_evidence/run_YYYYMMDD_HHMMSS
tar -czf ~/nsfc_results.tar.gz -C $RUN .

# 在本地
scp user@server:~/nsfc_results.tar.gz .
```

把 `evidence_report.json`、`dual_axis.json`、`cr_protocol.json`、`handoff.json` 发给我即可。

## 6. 手动分步运行

```bash
cd QIHC
source .venv/bin/activate
pip install -e ".[dev,hf,llm]"

# 主证据：bundled 约束集
python experiments/nsfc_evidence/run_dual_evidence.py \
  --source bundled --budget-steps 400 \
  --output-dir experiments/outputs/nsfc_evidence/manual/dual_bundled

# CR 协议 + 真实 LLM（HF）
python experiments/nsfc_evidence/run_cr_bbh.py \
  --source hf --use-llm \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --n-samples 50 --limit-per-task 30 \
  --output-dir experiments/outputs/nsfc_evidence/manual/cr_llm
```

## 7. 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | 0,1,2,3 | GPU |
| `MODEL_NAME` | Qwen/Qwen2.5-7B-Instruct | 7B 适合单卡；也可用 Llama-3.1-8B |
| `PROFILE` | server | smoke / server |
| `HF_HOME` | `.cache/huggingface` | HF 缓存目录 |

## 9. P0–P2 全量 Case（profile=p012）

一键在 **2× Pro 6000** 后台跑齐 P0–P2 + 机制证据：

```bash
cd /hdd/why/QIHC
chmod +x experiments/nsfc_evidence/run_on_server_p012.sh

# 后台运行（推荐）
nohup bash experiments/nsfc_evidence/run_on_server_p012.sh \
  > experiments/outputs/nsfc_evidence/p012_launcher.log 2>&1 &
echo $!   # 记录 PID

# 查看进度
tail -f experiments/outputs/nsfc_evidence/run_p012_*/console.log
```

| Case | 步骤名 | 脚本 | 说明 |
|------|--------|------|------|
| P0-A | case_a_cr_subset | run_case_a_cr_subset.py | CR 五模式 + 合成 200 子集 |
| P0-B | case_b_dual_synthetic | run_case_b_dual_synthetic.py | 可行率主表（合成 200） |
| P0-C | c_compute_budget | run_case_c_compute_budget.py | 同算力预算表（自动汇总） |
| P1-D | case_d_cr_by_task | run_case_d_cr_by_task.py | 分任务类型 CR 图 |
| P1-E | case_e_cr_bundled_full | run_case_e_cr_bundled_full.py | bundled 完整 CR 基线 |
| P1-F | case_f_vci_ablation | run_case_f_vci_ablation.py | vci-0/1/2/full 链 |
| P2-G | case_g_constrained_bbh | run_case_g_constrained_bbh.py | 约束版 HF BBH + 7B |
| P2-H | case_h_model_compare | run_case_h_model_compare.py | 7B vs 14B 合成集 |
| 机制 | handoff/pareto/F/sampler | 已有脚本 | handoff + Pareto 等 |

环境变量（Pro 6000 默认 2 卡）：

| 变量 | 默认 |
|------|------|
| `CUDA_VISIBLE_DEVICES` | 0,1 |
| `MODEL_NAME` | Qwen/Qwen2.5-7B-Instruct |
| `MODEL_NAME_14B` | Qwen/Qwen2.5-14B-Instruct |
| `PROFILE` | p012 |
| `SKIP_PYTEST` | 0（设为 1 跳过测试） |

预期耗时：**约 4–10 小时**（Case H 需加载 7B+14B 各一次）。


跑完后应能支撑：

1. **bundled 约束集**：VCI-2 可行率 > greedy（+5~10 pp），F 单调下降  
2. **CR 对齐**：quadratic ≥ linear；VCI-2 ≥ VCI-1 在含约束任务  
3. **handoff**：高 σ 区间 VCI-2 增益 +5~10%  
4. **采样**：PT 可行率 ≥ Gibbs（TAPT 叙事）  
5. **真实 LLM**：Qwen2.5-7B 在 HF BBH 子集上的同算力对比表
