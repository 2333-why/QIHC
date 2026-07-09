# QIHC 三律实验（NE1–NE9）运行说明

> Tier 0 / 1 / 2 全部代码已实现，可直接提交服务器运行。

## 一键服务器运行（推荐）

```bash
cd /path/to/QIHC

# 前台
bash experiments/nsfc_evidence/run_on_server_three_law.sh

# 后台
nohup bash experiments/nsfc_evidence/run_on_server_three_law.sh \
  > experiments/outputs/nsfc_evidence/three_law_launcher.log 2>&1 &
echo $!
```

可选环境变量：

```bash
export PROFILE=tier012   # smoke | tier0 | tier1 | tier2 | tier012 | full
export SKIP_PYTEST=1
export CUDA_VISIBLE_DEVICES=0,1
```

## 本地 / 分步运行

```bash
# 全套 smoke（~1–2 分钟，验证通路）
python experiments/nsfc_evidence/run_three_law_suite.py --profile smoke

# 正式 full（申请书前期证据）
python experiments/nsfc_evidence/run_three_law_suite.py --profile tier012

# 只跑某一层
python experiments/nsfc_evidence/run_three_law_suite.py --profile tier0
python experiments/nsfc_evidence/run_three_law_suite.py --profile tier1
python experiments/nsfc_evidence/run_three_law_suite.py --profile tier2

# 单实验
python experiments/nsfc_evidence/run_ne2_contraction.py --profile full
python experiments/nsfc_evidence/run_ne3_refresh.py --profile full
python experiments/nsfc_evidence/run_ne6_trust_gate.py --profile full
```

## 实验清单

| Tier | 编号 | 脚本 | 验证 |
|------|------|------|------|
| 0 | NE2 | `run_ne2_contraction.py` | 定理1c 收缩收敛 |
| 0 | NE4 | `run_ne4_stale_field.py` | 引理2 陈旧场界 |
| 0 | NE9 | `run_ne9_free_energy.py` | 定理4 自由能下降 |
| 1 | NE3 | `run_ne3_refresh.py` | 定理2 η_sc 阈值/幂律 |
| 1 | NE6 | `run_ne6_trust_gate.py` | 定理3b do-no-harm |
| 2 | NE1 | `run_ne1_division.py` | 定理1/1b 软硬分工 |
| 2 | NE5 | `run_ne5_snr_lambda.py` | 定理3a 维纳信任 |
| 2 | NE7 | `run_ne7_trust_proxy.py` | 可信度代理有效性 |
| 2 | NE8 | `run_ne8_pareto.py` | 三维 Pareto |

## 核心库

```
qihc/theory/
  mean_field.py      # 平均场自由能 / 收缩
  higher_order.py    # 高阶软摊派 / 二次化
  refresh.py         # η_sc / 陈旧场界 / 幂律拟合
  trust.py           # λ* / 门控 / 代理
```

## 输出

每次 suite 运行会在：

```
experiments/outputs/nsfc_evidence/run_three_law_<timestamp>/
  ne1_division/summary.json + *.png
  ...
  three_law_index.json
  nsfc_three_law_bundle.tar.gz   # 服务器脚本打包
```

## 预计耗时（full / tier012）

| 环境 | 估计 |
|------|------|
| 本地 CPU smoke | 1–3 分钟 |
| 服务器 CPU full（bundled 30–40 题） | 30–90 分钟 |
| 若后续接 7B logits | 另加数小时（本套默认合成/bundled logits，无需 GPU） |
