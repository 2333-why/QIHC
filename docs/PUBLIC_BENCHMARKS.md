# 公开测评数据下载与适配审计

核实日期：2026-10-21。当前完成的是数据获取和结构审计，**不是正式评测适配完成**。

## 已核实并默认下载

采用 [SIRL 官方仓库](https://github.com/Cardinal-Operations/SIRL) 指向的
[作者发布数据集](https://huggingface.co/datasets/chenyitian-shanshu/ORLMBenchmark)。
固定 revision：`54ca621042b4619b90ce23ba3edd710611822e02`。
全部七个文件（含 README）已实测从 `hf-mirror.com` 下载，字节数和 SHA256 与官方源一致。
本机使用代理进行连通性核实；服务器所在网络无需代理能否访问，应以服务器实际下载为准。

| 数据集/版本 | 核实条数 | 数值形式参考答案 | 当前用途 |
|---|---:|---:|---|
| NL4OPT（SIRL 版） | 245 | 228 | 建模外部测试候选；17 条非有限数值标签需单独定义判分 |
| MAMO EasyLP fixed | 642 | 642 | 线性优化建模测试；不能假定都是整数问题 |
| MAMO ComplexLP fixed | 203 | 203 | 更复杂线性建模测试；需变量类型审计 |
| IndustryOR fixedV2 | 100 | 100 | 行业建模测试；固定 V2，不混用旧版 |
| OptMATH Bench 166 | 166 | 166 | 综合建模测试；不与 193 题版本混报 |
| OptiBench | 605 | 605 | 综合建模测试；包含当前接口不能直接表达的问题 |
| 合计 | 1,961 | 1,944 | 规范化题面去重后 1,774 条，非语义去重 |

这些文件只有题目、参考答案，IndustryOR 另有 id/difficulty；没有可直接导入当前框架的独立约束验证器。
数值形式不等于参考值已被证明最优。题目重复组完整写入 audit.json，不擅自删除或修改官方样本。
各数据集分别报告原版本成绩；合并统计另报去重成绩。公开题目全部隔离为外部测试，不用于 SFT/DPO。
训练集与测试集仍需另外检查近重复和来源重叠；这里只检查本批题面的规范化完全重复。

## 暂不默认下载的候选

| 来源 | 核实情况 | 决策 |
|---|---|---|
| [NLP4LP / OptiMUS](https://github.com/teshnizi/OptiMUS) | HF `udell-lab/NLP4LP` 元数据可读；固定版本 train.jsonl 实际请求返回 401；卡片声明 CC-BY-NC-4.0 | 先核实访问授权，不能承诺匿名下载；不借镜像绕过访问限制 |
| [MIPLIB-NL](https://github.com/optsuite/MIPLIB-NL) | 官方说明 223 个工业 MILP，有题目、数学模型、数据及参考脚本；中位变量数 2,183 | 有价值的后续大规模测试；当前二元接口/512 编译位限制不适配，完整文件下载与授权审计待做 |
| [OptMATH 原仓库](https://github.com/optsuite/OptMATH) | 官方仓库可访问 | 本轮固定 SIRL 的 166 题发布版本；原仓库其他版本不混合 |

集合数据卡声明 Apache-2.0；这不自动证明所有上游素材均采用同一许可。
保留 README、来源和版本，论文引用原始 benchmark 与 SIRL 修订来源。再分发前核对上游条款。
没有核实到其他可信国内镜像，不使用未知 GitHub 代理站。

## CPU 联网端运行

先将本次新增的 `tools/benchmarks/` 更新到服务器代码目录，再运行：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash tools/benchmarks/prepare_public_cpu.sh
```

默认输出：`/inspire/hdd/global_user/czxs25210022/why/data/public_benchmarks_v1/<revision>/`。
默认国内镜像优先，失败重试后回退 HF 官方站；固定文件 SHA256 不一致不接受。
如服务器只能访问国内镜像，可禁止官方回退：

```bash
bash tools/benchmarks/prepare_public_cpu.sh --endpoint https://hf-mirror.com
```

不重新配置环境、不下载模型、不启动训练，不修改正在进行的实验或 prepared manifest。
脚本直接使用已有环境 Python，找不到时使用系统 Python 3，仅依赖标准库。
遵循固定版本/本地目录的 Hub 下载流程；小型静态文件采用匿名 HTTPS，不需安装或升级 hf CLI。
不发送 HF_TOKEN 或本地 Hub 登录凭据。代理如有必要由用户设置 `HTTPS_PROXY`；脚本不固化本机代理。
支持 `QIHC_BASE`、`QIHC_PUBLIC_DATA`、`QIHC_DOWNLOAD_PYTHON` 覆盖路径。

下载中断后重跑同一命令，正确文件跳过；损坏文件仅在新文件校验成功后替换。
不提供逐字节续传（文件均小于 1 MB）。若进程被强杀留下 `.download.lock`，确认没有下载进程后才能移除锁。

```bash
# 离线复核，无网络请求
bash tools/benchmarks/prepare_public_cpu.sh --verify-only
```

| 输出 | 作用 |
|---|---|
| raw/ | 原始数据和 README，保持字节不变 |
| source.lock.json | 源版本、字节数、SHA256、发布许可说明 |
| manifest.json | 本次准备状态及来源；先检查是否成功 |
| audit.json | 条数、字段、数值标签数、重复题组、适配状态 |
| questions.jsonl | 只有题面及来源标识，供后续模型输入适配 |
| reference_answers.DO_NOT_PROMPT.jsonl | 单独保存答案，禁止输入模型/训练/反馈 |

成功状态是 `download_verified_adaptation_pending`，不是 `formal_ready`。
失败时历史 audit 文件可能仍在，必须以本次 manifest.json 状态为准。

## 能量表示升级与正式测评门槛

当前 `qihc/generic/energy.py` 已支持二次目标、线性等式/不等式、整数系数、平方松弛惩罚。
限制为至多 96 个原始二元变量、128 条约束、编译后 512 比特（软件限制，并非 128 位 FPGA 配置）。

| 升级 | 方法 | 必须验证 |
|---|---|---|
| 有界整数原始变量 | 有限区间平移及精确二进制/有界权重编码 | 解码保持上下界，编译前后目标和可行集合一致 |
| 逻辑及高阶项 | 逻辑重述、辅助变量、降为二次项 | 辅助变量约束正确，位数和惩罚跨度不过度增长 |
| 混合连续变量 | 有界网格离散化，或离散 p-bit + 连续子问题求解器 | 明确近似误差；混合方法单独命名，不能称纯 p-bit |
| 超过硬件位数 | 固定块外变量求条件子问题，滚动分块搜索 | 跨块约束、全局解档案、收敛与预算，不声称分块等价全局最优 |
| int8 真机 | 局部缩放/量化、偏置校验、精度感知惩罚 | 量化后目标不被压为零、原始约束实测可行率、数值误差 |

建议先实现有界整数和规模/精度审计，再做连续混合及复杂非线性；不要仅放宽数字上限。
惩罚权重大并不保证有限时搜索成功，量化后也可能破坏理论能量间隔。
对可精确编译的小问题穷举原变量和辅助变量，验证对辅助变量取最小时能量与原模型一致。
对离散化问题区分原连续最优值与网格最优值。

正式测试之前按问题类型、编码后位数和验证器可用性预先固定覆盖范围，不能按求解成功与否筛选。
报告总数、覆盖数、不支持数、超时、可行率、目标值/差距及端到端成本。
参考约束模型需独立审核，不能用被评测 LLM 自己生成的约束证明它理解正确。
反馈、SFT、DPO比较使用相同测试题/预算；若底层更换为混合求解，所有消融保持一致。
