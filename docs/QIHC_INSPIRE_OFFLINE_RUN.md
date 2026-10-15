# Inspire：联网 CPU 准备，离线 GPU 一键实验

默认个人目录：`/inspire/hdd/global_user/czxs25210022/why`。

## 已完成准备：单卡一行启动

非 thinking 修复：CVRP 邻域、冷启动、直接大模型基线、语义解析（共用 selector）和后训练统一传入 `enable_thinking=False`，避免思考文字耗尽输出预算。审计新增生成 token 数和是否达到长度上限；不跳过真实模型验收，也不扩大 token 预算或把备用策略记成成功。

从已验收的上一版更新后，单卡入口会自动进行一次**离线代码复验**：只接受固定旧版→本次固定新版源码指纹，重新核对已有包版本、权重、数据，运行全套测试及微型真实 SFT/DPO/GRPO 与适配器推理；全部通过后备份原清单为 `artifacts/prepared.before-non-thinking.json`，记录迁移证据并更新代码指纹。失败不会改动原清单。该过程不下载、不安装、不重做基准数据准备，之后仍须通过 H100 上的真实模型试跑；未知版本或自行修改的源码不会被自动放行。

在已分配一张 H100 的离线 GPU 节点，仓库更新后，在仓库目录只需运行：

```bash
bash scripts/run_single_h100.sh
```

该入口固定使用一个 GPU 进程，自动启用已有 Conda 环境，核验准备清单后运行完整 formal 流水线，包括实验对比、教师造数、SFT/DPO/GRPO 和复评。无需手动激活环境或设置 `QIHC_NPROC`，也不自动申请 GPU。

获取更新先执行 `git pull --ff-only origin llm-cvrp`。已成功完成第一步的用户，通过上面的固定版本离线复验复用已有准备产物。模型仍为准备清单中的 9B，实验预算不缩减；单卡 GRPO 默认一次生成 4 条回答，长序列训练的实际显存需在服务器验收，不能保证一张 H100 上所有阶段均不溢出。

可用 `bash scripts/run_single_h100.sh --print-plan` 查看计划，或加 `--preflight-only` 仅验收。下文原始第二脚本仍保留双卡默认值；单卡用户使用上述新入口。

平台任务配置为一个 worker、一张 H100 时，可直接将 `bash scripts/run_single_h100.sh` 作为启动命令。入口接受平台注入的 rank=0、world size=1 元数据，确认不是多进程后清除外层分布式变量，由内部验收自行启动 torchrun。若声明多个进程或非零 rank，则仍拒绝启动，避免重复实验。`CUDA_VISIBLE_DEVICES` 保持平台原值。

GPU 容器没有 `nvidia-smi` 时，入口仅对该信息展示命令提供提示性兼容函数；不会跳过后续 PyTorch CUDA、BF16、NCCL 和 ZeRO-3 实际验收。若 CUDA 确实不可用，实验仍会失败。此次兼容修复仍只修改不参与准备源码指纹的入口和说明，不需重跑第一步。

## 两个脚本

- `scripts/server/01_prepare_online.sh`：安装 Miniconda、创建 Python 3.11 环境、安装 CUDA 版 PyTorch 与训练依赖、下载模型及评测数据、执行 CPU 检查。
- `scripts/server/02_run_offline_gpu.sh`：核验准备产物、检查 GPU 和分布式训练，运行完整 V4 流水线。GPU 阶段不安装依赖、不下载模型或评测数据。

默认配置为 **Qwen/Qwen3.5-9B、单节点两张 H100、完整 formal 实验**。两阶段无需额外环境变量，只需各运行一次对应脚本。环境使用 Torch 2.9.1、CUDA 12.8、Transformers 5.13.0、PEFT 0.21.2、TRL 1.14.2。固定核心版本见 `configs/server-requirements.lock.txt`，完整解析后的依赖另存 pip/conda 清单。

## 1. 在联网 CPU 节点准备

先在个人目录克隆**包含本次更新的完整代码版本**。示例假定目录名为 `qihc`，大小写或其他名称不影响脚本从自身位置定位仓库。

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc
bash scripts/server/01_prepare_online.sh
```

无需 sudo，不执行 `conda init`，不修改 `.bashrc`。Miniconda 使用固定安装包和官方 SHA256；环境使用 conda-forge/NVIDIA 显式渠道。脚本在 CPU 节点也安装 **GPU 版** torch，不能换成 CPU wheel。

若旧版在 Conda 激活时出现 `SYS_SYSROOT: unbound variable`，更新代码后重跑第一个脚本即可，无需删除环境。当前脚本仅在 Conda 钩子执行期间关闭 nounset，完成后恢复严格模式；真实的激活错误仍会中止。

模型下载使用明确的文件清单，包括全部权重分片，避免将多个 `--include` 模式误解析为文件名。下载后继续检查完整性，CLI 显示成功不等于准备完成。若旧版缺少 `click` 或只下载了配置文件，更新后重跑即可，保留已有缓存，无需删除模型目录。

**当前默认从 `https://hf-mirror.com` 下载公开模型**，获取清单与下载使用同一 endpoint，记录来源和不可变 revision。默认关闭 Xet，使用 1 个下载线程，最多重试 3 次；分片大小、LFS 哈希（源提供时）与 safetensors 索引检查不变。完整文件可复用，未完成文件是否续传取决于服务响应与下载缓存，无需主动删除缓存。

HF-Mirror 是第三方镜像，默认不向其转发环境或缓存里的 HF token。元数据请求明确匿名，下载子进程移除 token 环境变量并关闭隐式认证。公开 Qwen 模型无需 token。镜像元数据可达不代表节点的所有大文件连接都会稳定，文件地址也可能经过重定向。

准备步骤包括：

1. Miniconda、Conda Python、编译器和匹配 CUDA toolkit。
2. CUDA torch/torchvision、固定核心依赖和本项目。
3. 将 Hub revision 解析为不可变 commit，再由 `hf download` 下载本地文件。
4. 校验分片大小、SHA256、远端 LFS SHA256（提供时）、safetensors 索引与文件头。
5. 下载 CVRPLIB 的 200–399 与 400–1000 客户数据，各预备最多 24 个候选（12 主选＋12 备用，按可用数量截断），各准备 12 个带业务约束实例。保持原车辆数上限；主选参考解不合格时，补选规模最接近的合格备用实例。之后 GPU 节点只使用本地 JSONL。
6. 检查离线 tokenizer/processor/config、模型类、训练 API，执行全套测试、真实微型 SFT/DPO/GRPO 与适配器推理、全分支 CPU 冒烟。
7. 全部成功才生成 `artifacts/prepared.json`。它记录代码、包版本、权重与数据指纹。

CPU 节点不会加载完整 9B 模型做前向，也不能验证显卡驱动、NCCL 或正式训练显存。微型 GRPO 只检查执行链路，不表示能力提升。

目录组织：

```text
why/
  qihc/                         代码仓库（名称可不同）
  miniconda3/                   Conda 安装
  envs/qihc-cu128/              共享环境
  models/Qwen3.5-9B/            权重
  data/research_v4/             原始数据与本地 JSONL
  cache/                        Conda / pip / HF / 编译缓存
  artifacts/prepared.json       准备完成的核验清单
  artifacts/preparation-*/      CPU 检查、训练检查及版本记录
  logs/                         两阶段日志
  results/                      正式实验、训练权重与报告
```

保证两个节点能访问同一个共享目录，均为 Linux x86_64，并具有兼容的系统 ABI。需预留模型、CUDA 环境、下载缓存和实验输出空间，建议开始前检查至少约 200 GiB 可用空间及账户配额；实际占用以模型清单为准。

## 2. 在已分配 GPU 的离线节点执行

先通过平台正常申请单节点多卡资源，再在该节点运行。脚本不申请或提交调度任务，不能在普通 CPU 登录节点上运行。

第二个脚本默认自动验收后进入完整实验，无需另跑第三条命令。仅需诊断环境时，可单独验收：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc
bash scripts/server/02_run_offline_gpu.sh --preflight-only
```

验收包括：全部准备产物的校验、CUDA/BF16、NCCL all-reduce、真实 ZeRO-3 小模型优化步，以及完整 Qwen 模型的短生成。完整模型生成在一张卡上执行，与当前推理架构一致。

正式一键运行：

```bash
bash scripts/server/02_run_offline_gpu.sh
```

不指定 `QIHC_NPROC` 时固定使用 2 个 GPU 进程。少于 2 张可见卡时验收会报错，不会静默改成单卡。脚本尊重调度器设置的 `CUDA_VISIBLE_DEVICES`，不自行占用其他卡。**脚本只运行一次**，内部自行调用 torchrun；不要再按每卡启动一份脚本。

正式流程包括已有 V4 的方法/消融、预算比较、冷启动、语义前端、教师轨迹、训练数据检查、SFT/DPO/GRPO 与各阶段复评。训练使用单节点 torchrun＋ZeRO-3，推理实验使用独立 GPU 分片，不是模型张量并行。

**每个推理进程加载完整 9B 模型，两卡通过任务分片提高吞吐。** 后训练使用两卡 ZeRO-3。GRPO 全局生成批量默认设为 4，即双卡每卡 2 条回答，避免原自动设置一次生成过多回答；梯度累积仍为 8，生成批量和优化器累积解耦。完整方法、数据划分、训练阶段、上下文上限与输出 token 上限保持不变。

目标资源为两张独占的 H100 80GB。实际显存还取决于提示长度、模型、参考策略和生成预算，目标服务器上尚未完成正式长实验验收。短生成和微型 ZeRO-3 验收通过不等于全部长序列训练已通过，任何阶段显存或数据检查失败都会中止并保留日志。

## 参数与恢复

快速检查路径及执行计划，不下载、不启动任务：

```bash
bash scripts/server/01_prepare_online.sh --print-plan
bash scripts/server/02_run_offline_gpu.sh --print-plan
```

如果服务器驱动适合 CUDA 12.6 而非 12.8，两阶段统一指定：

```bash
QIHC_CUDA_WHEEL=cu126 bash scripts/server/01_prepare_online.sh
QIHC_CUDA_WHEEL=cu126 bash scripts/server/02_run_offline_gpu.sh
```

CUDA wheel/toolkit 不会安装或升级 NVIDIA 驱动。驱动不匹配时联系平台或选择兼容版本。

其他常用覆盖项：

- `QIHC_MODEL_ID`、`QIHC_MODEL_REVISION`、`QIHC_MODEL_DIR`：仅在联网准备阶段指定，离线阶段读取准备清单中的实际路径。
- `QIHC_HF_ENDPOINT`：覆盖默认镜像；例如设为 `https://huggingface.co` 使用官方源。脚本在下载前设置标准 `HF_ENDPOINT`。
- `QIHC_HF_DOWNLOAD_WORKERS=1`、`QIHC_HF_DOWNLOAD_ATTEMPTS=3`：下载线程和尝试次数（1–10）。元数据网络请求也有有界重试。
- `HF_TOKEN`：官方源可以使用。镜像默认匿名；只有明确设置 `QIHC_MIRROR_ALLOW_TOKEN=1` 才允许第三方接收认证信息，请自行判断信任边界。脚本不记录 token。
- `QIHC_PROFILE=pilot`：较小计算预算试跑，仍使用真实大模型并执行训练。
- `QIHC_TRAINING_STEPS=200`：每阶段训练步数，默认 200。
- `QIHC_GRPO_GENERATION_BATCH_SIZE=4`：全局 GRPO 生成批量，须同时被 GPU 进程数和每提示回答数 4 整除。
- `QIHC_OUTPUT=/完整路径`：指定结果目录。配置和代码不变时，复用同一路径可恢复已成功阶段；变化时必须用新目录。

预下载数据失败会中止，不以缺少实例的文件作为准备成功。代码、依赖或模型改变后，需重新执行联网准备，避免用旧清单运行。

数据准备仅使用本轮下载清单中的候选，不扫描混入的旧缓存。通过验证的原主选实例保留，备用实例只填补失败主选的位置；没有依据求解算法的表现挑选实例。`medium.manifest.json`、`hard.manifest.json` 保存候选、详细违反项及替换关系。不增加车辆数、不导出参考路线，不足 12 个合格实例仍会中止。

## 当前验证边界

本地已验证两个脚本的 Bash 语法与计划输出、Conda 激活兼容、双卡默认配置、GRPO 批量、下载清单、镜像 token 隔离及严格车辆上限下的备用补选，全套共 140 项通过。镜像实际元数据请求已返回既定 revision 和 4 个权重分片；此前同版本 CLI dry-run 已识别权重约 19.3 GB。本地未下载这些大权重，也未完成目标服务器 CUDA/NCCL/ZeRO-3 或正式长实验验收。

本次重新执行的 [微型真实训练记录](../deployment/results/server_launcher_training_api_20261010/summary.json)已完成三阶段训练、权重保存与适配器推理；[CPU 全分支记录](../deployment/results/server_launcher_smoke_20261010/research_results.json)包含 19 个分支、76 个任务，失败 0。这些是流程验证，不是正式模型能力提升或目标服务器验收。

双卡适配后，[显式 GRPO 生成批量微训练](../deployment/results/two_h100_generation_batch_api_20261010/summary.json)也已完成，[双卡完整命令计划](../deployment/results/two_h100_formal_plan_20261010/planned_execution.json)保留教师造数、SFT/DPO/GRPO 与全部复评。计划核验不是双卡真机执行记录。

一键入口运行的是现有 V4。此前讨论的分布蒸馏、加权 DPO 和联合进化等新后训练研究方案，尚未全部实现，不能将当前普通 DPO/GRPO 命名为那些新算法。

远端克隆请选择 `llm-cvrp` 分支。完整版本包含 `scripts/server/`、`experiments/server_readiness.py`、固定依赖清单和更新后的 V4 源码，不能只复制两个 shell 文件到旧版仓库。

依赖依据：[Miniconda 官方安装包与 SHA256](https://repo.anaconda.com/miniconda/)、[PyTorch 官方 CUDA wheel 版本](https://pytorch.org/get-started/previous-versions/)、[Hugging Face 固定 revision 下载](https://huggingface.co/docs/huggingface_hub/guides/download)、[DeepSpeed 离线 JIT 编译所需环境](https://www.deepspeed.ai/tutorials/advanced-install/)。
