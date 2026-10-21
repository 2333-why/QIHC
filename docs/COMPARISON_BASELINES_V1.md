# 首轮基线对比：不训练、不做消融

## 本轮范围

默认60个既有问题实例＋240道公开题目，共300题。公开数据按每组40题等距抽取，不按求解成功与否筛选。
既有问题每个规模5种子：背包16/32/64、顶点覆盖16/24/40、分配3×3/5×5/7×7、平衡割16/32/64。
顶点覆盖网格保留旧数据生成器的128条约束限制；这不是新仿真后端的硬件位数限制。
本轮不含先前的CVRP路线实验：该任务使用不同路线输出/验证协议，不能冒充已接入当前统一整数实验。

| 方法 | LLM调用/题 | 求解后端 | 信息条件 | 复现定位 |
|---|---:|---|---|---|
| Direct LLM | 1 | 无外部求解器 | 仅题面 | 基础端到端基线 |
| LLM + CP-SAT | 1 | CP-SAT，默认5秒 | 仅题面；与两种闭环共享初始模型输出 | 基础建模求解基线 |
| Solver-feedback + CP-SAT | 2 | 初次CP-SAT→建模复核→CP-SAT | 反馈来自预测模型，不看参考答案 | 受solver-agent工作启发的统一接口实现，**不是OptiMUS复现** |
| QIHC | 2 | 初次因子能量p-bit→建模复核→p-bit | 同一初始模型，反馈来自原始约束违反量 | 我们的完整两轮仿真流程 |
| Gold CP-SAT | 0 | CP-SAT | 给定正确数学模型，仅既有合成问题 | 求解器参考，不能与NL端到端混作公平输入 |
| Gold simulated annealing | 0 | 单比特Metropolis＋退火 | 同上 | 基础随机优化基线 |
| Gold random scan | 0 | 随机扫描比特 | 同上 | 随机搜索基线 |

不运行无反馈p-bit或自我复核等消融，不做SFT/DPO。所有LLM方法默认使用已有Qwen3-32B基座，无adapter。
默认p-bit/退火/随机：3种子，每种子16链×4096次更新；CP-SAT每次5秒且1CPU线程。
这不是等FLOPs或等耗时预算；逐阶段保存时长、token数、步数，并累加两轮成本，不能只报第二轮。
3个采样种子是重复测量，不是取最好一个作反馈：固定第一个种子提供诊断。
CPU/GPU使用的文件不同：GPU只看requests，不读取保存gold和发布答案的cases.CPU_ONLY.json。

## 相关研究核实（2026-10-22）

| 工作 | 官方实现与评测方式 | 本轮处理 |
|---|---|---|
| [SIRL](https://github.com/Cardinal-Operations/SIRL) | 发布Gurobi/COPT专门权重；按作者提示词生成代码并执行，仓库明确要求求解器许可 | 已固定仓库及权重revision；用户暂无许可，因此不启用，不伪称复现 |
| [ORLM](https://github.com/Cardinal-Operations/ORLM) | 官方评测生成并执行COPT代码；公开8B权重，许可为Llama3 | 不默认下载不可执行的额外权重；待许可、隔离执行环境齐备再接入 |
| [OptiMUS](https://github.com/teshnizi/OptiMUS) | 有多个版本，侧重建模与求解器协作 | 本轮借鉴建模—求解—复核思路，不宣称复刻其agent/RAG流程或论文成绩 |
| [Chain-of-Experts](https://github.com/xzymustbexzy/Chain-of-Experts) | 原仓库要求OPENAI_API_KEY，提供专家协作实验入口 | 不在离线节点自动发API请求；本轮未复现 |

版本记录在 `tools/benchmarks/related_methods.lock.json`。本轮没有复制作者公布的成绩到我们的实测表。
没有把“将SIRL权重强制改为输出我们的JSON”冒充原版SIRL；更换格式会改变其方法，应另命名适配实验。
Gurobi/COPT没有许可并不意味着这些论文不可用，而是本次环境尚不具备原版执行复现条件。
官方要求生成代码执行的方案，还需隔离环境；本轮仅解析严格JSON，不执行LLM生成的Python。

## 两个启动脚本

请先把新版代码放到独立checkout，复用原环境和权重。默认目录：
`/inspire/hdd/global_user/czxs25210022/why`。

**CPU联网计算节点**（不是不允许计算的登录节点）：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash tools/benchmarks/01_comparison_cpu.sh
```

执行数据下载、已有模型/Tokenizer/CP-SAT检查、准备试题，然后保持运行CPU队列工作进程。
国内镜像优先，官方源回退；固定公开数据版本和SHA256。已有完整Qwen3-32B权重直接复用，不重新跑旧第一步。
没有本地模型时才用固定revision调用hf CLI下载；本脚本发起的中断下载可重试。
若原有模型损坏或缺文件，检查失败并明确退出，不擅自覆盖用户已有模型。
不重装conda、不安装新包、不训练。仍要求已配置的 `envs/qihc-cu128` 环境包含现有依赖。

看到 `CPU queue worker ready` 后，在GPU分布式任务的**单worker启动命令**填写：

```bash
cd /inspire/hdd/global_user/czxs25210022/why/qihc && bash tools/benchmarks/02_comparison_gpu.sh
```

直接调用已有环境Python，不需要手动conda activate，不依赖nvidia-smi。使用PyTorch检查CUDA。
模型在可见GPU上device_map分布，适用于之前申请的单worker多卡推理配置；不要额外包torchrun。
本机测试未覆盖真正的32B GPU加载/生成，此项需目标GPU首跑验证。

两个进程需要共享results目录，CPU进程在GPU运行期间必须存活。
CPU联网准备结束后仍运行搜索；不应关闭CPU终端。如果平台不允许联网CPU节点持续计算，应把CPU工作进程提交到CPU计算资源：

```bash
/inspire/hdd/global_user/czxs25210022/why/envs/qihc-cu128/bin/python tools/benchmarks/comparison.py cpu --run /inspire/hdd/global_user/czxs25210022/why/results/comparison_v1
```

## 回收与中断策略

- GPU进程只加载模型、读取请求、推理和写结果，不执行长CP-SAT/p-bit搜索或下载。
- CPU收到初始输出立即生成反馈请求；GPU分批读取请求队列，所有已完成输出原子落盘。
- 没有可推理任务持续45秒时，GPU主动退出并写入`waiting_for_cpu_resubmit_same_command`，避免空占GPU。
  这不是实验完成；待CPU产生新请求后重提同一GPU命令续接。
- SIGTERM/SIGINT请求在当前检查点后退出；SIGKILL最多丢失当前未保存的一次调用/搜索，不损坏旧结果。
- 独占OS文件锁会在进程退出时释放；不需要手工删除陈旧PID锁。共享文件系统需要支持文件锁及原子重命名。
- 不提供空矩阵乘法等虚假保活；心跳日志也不能保证平台不回收。
- **无法保证平台绝不回收或抢占。** 本设计消除长CPU阶段占GPU的原因，并提供同命令续跑。
  首轮模型加载、真实利用率阈值、资源时限等仍需遵守平台策略；请先确认管理员的回收阈值。

CPU或GPU停止后分别重跑对应脚本；已存在manifest时CPU不会重新下载。源码、模型元数据或案例变化拒绝混用旧结果。
修改配置需要使用新 `QIHC_COMPARISON_RUN`，不能绕过指纹检查。

## 查看结果

```bash
cat /inspire/hdd/global_user/czxs25210022/why/results/comparison_v1/summary.json
cat /inspire/hdd/global_user/czxs25210022/why/results/comparison_v1/gpu_status.json
```

日志在 `why/logs/comparison-cpu-*.log` 和 `comparison-gpu-*.log`；逐题完整结果在 `comparison_v1/final/`。
`progress/`保存各阶段检查点，`responses/`保留原始LLM输出和token/耗时，`requests/`保留实际提示词。
summary按照数据集分组，标明completed/expected；部分完成不冒充全量成绩。
合成问题按独立gold约束验证可行性和目标，发布数据只能先报告目标答案一致率，不能称独立语义正确率。
直接LLM在公开题目上的解描述尚未独立验证；即便目标值吻合也不能称可行解已验证。
公开题目失败/不支持仍保留在分母；连续和缺数据题目不可偷偷改为整数或补造输入。

## 验证

```bash
python -m pytest tests/test_comparison.py tests/test_public_simulation.py tests/test_bounded_energy.py tests/test_generic_energy.py tests/test_semantic_training.py tests/test_public_benchmarks.py -q
```

覆盖队列流转、两轮反馈不含答案、独立gold验证、CPU/GPU职责分离、结果续跑、请求篡改检测、成本累计及默认合成题网格。
测试中的生成器/部分求解调用使用桩，不等于32B真实推理已验证。
