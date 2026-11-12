# p-bit × LLM × 求解器文献库（2026-11-12）

配套[深度调研与重合度分析](PBIT_LLM_SOLVER_FRONTIER_20261112.md)。本库用于查文献与复核，不把跨论文成绩合并排名。

纳入25项核心文献；22份开放PDF成功下载，并验证文件头、可解析页数与首页题名；3项没有本地PDF。另记录8项排除/降级示例。这不是穷尽检索或PRISMA系统综述，未统计全部原始搜索命中。

PDF保存在本机F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/，未纳入Git。文件名、来源和SHA256见[JSON清单](PBIT_LLM_SOLVER_SOURCES_20261112.json)。本地路径在其他机器上不自动存在；可从各行公开源获取。

## 范围与检索

纳入：自然语言到优化模型、语义反馈、求解器监督后训练、运行时控制、启发式发现，以及与条件采样/温度/惩罚控制直接有关的p-bit工作。排除：仅LLM量化、纯器件但无算法交集、二手文章无原文核实、无关物理任务。机制奠基论文不受近两年限制。

- 精确交集：“"p-bit" "large language model" optimization”；“"LLM" "p-bits" solver control”；“"probabilistic computing" "LLM" feedback”。
- 自然语言建模与修复：“"optimization modeling" "counterexample"”；“"Falsification-Based Verification" optimization”；“"LLM-QUBO" arxiv”。
- 求解反馈后训练：“"optimization modeling" "reinforcement learning" 2026”；“"solver-in-the-loop" "PEARL"”；“"Multi-Turn RL" "OptiMind"”。
- 在线搜索控制：“"Representing Search State" "Large Language Model"”；“"reinforcement learning" "annealing schedules"”；“"Controlled Online Optimization Learning"”。
- 算法生成与后训练：“"ReEvo" "Evolution of Heuristics"”；“"CALM" "Co-evolution of Algorithms"”。
- p-bit算法硬件：“p-bit probabilistic computing combinatorial optimization parallel tempering 2025 2026”；“"p-bit" "reinforcement learning" annealing”；“"p-bit" update dynamics”。

去重按DOI/arXiv编号优先、再按题名与版本关系；OMSO不同题名版本不重复计算。新预印本不标成已通过同行评审。

## 文献总表

作者依所核对版本列出；正式版本与本地预印本可能不同，逐篇说明。主要内容与创新点是简要转述，阅读深度见下一节。

| ID / 论文标题 | 关键词 | 作者 | 发表期刊/会议或状态 | 主要内容 | 创新点 | 论文链接 | PDF状态 | 本地PDF路径 |
|---|---|---|---|---|---|---|---|---|
| R01 COVER: Counterexample-Guided Repair for Natural-Language Constraint Modelling | 反例;typed IR;CP-SAT;局部修复 | Ziang Wu | SoMeT / FAIA，正式出版 | 验证器给反例，定位到约束后修补IR | 将语义证据转换为局部补丁 | [原始来源](https://journals.sagepub.com/doi/10.3233/FAIA260737) | 下载失败；HTML可读 | — |
| R02 ORSMT: Unleashing the Potential of Satisfiability Modulo Theories for Semantic Verification and Repair in LLM-Based Optimization | SMT;语义关系;反例;源约束 | Rui Zhang; Chenyu Zhou; Jianghao Lin; Yaoxin Wu; Dongdong Ge; Wen Song | arXiv v1，10-21，预印本 | 对数学模型、代码和报告建立精确关系查询 | 投影辅助变量；反例定位与修复后复核 | [原始来源](https://arxiv.org/abs/2610.29145v1) | 已验证，41页 | [ORSMT_R02_2026_2610.29145v1.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/ORSMT_R02_2026_2610.29145v1.pdf) |
| R03 SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling | 语义锚点;反向重建;局部纠错 | Yansen Zhang; Qingcan Kang; Yujie Chen; Yufei Wang; Xiongwei Han; Tao Zhong; Mingxuan Yuan; Chen Ma | arXiv v3；作者标注ICML2026录用 | 从代码反推语义并与原题锚点对齐 | 只修正不一致部分，无需额外训练 | [原始来源](https://arxiv.org/abs/2510.05115v3) | 已验证，17页 | [SAC-Opt_R03_2026_2510.05115v3.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/SAC-Opt_R03_2026_2510.05115v3.pdf) |
| R04 SemOPT: Fixing Semantic Errors in LLM-based Optimization Modeling via Reward-Guided Search | 语义奖励;分层搜索;纠错 | Zetong Zhou; Wentao Zhang; Jingyuan Wang; Yifan Yang; Zizhuo Wang; Shixi Hu | arXiv v1；作者标注EMNLP2026 Findings录用 | 学习语义奖励，指导建模空间的分层搜索 | 训练判别信号与推理纠错结合 | [原始来源](https://arxiv.org/abs/2609.37361v1) | 已验证，17页 | [SemOPT_R04_2026_2609.37361v1.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/SemOPT_R04_2026_2609.37361v1.pdf) |
| R05 Falsification-Based Verification of LLM-Generated Optimization Models: Sound Test Batteries and Their Detection Limits | 无参考;变形测试;敏感性;检测边界 | Haifeng Li; Mo Hai | arXiv v1，07-18，预印本 | 对参数变换后模型检查理论应满足的关系 | 明确可检测错误与不可检测盲区 | [原始来源](https://arxiv.org/abs/2607.16646v1) | 已验证，32页 | [Falsification_R05_2026_2607.16646v1.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/Falsification_R05_2026_2607.16646v1.pdf) |
| R06 Admission Without Answers: Label-Free Certification and Experience Learning for LLM-Based Optimization Modeling | 经验库;准入;多模型;重采样 | Junbo Jacob Lian; Huiling Chen; Hanzhang Qin; Chung-Piaw Teo | arXiv v6，11-07，预印本 | 跨模型族、参数重采样后的目标一致性决定经验准入 | 允许接受、弃权和升级核验 | [原始来源](https://arxiv.org/abs/2608.15565v6) | 已验证，28页 | [AdmitOR_R06_2026_2608.15565v6.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/AdmitOR_R06_2026_2608.15565v6.pdf) |
| R07 Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling | RLVR;求解验证;LP;造数 | Yitian Chen; Jingfan Xia; Siyu Shao; Dongdong Ge; Yinyu Ye | NeurIPS2025；arXiv v3 | 用执行、可行性和解质量信号训练建模器 | 求解器验证也用于合成训练数据 | [原始来源](https://arxiv.org/abs/2505.11792v3) | 已验证，37页 | [SIRL_R07_2025_2505.11792v3.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/SIRL_R07_2025_2505.11792v3.pdf) |
| R08 Learning to Model Optimization Problems from Solver Outcomes | 结果奖励;RL;跨求解器 | Runda Guan; Xiangqing Shen; Jiajun Zhang; Yifan Zhang; Jian Cheng; Rui Xia | arXiv v2，10-28，预印本 | 执行solver-specific代码并用结果标量奖励训练 | 测试Gurobi、OR-Tools、COPT间迁移 | [原始来源](https://arxiv.org/abs/2604.00442v2) | 已验证，10页 | [OMSO_R08_2026_2604.00442v2.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/OMSO_R08_2026_2604.00442v2.pdf) |
| R09 PEARL: Solver-in-the-Loop Interactive Optimization Modeling from Natural Language | 多轮RL;工具调用;修复;停止 | Hongliang Lu; Zhong Li; Yuxuan Chen; Yuan Lan; Fan Zhang; Zaiwen Wen | arXiv v2，11-08，预印本；首发日期字段异常见正文 | 学习何时执行、修订、结束建模 | 以完整交互轨迹训练策略 | [原始来源](https://arxiv.org/abs/2607.18256v2) | 已验证，34页 | [PEARL_R09_2026_2607.18256v2.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/PEARL_R09_2026_2607.18256v2.pdf) |
| R10 Multi-Turn RL Makes Small Language Model Competitive for Optimization Modeling | 多轮RL;数据清洗;类别先验;Gurobi | Xinzhi Zhang; Zeyi Chen; Humishka Zope; Hugo Barbalho; Konstantina Mellou; Marco Molinaro; Jana Kulkarni; Ishai Menache; Sirui Li | Microsoft官方页标注NeurIPS2026，10月；会场页抓取失败 | 清洗模型数据、接受目标集合、多轮修订训练 | 将数据审计和solver-grounded策略学习结合 | [原始来源](https://www.microsoft.com/en-us/research/publication/multi-turn-rl-makes-small-language-model-competitive-for-optimization-modeling/) | 未确认开放PDF | — |
| R11 Strategy-Aware Optimization Modeling with Reasoning LLMs | 多策略;验证SFT;分段GRPO;求解效率 | Ruiqing Zhao; Fengzhi Li; Yuan Zuo; Rui Liu; Yansong Liu; Yunfei Ma; Fanyu Meng; Junlan Feng | ICML2026 / PMLR306，正式论文集 | 验证多种建模策略，再用组合奖励后训练 | 显式策略与格式、正确性、效率共同学习 | [原始来源](https://proceedings.mlr.press/v306/zhao26z.html) | 已验证，25页 | [SAGE_R11_2026_zhao26z.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/SAGE_R11_2026_zhao26z.pdf) |
| R12 Reinforcement Learning via Self-Distillation | 自蒸馏;富反馈;token监督 | Jonas Hübotter; Frederike Lübeck; Lejs Behric; Anton Baumann; Marco Bagatella; Daniel Marta; Ido Hakimi; Idan Shenfeld; Thomas Kleine Buening; Carlos Guestrin; Andreas Krause | arXiv v2，02-16，预印本 | 反馈条件化自教师产生稠密训练信号 | 无外部教师的反馈蒸馏 | [原始来源](https://arxiv.org/abs/2601.20802v2) | 已验证，50页 | [SDPO_R12_2026_2601.20802v2.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/SDPO_R12_2026_2601.20802v2.pdf) |
| R13 LLM-QUBO: An End-to-End Framework for Automated QUBO Transformation from Natural Language Problem Descriptions | 自然语言;QUBO;Benders;经典求解 | Huixiang Zhang; Mahzabeen Emu; Salimur Choudhury | arXiv v1；另检出AAAI春季研讨会版本，未作同版认定 | 自然语言到结构模型、QUBO和分解求解 | 将表达转换与混合分解结合 | [原始来源](https://arxiv.org/abs/2509.00099v1) | 已验证，8页 | [LLM-QUBO_R13_2025_2509.00099v1.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/LLM-QUBO_R13_2025_2509.00099v1.pdf) |
| R14 Representing Search State to a Large Language Model: A Structured Telemetry Interface for Runtime Control of Adaptive Large Neighbourhood Search | 搜索遥测;运行时控制;ALNS;温度 | Mehdi Raoofi; Julien Maheut; Flavio Tonelli | Systems14(10),1200；09-25正式上线 | 分段状态摘要驱动有界控制动作 | LLM只调搜索控制，求解和可行性独立执行 | [原始来源](https://www.mdpi.com/2079-8954/14/10/1200) | 已验证，37页 | [LLM-ALNS_R14_2026_systems14101200.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/LLM-ALNS_R14_2026_systems14101200.pdf) |
| R15 Controlled Online Optimization Learning (COOL): Finding the ground state of spin Hamiltonians with reinforcement learning | PPO;Ising;退火;温度控制 | Kyle Mills; Pooya Ronagh; Isaac Tamblyn | Nature Machine Intelligence2,509–517；arXiv v2 | RL根据搜索状态学习加热和降温 | 学习非固定退火调度并测试泛化 | [原始来源](https://arxiv.org/abs/2003.00011v2) | 已验证，13页 | [COOL_R15_2020_2003.00011v2.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/COOL_R15_2020_2003.00011v2.pdf) |
| R16 Neural Simulated Annealing | SA;小网络;提议分布;RL | Alvaro H.C. Correia; Daniel E. Worrall; Roberto Bondesan | AISTATS2023 / PMLR206；预印本2022 | 学习SA的邻域提议分布 | 以固定预算提高解质量并考察跨规模 | [原始来源](https://arxiv.org/abs/2203.02201) | 已验证，17页 | [NeuralSA_R16_2023_2203.02201.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/NeuralSA_R16_2023_2203.02201.pdf) |
| R17 ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution | 反思;进化;启发式;代码 | Haoran Ye; Jiarui Wang; Zhiguang Cao; Federico Berto; Chuanbo Hua; Haeyeon Kim; Jinkyo Park; Guojie Song | NeurIPS2024，正式论文集 | LLM生成和反思启发式，进化择优 | 把语言反馈用于算法空间搜索 | [原始来源](https://arxiv.org/abs/2402.01145) | 已验证，38页 | [ReEvo_R17_2024_ReEvo2024.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/ReEvo_R17_2024_ReEvo2024.pdf) |
| R18 Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model | 思想;代码;进化;启发式 | Fei Liu; Xialiang Tong; Mingxuan Yuan; Xi Lin; Fu Luo; Zhenkun Wang; Zhichao Lu; Qingfu Zhang | ICML2024 / PMLR235 | 共同进化算法思路与实现 | 以可执行评价筛选启发式 | [原始来源](https://proceedings.mlr.press/v235/liu24bs.html) | 已验证，23页 | [EoH_R18_2024_liu24bs.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/EoH_R18_2024_liu24bs.pdf) |
| R19 CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design | 启发式;共进化;GRPO;7B | Ziyao Huang; Weiwei Wu; Kui Wu; Jianping Wang; Wei-Bin Lee | ICLR2026，正式论文集；预印本2025 | 启发式评价反馈用于LLM后训练 | 算法搜索与生成器参数共同改善 | [原始来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/75ed738c083f8941cfc990245e4f6265-Abstract-Conference.html) | 已验证，30页；2025预印本v1 | [CALM_R19_2025_2505.12285v1.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/CALM_R19_2025_2505.12285v1.pdf) |
| R20 Stochastic p-Bits for Invertible Logic | p-bit;钳位;Boltzmann;反向逻辑 | Kerem Yunus Camsari; Rafatul Faria; Brian M. Sutton; Supriyo Datta | Physical Review X7,031014 | 随机比特网络与钳位条件下的反向推断 | 可逆概率逻辑基础 | [原始来源](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.7.031014) | 已验证，19页 | [p-bits_R20_2017_PRX7031014.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/p-bits_R20_2017_PRX7031014.pdf) |
| R21 pc-COP: An Efficient and Configurable 2048-p-Bit Fully-Connected Probabilistic Computing Accelerator for Combinatorial Optimization | FPGA;2048p-bit;全连接;伪并行 | Kiran Magar; Shreya Bharathan; Utsav Banerjee | IEEE HPEC2024；预印本2025 | 可配置全连接概率求解加速器 | speculate-and-select更新架构 | [原始来源](https://arxiv.org/abs/2504.04543) | 已验证，10页 | [pc-COP_R21_2024_2504.04543.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/pc-COP_R21_2024_2504.04543.pdf) |
| R22 Extended-variable probabilistic computing with p-dits | p-dit;p-int;类别变量;整数 | Christian Duffee; Jordan Athas; Andrea Grimaldi; Deborah Volpe; Giovanni Finocchio; Ermin Wei; Pedram Khalili Amiri | arXiv v1；本报告按此版本分析 | 将概率单元扩展到多维和整数表示 | 减少复杂变量二值编码负担 | [原始来源](https://arxiv.org/abs/2506.00269v1) | 已验证，38页 | [p-dits_R22_2025_2506.00269v1.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/p-dits_R22_2025_2506.00269v1.pdf) |
| R23 Probabilistic Computers for MIMO Detection: From Sparsification to 2D Parallel Tempering | p-bit;PT;温度;惩罚;FPGA | M Mahmudul Hasan Sajeeb; Kevin Callahan-Coray; Corentin Delacour; Sanjay Seshan; Tathagata Srimani; Kerem Y. Camsari | arXiv v2，05-31，预印本 | 副本沿温度与约束强度两维交换 | 共同处理能量障碍和约束强度敏感性 | [原始来源](https://arxiv.org/abs/2601.09037v2) | 已验证，27页 | [2D-PT_R23_2026_2601.09037v2.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/2D-PT_R23_2026_2601.09037v2.pdf) |
| R24 Enhanced Convergence in p-bit Based Simulated Annealing with Partial Deactivation for Large-Scale Combinatorial Optimization Problems | 部分失活;振荡;TApSA;SpSA | Naoya Onizawa; Takahiro Hanyu | 关联Scientific Reports2024 DOI；arXiv2026版本，避免当全新2026方法 | 通过部分失活缓解并行更新振荡 | 时间平均与停滞式pSA | [原始来源](https://arxiv.org/abs/2601.15561v1) | 已验证，17页 | [TApSA-SpSA_R24_2024_2601.15561v1.pdf](F:/p-bit/QIHC-controller/output/literature_20261112/pdfs/TApSA-SpSA_R24_2024_2601.15561v1.pdf) |
| R25 A unified performance–cost landscape of parallel p-bit Ising machines based on update dynamics | 同步异步;时分复用;延迟;DAC | Naoya Onizawa; Takahiro Hanyu | Scientific Reports16,16643；04-08 | 比较更新策略、时延、复用与精度的成本质量关系 | 把硬件约束与调度联合分析 | [原始来源](https://www.nature.com/articles/s41598-026-47285-0) | 下载失败；HTML可读 | — |

## 阅读深度与使用限制

| ID / 方法 | 本轮核查深度 | 限制与注意事项 |
|---|---|---|
| R01 COVER | 正文方法及实验 | oracle模式需要参考模型；测试模式保证较弱 |
| R02 ORSMT | 正文方法及附录 | 制品一致不等于原题正确；源适配器有未决项 |
| R03 SAC-Opt | 摘要及作者元数据 | LLM语义判断不构成精确证明 |
| R04 SemOPT | 摘要 | 奖励模型可能误判；不能视为形式验证 |
| R05 Falsification | 正文方法、检测限制及Remark4.5 | 实现A5正确性依赖探测尺度；测试通过不证明全部正确 |
| R06 AdmitOR | 摘要 | 仍需solver-verified校准；界在基准流未成立 |
| R07 SIRL | 摘要 | 不能将执行或目标一致自动解释为完整语义等价 |
| R08 OMSO | 摘要及正文结构 | 是建模策略训练，不能直接当搜索温度控制 |
| R09 PEARL | 正文3及附录A/B | 终端主要核对参考目标；非逐约束语义认证 |
| R10 OptiMind | 官方摘要；未取得全文 | 完整训练数据与模型在页面中仍列待发布 |
| R11 SAGE | 官方摘要 | 仍要区分模型正确和最终求解成本 |
| R12 SDPO | 摘要 | 通用学习机制；不是p-bit或优化建模专用 |
| R13 LLM-QUBO | 正文框架及实验 | 实验主问题和子问题都用Gurobi；未证明量子/p-bit加速 |
| R14 LLM-ALNS | 正文方法、实验及提示附录 | 免训练；9个实例；同迭代优势不等于同墙钟优势 |
| R15 COOL | 摘要 | 非LLM；只覆盖所测Hamiltonian族 |
| R16 NeuralSA | 摘要 | 不是温度控制；适合作邻域学习对照 |
| R17 ReEvo | 官方论文摘要 | 离线算法设计与每段在线决策不同 |
| R18 EoH | 官方摘要 | 不是逐状态LLM控制或模型权重训练 |
| R19 CALM | 官方摘要、论文首页及作者代码 | 训练对象为启发式程序生成，不是固定p-bit动作策略；本地PDF为2025预印本v1；2026发表信息另据官方论文集，未声称两版完全相同 |
| R20 p-bits | 官方摘要 | 未涉及LLM；条件采样不是新提出的机制 |
| R21 pc-COP | 摘要及正式发表元数据 | 设备结果不可迁移成CPU模拟的加速宣称 |
| R22 p-dits | 摘要 | 特定问题/硬件收益不能外推至全部ILP |
| R23 2D-PT | 正文结构、摘要及硬件方法 | FPGA实测与ASIC投影须分列；非学习控制 |
| R24 TApSA-SpSA | 摘要及关联DOI | 规则控制；需匹配更新语义再比较 |
| R25 p-bit-update | 正文引言、方法与摘要 | 主要为模拟与规范化硬件成本；不是LLM后训练 |

取得PDF只表示能够进一步阅读，不表示已经逐页精读。COVER没有本地PDF，但出版社HTML提供正文，方法和实验已从该来源核查。OptiMind仅有官方摘要，不能据此补写训练细节。

## 优先阅读与项目映射

| 优先级 | 文献 | 对应项目问题 |
|---|---|---|
| 第一批 | R14 LLM-ALNS、R15 COOL、R23 2D-PT | E005-1.11：LLM控制、学习调温与强非学习对照 |
| 第一批 | R01 COVER、R02 ORSMT、R09 PEARL、R12 SDPO | P-VCD：反例修复、语义认证、交互训练与反馈蒸馏 |
| 第二批 | R19 CALM、R05 Falsification、R06 AdmitOR | 求解经验后训练、无参考验证、数据准入 |
| 第二批 | R07 SIRL、R08 OMSO、R10 OptiMind、R11 SAGE | 常见与最新建模后训练路线 |
| 背景 | R20–R25，及R17–R18 | p-bit的机制/硬件边界与启发式生成 |

## 排除/降级记录（8项示例）

以下是定向筛选记录，不代表其论文质量低；未核实的候选不用于关键论证。

| 候选 | 处理原因 |
|---|---|
| [OptiChat](https://pubsonline.informs.org/doi/full/10.1287/ijds.2025.0074) | 问题解释/交互支持相关，未列为本轮最接近的机制主对照 |
| [Counterexample Guided Inductive Synthesis Using Large Language Models and Satisfiability Solving](https://users.cs.fiu.edu/~sjha/papers/2023_Jha_MILCOM_Planning.pdf) | 保留为CEGIS背景；主表聚焦优化建模而非规划 |
| [RL-QESA](https://openreview.net/forum?id=4eAAIIBt3Q) | 连续函数/原子团簇，优先选择更贴近离散Ising的COOL |
| [Hardware Acceleration of Probabilistic Computing for Max-Cut Problems](https://ieeexplore.ieee.org/document/11668769/) | 硬件候选；本轮未完成全文核验，不作核心方法结论 |
| [A Noise-Based CMOS Probabilistic Bit for Combinatorial Optimization Problems](https://www.mdpi.com/2079-9292/15/12/2510) | 器件电路为主；本轮优先算法、控制及编译交集 |
| [Improving deep neural network performance through sampling](https://www.nature.com/articles/s44335-026-00063-7) | 采样用于神经网络推理，与LLM协同求组合优化的任务方向不同 |
| [IBM p-kit language-model demos](https://github.com/IBM/p-kit) | 软件项目线索，非同一研究任务；不作为已发表论文计数 |
| [Tree-Ising](https://arxivsignals.io/papers/2611.14210) | 仅检出二手索引且任务指向权重编辑，未核实原论文，排除 |

## 版本与可获得性审计

- R01：PDF请求超时；使用出版社HTML正文，不保存错误响应。
- R10：未确认本次2026论文的开放PDF；官方摘要标注NeurIPS2026，会议链接无法抓取，完整训练资源仍需核实。
- R19：OpenReview正式PDF下载失败，保存作者arXiv2025 v1；正式发表信息依据ICLR2026官方论文集，不假定两个PDF完全相同。
- R25：PDF URL返回非PDF内容，文件头校验拒绝保存；正文可从出版社HTML阅读。
- PEARL首发日期字段与编号前缀不一致，本库按v2修订时间引用；不据此判断首发优先权。
- pc-COP的正式年为2024，TApSA/SpSA关联2024 DOI；arXiv较晚上传不等于更晚发明。
- 22份成功文件均有SHA256；本地正文抽取仅用于核查，不把大段原文作为项目文档提交。

## 推荐Notion字段

标题、方法简称、作者、年份、正式发表状态、版本日期、研究路线、与QIHC重合度、监督来源、是否测试时使用参考、是否LLM权重更新、是否在线控制、是否真实硬件、阅读深度、代码可用性、论文URL、PDF路径、SHA256、重要限制、建议对照、下一步。

本文未执行Notion导入或同步。
