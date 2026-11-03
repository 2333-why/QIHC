# 独立实验入口

新实验采用 `experiments/<实验名>_vN/` 独立目录，包含配置、CPU 准备、GPU 运行、报告入口及说明。

- [prompt_efficiency_v1](prompt_efficiency_v1/README.md)：五组提示词，固定编译与求解后端，评估建模质量和实际成本。
- [prompt_efficiency_v2](prompt_efficiency_v2/README.md)：保留原版一次建模，比较审计分离、类型明确的数据引用、显式编号映射和局部字段纠错。
- [feedback_pilot_v1](feedback_pilot_v1/README.md)：E004-1.2，18 题、四组、两轮反馈的快速筛查，复用 E003 同题七求解器评分；独立结果目录，不改变历史 runtime。
- [feedback_witness_v1](feedback_witness_v1/README.md)：E004-1.4，18 题六组两轮的主动语义反例反馈、原子规则替换、八小时运行监督与七后端交叉评价；结果已返回：建模无完整修复，缓存竞争已定位并在下一迭代修复。

- [feedback_witness_v2](feedback_witness_v2/README.md)：E004-1.5，六小时修复版，任务去重、紧凑代数、稳定来源片段、六组与 C0/GOLD 七后端对照；本地回归通过，服务器待运行。

通用建模位于 `qihc/modeling/`，共享实验基础组件位于 `qihc/experiment_runtime/`，
搜索算法继续位于 `qihc/generic/`。实验设置不得通过修改这些组件的模块全局变量临时注入。
每次改变共享代码都必须固定新的代码提交并使用独立结果目录，禁止混用旧检查点。

已有平铺脚本和 `tools/runtime/modeling_v2` 至 `modeling_v6` 作为历史可复现入口保留，
本次不迁移或删除，避免影响运行中实验及其源代码校验。
实验文档和时间线统一保存于 [docs/experiments](../docs/experiments/README.md)。
