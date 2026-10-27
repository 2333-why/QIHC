# 通用自然语言建模组件

- `protocol.py`、`primitives.py`：格式解析、数学合法性和原文引用审计。
- `sources.py`、`source_view.py`、`semantics.py`：精确字面数据引用、紧凑目录、模型侧诊断。
- `builder.py`：需求计划、目标、约束组、局部修订的确定性组装和校验。
- `prompts.py`：通用短提示和语义检查清单；不按题型选择公式。
- `backend.py`：离线 Transformers 模型加载与文本输入。
- `formatting.py`、`token_bridge.py`：LMFE 核心与公开生成 API 的连接。
- `generation.py`、`workflow.py`：有限预算生成、按 schema 合批、失败重试及子步骤检查点。
- `typed_sources.py`、`typed_schema.py`：按原文数据定义类型明确的引用与显式编号映射，不猜测编号偏移。
- `typed_builder.py`、`typed_workflow.py`、`local_repair.py`：选择启用的审计分离、逐行 schema 和局部字段纠错，默认旧路径保持兼容。

从历史 v5/v6 抽取，历史模块冻结以支持旧实验续跑。新代码不依赖 `tools.runtime` 或 CVRP 模块。
历史提示仍作为具名基线保留；新提示由请求中的 `prompt_style` 显式选择。
请求仅携带原题、共享变量布局与源数据目录；独立评分数据不进入此层。
格式成功、数学合法、自报语义审计通过与原题正确性是不同指标。
