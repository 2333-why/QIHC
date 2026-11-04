---
pretty_name: NLCO
tags:
- benchmark
- combinatorial-optimization
- reasoning
- llm
license: mit
task_categories:
- text-generation
size_categories:
- 1K<n<10K
---

# NLCO Benchmark Dataset

This directory contains the normalized release of **NLCO**: *Reasoning in a Combinatorial and Constrained World: Benchmarking LLMs on Natural-Language Combinatorial Optimization*.

## Summary

- Tasks: `43`
- CSV files: `129`
- Samples per file: `50`
- Total samples: `6450`
- Difficulty tiers from the paper: `Set-S`, `Set-M`, `Set-L`
- Dataset size buckets: `S`, `M`, `L` mapped to `Set-S`, `Set-M`, `Set-L`
- Split policy: this is an evaluation benchmark; `S/M/L` are size buckets, not train/validation/test splits
- Benchmark design: four-layer taxonomy over `var_sort`, `constraint_family`, `global_patterns`, and `objective_class`
- Prompt construction: numerical instances are contextualized into natural-language scenarios and diversified across `NL`, `JSON`, `CSV`, and `MD` surface formats

Task codes:

`2SP`, `AP3`, `BPP`, `CFLP`, `CMP`, `CSP`, `CVRP`, `FSP`, `GAP`, `GCP`, `HSP`, `JSP`, `KMST`, `KP`, `LOP`, `MAXCUT`, `MCP`, `MDP`, `MDS`, `MIS`, `MkC`, `MLP`, `MVC`, `OP`, `OSP`, `PCENTER`, `PCTSP`, `PDP`, `PMED`, `PMS`, `QAP`, `QKP`, `QSPP`, `RCPSP`, `SCP`, `SFP`, `SMTWT`, `SP`, `SPP`, `STP`, `TSP`, `TSPTW`, `UFLP`


## CSV Schema

Every CSV uses the same columns and column order:

| column | meaning |
|---|---|
| `id` | Global unique identifier in the form `{task_id}_{difficulty_tier}_{example_index:03d}` |
| `task_id` | Task code |
| `difficulty_tier` | Instance size bucket: `S`, `M`, or `L` |
| `example_index` | 1-based example index within the task-tier file |
| `prompt` | Natural-language benchmark prompt |
| `surface_format` | Original rendered instance format such as `nl`, `json`, `csv`, or `markdown_table` |
| `indexing_scheme` | Identifier style used in the prompt: `zero_based`, `one_based`, or `names` |
| `instance_canonical_json` | Canonical instance payload as valid JSON |
| `reference_solution_canonical_json` | Canonical solver-annotated reference answer as valid JSON |
| `reference_objective_value` | Solver-annotated reference objective value for the instance |
| `instance_surface_json` | Instance payload aligned with the prompt identifier scheme |
| `reference_solution_surface_json` | Solver-annotated reference answer aligned with the prompt identifier scheme |


## Evaluation Contract

- `solution_json` is the canonical task-specific answer format for automatic evaluation.
- `objective_value` is the solver-annotated reference objective for the row.
- The paper describes NLCO labels as solver-annotated `(near-)optimal` references. Exactness therefore depends on the task-level solver, not on a single benchmark-wide guarantee.
- Tasks backed by `LKH3`, `HGS`, or mixed `GA/Gurobi` pipelines should be treated as heuristic or mixed references rather than globally certified optima.
- Task-specific taxonomy labels, data sources, solvers, difficulty scales, validity rules, and reference-quality notes are stored in [benchmark_task_specs.json].

Examples:

- Routing tasks use route arrays in `solution_json`.
- Facility-location tasks use JSON objects with opened facilities and assignments.
- Scheduling tasks use JSON objects or arrays with explicit start and end times.

## Loading

With pandas:

```python
import pandas as pd

df = pd.read_csv("HSP/HSP_L_context.csv")
```

With Hugging Face datasets:

```python
from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files=["HSP/HSP_S_context.csv", "HSP/HSP_M_context.csv", "HSP/HSP_L_context.csv"],
    split="train",
)
```

## Task Metadata

The task spec file now mirrors the paper more closely. For each task it includes:

- `paper_name`
- `taxonomy.var_sort`
- `taxonomy.constraint_family`
- `taxonomy.global_patterns`
- `taxonomy.canonical_global_constraints`
- `taxonomy.objective_class`
- `data_source`
- `utilized_solver`
- `instance_scale`
- `reference_solution_quality`

## Citation

If you use NLCO, please cite the paper:

- Xia Jiang, Jing Chen, Cong Zhang, Jie Gao, Chengpeng Hu, Chenhao Zhang, Yaoxin Wu, Yingqian Zhang. "Reasoning in a Combinatorial and Constrained World: Benchmarking LLMs on Natural-Language Combinatorial Optimization." arXiv:2602.02188 (2026). https://arxiv.org/abs/2602.02188

BibTeX:

```bibtex
@article{jiang2026nlco,
  title        = {Reasoning in a Combinatorial and Constrained World: Benchmarking LLMs on Natural-Language Combinatorial Optimization},
  author       = {Jiang, Xia and Chen, Jing and Zhang, Cong and Gao, Jie and Hu, Chengpeng and Zhang, Chenhao and Wu, Yaoxin and Zhang, Yingqian},
  journal      = {arXiv preprint arXiv:2602.02188},
  year         = {2026}
}
```