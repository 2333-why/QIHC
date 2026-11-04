---
license: apache-2.0
task_categories:
- text-generation
tags:
- code
size_categories:
- n<1K
language:
- en
configs:
- config_name: default
  data_files:
  - split: original
    path: "dataset.jsonl"
  - split: verified
    path: "dataset_verified.jsonl"
---

> [!IMPORTANT]
> CP-Bench has been superseded by **DCP-Bench-Open**. This repository is kept for archival and reproducibility purposes only.
>
> Please use the current version here: [DCP-Bench/DCP-Bench-Open](https://github.com/DCP-Bench/DCP-Bench-Open).

# CP-Bench: A dataset for evaluating LLM-driven constraint modelling

[![Hugging Face Space](https://img.shields.io/badge/Leaderboard-HF%20Space-blue?logo=huggingface)](https://huggingface.co/spaces/kostis-init/CP-Bench-Leaderboard)

This dataset is designed to facilitate the evaluation of LLM-based methods for translating natural language problem descriptions into accurate constraint specifications. It contains diverse combinatorial problems, and is sourced from various well-established sources from the Constraint Programming community.

---

## Dataset Versions

[//]: # (> **tl;dr:** Use `v1_verified`, it's the latest!)

You will notice that the dataset contains various splits (which are not exactly splits, but rather different versions of the dataset):
- `original`: The original dataset, which contains all problems initially designed, including those with known issues. More details about the issues can be found in the [changelog](cp_bench_changes.md).
- `verified`: A stripped-down version of the original dataset that has been verified to contain complete problem specifications with matching ground-truth models. This version is recommended for use in evaluations.

---

## 📊 Leaderboard

You can easily submit your results and view the global leaderboard here:  
👉 **[CP-Bench Leaderboard](https://huggingface.co/spaces/kostis-init/CP-Bench-Leaderboard)**

---

## Dataset Breakdown

The dataset contains problems from the following sources:

- `aplai_course`: Problems from the APLAI course of KU Leuven, 2023-2024. As modelled [here](https://github.com/kostis-init/CP-LLMs-ICL/tree/main/data/APLAI_course).
- `cpmpy_examples`: Problems from the [CPMpy repository](https://github.com/CPMpy/cpmpy/tree/master/examples)
  - All included, except for the ones that require enumeration of all solutions (e.g. `solveAll`).
- [`csplib`](https://www.csplib.org/Problems/)
  - For now, only the ones modelled in the [CPMpy repository](https://github.com/CPMpy/cpmpy/tree/master/examples/csplib) are included, and the ones modelled by [Hakan Kjellerstrand](http://www.hakank.org/cpmpy/).
- `hakan_examples`: Models created by [Hakan Kjellerstrand](http://www.hakank.org/cpmpy/)
  - In progress with alphabetical order, excluding the following:
    - Those already modelled from other sources (e.g. aplai_course, cpmpy_examples, csplib)
    - Those that contain `solveAll` (counting solutions).
    - Global constraints tests, e.g. http://www.hakank.org/cpmpy/atmost_test.py

---

## Diversity

We attempted to include unique problems from different sources, in order to provide a diverse set of problems.
However, as this was a manual process, there might be duplicates or similar problems. If you notice any issues, please let us know.

---

## Citation

If you found our work useful, please consider citing it:

```bib
@misc{michailidis2025cpbench,
  title={CP-Bench: Evaluating Large Language Models for Constraint Modelling}, 
  author={Kostis Michailidis and Dimos Tsouros and Tias Guns},
  year={2025},
  eprint={2506.06052},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.06052}, 
}
```