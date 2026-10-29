# Fixed public NLCO snapshot

This bundle contains only the 18 original CSV files for KP, MIS, MVC, MAXCUT,
MCP and MDS (S/M/L, 50 rows each). No user results, model weights or credentials
are included. CSV bytes are unchanged from the pinned author release.

- Source: https://huggingface.co/datasets/summer142857jiang/NLCO
- Revision: `6a7dff75171906e0054ba6c92aa1ef8015b47c36`
- Upstream license declaration: **MIT**, preserved in `UPSTREAM_README.md`.
- Authors: Xia Jiang, Jing Chen, Cong Zhang, Jie Gao, Chengpeng Hu, Chenhao Zhang,
  Yaoxin Wu, Yingqian Zhang.
- Paper: https://aclanthology.org/2026.findings-acl.1529.pdf
- Exact file and archive SHA-256 hashes: `nlco.lock.json`.

The deterministic gzip/tar packaging is provided so users who can update this
Git repository can prepare the benchmark even when HF mirrors return 403 or the
official host times out. The loader validates the archive and every CSV, reads
only the exact locked regular members, and preserves already valid cached files.
This is the six-task subset, not a new dataset or the full NLCO benchmark.
