# Changelog

## [0.2.0] — 2026-07-05

### Added
- **VCI** (Variational Co-Inference): `free_energy.py`, `vci_scheduler.py`, `frontend.refine()`
- Case A toy benchmark: `experiments/run_vci_reasoning.py`
- **BBH mini-set** (40 tasks): `qihc/data/bbh_subset.json`, `qihc/orchestrator/bbh.py`
- BBH experiment: `experiments/run_vci_bbh.py` (same p-bit budget)
- Handoff / temperature probe: `experiments/run_vci_handoff.py`
- TTS scaling: `experiments/run_sampler_scaling.py`
- Architecture diagram: `docs/QIHC_VCI_architecture.png`
- Docs: `docs/VCI_GUIDE.md`

### Changed
- Version bump 0.1.0 → 0.2.0
- README: VCI section, experiment commands, roadmap

### Tests
- `tests/test_vci.py`, `tests/test_bbh.py` (21+ tests total)

## [0.1.0] — initial

- p-bit Ising samplers (Gibbs / PT / SQA / SA)
- QIHC orchestrator + MoE PoC
- Max-Cut benchmark
