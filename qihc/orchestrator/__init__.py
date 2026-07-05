"""QIHC orchestrator: AI frontend + p-bit backend heterogeneous loop."""

from qihc.orchestrator.config import QIHCConfig
from qihc.orchestrator.free_energy import FreeEnergyResult, compute_free_energy
from qihc.orchestrator.bbh import load_bbh_problems, load_bbh_tasks
from qihc.orchestrator.reasoning import SubsetProblem, demo_problem, generate_toy_problems
from qihc.orchestrator.scheduler import QIHCOrchestrator
from qihc.orchestrator.types import RoutingBatchResult, RoutingDecision
from qihc.orchestrator.vci_scheduler import VCIConfig, VCIOrchestrator, VCIResult

__all__ = [
    "QIHCConfig",
    "QIHCOrchestrator",
    "RoutingBatchResult",
    "RoutingDecision",
    "VCIConfig",
    "VCIOrchestrator",
    "VCIResult",
    "FreeEnergyResult",
    "compute_free_energy",
    "SubsetProblem",
    "demo_problem",
    "generate_toy_problems",
    "load_bbh_problems",
    "load_bbh_tasks",
]
