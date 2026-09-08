"""Capacitated vehicle-routing problem support for QIHC-LNS."""

from qihc.problems.cvrp.instance import (
    CVRPInstance,
    ConstraintSpec,
    Customer,
    RouteSolution,
    generate_synthetic_instance,
    load_cvrplib,
)
from qihc.problems.cvrp.neighborhood import (
    NeighborhoodProposal,
    KNNNeighborhoodSelector,
    RandomNeighborhoodSelector,
    safely_expand_proposal,
)
from qihc.problems.cvrp.qubo import AssignmentQUBO, build_assignment_qubo
from qihc.problems.cvrp.scheduler import LNSConfig, LNSResult, QIHCLNSSolver
from qihc.problems.cvrp.verifier import VerificationResult, greedy_initial_solution, verify_solution
from qihc.problems.cvrp.logit_feedback import PBitLogitFeedback

__all__ = [
    "AssignmentQUBO",
    "CVRPInstance",
    "ConstraintSpec",
    "Customer",
    "KNNNeighborhoodSelector",
    "LNSConfig",
    "LNSResult",
    "NeighborhoodProposal",
    "RandomNeighborhoodSelector",
    "safely_expand_proposal",
    "RouteSolution",
    "QIHCLNSSolver",
    "VerificationResult",
    "build_assignment_qubo",
    "generate_synthetic_instance",
    "greedy_initial_solution",
    "load_cvrplib",
    "verify_solution",
    "PBitLogitFeedback",
]
