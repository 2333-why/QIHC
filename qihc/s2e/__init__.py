"""QIHC-S2E: verified constraint synthesis and solver-feedback learning."""

from .compiler import CompilationPlan, Representation, RepresentationSelector, compile_cpp
from .cpp import ConstraintProgram, ConstraintProgramPackage
from .feedback import FeedbackRecord, build_dpo_pairs, build_grpo_records, build_sft_records
from .repair import repair_cpp
from .hybrid_sampler import HybridSampleBatch, PDitMFCSampler, TorchPDitMFCSampler
from .synthesizer import ConstraintSynthesizer, HeuristicConstraintBackend
from .validator import CPPValidationReport, CPPValidator, ValidationStage

__all__ = [
    "CPPValidationReport",
    "CPPValidator",
    "CompilationPlan",
    "ConstraintProgram",
    "ConstraintProgramPackage",
    "ConstraintSynthesizer",
    "FeedbackRecord",
    "HeuristicConstraintBackend",
    "Representation",
    "RepresentationSelector",
    "ValidationStage",
    "build_dpo_pairs",
    "build_grpo_records",
    "build_sft_records",
    "compile_cpp",
    "repair_cpp",
    "HybridSampleBatch",
    "PDitMFCSampler",
    "TorchPDitMFCSampler",
]
