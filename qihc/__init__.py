"""QIHC: Quantum-Inspired Intelligence Heterogeneous Computing."""

__version__ = "0.1.0"

from qihc.ising.model import IsingModel
from qihc.ising import maxcut

__all__ = ["IsingModel", "maxcut", "__version__"]
