"""QIHC: Quantum-Inspired Intelligence Heterogeneous Computing."""

__version__ = "0.1.0"

__all__ = ["IsingModel", "maxcut", "__version__"]


def __getattr__(name):
    """Keep legacy convenience exports without loading plotting on CPU workers."""
    if name == "IsingModel":
        from qihc.ising.model import IsingModel
        return IsingModel
    if name == "maxcut":
        from importlib import import_module
        return import_module("qihc.ising.maxcut")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
