"""Stable v5 compiler; v6 changes the interaction, not the mathematical semantics."""
from tools.runtime.modeling_v5.protocol import (  # noqa: F401
    compile_math, compile_object, inspect, load_object, gate_cases, output_budget)
from tools.runtime.modeling_v5.protocol import modeling_prompt as _legacy_prompt
from .source_view import render_catalog


def modeling_prompt(case):
    # Replace only the final generated catalog. The original question, including
    # any marker-like text it contains, and all compiler instructions stay intact.
    prefix, marker, _ = _legacy_prompt(case).rpartition('\nSOURCE CATALOG:\n')
    if not marker:
        raise RuntimeError('Legacy modeling prompt has no final source catalog')
    return prefix + marker + render_catalog(case['question'])
