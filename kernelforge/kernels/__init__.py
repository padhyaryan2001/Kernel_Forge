"""Trusted kernel sources, launch binders and baselines for each operator.

Kernel sources are plain text so the CPU-only controller can hash, validate and dispatch
them; CUDA/Triton are imported only inside the binder functions on the worker.
"""
from __future__ import annotations

from . import layernorm, mrope

OPERATOR_MODULES = {"layernorm": layernorm, "mrope": mrope}


def module_for(operator: str):
    try:
        return OPERATOR_MODULES[operator]
    except KeyError as exc:
        raise ValueError(f"unknown operator {operator!r}") from exc


class LaunchConfigError(ValueError):
    """The requested launch configuration is outside what the trusted wrapper allows."""
