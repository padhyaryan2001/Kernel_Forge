"""Deliberately wrong or failing candidates.

They protect the measurement system: the evaluator must reject every one of them before
performance ranking. Each entry names the stage expected to reject it.
"""
from __future__ import annotations

from ..contracts import CandidateSpec, LaunchPolicy
from . import layernorm as ln
from . import mrope as mr

_LN_HEADER = '''\
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd(X, Y, W, B, stride_x_row, stride_y_row, M, N, eps,
                  BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
'''

_LN_BODIES = {
    # Sample variance (N - 1) instead of population variance.
    "ln_unbiased_variance": '''\
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / (N - 1)
    y = xc / tl.sqrt(var + eps) * w + b
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
''',
    # Divides by the padded block size: only correct when N is a power of two.
    "ln_tail_mean": '''\
    mean = tl.sum(x, axis=0) / BLOCK_SIZE
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / BLOCK_SIZE
    y = xc / tl.sqrt(var + eps) * w + b
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
''',
    # Never writes the output (no-op): caught by the NaN-filled output sentinel.
    "ln_noop": '''\
    y = x * w + b
''',
    # Hardcoded output.
    "ln_hardcoded_zeros": '''\
    y = x * 0.0
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
''',
    # Correct output, but scribbles over its input.
    "ln_mutates_input": '''\
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    y = xc / tl.sqrt(var + eps) * w + b
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(X + row * stride_x_row + cols, y.to(X.dtype.element_ty), mask=mask)
''',
    # Passes the AST policy but cannot compile (tl.arange needs a power-of-two range).
    "ln_compile_error": '''\
    extra = tl.arange(0, BLOCK_SIZE + 1)
    y = x * w + b
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
''',
}

_HOST_CODE = '''\
import os
import triton
import triton.language as tl


def helper():
    return os.environ


@triton.jit
def layernorm_fwd(X, Y, W, B, stride_x_row, stride_y_row, M, N, eps,
                  BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    pass
'''

_BASIC_POLICY = LaunchPolicy.single(BLOCK_SIZE=0, ROWS_PER_PROGRAM=1, num_warps=4, num_stages=1)


def _ln_spec(name: str, source: str) -> CandidateSpec:
    return CandidateSpec(operator="layernorm", kernel_source=source, entry_point=ln.ENTRY_POINT,
                         launch=_BASIC_POLICY, origin="canary", search_mode="manual", label=name)


# Interleaved (even/odd) rotation instead of split-half: wrong convention.
_MR_INTERLEAVED = mr.MANUAL_SOURCE.replace(
    "x2 = tl.load(Q + offs + HALF_D", "x2 = tl.load(Q + offs + 1"
).replace("tl.store(QO + offs + HALF_D", "tl.store(QO + offs + 1")

# Uses the temporal axis for every pair: correct for text positions only.
_MR_SINGLE_AXIS = mr.MANUAL_SOURCE.replace(
    "pos = tl.where(dd < SEC_T, p_t[:, None], tl.where(dd < SEC_TH, p_h[:, None], p_w[:, None]))",
    "pos = tl.where(dd < HALF_D, p_t[:, None], p_w[:, None])",
)


def canaries() -> dict[str, tuple[CandidateSpec, str]]:
    """name -> (spec, stage expected to reject it)."""
    out: dict[str, tuple[CandidateSpec, str]] = {}
    expected = {
        "ln_unbiased_variance": "correctness", "ln_tail_mean": "correctness", "ln_noop": "correctness",
        "ln_hardcoded_zeros": "correctness", "ln_mutates_input": "correctness",
        "ln_compile_error": "compile",
    }
    for name, body in _LN_BODIES.items():
        out[name] = (_ln_spec(name, _LN_HEADER + body), expected[name])
    out["ln_host_code"] = (_ln_spec("ln_host_code", _HOST_CODE), "validation")
    out["mr_interleaved"] = (CandidateSpec(operator="mrope", kernel_source=_MR_INTERLEAVED,
                                           entry_point=mr.ENTRY_POINT, launch=mr.MANUAL_POLICY,
                                           origin="canary", search_mode="manual", label="mr_interleaved"),
                             "correctness")
    out["mr_single_axis"] = (CandidateSpec(operator="mrope", kernel_source=_MR_SINGLE_AXIS,
                                           entry_point=mr.ENTRY_POINT, launch=mr.MANUAL_POLICY,
                                           origin="canary", search_mode="manual", label="mr_single_axis"),
                             "correctness")
    assert _MR_INTERLEAVED != mr.MANUAL_SOURCE and _MR_SINGLE_AXIS != mr.MANUAL_SOURCE
    return out
