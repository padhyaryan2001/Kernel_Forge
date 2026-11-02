"""LayerNorm forward (inference only; backward/gradients are out of scope).

Structure adapted from the Triton LayerNorm tutorial (MIT license, see NOTICE): one row per
program with masked loads, FP32 reductions and a fused affine transform. The tuned kernel
adds multi-row 2D tiles and a per-N launch policy.
"""
from __future__ import annotations

from ..contracts import CandidateSpec, LaunchPolicy, LaunchTier, LAUNCH_DEFAULTS

ENTRY_POINT = "layernorm_fwd"
OPERATOR = "layernorm"
MAX_TILE_ELEMENTS = 65536
PRIMARY_BASELINE = "torch_eager"
BASELINES = ("torch_eager", "triton_basic", "torch_compile")

# Basic, untuned Triton baseline: one row per program, whole row in one power-of-two block,
# fixed num_warps=4. Labeled separately as "triton_basic".
BASIC_SOURCE = '''\
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd(X, Y, W, B, stride_x_row, stride_y_row, M, N, eps,
                  BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(Y + row * stride_y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
'''

# Manual optimized kernel: ROWS_PER_PROGRAM rows per program as a 2D tile so that small-N
# rows share one program, gamma/beta loaded once per tile, 64-bit row offsets, two-pass
# (centered) variance from registers.
MANUAL_SOURCE = '''\
import triton
import triton.language as tl


@triton.jit
def layernorm_fwd(X, Y, W, B, stride_x_row, stride_y_row, M, N, eps,
                  BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]
    rows64 = rows.to(tl.int64)
    x = tl.load(X + rows64[:, None] * stride_x_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / N
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = xc * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(Y + rows64[:, None] * stride_y_row + cols[None, :], y.to(Y.dtype.element_ty), mask=mask)
'''

BASIC_SPEC = CandidateSpec(
    operator=OPERATOR, kernel_source=BASIC_SOURCE, entry_point=ENTRY_POINT,
    launch=LaunchPolicy.single(BLOCK_SIZE=0, ROWS_PER_PROGRAM=1, num_warps=4, num_stages=1),
    explanation="Untuned baseline: one row per program, BLOCK_SIZE=next_pow2(N), num_warps=4.",
    origin="baseline_triton", search_mode="manual", label="triton_basic",
)

# Frozen from `kernelforge tune --operator layernorm` (experiment tune-layernorm-20261101-062214-3c06):
# per-N-tier best CUDA-graph geomean vs F.layer_norm on fp16 dev shapes (M in {256, 2048}).
# The pre-tuning hand-chosen policy was ROWS 8/4/1/1 with warps 2/4/8/16.
MANUAL_POLICY = LaunchPolicy(tiers=[
    LaunchTier(upto=256, params=dict(BLOCK_SIZE=0, ROWS_PER_PROGRAM=4, num_warps=2, num_stages=1)),
    LaunchTier(upto=1024, params=dict(BLOCK_SIZE=0, ROWS_PER_PROGRAM=2, num_warps=2, num_stages=1)),
    LaunchTier(upto=4096, params=dict(BLOCK_SIZE=0, ROWS_PER_PROGRAM=2, num_warps=8, num_stages=1)),
    LaunchTier(upto=None, params=dict(BLOCK_SIZE=0, ROWS_PER_PROGRAM=1, num_warps=8, num_stages=1)),
])

MANUAL_SPEC = CandidateSpec(
    operator=OPERATOR, kernel_source=MANUAL_SOURCE, entry_point=ENTRY_POINT, launch=MANUAL_POLICY,
    explanation="2D multi-row tiles for small N, gamma/beta loaded once per tile, rsqrt, per-N warps.",
    origin="manual", search_mode="manual", label="manual",
)

TUNING_GRID = {
    "ROWS_PER_PROGRAM": [1, 2, 4, 8, 16],
    "num_warps": [1, 2, 4, 8, 16],
}
TUNING_TIERS = [256, 1024, 4096, None]

SIGNATURE_DOC = (
    "@triton.jit\n"
    "def layernorm_fwd(X, Y, W, B, stride_x_row, stride_y_row, M, N, eps,\n"
    "                  BLOCK_SIZE: tl.constexpr, ROWS_PER_PROGRAM: tl.constexpr)"
)

SPEC_DOC = """\
Operator: LayerNorm forward over the last dim of a contiguous row-major x[M, N].
mean = sum(x)/N; var = sum((x-mean)^2)/N (population); y = (x-mean)/sqrt(var+eps)*W + B.
Accumulate in float32; store y in Y's dtype (Y.dtype.element_ty). x/W/B/Y share dtype float32, float16 or bfloat16.
Trusted host wrapper launches grid = (cdiv(M, ROWS_PER_PROGRAM),) and passes
BLOCK_SIZE (power of two; 0 in the policy means next_pow2(N)) and ROWS_PER_PROGRAM (power of two).
Rows/cols beyond M/N must be masked. Do not write to X, W or B. Every Y element must be written."""


def launch_selector(case) -> int:
    return case.N


# ---------------------------------------------------------------------------
# Worker-side binders (torch / triton imported lazily)
# ---------------------------------------------------------------------------

def validate_inputs(inputs, case) -> None:
    import torch

    x, w, b = inputs["x"], inputs["gamma"], inputs["beta"]
    if x.dim() != 2 or tuple(x.shape) != (case.M, case.N):
        raise ValueError(f"x must have shape [{case.M}, {case.N}], got {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("non-contiguous x is not supported (no hidden copies are made)")
    for name, t in (("gamma", w), ("beta", b)):
        if tuple(t.shape) != (case.N,) or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous with shape [{case.N}]")
        if t.dtype != x.dtype or t.device != x.device:
            raise ValueError(f"{name} must match x dtype/device")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")


def allocate_outputs(inputs, fill_nan: bool = False):
    import torch

    y = torch.empty_like(inputs["x"])
    if fill_nan:
        y.fill_(float("nan"))
    return {"y": y}


def launch(kernel, policy: LaunchPolicy, inputs, outputs, case) -> dict:
    import triton

    from . import LaunchConfigError

    x, w, b, y = inputs["x"], inputs["gamma"], inputs["beta"], outputs["y"]
    M, N = x.shape
    p = {**LAUNCH_DEFAULTS[OPERATOR], **policy.select(N)}
    block = p["BLOCK_SIZE"] or max(16, triton.next_power_of_2(N))
    rows = p["ROWS_PER_PROGRAM"]
    if rows * block > MAX_TILE_ELEMENTS:
        raise LaunchConfigError(f"ROWS_PER_PROGRAM*BLOCK_SIZE={rows * block} exceeds {MAX_TILE_ELEMENTS}")
    grid = (triton.cdiv(M, rows),)
    kernel[grid](x, y, w, b, x.stride(0), y.stride(0), M, N, case.eps,
                 BLOCK_SIZE=block, ROWS_PER_PROGRAM=rows,
                 num_warps=p["num_warps"], num_stages=p["num_stages"])
    return {"BLOCK_SIZE": block, "ROWS_PER_PROGRAM": rows, "num_warps": p["num_warps"],
            "num_stages": p["num_stages"], "grid": grid[0]}


def torch_eager(inputs, case):
    from ..references import layernorm_torch_eager

    return {"y": layernorm_torch_eager(inputs["x"], inputs["gamma"], inputs["beta"], case.eps)}


def torch_compiled():
    """Returns a callable(inputs, case) wrapping torch.compile(F.layer_norm). Compile time is
    excluded from timing by the harness warmup; setup: default inductor mode, dynamic=False."""
    import torch

    _raise_recompile_limit(torch)
    from ..references import layernorm_torch_eager

    compiled = torch.compile(layernorm_torch_eager, dynamic=False)

    def run(inputs, case):
        return {"y": compiled(inputs["x"], inputs["gamma"], inputs["beta"], case.eps)}

    run.setup = {"api": "torch.compile", "mode": "default", "dynamic": False}
    return run


def _raise_recompile_limit(torch) -> None:
    """dynamic=False compiles once per shape; past dynamo's recompile limit it would silently run eager,
    so the limit is raised to cover every declared workload shape."""
    import torch._dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 4096)
