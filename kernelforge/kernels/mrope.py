"""Fused Qwen2-VL-style multimodal RoPE (temporal / height / width sections).

The fused kernel loads the three position rows once per token tile, selects the axis per
rotary pair, computes cos/sin once per (token, pair) and reuses them for every query and
key head. It writes only the two outputs: no expanded trig tensors, no rotate_half
concatenation, no FP32 copies of q/k. Hq != Hkv is handled by two head loops.
"""
from __future__ import annotations

from ..contracts import CandidateSpec, LaunchPolicy, LaunchTier, LAUNCH_DEFAULTS

ENTRY_POINT = "mrope_fwd"
OPERATOR = "mrope"
PRIMARY_BASELINE = "torch_eager"
BASELINES = ("torch_eager", "triton_basic", "torch_compile")

MANUAL_SOURCE = '''\
import triton
import triton.language as tl


@triton.jit
def mrope_fwd(Q, K, QO, KO, POS, INV_FREQ, COS, SIN, S, HQ, HKV, HALF_D, SEC_T, SEC_TH,
              stride_qb, stride_qh, stride_qs, stride_kb, stride_kh, stride_ks,
              stride_pa, stride_pb, stride_cb,
              USE_PRECOMPUTED: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
              HEAD_SPLITS: tl.constexpr):
    pid_s = tl.program_id(0)
    b = tl.program_id(1)
    pid_h = tl.program_id(2)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    s_mask = s < S
    d_mask = d < HALF_D
    mask = s_mask[:, None] & d_mask[None, :]
    if USE_PRECOMPUTED:
        c_offs = b * stride_cb + s[:, None] * HALF_D + d[None, :]
        cos = tl.load(COS + c_offs, mask=mask, other=0.0)
        sin = tl.load(SIN + c_offs, mask=mask, other=0.0)
    else:
        p_base = POS + b * stride_pb + s
        p_t = tl.load(p_base, mask=s_mask, other=0).to(tl.float32)
        p_h = tl.load(p_base + stride_pa, mask=s_mask, other=0).to(tl.float32)
        p_w = tl.load(p_base + 2 * stride_pa, mask=s_mask, other=0).to(tl.float32)
        dd = d[None, :]
        pos = tl.where(dd < SEC_T, p_t[:, None], tl.where(dd < SEC_TH, p_h[:, None], p_w[:, None]))
        inv = tl.load(INV_FREQ + d, mask=d_mask, other=0.0)
        theta = pos * inv[None, :]
        cos = tl.cos(theta)
        sin = tl.sin(theta)
    for h in range(pid_h, HQ, HEAD_SPLITS):
        offs = b * stride_qb + h * stride_qh + s[:, None] * stride_qs + d[None, :]
        x1 = tl.load(Q + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(Q + offs + HALF_D, mask=mask, other=0.0).to(tl.float32)
        tl.store(QO + offs, (x1 * cos - x2 * sin).to(QO.dtype.element_ty), mask=mask)
        tl.store(QO + offs + HALF_D, (x1 * sin + x2 * cos).to(QO.dtype.element_ty), mask=mask)
    for h in range(pid_h, HKV, HEAD_SPLITS):
        offs = b * stride_kb + h * stride_kh + s[:, None] * stride_ks + d[None, :]
        x1 = tl.load(K + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(K + offs + HALF_D, mask=mask, other=0.0).to(tl.float32)
        tl.store(KO + offs, (x1 * cos - x2 * sin).to(KO.dtype.element_ty), mask=mask)
        tl.store(KO + offs + HALF_D, (x1 * sin + x2 * cos).to(KO.dtype.element_ty), mask=mask)
'''

# Basic, untuned Triton baseline ("triton_basic"): same contract, but positions, angles and
# cos/sin are recomputed for every head, and one program walks all heads (HEAD_SPLITS=1).
BASIC_SOURCE = '''\
import triton
import triton.language as tl


@triton.jit
def mrope_fwd(Q, K, QO, KO, POS, INV_FREQ, COS, SIN, S, HQ, HKV, HALF_D, SEC_T, SEC_TH,
              stride_qb, stride_qh, stride_qs, stride_kb, stride_kh, stride_ks,
              stride_pa, stride_pb, stride_cb,
              USE_PRECOMPUTED: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
              HEAD_SPLITS: tl.constexpr):
    pid_s = tl.program_id(0)
    b = tl.program_id(1)
    s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)
    mask = (s < S)[:, None] & (d < HALF_D)[None, :]
    for h in range(0, HQ + HKV):
        if USE_PRECOMPUTED:
            c_offs = b * stride_cb + s[:, None] * HALF_D + d[None, :]
            cos = tl.load(COS + c_offs, mask=mask, other=0.0)
            sin = tl.load(SIN + c_offs, mask=mask, other=0.0)
        else:
            axis = tl.where(d < SEC_T, 0, tl.where(d < SEC_TH, 1, 2))
            pos = tl.load(POS + axis[None, :] * stride_pa + b * stride_pb + s[:, None], mask=mask, other=0)
            inv = tl.load(INV_FREQ + d, mask=d < HALF_D, other=0.0)
            theta = pos.to(tl.float32) * inv[None, :]
            cos = tl.cos(theta)
            sin = tl.sin(theta)
        if h < HQ:
            offs = b * stride_qb + h * stride_qh + s[:, None] * stride_qs + d[None, :]
            x1 = tl.load(Q + offs, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(Q + offs + HALF_D, mask=mask, other=0.0).to(tl.float32)
            tl.store(QO + offs, (x1 * cos - x2 * sin).to(QO.dtype.element_ty), mask=mask)
            tl.store(QO + offs + HALF_D, (x1 * sin + x2 * cos).to(QO.dtype.element_ty), mask=mask)
        else:
            offs = b * stride_kb + (h - HQ) * stride_kh + s[:, None] * stride_ks + d[None, :]
            x1 = tl.load(K + offs, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(K + offs + HALF_D, mask=mask, other=0.0).to(tl.float32)
            tl.store(KO + offs, (x1 * cos - x2 * sin).to(KO.dtype.element_ty), mask=mask)
            tl.store(KO + offs + HALF_D, (x1 * sin + x2 * cos).to(KO.dtype.element_ty), mask=mask)
'''

BASIC_SPEC = CandidateSpec(
    operator=OPERATOR, kernel_source=BASIC_SOURCE, entry_point=ENTRY_POINT,
    launch=LaunchPolicy.single(BLOCK_S=16, HEAD_SPLITS=1, num_warps=4, num_stages=1),
    explanation="Untuned baseline: trig recomputed per head, one program walks every head.",
    origin="baseline_triton", search_mode="manual", label="triton_basic",
)

# Frozen from `kernelforge tune --operator mrope` (experiment tune-mrope-20261101-062918-35dc): per-S-tier
# best CUDA-graph geomean vs eager PyTorch on fp16/bf16 positions-to-output dev shapes.
# The pre-tuning hand-chosen policy was BLOCK_S 16/32, HEAD_SPLITS 4/2, num_warps 4.
MANUAL_POLICY = LaunchPolicy(tiers=[
    LaunchTier(upto=256, params=dict(BLOCK_S=8, HEAD_SPLITS=4, num_warps=8, num_stages=1)),
    LaunchTier(upto=1024, params=dict(BLOCK_S=16, HEAD_SPLITS=4, num_warps=4, num_stages=1)),
    LaunchTier(upto=None, params=dict(BLOCK_S=16, HEAD_SPLITS=2, num_warps=4, num_stages=1)),
])

MANUAL_SPEC = CandidateSpec(
    operator=OPERATOR, kernel_source=MANUAL_SOURCE, entry_point=ENTRY_POINT, launch=MANUAL_POLICY,
    explanation="Fused axis selection + trig + rotation; trig computed once per token tile and reused "
                "across all q and k heads; no intermediate tensors.",
    origin="manual", search_mode="manual", label="manual",
)

TUNING_GRID = {
    "BLOCK_S": [8, 16, 32, 64],
    "HEAD_SPLITS": [1, 2, 4, 8],
    "num_warps": [2, 4, 8],
}
TUNING_TIERS = [256, 1024, None]

SIGNATURE_DOC = (
    "@triton.jit\n"
    "def mrope_fwd(Q, K, QO, KO, POS, INV_FREQ, COS, SIN, S, HQ, HKV, HALF_D, SEC_T, SEC_TH,\n"
    "              stride_qb, stride_qh, stride_qs, stride_kb, stride_kh, stride_ks,\n"
    "              stride_pa, stride_pb, stride_cb,\n"
    "              USE_PRECOMPUTED: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,\n"
    "              HEAD_SPLITS: tl.constexpr)"
)

SPEC_DOC = """\
Operator: Qwen2-VL multimodal RoPE on contiguous Q[B,HQ,S,D] and K[B,HKV,S,D] (HQ may differ from HKV),
writing QO/KO (same shapes/dtype, contiguous). HALF_D = D/2. Pair i in [0,HALF_D) uses position axis
0 if i < SEC_T, 1 if i < SEC_TH, else 2. POS is int64 [3,B,S] (axis stride stride_pa, batch stride stride_pb).
theta = float32(POS[axis,b,s]) * INV_FREQ[i] (float32). Split-half rotation:
  out[i] = x[i]*cos(theta) - x[i+HALF_D]*sin(theta); out[i+HALF_D] = x[i]*sin(theta) + x[i+HALF_D]*cos(theta).
If USE_PRECOMPUTED: POS/INV_FREQ are None; use COS/SIN float32 [B,S,HALF_D] (batch stride stride_cb).
Math in float32, store in QO.dtype.element_ty. Never write Q or K.
Trusted wrapper grid = (cdiv(S, BLOCK_S), B, HEAD_SPLITS); BLOCK_D = next_pow2(HALF_D)."""


def launch_selector(case) -> int:
    return case.S


def validate_inputs(inputs, case) -> None:
    import torch

    q, k = inputs["q"], inputs["k"]
    if tuple(q.shape) != (case.B, case.Hq, case.S, case.D) or tuple(k.shape) != (case.B, case.Hkv, case.S, case.D):
        raise ValueError("q/k shapes do not match the declared workload")
    if not (q.is_contiguous() and k.is_contiguous()):
        raise ValueError("non-contiguous q/k are not supported (no hidden copies are made)")
    if q.dtype != k.dtype or q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("q and k must share a supported dtype")
    if case.contract == "precomputed_trig":
        for name in ("cos", "sin"):
            t = inputs[name]
            if t.dtype != torch.float32 or tuple(t.shape) != (case.B, case.S, case.D // 2) or not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous float32 [B, S, D/2]")
    else:
        pos, inv = inputs["position_ids"], inputs["inv_freq"]
        if pos.dtype != torch.int64 or tuple(pos.shape) != (3, case.B, case.S) or not pos.is_contiguous():
            raise ValueError("position_ids must be contiguous int64 [3, B, S]")
        if inv.dtype != torch.float32 or tuple(inv.shape) != (case.D // 2,):
            raise ValueError("inv_freq must be float32 [D/2]")


def allocate_outputs(inputs, fill_nan: bool = False):
    import torch

    qo, ko = torch.empty_like(inputs["q"]), torch.empty_like(inputs["k"])
    if fill_nan:
        qo.fill_(float("nan"))
        ko.fill_(float("nan"))
    return {"q_out": qo, "k_out": ko}


def launch(kernel, policy: LaunchPolicy, inputs, outputs, case) -> dict:
    import triton

    q, k, qo, ko = inputs["q"], inputs["k"], outputs["q_out"], outputs["k_out"]
    B, HQ, S, D = q.shape
    HKV = k.shape[1]
    half = D // 2
    p = {**LAUNCH_DEFAULTS[OPERATOR], **policy.select(S)}
    block_d = triton.next_power_of_2(half)
    sec_t, sec_h, _ = case.sections
    grid = (triton.cdiv(S, p["BLOCK_S"]), B, p["HEAD_SPLITS"])
    if case.contract == "precomputed_trig":
        pos = inv = None
        cos, sin = inputs["cos"], inputs["sin"]
        stride_pa = stride_pb = 0
        stride_cb = cos.stride(0)
        use_pre = True
    else:
        pos, inv = inputs["position_ids"], inputs["inv_freq"]
        cos = sin = None
        stride_pa, stride_pb = pos.stride(0), pos.stride(1)
        stride_cb = 0
        use_pre = False
    kernel[grid](q, k, qo, ko, pos, inv, cos, sin, S, HQ, HKV, half, sec_t, sec_t + sec_h,
                 q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                 stride_pa, stride_pb, stride_cb,
                 USE_PRECOMPUTED=use_pre, BLOCK_S=p["BLOCK_S"], BLOCK_D=block_d,
                 HEAD_SPLITS=p["HEAD_SPLITS"], num_warps=p["num_warps"], num_stages=p["num_stages"])
    return {"BLOCK_S": p["BLOCK_S"], "BLOCK_D": block_d, "HEAD_SPLITS": p["HEAD_SPLITS"],
            "num_warps": p["num_warps"], "num_stages": p["num_stages"],
            "grid": list(grid)}


def torch_eager(inputs, case):
    from ..references import mrope_torch_eager, mrope_torch_eager_precomputed

    if case.contract == "precomputed_trig":
        qo, ko = mrope_torch_eager_precomputed(inputs["q"], inputs["k"], inputs["cos"], inputs["sin"])
    else:
        qo, ko = mrope_torch_eager(inputs["q"], inputs["k"], inputs["position_ids"], inputs["inv_freq"],
                                   case.sections)
    return {"q_out": qo, "k_out": ko}


def torch_compiled():
    import torch

    _raise_recompile_limit(torch)
    from ..references import mrope_torch_eager, mrope_torch_eager_precomputed

    pos_fn = torch.compile(mrope_torch_eager, dynamic=False)
    pre_fn = torch.compile(mrope_torch_eager_precomputed, dynamic=False)

    def run(inputs, case):
        if case.contract == "precomputed_trig":
            qo, ko = pre_fn(inputs["q"], inputs["k"], inputs["cos"], inputs["sin"])
        else:
            qo, ko = pos_fn(inputs["q"], inputs["k"], inputs["position_ids"], inputs["inv_freq"],
                            tuple(case.sections))
        return {"q_out": qo, "k_out": ko}

    run.setup = {"api": "torch.compile", "mode": "default", "dynamic": False}
    return run


def _raise_recompile_limit(torch) -> None:
    """dynamic=False compiles once per shape; past dynamo's recompile limit it would silently run eager,
    so the limit is raised to cover every declared workload shape."""
    import torch._dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 4096)
