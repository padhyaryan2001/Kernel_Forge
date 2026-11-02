"""Independent PyTorch implementations: float64 mathematical references, input generators,
and the eager / compiled PyTorch baselines.

The references are written directly from the mathematical contracts and do not call the
baselines (or torch.nn.functional.layer_norm), so they can arbitrate between them on CPU.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .contracts import LayerNormCase, MRoPECase

TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

# Upstream convention this module transcribes (see hf_qwen2vl_mrope_transcription).
QWEN2VL_UPSTREAM = {
    "repository": "huggingface/transformers",
    "path": "src/transformers/models/qwen2_vl/modeling_qwen2_vl.py",
    "revision": "f90e93c4f83974f634860e246cdfd2c68666d605",
    "revision_date": "2026-10-30",
    "functions": ["Qwen2VLRotaryEmbedding.forward", "recomposition_frequencies", "rotate_half",
                  "apply_rotary_pos_emb"],
}


def _generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

def make_layernorm_inputs(case: LayerNormCase) -> dict[str, torch.Tensor]:
    """Deterministic CPU inputs in the declared dtype."""
    g = _generator(case.seed)
    M, N = case.M, case.N
    if case.input_kind == "normal":
        x = torch.randn(M, N, generator=g) * 1.5 + 0.25
    elif case.input_kind == "constant_rows":
        # Dyadic constants are exact in every dtype, so the mathematically exact answer is beta.
        palette = torch.tensor([-2.25, -1.0, 0.0, 0.5, 3.75])
        idx = torch.randint(0, len(palette), (M, 1), generator=g)
        x = palette[idx].expand(M, N).clone()
    elif case.input_kind == "low_variance":
        x = 1.0 + 1e-3 * torch.randn(M, N, generator=g)
    elif case.input_kind == "negative":
        x = -3.0 - 5.0 * torch.rand(M, N, generator=g)
    elif case.input_kind == "large_offset":
        x = 1000.0 + torch.randn(M, N, generator=g)
    else:  # pragma: no cover
        raise ValueError(case.input_kind)
    gamma = 1.0 + 0.1 * torch.randn(N, generator=g)
    beta = 0.1 * torch.randn(N, generator=g)
    dt = TORCH_DTYPES[case.dtype]
    return {"x": x.to(dt), "gamma": gamma.to(dt), "beta": beta.to(dt)}


def layernorm_reference(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """float64 reference: population variance over the last dimension."""
    xd = x.detach().to("cpu", torch.float64)
    n = xd.shape[-1]
    mean = xd.sum(dim=-1, keepdim=True) / n
    centered = xd - mean
    var = (centered * centered).sum(dim=-1, keepdim=True) / n
    y = centered / torch.sqrt(var + eps)
    return y * gamma.detach().to("cpu", torch.float64) + beta.detach().to("cpu", torch.float64)


def layernorm_torch_eager(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    """Primary practical baseline: torch.nn.functional.layer_norm."""
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), gamma, beta, eps)


# ---------------------------------------------------------------------------
# MRoPE
# ---------------------------------------------------------------------------

def inv_freq_for(D: int, rope_theta: float) -> torch.Tensor:
    # Same construction as upstream Qwen2-VL default RoPE init (FP32).
    return 1.0 / (rope_theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D))


def make_positions(case: MRoPECase) -> torch.Tensor:
    """Synthetic [3, B, S] int64 positions.

    text:  all three axes equal the token index.
    image: a short text prefix, then one image grid at a fixed temporal position with
           height/width indices offset by the prefix length (Qwen2-VL layout).
    video: a short text prefix, then 8x8-patch frames whose temporal index advances by 2
           per frame (changing temporal positions).
    Batch row b is shifted by b so batches are not identical.
    """
    B, S, off = case.B, case.S, case.position_offset
    pos = torch.empty(3, B, S, dtype=torch.int64)
    for b in range(B):
        base = off + b
        idx = torch.arange(S, dtype=torch.int64)
        if case.position_kind == "text":
            t = h = w = base + idx
        else:
            prefix = min(16, S // 8)
            t = (base + idx).clone()
            h = t.clone()
            w = t.clone()
            n_vis = S - prefix
            v = torch.arange(n_vis, dtype=torch.int64)
            start = base + prefix
            if case.position_kind == "image":
                grid_w = max(1, 1 << int(math.log2(max(1, int(math.sqrt(n_vis))))))
                t[prefix:] = start
                h[prefix:] = start + v // grid_w
                w[prefix:] = start + v % grid_w
            else:
                per_frame = 64
                frame = v // per_frame
                inner = v % per_frame
                t[prefix:] = start + 2 * frame
                h[prefix:] = start + inner // 8
                w[prefix:] = start + inner % 8
        pos[0, b], pos[1, b], pos[2, b] = t, h, w
    return pos


def axis_of_pair(sections: tuple[int, int, int]) -> list[int]:
    nt, nh, nw = sections
    return [0] * nt + [1] * nh + [2] * nw


def mrope_theta_fp32(position_ids: torch.Tensor, inv_freq: torch.Tensor, sections: tuple[int, int, int]) -> torch.Tensor:
    """theta[b, s, i] = fp32(position_ids[axis(i), b, s]) * inv_freq[i], rounded to FP32. Shape [B, S, D/2]."""
    axes = torch.tensor(axis_of_pair(sections), dtype=torch.int64)
    pos = position_ids.to("cpu")
    selected = pos[axes]  # [D/2, B, S]
    theta = selected.to(torch.float32) * inv_freq.to("cpu", torch.float32)[:, None, None]
    return theta.permute(1, 2, 0).contiguous()


def precomputed_trig(position_ids: torch.Tensor, inv_freq: torch.Tensor, sections: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 cos/sin [B, S, D/2] shared by both sides of the precomputed-trig contract."""
    theta = mrope_theta_fp32(position_ids, inv_freq, sections).to(torch.float64)
    return torch.cos(theta).to(torch.float32), torch.sin(theta).to(torch.float32)


def make_mrope_inputs(case: MRoPECase) -> dict[str, torch.Tensor]:
    g = _generator(case.seed)
    dt = TORCH_DTYPES[case.dtype]
    q = torch.randn(case.B, case.Hq, case.S, case.D, generator=g).to(dt)
    k = torch.randn(case.B, case.Hkv, case.S, case.D, generator=g).to(dt)
    pos = make_positions(case)
    inv = inv_freq_for(case.D, case.rope_theta)
    if case.contract == "precomputed_trig":
        cos, sin = precomputed_trig(pos, inv, case.sections)
        return {"q": q, "k": k, "cos": cos, "sin": sin}
    return {"q": q, "k": k, "position_ids": pos, "inv_freq": inv}


def _rotate_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x1 * sin + x2 * cos
    return out


def mrope_reference(inputs: dict[str, torch.Tensor], case: MRoPECase) -> dict[str, torch.Tensor]:
    """float64 reference. theta follows the FP32 product contract; trig is evaluated in float64."""
    if case.contract == "precomputed_trig":
        cos = inputs["cos"].to("cpu", torch.float64)
        sin = inputs["sin"].to("cpu", torch.float64)
    else:
        theta = mrope_theta_fp32(inputs["position_ids"], inputs["inv_freq"], case.sections).to(torch.float64)
        cos, sin = torch.cos(theta), torch.sin(theta)
    cos, sin = cos[:, None], sin[:, None]  # broadcast over heads
    q = inputs["q"].to("cpu", torch.float64)
    k = inputs["k"].to("cpu", torch.float64)
    return {"q_out": _rotate_reference(q, cos, sin), "k_out": _rotate_reference(k, cos, sin)}


def _section_bounds(sections) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for n in sections:
        bounds.append((start, start + n))
        start += n
    return bounds


def _rotate_fp32(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)


def mrope_torch_eager(q, k, position_ids, inv_freq, sections) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean vectorized PyTorch baseline (positions-to-output contract, FP32 math, explicit cast)."""
    freqs = position_ids[..., None].to(torch.float32) * inv_freq  # [3, B, S, D/2]
    theta = torch.cat([freqs[a, ..., lo:hi] for a, (lo, hi) in enumerate(_section_bounds(sections))], dim=-1)
    cos = theta.cos()[:, None]
    sin = theta.sin()[:, None]
    return _rotate_fp32(q, cos, sin), _rotate_fp32(k, cos, sin)


def mrope_torch_eager_precomputed(q, k, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
    """Same rotation, consuming the shared precomputed FP32 cos/sin [B, S, D/2]."""
    c, s = cos[:, None], sin[:, None]
    return _rotate_fp32(q, c, s), _rotate_fp32(k, c, s)


def hf_qwen2vl_mrope_transcription(q, k, position_ids, inv_freq, sections):
    """Line-by-line transcription of the pinned upstream Qwen2-VL path (QWEN2VL_UPSTREAM).

    Note the upstream casts cos/sin to the activation dtype before rotating; it is used to
    verify the axis/rotation convention, not as a benchmark baseline.
    """
    freqs = position_ids[..., None].float() * inv_freq.to(dtype=torch.float)
    cos = freqs.cos()
    sin = freqs.sin()

    def recompose(freq):
        freq = torch.cat([m[i % 3] for i, m in enumerate(freq.split(list(sections), dim=-1))], dim=-1)
        return torch.cat((freq, freq), dim=-1)

    cos, sin = recompose(cos).to(dtype=q.dtype), recompose(sin).to(dtype=q.dtype)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ---------------------------------------------------------------------------
# Uniform per-operator access
# ---------------------------------------------------------------------------

def make_inputs(case) -> dict[str, torch.Tensor]:
    return make_layernorm_inputs(case) if case.operator == "layernorm" else make_mrope_inputs(case)


def reference(case, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if case.operator == "layernorm":
        return {"y": layernorm_reference(inputs["x"], inputs["gamma"], inputs["beta"], case.eps)}
    return mrope_reference(inputs, case)


def compare(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float, rel_floor: float) -> dict[str, Any]:
    """Elementwise |out - ref| <= atol + rtol * |ref| on float64 CPU copies."""
    o = out.detach().to("cpu", torch.float64)
    r = ref.detach().to("cpu", torch.float64)
    finite = bool(torch.isfinite(o).all().item())
    diff = (o - r).abs()
    if not finite:
        diff = torch.where(torch.isfinite(diff), diff, torch.full_like(diff, float("inf")))
    bound = atol + rtol * r.abs()
    mismatches = int((~(diff <= bound)).sum().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    max_rel = float((diff / r.abs().clamp_min(rel_floor)).max().item()) if diff.numel() else 0.0
    return {"finite": finite, "mismatch_count": mismatches, "element_count": int(diff.numel()),
            "max_abs_error": max_abs, "max_rel_error": max_rel, "passed": finite and mismatches == 0}
