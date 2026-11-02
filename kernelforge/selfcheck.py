"""CPU checks of the independent references (`kernelforge test-reference --device cpu`).

They exercise the mathematics only (no Triton, no GPU) and are also covered by pytest.
"""
from __future__ import annotations

import math


def run_reference_checks() -> list[dict]:
    import torch

    from .contracts import LayerNormCase, MRoPECase
    from .references import (axis_of_pair, hf_qwen2vl_mrope_transcription, inv_freq_for, layernorm_reference,
                             layernorm_torch_eager, make_layernorm_inputs, make_mrope_inputs, make_positions,
                             mrope_reference, mrope_torch_eager, mrope_torch_eager_precomputed, precomputed_trig)

    results: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        results.append({"name": name, "passed": bool(passed), "detail": detail})

    # LayerNorm: reference vs torch.nn.functional.layer_norm in float64
    for M, N, kind in [(4, 128, "normal"), (3, 1000, "negative"), (5, 4096, "large_offset"), (2, 333, "low_variance")]:
        case = LayerNormCase(workload_id="cpu", M=M, N=N, dtype="float32", seed=7, input_kind=kind)
        inp = {k: v.double() for k, v in make_layernorm_inputs(case).items()}
        ref = layernorm_reference(inp["x"], inp["gamma"], inp["beta"], case.eps)
        base = layernorm_torch_eager(inp["x"], inp["gamma"], inp["beta"], case.eps)
        err = (ref - base).abs().max().item()
        check(f"layernorm ref == F.layer_norm (M={M}, N={N}, {kind})", err < 1e-10, f"max abs diff {err:.2e}")

    case = LayerNormCase(workload_id="cpu", M=6, N=768, dtype="float32", seed=3, input_kind="constant_rows")
    inp = make_layernorm_inputs(case)
    ref = layernorm_reference(inp["x"], inp["gamma"], inp["beta"], case.eps)
    err = (ref - inp["beta"].double()).abs().max().item()
    check("layernorm constant rows -> beta", err == 0.0, f"max abs diff {err:.2e}")

    x = torch.randn(2, 64, dtype=torch.float64) * 1e-3
    g, b = torch.ones(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64)
    lo, hi = layernorm_reference(x, g, b, 1e-6), layernorm_reference(x, g, b, 1e-2)
    check("layernorm epsilon changes low-variance output", (lo.abs().max() > 5 * hi.abs().max()).item(),
          f"|y| eps=1e-6: {lo.abs().max().item():.3f}, eps=1e-2: {hi.abs().max().item():.3f}")

    y = layernorm_reference(torch.randn(3, 512, dtype=torch.float64), g.new_ones(512), g.new_zeros(512), 1e-5)
    check("layernorm population variance (mean 0, var ~1)",
          abs(y.mean().item()) < 1e-12 and abs(y.var(unbiased=False, dim=-1).mean().item() - 1) < 1e-3,
          f"mean {y.mean().item():.1e}, var {y.var(unbiased=False, dim=-1).mean().item():.5f}")

    # MRoPE
    for kind in ("text", "image", "video"):
        case = MRoPECase(workload_id="cpu", B=2, Hq=4, Hkv=2, S=96, D=64, sections=(8, 12, 12), dtype="float32",
                         seed=5, position_kind=kind)
        inp = make_mrope_inputs(case)
        ref = mrope_reference(inp, case)
        qo, ko = mrope_torch_eager(inp["q"], inp["k"], inp["position_ids"], inp["inv_freq"], case.sections)
        err = max((qo.double() - ref["q_out"]).abs().max().item(), (ko.double() - ref["k_out"]).abs().max().item())
        check(f"mrope ref == eager baseline ({kind}, Hq=4, Hkv=2)", err < 1e-5, f"max abs diff {err:.2e}")
        hq, hk = hf_qwen2vl_mrope_transcription(inp["q"].double(), inp["k"].double(), inp["position_ids"],
                                                inp["inv_freq"], case.sections)
        err = max((hq - ref["q_out"]).abs().max().item(), (hk - ref["k_out"]).abs().max().item())
        # upstream evaluates cos/sin in FP32 (~1e-7); a convention mismatch would be O(1)
        check(f"mrope ref == pinned Qwen2-VL convention ({kind})", err < 1e-5, f"max abs diff {err:.2e}")

    # text positions: every axis equal -> ordinary 1D split-half RoPE
    case = MRoPECase(workload_id="cpu", B=1, Hq=2, Hkv=2, S=50, D=128, sections=(16, 24, 24), dtype="float32", seed=9)
    inp = make_mrope_inputs(case)
    ref = mrope_reference(inp, case)
    pos = torch.arange(50, dtype=torch.float64)
    theta = pos[:, None] * inp["inv_freq"].double()[None, :]
    q = inp["q"].double()
    rot = torch.cat([q[..., :64] * theta.cos() - q[..., 64:] * theta.sin(),
                     q[..., :64] * theta.sin() + q[..., 64:] * theta.cos()], dim=-1)
    err = (rot - ref["q_out"]).abs().max().item()
    check("mrope text positions == 1D RoPE", err < 1e-5, f"max abs diff {err:.2e}")

    # split-half vs interleaved must differ (the evaluator would catch a convention mix-up)
    inter = torch.empty_like(q)
    inter[..., 0::2] = q[..., 0::2] * theta.cos() - q[..., 1::2] * theta.sin()
    inter[..., 1::2] = q[..., 0::2] * theta.sin() + q[..., 1::2] * theta.cos()
    check("interleaved rotation differs from split-half", (inter - ref["q_out"]).abs().max().item() > 0.1,
          f"max abs diff {(inter - ref['q_out']).abs().max().item():.2f}")

    check("sections map pairs to t/h/w axes", axis_of_pair((2, 3, 1)) == [0, 0, 1, 1, 1, 2], str(axis_of_pair((2, 3, 1))))

    case = MRoPECase(workload_id="cpu", B=1, Hq=1, Hkv=1, S=200, D=64, sections=(8, 12, 12), dtype="float32", seed=1,
                     position_kind="video")
    p = make_positions(case)
    check("video positions: temporal axis advances, h/w repeat per frame",
          p[0, 0, -1] > p[0, 0, 30] and len(set(p[1, 0, 25:].tolist())) <= 8 + 1,
          f"t range {p[0, 0].min().item()}..{p[0, 0].max().item()}")

    case = MRoPECase(workload_id="cpu", B=1, Hq=2, Hkv=1, S=64, D=64, sections=(8, 12, 12), dtype="float32", seed=2,
                     position_kind="video", position_offset=750_000)
    inp = make_mrope_inputs(case)
    ref = mrope_reference(inp, case)
    qo, ko = mrope_torch_eager(inp["q"], inp["k"], inp["position_ids"], inp["inv_freq"], case.sections)
    err = (qo.double() - ref["q_out"]).abs().max().item()
    check("mrope large positions (offset 750k): eager fp32 matches fp32-theta contract", err < 1e-5,
          f"max abs diff {err:.2e}")

    cos, sin = precomputed_trig(inp["position_ids"], inp["inv_freq"], case.sections)
    pq, pk = mrope_torch_eager_precomputed(inp["q"], inp["k"], cos, sin)
    err = (pq - qo).abs().max().item()
    check("precomputed-trig contract == positions contract", err < 1e-6, f"max abs diff {err:.2e}")

    inv = inv_freq_for(128, 1_000_000.0)
    check("inv_freq construction", math.isclose(inv[0].item(), 1.0) and inv[-1].item() < 1e-5,
          f"first {inv[0].item()}, last {inv[-1].item():.2e}")
    return results
