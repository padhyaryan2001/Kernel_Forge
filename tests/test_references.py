from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kernelforge.contracts import LayerNormCase, MRoPECase, TOLERANCES  # noqa: E402
from kernelforge.references import (QWEN2VL_UPSTREAM, compare, hf_qwen2vl_mrope_transcription,  # noqa: E402
                                    layernorm_reference, layernorm_torch_eager, make_layernorm_inputs,
                                    make_mrope_inputs, mrope_reference, mrope_torch_eager,
                                    mrope_torch_eager_precomputed, precomputed_trig)
from kernelforge.selfcheck import run_reference_checks  # noqa: E402


def test_all_cpu_reference_checks_pass():
    results = run_reference_checks()
    failed = [r for r in results if not r["passed"]]
    assert not failed, failed


@pytest.mark.parametrize("kind", ["normal", "constant_rows", "low_variance", "negative", "large_offset"])
@pytest.mark.parametrize("N", [128, 1000, 3072])
def test_layernorm_reference_matches_functional(kind, N):
    case = LayerNormCase(workload_id="t", M=5, N=N, dtype="float32", seed=11, input_kind=kind)
    inp = {k: v.double() for k, v in make_layernorm_inputs(case).items()}
    ref = layernorm_reference(inp["x"], inp["gamma"], inp["beta"], case.eps)
    out = layernorm_torch_eager(inp["x"], inp["gamma"], inp["beta"], case.eps)
    assert torch.allclose(ref, out, atol=1e-10, rtol=1e-10)


def test_inputs_are_deterministic_and_dtype_rounded():
    case = LayerNormCase(workload_id="t", M=3, N=64, dtype="bfloat16", seed=4)
    a, b = make_layernorm_inputs(case), make_layernorm_inputs(case)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert a["x"].dtype == torch.bfloat16


def test_epsilon_is_used():
    case = LayerNormCase(workload_id="t", M=2, N=256, dtype="float32", seed=1, input_kind="low_variance")
    inp = make_layernorm_inputs(case)
    a = layernorm_reference(inp["x"], inp["gamma"], inp["beta"], 1e-6)
    b = layernorm_reference(inp["x"], inp["gamma"], inp["beta"], 1e-1)
    assert (a - b).abs().max() > 0.1


@pytest.mark.parametrize("kind", ["text", "image", "video"])
@pytest.mark.parametrize("D,sections", [(64, (8, 12, 12)), (128, (16, 24, 24))])
def test_mrope_baseline_matches_reference_fp32(kind, D, sections):
    case = MRoPECase(workload_id="t", B=2, Hq=8, Hkv=2, S=80, D=D, sections=sections, dtype="float32", seed=3,
                     position_kind=kind)
    inp = make_mrope_inputs(case)
    ref = mrope_reference(inp, case)
    qo, ko = mrope_torch_eager(inp["q"], inp["k"], inp["position_ids"], inp["inv_freq"], sections)
    tol = TOLERANCES["mrope"]["float32"]
    assert compare(qo, ref["q_out"], tol.atol, tol.rtol, 1e-3)["passed"]
    assert compare(ko, ref["k_out"], tol.atol, tol.rtol, 1e-3)["passed"]


def test_mrope_convention_matches_pinned_upstream_transcription():
    assert QWEN2VL_UPSTREAM["revision"]
    case = MRoPECase(workload_id="t", B=1, Hq=4, Hkv=4, S=64, D=128, sections=(16, 24, 24), dtype="float32", seed=8,
                     position_kind="video")
    inp = make_mrope_inputs(case)
    ref = mrope_reference(inp, case)
    hq, hk = hf_qwen2vl_mrope_transcription(inp["q"].double(), inp["k"].double(), inp["position_ids"],
                                            inp["inv_freq"], case.sections)
    assert (hq - ref["q_out"]).abs().max() < 1e-5 and (hk - ref["k_out"]).abs().max() < 1e-5


def test_mrope_matches_installed_transformers_if_available():
    mod = pytest.importorskip("transformers.models.qwen2_vl.modeling_qwen2_vl")
    rotate_half = getattr(mod, "rotate_half")
    case = MRoPECase(workload_id="t", B=1, Hq=2, Hkv=2, S=40, D=64, sections=(8, 12, 12), dtype="float32", seed=2,
                     position_kind="image")
    inp = make_mrope_inputs(case)
    ref = mrope_reference(inp, case)
    freqs = inp["position_ids"][..., None].double() * inp["inv_freq"].double()
    bounds = [(0, 8), (8, 20), (20, 32)]
    theta = torch.cat([freqs[a, ..., lo:hi] for a, (lo, hi) in enumerate(bounds)], dim=-1)
    emb = torch.cat([theta, theta], dim=-1)[:, None]
    q = inp["q"].double()
    out = q * emb.cos() + rotate_half(q) * emb.sin()
    assert (out - ref["q_out"]).abs().max() < 1e-5


def test_precomputed_trig_contract_is_equivalent():
    case = MRoPECase(workload_id="t", B=2, Hq=4, Hkv=1, S=33, D=64, sections=(8, 12, 12), dtype="float32", seed=6,
                     position_kind="video", position_offset=300_000)
    inp = make_mrope_inputs(case)
    cos, sin = precomputed_trig(inp["position_ids"], inp["inv_freq"], case.sections)
    a = mrope_torch_eager(inp["q"], inp["k"], inp["position_ids"], inp["inv_freq"], case.sections)
    b = mrope_torch_eager_precomputed(inp["q"], inp["k"], cos, sin)
    assert torch.allclose(a[0], b[0], atol=1e-6) and torch.allclose(a[1], b[1], atol=1e-6)


def test_inputs_not_mutated_by_baselines():
    case = MRoPECase(workload_id="t", B=1, Hq=2, Hkv=1, S=16, D=64, sections=(8, 12, 12), dtype="float16", seed=1)
    inp = make_mrope_inputs(case)
    snap = {k: v.clone() for k, v in inp.items()}
    mrope_torch_eager(inp["q"], inp["k"], inp["position_ids"], inp["inv_freq"], case.sections)
    assert all(torch.equal(inp[k], snap[k]) for k in inp)


def test_compare_flags_nan_and_mismatch():
    ref = torch.ones(10, dtype=torch.float64)
    out = ref.clone().float()
    assert compare(out, ref, 1e-5, 1e-5, 1e-3)["passed"]
    out[3] = float("nan")
    r = compare(out, ref, 1e-5, 1e-5, 1e-3)
    assert not r["passed"] and not r["finite"]
    out = ref.clone().float() + 1e-3
    r = compare(out, ref, 1e-5, 1e-5, 1e-3)
    assert not r["passed"] and r["mismatch_count"] == 10 and r["max_abs_error"] == pytest.approx(1e-3, rel=1e-3)
