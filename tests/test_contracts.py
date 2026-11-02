from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from kernelforge.config import KernelForgeConfig, TargetsConfig
from kernelforge.contracts import (CandidateSpec, EvaluationRecord, LaunchPolicy, LaunchTier, MRoPECase,
                                   LayerNormCase, latency_reduction_percent, memory_reduction_percent, speedup,
                                   speedup_to_latency_reduction)
from kernelforge.kernels import layernorm as ln, mrope as mr
from kernelforge.workloads import manifest, shape_key


def test_target_definitions_are_precise():
    # 22% faster == 1.22x speedup == ~18.0% lower latency, not a 22% latency reduction
    assert speedup_to_latency_reduction(1.22) == pytest.approx(18.03, abs=0.01)
    assert latency_reduction_percent(1.22, 1.0) == pytest.approx(18.03, abs=0.01)
    assert speedup(1.22, 1.0) == pytest.approx(1.22)
    assert memory_reduction_percent(100, 40) == pytest.approx(60.0)
    text = " ".join(t["definition"] for t in TargetsConfig().describe())
    assert "18.0% lower latency" in text and "not whole-model" in text


def test_targets_live_only_in_targets_config():
    cfg = KernelForgeConfig()
    assert cfg.targets.layernorm_speedup == 1.22
    # nothing in the evaluation record schema can carry a target value
    assert not any("target" in f for f in EvaluationRecord.model_fields)


def test_unmeasured_metrics_are_null_not_zero():
    base = dict(experiment_id="e", workload_id="w", candidate_hash="h", operator="layernorm", evaluation_split="dev",
                status="not_run", dtype="float16", shapes={"x": [1, 2]}, contract="c", baseline_id="torch_eager")
    rec = EvaluationRecord(**base)
    assert rec.candidate_median_ms is None and rec.speedup is None and rec.max_abs_error is None
    with pytest.raises(ValidationError):
        EvaluationRecord(**base, candidate_median_ms=0.0)
    with pytest.raises(ValidationError):
        EvaluationRecord(**base, speedup=0.0)


def test_launch_policy_bounds_and_selection():
    pol = LaunchPolicy(tiers=[LaunchTier(upto=1024, params={"ROWS_PER_PROGRAM": 4, "num_warps": 4}),
                              LaunchTier(upto=None, params={"num_warps": 8})])
    assert pol.validate_for("layernorm") == []
    assert pol.select(512)["ROWS_PER_PROGRAM"] == 4
    assert pol.select(8192) == {"num_warps": 8}
    assert LaunchPolicy.single(num_warps=3).validate_for("layernorm")  # not a power of two
    assert LaunchPolicy.single(BOGUS=1).validate_for("layernorm")
    no_catch_all = LaunchPolicy(tiers=[LaunchTier(upto=10, params={})])
    assert any("catch-all" in p for p in no_catch_all.validate_for("layernorm"))
    assert ln.MANUAL_SPEC.launch.validate_for("layernorm") == []
    assert mr.MANUAL_SPEC.launch.validate_for("mrope") == []


def test_candidate_hash_is_stable_and_content_addressed():
    a = ln.MANUAL_SPEC
    b = CandidateSpec(**a.model_dump())
    assert a.source_hash == b.source_hash
    c = a.model_copy(update={"launch": LaunchPolicy.single(num_warps=2)})
    assert c.source_hash != a.source_hash
    # explanation is metadata, not identity
    assert a.model_copy(update={"explanation": "x"}).source_hash == a.source_hash


def test_mrope_case_validation():
    MRoPECase(workload_id="w", B=1, Hq=4, Hkv=2, S=8, D=64, sections=(8, 12, 12), dtype="float16", seed=0)
    with pytest.raises(ValidationError):
        MRoPECase(workload_id="w", B=1, Hq=4, Hkv=2, S=8, D=64, sections=(8, 12, 13), dtype="float16", seed=0)
    with pytest.raises(ValidationError):
        LayerNormCase(workload_id="w", M=1, N=8, dtype="float16", seed=0, eps=0.0)


@pytest.mark.parametrize("op", ["layernorm", "mrope"])
def test_manifests_heldout_disjoint_and_fit_budget(op):
    dev = manifest(op, "dev")
    held = manifest(op, "heldout")
    assert {shape_key(c) for c in dev.cases}.isdisjoint({shape_key(c) for c in held.cases})
    assert {c.seed for c in dev.cases}.isdisjoint({c.seed for c in held.cases})
    for split in ("smoke", "search", "dev", "heldout"):
        m = manifest(op, split)
        assert len({c.workload_id for c in m.cases}) == len(m.cases)
        assert all(c.estimated_live_bytes() <= 1 << 30 for c in m.cases)
        assert m.manifest_hash == manifest(op, split).manifest_hash  # deterministic


def test_dev_manifest_covers_declared_matrix():
    dev = manifest("layernorm", "dev").cases
    assert {c.M for c in dev} >= {1, 32, 256, 2048}
    assert {c.N for c in dev} >= {128, 768, 1024, 3072, 4096, 8192}
    assert {c.input_kind for c in dev} == {"normal", "constant_rows", "low_variance", "negative", "large_offset"}
    assert {c.N for c in manifest("layernorm", "heldout").cases} >= {1536, 5120}
    mdev = manifest("mrope", "dev").cases
    assert {c.S for c in mdev} >= {128, 1024, 4096} and {c.D for c in mdev} == {64, 128}
    assert {c.position_kind for c in mdev} == {"text", "image", "video"}
    assert {c.contract for c in mdev} == {"positions_to_output", "precomputed_trig"}
    assert any(c.Hq != c.Hkv for c in mdev) and any(c.position_offset >= 100_000 for c in mdev)


def test_config_rejects_concurrent_gpu_jobs():
    with pytest.raises(ValidationError):
        KernelForgeConfig(worker={"concurrent_jobs": 2})


def test_rl_example_config_loads_and_reward_is_bounded():
    from pathlib import Path

    from kernelforge.train_rl import INSTRUCTIONS, build_prompt, load_rl_config, reference_completion

    rl = load_rl_config(str(Path(__file__).resolve().parent.parent / "rl.example.yaml"))
    assert rl.num_generations == 2 and rl.max_prompt_tokens == 768 and rl.max_completion_tokens == 640
    assert "0.5B" in rl.model and rl.model_revision
    train, held = INSTRUCTIONS[:4], INSTRUCTIONS[4:]
    assert set(train).isdisjoint(held)
    msgs = build_prompt("layernorm", held[0])
    assert held[0] in msgs[1]["content"] and "```python" in reference_completion("layernorm")
    # reward formula from the brief: every correct candidate beats any failure (-1)
    clip = lambda s: max(-0.5, min(2.0, math.log2(s)))  # noqa: E731
    assert clip(0.01) == -0.5 > -1 and clip(1000) == 2.0 and clip(1.0) == 0.0


def test_example_config_loads():
    from pathlib import Path

    from kernelforge.config import load_config

    cfg = load_config(Path(__file__).resolve().parent.parent / "config.example.yaml")
    assert cfg.generator.context_tokens == 4096 and cfg.generator.max_output_tokens == 1536
    assert cfg.worker.max_tensor_bytes == 1 << 30 and cfg.independent_sessions == 3
    assert math.isclose(cfg.targets.generated_kernel_speedup, 2.2)
