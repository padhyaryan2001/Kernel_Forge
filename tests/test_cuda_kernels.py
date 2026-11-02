"""Real-GPU tests through the trusted worker path. Skipped (reported as skipped, never passed)
without a CUDA device and Triton."""
from __future__ import annotations

import sys

import pytest

from kernelforge.config import KernelForgeConfig
from kernelforge.kernels import layernorm as ln, mrope as mr
from kernelforge.kernels.canaries import canaries
from kernelforge.workloads import manifest

pytestmark = pytest.mark.cuda


@pytest.fixture(scope="module")
def executor(tmp_path_factory):
    from kernelforge.executor import Executor
    from kernelforge.store import Store

    cfg = KernelForgeConfig(runs_dir=str(tmp_path_factory.mktemp("runs")),
                            worker={"transport": "local_process", "python": sys.executable})
    return Executor(cfg, Store(cfg.runs_path()))


def test_blackwell_smoke(executor):
    res = executor.smoke("pytest")
    assert res.status == "ok", res.message
    assert res.probe["matmul_finite"] and res.probe["triton_vector_add_exact"]


@pytest.mark.parametrize("spec", [ln.BASIC_SPEC, ln.MANUAL_SPEC, mr.BASIC_SPEC, mr.MANUAL_SPEC],
                         ids=["ln_basic", "ln_manual", "mr_basic", "mr_manual"])
def test_trusted_kernels_pass_dev_manifest(executor, spec):
    m = manifest(spec.operator, "dev")
    res = executor.run_task(executor.make_task("pytest", "correctness", candidate=spec, workloads=m.cases,
                                               split="dev"), 1800)
    bad = [(c.workload_id, c.message) for c in res.correctness if c.status not in ("passed", "skipped")]
    assert res.status == "ok" and not bad, bad


@pytest.mark.parametrize("name", list(canaries()))
def test_evaluator_rejects_canaries(executor, name):
    spec, expected = canaries()[name]
    m = manifest(spec.operator, "dev")
    res = executor.run_task(executor.make_task("pytest", "correctness", candidate=spec, workloads=m.cases,
                                               split="dev"), 1800)
    stages = {c.failure_stage for c in res.correctness if c.status != "passed"}
    assert res.status != "ok" and expected in stages


def test_timing_task_returns_raw_samples_and_memory(executor):
    from kernelforge.contracts import BenchmarkSettings

    cases = manifest("layernorm", "smoke").cases[:2]
    settings = BenchmarkSettings(warmup_iterations=3, samples=6, calls_per_sample=5)
    res = executor.run_task(executor.make_task("pytest", "timing", candidate=ln.MANUAL_SPEC, workloads=cases,
                                               split="smoke", baselines=["torch_eager", "triton_basic"],
                                               benchmark=settings), 900)
    assert res.status == "ok", res.message
    t = res.timing[0]
    scopes = {(s.impl_id, s.scope) for s in t.stats}
    assert {("candidate", "operator"), ("torch_eager", "operator"), ("candidate", "kernel"),
            ("triton_basic", "kernel"), ("candidate", "graph")} <= scopes
    assert all(len(s.samples_ms) == 6 and min(s.samples_ms) > 0 for s in t.stats)
    mem = executor.run_task(executor.make_task("pytest", "memory", candidate=ln.MANUAL_SPEC, implementation="candidate",
                                               workloads=cases, split="smoke"), 600)
    m0 = mem.memory[0]
    assert m0.status == "passed" and m0.incremental_peak_bytes >= m0.output_bytes > 0


def test_non_contiguous_input_is_rejected_not_copied():
    import torch

    from kernelforge.contracts import LayerNormCase

    case = LayerNormCase(workload_id="t", M=4, N=64, dtype="float32", seed=0)
    x = torch.randn(64, 4, device="cuda").t()
    with pytest.raises(ValueError, match="non-contiguous"):
        ln.validate_inputs({"x": x, "gamma": torch.ones(64, device="cuda"), "beta": torch.zeros(64, device="cuda")},
                           case)
