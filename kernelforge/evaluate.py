"""Trusted correctness checking and task execution inside the GPU worker process.

Owns tensor generation, the correctness decision, timing and memory fields. Candidate
source is written to the task's scratch directory and imported only here, inside the
disposable worker, after the AST policy accepts it again.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .benchmark import capture_graph, measure_memory, paired_stats, time_implementations
from .config import is_wsl, nvidia_smi_query, os_release, package_version
from .contracts import (REL_ERROR_FLOOR, TOLERANCES, CandidateSpec, CorrectnessOutcome, MemoryOutcome,
                        ResultBundle, TaskBundle, TimingOutcome, utc_now)
from .kernels import LaunchConfigError, module_for
from .source_policy import check_source

PACKAGE_DIR = Path(__file__).resolve().parent


class WorkerQuarantine(RuntimeError):
    """The CUDA context is unhealthy; no further measurements from this process are valid."""


class CandidateValidationError(ValueError):
    pass


def evaluator_source_hash() -> str:
    """Identity of the trusted evaluator code (every .py file in the package)."""
    h = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        h.update(str(path.relative_to(PACKAGE_DIR)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def software_versions() -> dict[str, Any]:
    import torch

    info = {
        "kernelforge": __version__,
        "evaluator_sha256": evaluator_source_hash(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": package_version("triton"),
        "platform": "wsl2" if is_wsl() else platform.system().lower(),
        "distribution": os_release(),
    }
    return info


def hardware_info(gpu_index: int = 0) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, compute_capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count,
                    total_memory_bytes=p.total_memory, l2_cache_bytes=getattr(p, "L2_cache_size", None),
                    bf16_supported=torch.cuda.is_bf16_supported())
    smi = nvidia_smi_query(gpu_index)
    for key in ("uuid", "driver_version", "driver_cuda_version", "display_active"):
        if key in smi:
            info[key] = smi[key]
    return info


# ---------------------------------------------------------------------------
# Candidate loading and implementations
# ---------------------------------------------------------------------------

def load_kernel(spec: CandidateSpec, scratch: Path):
    report = check_source(spec.kernel_source, spec.operator, spec.entry_point)
    if not report.ok:
        raise CandidateValidationError("; ".join(report.problems))
    problems = spec.launch.validate_for(spec.operator)
    if problems:
        raise CandidateValidationError("; ".join(problems))
    scratch.mkdir(parents=True, exist_ok=True)
    mod_name = f"kf_candidate_{spec.short_hash}"
    path = scratch / f"{mod_name}.py"
    path.write_text(spec.kernel_source, encoding="utf-8")
    module_spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[mod_name] = module
    module_spec.loader.exec_module(module)
    kernel = getattr(module, spec.entry_point)
    from triton.runtime.jit import JITFunction

    if not isinstance(kernel, JITFunction):
        raise CandidateValidationError(f"{spec.entry_point} is not a triton.jit function")
    return kernel


class Impl:
    """A uniform callable view over Triton specs and PyTorch baselines."""

    def __init__(self, impl_id: str, operator: str, run: Callable, launch_into: Callable | None = None,
                 launch_config: Callable | None = None, setup: dict | None = None):
        self.impl_id = impl_id
        self.operator = operator
        self._run = run
        self._launch_into = launch_into
        self._launch_config = launch_config
        self.setup = setup or {}

    @property
    def has_kernel_scope(self) -> bool:
        return self._launch_into is not None

    def run(self, inputs, case):
        return self._run(inputs, case)

    def launch_into(self, inputs, outputs, case):
        return self._launch_into(inputs, outputs, case)

    def launch_config(self, inputs, case) -> dict:
        return self._launch_config(inputs, case) if self._launch_config else {}


def triton_impl(impl_id: str, spec: CandidateSpec, scratch: Path) -> Impl:
    mod = module_for(spec.operator)
    kernel = load_kernel(spec, scratch)
    last: dict[str, Any] = {}

    def launch_into(inputs, outputs, case):
        last.update(mod.launch(kernel, spec.launch, inputs, outputs, case))
        return outputs

    def run(inputs, case):
        outputs = mod.allocate_outputs(inputs)
        return launch_into(inputs, outputs, case)

    def config(inputs, case):
        return dict(last)

    return Impl(impl_id, spec.operator, run, launch_into, config,
                setup={"source_hash": spec.source_hash, "origin": spec.origin})


def build_impl(operator: str, impl_id: str, candidate: CandidateSpec | None, scratch: Path) -> Impl:
    mod = module_for(operator)
    if impl_id == "candidate":
        if candidate is None:
            raise ValueError("task has no candidate")
        return triton_impl("candidate", candidate, scratch)
    if impl_id == "manual":
        return triton_impl("manual", mod.MANUAL_SPEC, scratch)
    if impl_id == "triton_basic":
        if not hasattr(mod, "BASIC_SPEC"):
            raise ValueError(f"{operator} has no triton_basic baseline")
        return triton_impl("triton_basic", mod.BASIC_SPEC, scratch)
    if impl_id == "torch_eager":
        return Impl("torch_eager", operator, mod.torch_eager, setup={"api": "eager"})
    if impl_id == "torch_compile":
        fn = mod.torch_compiled()
        return Impl("torch_compile", operator, fn, setup=fn.setup)
    raise ValueError(f"unknown implementation {impl_id!r}")


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _is_compile_error(exc: BaseException) -> bool:
    try:
        from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, (CompilationError, CompileTimeAssertionFailure, SyntaxError, NameError)):
        return True
    return type(exc).__module__.startswith("triton.compiler")


def cuda_healthy() -> bool:
    import torch

    try:
        torch.cuda.synchronize()
        torch.zeros(1, device="cuda").add_(1).item()
        return True
    except Exception:
        return False


def _short(exc: BaseException, limit: int = 1500) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[:limit] + " ..."


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def _budget_skip(case, task: TaskBundle) -> str | None:
    import torch

    if case.estimated_live_bytes() > task.max_tensor_bytes:
        return f"estimated live bytes {case.estimated_live_bytes()} exceed max_tensor_bytes {task.max_tensor_bytes}"
    free, _ = torch.cuda.mem_get_info()
    if case.estimated_live_bytes() > 0.8 * free:
        return f"estimated live bytes {case.estimated_live_bytes()} exceed 80% of free VRAM ({free})"
    if case.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        return "bfloat16 not supported by this GPU"
    return None


def check_correctness(impl: Impl, case, task: TaskBundle) -> CorrectnessOutcome:
    import torch

    from .references import compare, make_inputs, reference

    tol = TOLERANCES[case.operator][case.dtype]
    reason = _budget_skip(case, task)
    if reason:
        return CorrectnessOutcome(workload_id=case.workload_id, status="skipped", failure_stage="unsupported",
                                  message=reason, tolerance=tol)
    mod = module_for(case.operator)
    cpu_inputs = make_inputs(case)
    ref = reference(case, cpu_inputs)
    inputs = {k: v.to("cuda") for k, v in cpu_inputs.items()}
    snapshot = {k: v.clone() for k, v in inputs.items()}
    try:
        mod.validate_inputs(inputs, case)
    except ValueError as exc:
        return CorrectnessOutcome(workload_id=case.workload_id, status="error", failure_stage="validation",
                                  message=str(exc), tolerance=tol)
    try:
        if impl.has_kernel_scope:
            outputs = mod.allocate_outputs(inputs, fill_nan=True)  # no-op kernels leave NaN behind
            impl.launch_into(inputs, outputs, case)
        else:
            outputs = impl.run(inputs, case)
        torch.cuda.synchronize()
    except LaunchConfigError as exc:
        return CorrectnessOutcome(workload_id=case.workload_id, status="failed", failure_stage="validation",
                                  message=_short(exc), passed=False, tolerance=tol)
    except Exception as exc:  # noqa: BLE001 - classified below
        if _is_compile_error(exc):
            return CorrectnessOutcome(workload_id=case.workload_id, status="failed", failure_stage="compile",
                                      message=_short(exc), passed=False, tolerance=tol)
        if not cuda_healthy():
            raise WorkerQuarantine(_short(exc)) from exc
        return CorrectnessOutcome(workload_id=case.workload_id, status="failed", failure_stage="execution",
                                  message=_short(exc), passed=False, tolerance=tol)

    checks: dict[str, bool] = {}
    max_abs = max_rel = 0.0
    mismatches = elements = 0
    finite = True
    for name, expected in ref.items():
        out = outputs.get(name)
        checks[f"{name}_present"] = out is not None
        if out is None:
            continue
        checks[f"{name}_shape"] = tuple(out.shape) == tuple(expected.shape)
        checks[f"{name}_dtype"] = out.dtype == inputs[next(iter(inputs))].dtype
        checks[f"{name}_device"] = out.device.type == "cuda"
        if not checks[f"{name}_shape"]:
            continue
        cmp = compare(out, expected, tol.atol, tol.rtol, REL_ERROR_FLOOR)
        finite &= cmp["finite"]
        max_abs = max(max_abs, cmp["max_abs_error"])
        max_rel = max(max_rel, cmp["max_rel_error"])
        mismatches += cmp["mismatch_count"]
        elements += cmp["element_count"]
        checks[f"{name}_numeric"] = cmp["passed"]
    checks["finite"] = finite
    checks["inputs_preserved"] = all(torch.equal(inputs[k], snapshot[k]) for k in inputs)
    checks["no_input_aliasing"] = all(
        o.untyped_storage().data_ptr() != i.untyped_storage().data_ptr()
        for o in outputs.values() for i in inputs.values())
    passed = all(checks.values())
    failed_checks = [k for k, v in checks.items() if not v]
    return CorrectnessOutcome(
        workload_id=case.workload_id, status="passed" if passed else "failed",
        failure_stage=None if passed else "correctness",
        message=None if passed else f"failed checks: {', '.join(failed_checks)}",
        passed=passed, checks=checks, max_abs_error=max_abs if elements else None,
        max_rel_error=max_rel if elements else None, mismatch_count=mismatches, element_count=elements,
        tolerance=tol)


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------

def run_correctness(task: TaskBundle, scratch: Path, result: ResultBundle) -> None:
    operator = task.candidate.operator if task.candidate else task.workloads[0].operator
    try:
        impl = build_impl(operator, task.implementation, task.candidate, scratch)
    except CandidateValidationError as exc:
        result.status, result.failure_stage, result.message = "failed", "validation", str(exc)
        result.correctness = [CorrectnessOutcome(workload_id=c.workload_id, status="not_run",
                                                 failure_stage="validation", message="candidate rejected by policy")
                              for c in task.workloads]
        return
    except Exception as exc:  # import-time failure of candidate module
        result.status, result.failure_stage, result.message = "failed", "compile", _short(exc)
        result.correctness = [CorrectnessOutcome(workload_id=c.workload_id, status="not_run",
                                                 failure_stage="compile", message="candidate module failed to load")
                              for c in task.workloads]
        return
    for i, case in enumerate(task.workloads):
        try:
            result.correctness.append(check_correctness(impl, case, task))
        except WorkerQuarantine as exc:
            result.status, result.failure_stage, result.message = "quarantined", "execution", str(exc)
            result.correctness.append(CorrectnessOutcome(workload_id=case.workload_id, status="error",
                                                         failure_stage="execution", message=str(exc), passed=False))
            for rest in task.workloads[i + 1:]:
                result.correctness.append(CorrectnessOutcome(workload_id=rest.workload_id, status="not_run",
                                                             failure_stage="worker",
                                                             message="worker quarantined after CUDA fault"))
            return
        gc.collect()
    if any(c.status not in ("passed", "skipped") for c in result.correctness):
        result.status = "failed"
        result.failure_stage = next(c.failure_stage for c in result.correctness if c.status not in ("passed", "skipped"))


def run_timing(task: TaskBundle, scratch: Path, result: ResultBundle) -> None:
    import torch

    from .references import compare, make_inputs, reference

    operator = task.workloads[0].operator
    impl_ids = ["candidate"] + [b for b in task.baselines if b != "candidate"] if task.candidate else list(task.baselines)
    impls = {i: build_impl(operator, i, task.candidate, scratch) for i in impl_ids}
    settings = task.benchmark
    gpu_index = int(os.environ.get("KERNELFORGE_GPU_INDEX", "0"))
    mod = module_for(operator)
    for case in task.workloads:
        reason = _budget_skip(case, task)
        if reason:
            result.timing.append(TimingOutcome(workload_id=case.workload_id, status="skipped",
                                               failure_stage="unsupported", message=reason))
            continue
        cpu_inputs = make_inputs(case)
        inputs = {k: v.to("cuda") for k, v in cpu_inputs.items()}
        try:
            mod.validate_inputs(inputs, case)
            # The evaluator re-checks every implementation on this exact workload before timing it.
            ref = reference(case, cpu_inputs)
            tol = TOLERANCES[case.operator][case.dtype]
            for impl in impls.values():
                out = impl.run(inputs, case)
                torch.cuda.synchronize()
                bad = [n for n, r in ref.items()
                       if not compare(out[n], r, tol.atol, tol.rtol, REL_ERROR_FLOOR)["passed"]]
                if bad:
                    raise AssertionError(f"{impl.impl_id} output {bad} disagrees with the reference")
                del out
            outcome = TimingOutcome(workload_id=case.workload_id, status="passed")
            for scope in settings.timing_scopes:
                inner, scope_settings, graphs = 1, settings, {}
                if scope == "operator":
                    fns = {i: (lambda impl=impl: impl.run(inputs, case)) for i, impl in impls.items()}
                elif scope == "kernel":
                    scoped = {i: impl for i, impl in impls.items() if impl.has_kernel_scope}
                    if len(scoped) < 2 or "candidate" not in scoped and task.candidate:
                        continue
                    outs = {i: mod.allocate_outputs(inputs) for i in scoped}
                    fns = {i: (lambda impl=impl, o=outs[i]: impl.launch_into(inputs, o, case))
                           for i, impl in scoped.items()}
                else:  # graph: the same operator calls replayed from a CUDA graph (no host dispatch cost)
                    inner = settings.calls_per_sample
                    for i, impl in impls.items():
                        try:
                            graphs[i] = capture_graph(lambda impl=impl: impl.run(inputs, case), inner)
                        except Exception as exc:  # noqa: BLE001 - recorded, scope skipped for this impl
                            if not cuda_healthy():
                                raise
                            outcome.telemetry.setdefault("graph_capture_failed", {})[i] = _short(exc, 300)
                    if "candidate" not in graphs or len(graphs) < 2:
                        continue
                    fns = {i: g.replay for i, g in graphs.items()}
                    scope_settings = settings.model_copy(update={
                        "calls_per_sample": 1, "warmup_iterations": max(2, settings.warmup_iterations // inner),
                        "min_batch_ms": settings.min_batch_ms})
                stats, flags, telemetry = time_implementations(fns, scope, scope_settings, gpu_index,
                                                               task.generator_resident, inner_calls=inner)
                del graphs
                outcome.stats.extend(stats)
                outcome.contamination_flags = sorted(set(outcome.contamination_flags) | set(flags))
                outcome.telemetry[scope] = telemetry
                by_id = {s.impl_id: s for s in stats}
                if "candidate" in by_id:
                    for bid, bstats in by_id.items():
                        if bid != "candidate":
                            outcome.paired.append(paired_stats(bstats, by_id["candidate"], seed=settings.seed))
            launch = impls["candidate"].launch_config(inputs, case) if "candidate" in impls else {}
            outcome.telemetry["launch_config"] = launch
            outcome.telemetry["impl_setup"] = {i: impl.setup for i, impl in impls.items()}
            result.timing.append(outcome)
        except AssertionError as exc:
            result.timing.append(TimingOutcome(workload_id=case.workload_id, status="failed",
                                               failure_stage="correctness", message=str(exc)))
        except Exception as exc:  # noqa: BLE001
            if not cuda_healthy():
                result.status, result.failure_stage, result.message = "quarantined", "execution", _short(exc)
                result.timing.append(TimingOutcome(workload_id=case.workload_id, status="error",
                                                   failure_stage="execution", message=_short(exc)))
                return
            stage = "compile" if _is_compile_error(exc) else "benchmark"
            result.timing.append(TimingOutcome(workload_id=case.workload_id, status="error",
                                               failure_stage=stage, message=_short(exc)))
        finally:
            del inputs
            gc.collect()
            torch.cuda.empty_cache()
    if any(t.status not in ("passed", "skipped") for t in result.timing):
        result.status = "failed" if result.status == "ok" else result.status
        result.failure_stage = result.failure_stage or "benchmark"


def run_memory(task: TaskBundle, scratch: Path, result: ResultBundle) -> None:
    import torch

    from .references import make_inputs

    operator = task.workloads[0].operator
    impl = build_impl(operator, task.implementation, task.candidate, scratch)
    mod = module_for(operator)
    for case in task.workloads:
        reason = _budget_skip(case, task)
        if reason:
            result.memory.append(MemoryOutcome(workload_id=case.workload_id, impl_id=impl.impl_id,
                                               status="skipped", failure_stage="unsupported", message=reason))
            continue
        inputs = {k: v.to("cuda") for k, v in make_inputs(case).items()}
        try:
            mod.validate_inputs(inputs, case)
            result.memory.append(measure_memory(lambda: impl.run(inputs, case), inputs, case.workload_id,
                                                impl.impl_id))
        except Exception as exc:  # noqa: BLE001
            if not cuda_healthy():
                result.status, result.failure_stage, result.message = "quarantined", "execution", _short(exc)
                return
            result.memory.append(MemoryOutcome(workload_id=case.workload_id, impl_id=impl.impl_id, status="error",
                                               failure_stage="memory", message=_short(exc)))
        finally:
            del inputs
            gc.collect()
            torch.cuda.empty_cache()


def run_probe(task: TaskBundle, result: ResultBundle) -> None:
    import torch

    free, total = torch.cuda.mem_get_info()
    probe: dict[str, Any] = {"free_bytes": free, "total_bytes": total}
    # Allocation probe: confirm the reported free memory is actually allocatable.
    want = int(min(1 << 30, free * 0.5))
    try:
        buf = torch.empty(want, dtype=torch.uint8, device="cuda")
        buf.fill_(1)
        torch.cuda.synchronize()
        probe["allocation_probe_bytes"] = want
        probe["allocation_probe_ok"] = True
        del buf
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        probe["allocation_probe_ok"] = False
        probe["allocation_probe_error"] = _short(exc)
    result.probe = probe


def run_smoke(task: TaskBundle, result: ResultBundle) -> None:
    """Blackwell smoke checks: a CUDA matmul and a compiled Triton vector add on the device."""
    import torch
    import triton
    import triton.language as tl

    checks: dict[str, Any] = {"arch_list": torch.cuda.get_arch_list()}
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    checks["matmul_finite"] = bool(torch.isfinite(y).all().item())
    ref = (x.double() @ x.double()).float()
    checks["matmul_max_abs_err"] = float((y - ref).abs().max().item())

    @triton.jit
    def _add(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(c_ptr + offs, tl.load(a_ptr + offs, mask=m) + tl.load(b_ptr + offs, mask=m), mask=m)

    n = 98_765
    a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    c = torch.empty_like(a)
    _add[(triton.cdiv(n, 1024),)](a, b, c, n, BLOCK=1024)
    torch.cuda.synchronize()
    checks["triton_vector_add_exact"] = bool(torch.equal(c, a + b))
    cap = torch.cuda.get_device_capability(0)
    checks["compute_capability"] = f"{cap[0]}.{cap[1]}"
    checks["sm_target_in_arch_list"] = f"sm_{cap[0]}{cap[1]}" in checks["arch_list"]
    ok = checks["matmul_finite"] and checks["triton_vector_add_exact"]
    result.probe = checks
    if not ok:
        result.status, result.failure_stage, result.message = "failed", "execution", "smoke check failed"


def run_selftest(task: TaskBundle, scratch: Path, result: ResultBundle) -> None:
    """Baselines must agree with the independent reference on the device."""
    task_copy = task.model_copy(update={"implementation": "torch_eager"})
    run_correctness(task_copy, scratch, result)


def execute_task(task: TaskBundle, scratch: Path) -> ResultBundle:
    result = ResultBundle(task_id=task.task_id, kind=task.kind, status="ok",
                          isolation={"transport": os.environ.get("KERNELFORGE_TRANSPORT", "unknown"),
                                     "pid": os.getpid()})
    try:
        import torch

        if not torch.cuda.is_available():
            result.status, result.failure_stage = "error", "unsupported"
            result.message = "CUDA is not available in the worker; GPU tasks cannot run (not a pass)"
            result.software = {"python": sys.version.split()[0], "torch": torch.__version__}
            return result
        result.software = software_versions()
        result.hardware = hardware_info(int(os.environ.get("KERNELFORGE_GPU_INDEX", "0")))
        if task.kind == "probe":
            run_probe(task, result)
        elif task.kind == "smoke":
            run_smoke(task, result)
        elif task.kind == "selftest":
            run_selftest(task, scratch, result)
        elif task.kind == "correctness":
            run_correctness(task, scratch, result)
        elif task.kind == "timing":
            run_timing(task, scratch, result)
        elif task.kind == "memory":
            run_memory(task, scratch, result)
    except WorkerQuarantine as exc:
        result.status, result.failure_stage, result.message = "quarantined", "execution", str(exc)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed silently
        result.status, result.failure_stage = "error", "worker"
        result.message = _short(exc) + "\n" + traceback.format_exc(limit=8)[-2000:]
    result.finished_at = utc_now()
    return result
