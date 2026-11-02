"""Dispatch of validated task bundles to the isolated worker, and controller-side assembly
of evaluation records.

Transports
- local_process: a fresh `python -m kernelforge.worker` process per task with a scrubbed
  environment (no API keys), its own HOME/TMPDIR, rlimits and a wall-clock timeout. Shares
  the host network namespace and filesystem permissions: development isolation only.
- local_container: a disposable `docker run` with --network none, read-only root,
  dropped capabilities, no-new-privileges, pids/memory limits, non-root user and only the
  task directory mounted. The evaluator code is baked into the image.
- remote: optional SSH adapter. Host keys must already be known (StrictHostKeyChecking=yes);
  the task bundle travels on stdin, so no shell command is ever built from model output.

Exactly one GPU job runs at a time (file lock under runs/).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import PROJECT_ROOT, KernelForgeConfig
from .contracts import (MEMORY_METRIC, MEMORY_METRIC_DEFINITION, CandidateSpec, EvaluationRecord, Manifest,
                        ResultBundle, TaskBundle, TimingStats, geomean, memory_reduction_percent, utc_now)
from .store import Store

SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
ENV_ALLOW = ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "TZ")


def scrubbed_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k in ENV_ALLOW and not any(m in k.upper() for m in SECRET_MARKERS)}
    if "PATH" in env:
        # Under WSL, interop appends Windows directories (/mnt/c/...). The worker must not run Windows
        # binaries, and an unreadable entry turns tool probes (e.g. Inductor's `nvcc` lookup) into
        # PermissionError instead of a clean "not found".
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if p and not p.startswith("/mnt/"))
    env.update(extra)
    return env


def _limits() -> None:  # pragma: no cover - runs in the child
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))


@contextmanager
def gpu_lock(runs_dir: Path) -> Iterator[None]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / ".gpu.lock"
    fh = open(path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.5)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


class Executor:
    def __init__(self, cfg: KernelForgeConfig, store: Store | None = None):
        self.cfg = cfg
        self.store = store
        self.runs_dir = cfg.runs_path()
        self.last_quarantine: str | None = None

    # -- transports ---------------------------------------------------------
    def _task_dir(self, task: TaskBundle) -> Path:
        d = self.runs_dir / (task.experiment_id or "_system") / "tasks" / task.task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _local_process(self, task_dir: Path, timeout_s: float) -> tuple[int | None, str]:
        w = self.cfg.worker
        exp_dir = task_dir.parent.parent
        for sub in ("home", "tmp", "scratch"):
            (task_dir / sub).mkdir(exist_ok=True)
        env = scrubbed_env({
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(task_dir / "home"),
            "TMPDIR": str(task_dir / "tmp"),
            "KERNELFORGE_SCRATCH": str(task_dir / "scratch"),
            "KERNELFORGE_TRANSPORT": "local_process",
            "KERNELFORGE_GPU_INDEX": str(w.gpu_index),
            "CUDA_VISIBLE_DEVICES": str(w.gpu_index),
            "TRITON_CACHE_DIR": str(exp_dir / "cache" / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(exp_dir / "cache" / "inductor"),
            "MAX_JOBS": str(w.max_compile_jobs),
        })
        cmd = [w.python or sys.executable, "-m", "kernelforge.worker", "run",
               "--task", str(task_dir / "task.json"), "--result", str(task_dir / "result.json")]
        with open(task_dir / "worker.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=task_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    preexec_fn=_limits if sys.platform != "win32" else None)
            try:
                return proc.wait(timeout=timeout_s), "local_process"
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, AttributeError):
                    proc.kill()
                proc.wait(timeout=30)
                return None, "local_process"

    def _container(self, task_dir: Path, timeout_s: float) -> tuple[int | None, str]:
        w = self.cfg.worker
        name = f"kf-{task_dir.name}"[:60]
        cmd = ["docker", "run", "--rm", "--name", name, "--gpus", f"device={w.gpu_index}",
               "--network", "none", "--read-only", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "512", "--memory", w.container_memory,
               "--user", "1000:1000", "--tmpfs", "/tmp:rw,exec,size=2g",
               "-v", f"{task_dir}:/task:rw",
               "-e", "KERNELFORGE_TRANSPORT=local_container", "-e", "KERNELFORGE_GPU_INDEX=0",
               "-e", "TRITON_CACHE_DIR=/tmp/triton", "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
               "-e", "KERNELFORGE_SCRATCH=/tmp/scratch", "-e", "HOME=/tmp", "-e", f"MAX_JOBS={w.max_compile_jobs}",
               w.container_image, "python", "-m", "kernelforge.worker", "run",
               "--task", "/task/task.json", "--result", "/task/result.json"]
        with open(task_dir / "worker.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                return proc.wait(timeout=timeout_s), "local_container"
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=60)
                proc.wait(timeout=60)
                return None, "local_container"

    def _remote(self, task_dir: Path, timeout_s: float) -> tuple[int | None, str]:
        w = self.cfg.worker
        if not w.remote_host:
            raise RuntimeError("worker.transport=remote requires worker.remote_host; the remote adapter is optional "
                               "and not needed for the local workflow")
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", w.remote_host,
               w.remote_python, "-m", "kernelforge.worker", "run-stdin"]
        task_json = (task_dir / "task.json").read_text(encoding="utf-8")
        try:
            out = subprocess.run(cmd, input=task_json, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None, "remote"
        (task_dir / "worker.log").write_text(out.stderr, encoding="utf-8")
        if out.stdout.strip():
            (task_dir / "result.json").write_text(out.stdout, encoding="utf-8")
        return out.returncode, "remote"

    # -- public API ---------------------------------------------------------
    def run_task(self, task: TaskBundle, timeout_s: float, candidate_id: str | None = None) -> ResultBundle:
        task_dir = self._task_dir(task)
        task_path, result_path = task_dir / "task.json", task_dir / "result.json"
        task_path.write_text(task.model_dump_json(indent=1), encoding="utf-8")
        transport = self.cfg.worker.transport
        runner = {"local_process": self._local_process, "local_container": self._container,
                  "remote": self._remote}[transport]
        with gpu_lock(self.runs_dir):
            started = time.monotonic()
            code, _ = runner(task_dir, timeout_s)
            elapsed = time.monotonic() - started
        log_tail = ""
        if (task_dir / "worker.log").exists():
            log_tail = (task_dir / "worker.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        if code is None:
            result = ResultBundle(task_id=task.task_id, kind=task.kind, status="timeout", failure_stage="timeout",
                                  message=f"worker exceeded {timeout_s:.0f}s wall-clock limit and was killed",
                                  finished_at=utc_now())
        elif result_path.exists() and code in (0, 3):
            result = ResultBundle.model_validate_json(result_path.read_text(encoding="utf-8"))
        else:
            result = ResultBundle(task_id=task.task_id, kind=task.kind, status="error", failure_stage="worker",
                                  message=f"worker exited with code {code} and no valid result.\n{log_tail}",
                                  finished_at=utc_now())
        result.isolation.update({"transport": transport, "wall_seconds": round(elapsed, 2)})
        if result.status == "quarantined":
            self.last_quarantine = result.message
        if self.store is not None:
            if not result_path.exists():
                result_path.write_text(result.model_dump_json(indent=1), encoding="utf-8")
            self.store.record_task(task.task_id, task.experiment_id, candidate_id, task.kind, result.status,
                                   task_path, result_path)
        return result

    def make_task(self, experiment_id: str, kind: str, **kw: Any) -> TaskBundle:
        return TaskBundle(task_id=f"{kind}-{uuid.uuid4().hex[:10]}", experiment_id=experiment_id, kind=kind,
                          max_tensor_bytes=self.cfg.worker.max_tensor_bytes, **kw)

    def probe(self, experiment_id: str = "_system") -> dict[str, Any]:
        res = self.run_task(self.make_task(experiment_id, "probe"), timeout_s=180)
        return {"status": res.status, "message": res.message, "hardware": res.hardware,
                "software": res.software, "probe": res.probe, "isolation": res.isolation}

    def smoke(self, experiment_id: str = "_system") -> ResultBundle:
        return self.run_task(self.make_task(experiment_id, "smoke"), timeout_s=600)

    def ensure_healthy(self) -> None:
        """After a CUDA fault, require a clean probe before trusting further measurements."""
        if self.last_quarantine is None:
            return
        probe = self.probe()
        if probe["status"] != "ok":
            raise RuntimeError(f"worker remains unhealthy after CUDA fault: {probe['message']}")
        self.last_quarantine = None


# ---------------------------------------------------------------------------
# Controller-side orchestration of one candidate evaluation
# ---------------------------------------------------------------------------

class EvaluationSummary(dict):
    """correct_all, n_passed, n_failed, failures, speedups{workload: x}, geomean_speedup, records."""


class CandidateEvaluator:
    def __init__(self, cfg: KernelForgeConfig, store: Store, executor: Executor):
        self.cfg, self.store, self.executor = cfg, store, executor

    def _timeouts(self, n: int, factor: float = 1.0) -> float:
        s = self.cfg.search
        return s.max_compile_seconds + factor * s.max_execution_seconds * max(n, 1) + 90

    def correctness(self, experiment_id: str, spec: CandidateSpec, cases: list, split: str,
                    candidate_id: str | None = None) -> ResultBundle:
        task = self.executor.make_task(experiment_id, "correctness", candidate=spec, workloads=cases, split=split)
        return self.executor.run_task(task, self._timeouts(len(cases)), candidate_id)

    def timing(self, experiment_id: str, spec: CandidateSpec, cases: list, split: str, baselines: list[str],
               session: int, candidate_id: str | None = None, generator_resident: bool = False,
               measurement_kind: str = "clean") -> ResultBundle:
        settings = self.cfg.benchmark_settings(seed=self.cfg.search.seed * 1000 + session)
        task = self.executor.make_task(experiment_id, "timing", candidate=spec, workloads=cases, split=split,
                                       baselines=baselines, benchmark=settings, session_index=session,
                                       generator_resident=generator_resident, measurement_kind=measurement_kind)
        n_impl = len(baselines) + 1
        return self.executor.run_task(task, self._timeouts(len(cases), factor=2.0 * n_impl), candidate_id)

    def memory(self, experiment_id: str, spec: CandidateSpec, impl_id: str, cases: list, split: str,
               candidate_id: str | None = None) -> ResultBundle:
        task = self.executor.make_task(experiment_id, "memory", candidate=spec, implementation=impl_id,
                                       workloads=cases, split=split)
        return self.executor.run_task(task, self._timeouts(len(cases)), candidate_id)

    def evaluate(self, experiment_id: str, candidate_id: str, spec: CandidateSpec, correctness_manifest: Manifest,
                 timing_cases: list | None, baselines: list[str], sessions: int = 1, memory: bool = False,
                 measurement_kind: str = "clean", generator_resident: bool = False) -> EvaluationSummary:
        """Correctness on the manifest (+ timing cases), then timing/memory on correct timing cases only."""
        split = correctness_manifest.split
        summary = EvaluationSummary(correct_all=False, n_passed=0, n_failed=0, failures=[], speedups={},
                                    geomean_speedup=None, records=[], stage=None, quarantined=False)
        self.executor.ensure_healthy()
        timing_cases = list(timing_cases if timing_cases is not None else correctness_manifest.cases)
        cases = {c.workload_id: c for c in correctness_manifest.cases}
        for c in timing_cases:
            cases.setdefault(c.workload_id, c)
        corr = self.correctness(experiment_id, spec, list(cases.values()), split, candidate_id)
        summary["correctness_status"] = corr.status
        by_case = {c.workload_id: c for c in corr.correctness}
        for wid in cases:
            outcome = by_case.get(wid)
            if outcome is None or outcome.status in ("failed", "error", "not_run"):
                summary["n_failed"] += 1
                msg = outcome.message if outcome else (corr.message or "no correctness result")
                stage = (outcome.failure_stage if outcome else None) or corr.failure_stage or "worker"
                summary["failures"].append({"workload_id": wid, "stage": stage, "message": (msg or "")[:1500],
                                            "max_abs_error": outcome.max_abs_error if outcome else None})
            elif outcome.status == "passed":
                summary["n_passed"] += 1
        if corr.status == "timeout":
            summary["stage"] = "timeout"
        elif corr.status == "error" and not corr.correctness:
            summary["stage"] = "worker"  # infrastructure failure, not the candidate's fault (retryable)
        elif corr.status == "quarantined":
            summary["stage"], summary["quarantined"] = "execution", True
        elif summary["failures"]:
            stages = {f["stage"] for f in summary["failures"]}
            summary["stage"] = "validation" if "validation" in stages else (
                "compile" if "compile" in stages else ("execution" if "execution" in stages else "correctness"))
        summary["correct_all"] = summary["n_failed"] == 0 and summary["n_passed"] > 0

        primary = baselines[0] if baselines else "none"
        timed_ids = {c.workload_id for c in timing_cases}
        passed_cases = [c for c in timing_cases if by_case.get(c.workload_id) and by_case[c.workload_id].status == "passed"]
        run_timing = summary["correct_all"] and bool(passed_cases) and bool(baselines)
        # Every workload gets an evidence record; unmeasured timing fields stay null, never zero.
        for wid, case in cases.items():
            if run_timing and wid in timed_ids and by_case[wid].status == "passed":
                continue
            summary["records"].append(self._record(experiment_id, candidate_id, spec, case, split_of(case, split),
                                                   corr, by_case.get(wid), primary, None, None, measurement_kind))
        if not run_timing:
            self._persist(summary, candidate_id)
            return summary

        timing_results = [self.timing(experiment_id, spec, passed_cases, split, baselines, s, candidate_id,
                                      generator_resident, measurement_kind) for s in range(sessions)]
        mem_results: dict[str, ResultBundle] = {}
        if memory:  # one fresh worker process per implementation
            for impl in ["candidate"] + baselines:
                mem_results[impl] = self.memory(experiment_id, spec, impl, passed_cases, split, candidate_id)

        for case in passed_cases:
            corr_outcome = by_case.get(case.workload_id)
            for bid in baselines:
                for scope in ("operator", "kernel", "graph"):
                    rec = self._timed_record(experiment_id, candidate_id, spec, case, split_of(case, split), corr,
                                             corr_outcome,
                                             bid, scope, timing_results, mem_results, measurement_kind)
                    if rec is not None:
                        summary["records"].append(rec)
                        if (bid == self.cfg.search.primary_baseline and scope == self.cfg.search.score_scope
                                and rec.speedup is not None):
                            summary["speedups"][case.workload_id] = rec.speedup
        timing_failures = [t for r in timing_results for t in r.timing if t.status not in ("passed", "skipped")]
        for t in timing_failures:
            summary["failures"].append({"workload_id": t.workload_id, "stage": t.failure_stage, "message": t.message})
        if any(r.status in ("timeout", "quarantined", "error") for r in timing_results) or timing_failures:
            summary["benchmark_failed"] = True
        summary["geomean_speedup"] = geomean(list(summary["speedups"].values()))
        self._persist(summary, candidate_id)
        return summary

    def _persist(self, summary: EvaluationSummary, candidate_id: str) -> None:
        paths = []
        for rec in summary["records"]:
            paths.append(self.store.rel(self.store.add_evaluation(rec, candidate_id)))
        summary["record_paths"] = paths

    def _base_record(self, experiment_id: str, spec: CandidateSpec, case, split: str, bundle: ResultBundle,
                     baseline_id: str, measurement_kind: str) -> dict[str, Any]:
        return dict(experiment_id=experiment_id, workload_id=case.workload_id, candidate_hash=spec.source_hash,
                    operator=case.operator, evaluation_split=split, hardware=bundle.hardware,
                    software_versions={**bundle.software, "isolation": bundle.isolation}, dtype=case.dtype,
                    shapes=case.shapes(), contract=getattr(case, "contract", "forward_inference"),
                    baseline_id=baseline_id, measurement_kind=measurement_kind, workload=case.model_dump())

    def _record(self, experiment_id, candidate_id, spec, case, split, bundle, outcome, baseline_id,
                timing, memory, measurement_kind) -> EvaluationRecord:
        base = self._base_record(experiment_id, spec, case, split, bundle, baseline_id, measurement_kind)
        if outcome is None:
            status, stage, msg = ("error" if bundle.status != "ok" else "not_run",
                                  bundle.failure_stage or "worker", bundle.message)
            return EvaluationRecord(**base, status=status, failure_stage=stage, failure_message=(msg or "")[:2000])
        status = outcome.status if outcome.status != "passed" else "passed"
        return EvaluationRecord(**base, status=status, failure_stage=outcome.failure_stage,
                                failure_message=(outcome.message or None),
                                correctness_pass=outcome.passed, max_abs_error=outcome.max_abs_error,
                                max_rel_error=outcome.max_rel_error, tolerance=outcome.tolerance)

    def _timed_record(self, experiment_id, candidate_id, spec, case, split, corr, corr_outcome, baseline_id, scope,
                      timing_results: list[ResultBundle], mem_results: dict[str, ResultBundle],
                      measurement_kind: str) -> EvaluationRecord | None:
        from .benchmark import dedupe_flags, paired_stats, summarize

        cand_stats: list[TimingStats] = []
        base_stats: list[TimingStats] = []
        flags: set[str] = set()
        launch: dict = {}
        failure = None
        bundle = timing_results[0]
        for res in timing_results:
            outcome = next((t for t in res.timing if t.workload_id == case.workload_id), None)
            if outcome is None or outcome.status != "passed":
                failure = outcome or failure
                continue
            c = next((s for s in outcome.stats if s.impl_id == "candidate" and s.scope == scope), None)
            b = next((s for s in outcome.stats if s.impl_id == baseline_id and s.scope == scope), None)
            if c and b:
                cand_stats.append(c)
                base_stats.append(b)
                flags |= set(outcome.contamination_flags)
            launch = outcome.telemetry.get("launch_config", launch)
            bundle = res
        if not cand_stats:
            if scope != "operator":
                return None  # this baseline has no kernel-only / graph measurement
            rec = self._record(experiment_id, candidate_id, spec, case, split, corr, corr_outcome, baseline_id,
                               None, None, measurement_kind)
            data = rec.model_dump()
            data.update(status="error", failure_stage=(failure.failure_stage if failure else "benchmark"),
                        failure_message=(failure.message if failure else "no timing result"))
            return EvaluationRecord(**data)
        cand_pool = summarize("candidate", scope, cand_stats[0].calls_per_sample,
                              [x for s in cand_stats for x in s.samples_ms],
                              [x for s in cand_stats for x in s.host_samples_ms])
        base_pool = summarize(baseline_id, scope, base_stats[0].calls_per_sample,
                              [x for s in base_stats for x in s.samples_ms],
                              [x for s in base_stats for x in s.host_samples_ms])
        paired = paired_stats(base_pool, cand_pool, seed=self.cfg.search.seed)
        base = self._base_record(experiment_id, spec, case, split, bundle, baseline_id, measurement_kind)
        rec = dict(base, status="passed", correctness_pass=corr_outcome.passed if corr_outcome else None,
                   max_abs_error=corr_outcome.max_abs_error if corr_outcome else None,
                   max_rel_error=corr_outcome.max_rel_error if corr_outcome else None,
                   tolerance=corr_outcome.tolerance if corr_outcome else None,
                   launch_config={k: v for k, v in launch.items() if isinstance(v, int)},
                   raw_timing_samples={
                       "baseline": base_pool.samples_ms, "candidate": cand_pool.samples_ms,
                       "baseline_host": base_pool.host_samples_ms, "candidate_host": cand_pool.host_samples_ms,
                       "session_speedups": [b.median_ms / c.median_ms for b, c in zip(base_stats, cand_stats)],
                   },
                   timing_scope=scope, calls_per_sample=cand_pool.calls_per_sample,
                   baseline_median_ms=base_pool.median_ms, candidate_median_ms=cand_pool.median_ms,
                   baseline_p10_p90_ms=[base_pool.p10_ms, base_pool.p90_ms],
                   candidate_p10_p90_ms=[cand_pool.p10_ms, cand_pool.p90_ms],
                   speedup=paired.speedup_of_medians, speedup_ci95=[paired.ci95_low, paired.ci95_high],
                   sessions=len(cand_stats), contamination_flags=dedupe_flags(flags))
        if scope == "operator" and "candidate" in mem_results and baseline_id in mem_results:
            cm = next((m for m in mem_results["candidate"].memory if m.workload_id == case.workload_id), None)
            bm = next((m for m in mem_results[baseline_id].memory if m.workload_id == case.workload_id), None)
            if cm and bm and cm.status == "passed" and bm.status == "passed":
                rec.update(memory_metric=MEMORY_METRIC, baseline_peak_bytes=bm.incremental_peak_bytes,
                           candidate_peak_bytes=cm.incremental_peak_bytes,
                           memory_reduction_percent=memory_reduction_percent(bm.incremental_peak_bytes,
                                                                             cm.incremental_peak_bytes),
                           memory_detail={"definition": MEMORY_METRIC_DEFINITION,
                                          "baseline": bm.model_dump(), "candidate": cm.model_dump(),
                                          "total_working_reduction_percent": memory_reduction_percent(
                                              bm.peak_allocated_bytes, cm.peak_allocated_bytes)})
        return EvaluationRecord(**rec)


def split_of(case, default: str) -> str:
    """Workload ids encode their manifest split: '<ln|mr>-<split>-<idx>-...'."""
    parts = case.workload_id.split("-")
    return parts[1] if len(parts) > 2 and parts[1] in ("smoke", "search", "dev", "heldout") else default
