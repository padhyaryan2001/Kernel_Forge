"""Configuration models and the hardware/software preflight (`kernelforge doctor`).

The controller never initialises a CUDA context: GPU facts come from `nvidia-smi` and
from a probe task executed by the worker process.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts import BenchmarkSettings, speedup_to_latency_reduction

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WorkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transport: Literal["local_process", "local_container", "remote"] = "local_process"
    gpu_index: int = 0
    concurrent_jobs: int = 1
    max_tensor_bytes: int = 1 << 30
    # Interpreter of the evaluator environment (local_process). Defaults to the current one.
    python: str | None = None
    container_image: str = "kernelforge-worker:latest"
    container_memory: str = "6g"
    remote_host: str | None = None
    remote_python: str = "python3"
    max_compile_jobs: int = 2

    @field_validator("concurrent_jobs")
    @classmethod
    def _one_job(cls, v: int) -> int:
        if v != 1:
            raise ValueError("exactly one GPU evaluation job may run at a time on this machine")
        return v


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["openai_compatible", "fixture"] = "openai_compatible"
    runtime: Literal["llama_cpp", "external"] = "llama_cpp"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local-coder"
    source_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
    source_file: str = "qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    source_revision: str | None = None
    source_sha256: str | None = None
    quantization: str = "Q4_K_M"
    context_tokens: int = 4096
    max_output_tokens: int = 1536
    max_concurrent_requests: int = 1
    managed_process: bool = True
    stop_before_gpu_evaluation: bool = True
    temperature: float = 0.7
    server_binary: str | None = None
    # Extra shared-library directories for the server (e.g. the CUDA runtime shipped with llama.cpp).
    server_lib_dirs: list[str] = []
    model_path: str | None = None
    n_gpu_layers: int = 99
    server_extra_args: list[str] = []
    startup_timeout_s: float = 180.0
    shutdown_timeout_s: float = 30.0
    request_timeout_s: float = 600.0
    # Free VRAM must recover to within this many MiB of the pre-start value after shutdown.
    release_tolerance_mib: int = 256

    @field_validator("base_url")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        return v.rstrip("/")


class CriticsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    openai_model: str | None = None
    anthropic_model: str | None = None
    google_model: str | None = None
    self_critique: bool = True
    max_output_tokens: int = 1024
    # Optional documented pricing (USD per 1M tokens) used only when actual usage exists.
    pricing: dict[str, dict[str, Any]] = {}


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidates_per_round: int = Field(default=2, ge=1, le=16)
    max_rounds: int = Field(default=4, ge=1, le=50)
    max_compile_seconds: float = 120.0
    max_execution_seconds: float = 30.0
    max_total_generator_tokens: int = 200_000
    max_experiment_minutes: float = 180.0
    mode: Literal["full", "launch_only"] = "full"
    correctness_suite: Literal["smoke", "dev"] = "dev"
    benchmark_suite: Literal["smoke", "search", "dev"] = "search"
    primary_baseline: str = "torch_eager"
    # Timing scope used for the development score: graph (CUDA-graph replay, device time) or operator
    # (eager calls incl. host dispatch). Declared before the agent runs; every scope is still recorded.
    # Default graph: in the pilot (optimize-layernorm-20261101-064550-8fc8) an identical kernel timed
    # against itself read 0.93x in operator scope but 1.00x in kernel/graph scope (host-dispatch noise).
    score_scope: Literal["operator", "graph"] = "graph"
    seed: int = 0


class TargetsConfig(BaseModel):
    """Aspirational experiment targets. Never results; never pre-populated as achieved."""

    model_config = ConfigDict(extra="forbid")
    layernorm_speedup: float = 1.22
    mrope_memory_reduction_fraction: float = 0.60
    generated_kernel_speedup: float = 2.20

    def describe(self) -> list[dict[str, str]]:
        return [
            {
                "target": "LayerNorm forward latency",
                "value": f"{self.layernorm_speedup:.2f}x speedup vs F.layer_norm",
                "definition": (f"{self.layernorm_speedup:.2f}x == "
                               f"{speedup_to_latency_reduction(self.layernorm_speedup):.1f}% lower latency "
                               "(a 22% latency reduction would be a different, larger number)"),
            },
            {
                "target": "MRoPE operator memory",
                "value": f"{self.mrope_memory_reduction_fraction:.0%} lower incremental peak",
                "definition": "Operator-level incremental peak memory (outputs + temporaries) vs the eager "
                              "PyTorch baseline; not whole-model memory.",
            },
            {
                "target": "Generated kernel speedup",
                "value": f"{self.generated_kernel_speedup:.2f}x",
                "definition": "Kernel/operator-level speedup of an agent-generated kernel vs the primary "
                              "baseline on a named workload group; not full-model throughput.",
            },
        ]


class KernelForgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["reference", "cuda", "remote"] = "cuda"
    runs_dir: str = "runs"
    worker: WorkerConfig = WorkerConfig()
    generator: GeneratorConfig = GeneratorConfig()
    critics: CriticsConfig = CriticsConfig()
    search: SearchConfig = SearchConfig()
    benchmark: dict[str, Any] = {}
    targets: TargetsConfig = TargetsConfig()

    def benchmark_settings(self, **overrides: Any) -> BenchmarkSettings:
        data = {k: v for k, v in self.benchmark.items() if k != "independent_sessions"}
        data.update(overrides)
        return BenchmarkSettings(**data)

    @property
    def independent_sessions(self) -> int:
        return int(self.benchmark.get("independent_sessions", 3))

    def runs_path(self) -> Path:
        p = Path(self.runs_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | os.PathLike | None) -> KernelForgeConfig:
    if path is None:
        default = PROJECT_ROOT / "config.yaml"
        if not default.exists():
            return KernelForgeConfig()
        path = default
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return KernelForgeConfig(**data)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
}


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def os_release() -> str | None:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        return None
    return None


def _vmstat_swaps() -> tuple[int, int] | None:
    try:
        vals = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
        return int(vals.get("pswpin", 0)), int(vals.get("pswpout", 0))
    except (OSError, ValueError):
        return None


def swap_activity(window_s: float = 1.0) -> dict[str, Any]:
    """Pages swapped in/out during a short window (Linux only)."""
    a = _vmstat_swaps()
    if a is None:
        return {"available": False}
    time.sleep(window_s)
    b = _vmstat_swaps()
    pin, pout = b[0] - a[0], b[1] - a[1]
    return {"available": True, "window_s": window_s, "pages_in": pin, "pages_out": pout,
            "active": (pin + pout) > 0}


def nvidia_smi_query(gpu_index: int = 0) -> dict[str, Any]:
    fields = ["name", "uuid", "driver_version", "memory.total", "memory.used", "memory.free",
              "compute_cap", "utilization.gpu", "clocks.sm", "clocks.max.sm", "temperature.gpu",
              "display_active"]
    exe = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else None)
    if exe is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        out = subprocess.run([exe, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
                              "-i", str(gpu_index)], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "reason": str(exc)}
    if out.returncode != 0:
        return {"available": False, "reason": out.stderr.strip()[:300]}
    values = [v.strip() for v in out.stdout.strip().split(",")]
    info: dict[str, Any] = {"available": True}
    for key, val in zip(fields, values):
        info[key] = val
    try:
        cuda_line = subprocess.run([exe], capture_output=True, text=True, timeout=20).stdout
        for token in ("CUDA Version:", "CUDA UMD Version:"):
            if token in cuda_line:
                info["driver_cuda_version"] = cuda_line.split(token, 1)[1].split()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def gpu_free_mib(gpu_index: int = 0) -> int | None:
    info = nvidia_smi_query(gpu_index)
    try:
        return int(float(info["memory.free"]))
    except (KeyError, ValueError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def host_report(cfg: KernelForgeConfig) -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    runs = cfg.runs_path()
    disk_target = runs if runs.exists() else PROJECT_ROOT
    du = shutil.disk_usage(disk_target)
    report: dict[str, Any] = {
        "platform": "wsl2" if is_wsl() else platform.system().lower(),
        "distribution": os_release(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "ram_total_gib": round(vm.total / 2**30, 2),
        "ram_available_gib": round(vm.available / 2**30, 2),
        "swap_total_gib": round(sw.total / 2**30, 2),
        "swap_used_gib": round(sw.used / 2**30, 2),
        "swap_activity": swap_activity(1.0),
        "disk_free_gib": round(du.free / 2**30, 1),
        "cpu_count": os.cpu_count(),
        "packages": {p: package_version(p) for p in ("torch", "triton", "pydantic", "pyyaml", "psutil")},
    }
    if report["platform"] == "wsl2":
        report["wsl_note"] = ("WSL2 exposes only the memory configured for the Linux VM; visible RAM can be "
                              "lower than the physical 16 GB. Tune %UserProfile%\\.wslconfig if needed.")
    return report


def generator_report(cfg: KernelForgeConfig, hash_model: bool = False) -> dict[str, Any]:
    g = cfg.generator
    report: dict[str, Any] = {
        "backend": g.backend,
        "runtime": g.runtime,
        "serving_alias": g.model,
        "source_model": g.source_model,
        "source_file": g.source_file,
        "source_revision": g.source_revision,
        "declared_sha256": g.source_sha256,
        "quantization": g.quantization,
        "context_tokens": g.context_tokens,
        "max_output_tokens": g.max_output_tokens,
        "base_url": g.base_url,
        "loopback_only": any(h in g.base_url for h in ("127.0.0.1", "localhost", "[::1]")),
    }
    if g.server_binary:
        binary = Path(os.path.expanduser(g.server_binary))
        report["server_binary_exists"] = binary.exists()
        if binary.exists():
            try:
                out = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=30,
                                     env=server_env(g))
                report["server_version"] = (out.stdout + out.stderr).strip().splitlines()[-2:]
            except (OSError, subprocess.TimeoutExpired) as exc:
                report["server_version_error"] = str(exc)
    if g.model_path:
        path = Path(os.path.expanduser(g.model_path))
        report["model_file_exists"] = path.exists()
        if path.exists():
            report["model_file_bytes"] = path.stat().st_size
            if hash_model:
                from .store import sha256_file

                digest = sha256_file(path)
                report["model_file_sha256"] = digest
                if g.source_sha256:
                    report["sha256_matches_declared"] = digest == g.source_sha256
    return report


def server_env(g: GeneratorConfig) -> dict[str, str]:
    """Environment for llama-server: its own directory plus configured library dirs on LD_LIBRARY_PATH."""
    env = dict(os.environ)
    dirs = [str(Path(os.path.expanduser(g.server_binary or "")).parent)]
    dirs += [os.path.expanduser(d) for d in g.server_lib_dirs]
    if env.get("LD_LIBRARY_PATH"):
        dirs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(dirs)
    return env


def doctor(cfg: KernelForgeConfig, probe_gpu: bool = True, hash_model: bool = False) -> dict[str, Any]:
    """Full preflight. The GPU probe runs in a separate worker process."""
    report: dict[str, Any] = {
        "mode": cfg.mode,
        "host": host_report(cfg),
        "nvidia_smi": nvidia_smi_query(cfg.worker.gpu_index),
        "api_keys": {name: ("set" if os.environ.get(env) else "not set") for name, env in API_KEY_ENV.items()},
        "generator": generator_report(cfg, hash_model=hash_model),
        "worker": {"transport": cfg.worker.transport, "gpu_index": cfg.worker.gpu_index,
                   "python": cfg.worker.python or sys.executable,
                   "max_tensor_bytes": cfg.worker.max_tensor_bytes},
    }
    if probe_gpu and cfg.mode == "cuda":
        from .executor import Executor

        report["evaluator_probe"] = Executor(cfg).probe()
    elif cfg.mode != "cuda":
        report["evaluator_probe"] = {"status": "not_run", "reason": f"mode={cfg.mode}"}
    return report
