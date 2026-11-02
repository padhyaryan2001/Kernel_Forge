"""Trusted timing and memory measurement (worker side).

Timing: warmup (compilation/autotuning excluded), symmetric batch-length calibration,
then `samples` batches per implementation in a randomized interleaved order. Each batch is
`calls_per_sample` back-to-back calls bracketed by CUDA events (device time) and
perf_counter after a synchronize (host wall time). L2 is not flushed between calls; both
sides see the same warm-cache conditions.

Memory: one implementation per fresh process; peak allocated bytes above the resident
inputs during a single call (outputs + temporaries), see contracts.MEMORY_METRIC_DEFINITION.
"""
from __future__ import annotations

import gc
import random
import statistics
import time
from typing import Any, Callable

from .config import nvidia_smi_query, _vmstat_swaps
from .contracts import BenchmarkSettings, MemoryOutcome, PairedStats, TimingStats


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def summarize(impl_id: str, scope: str, calls: int, dev: list[float], host: list[float]) -> TimingStats:
    s = sorted(dev)
    mean = statistics.fmean(dev)
    cv = statistics.pstdev(dev) / mean if mean > 0 else float("nan")
    return TimingStats(impl_id=impl_id, scope=scope, calls_per_sample=calls, samples_ms=dev, host_samples_ms=host,
                       median_ms=statistics.median(dev), p10_ms=_percentile(s, 0.10), p90_ms=_percentile(s, 0.90),
                       mean_ms=mean, cv=cv, host_median_ms=statistics.median(host))


def paired_stats(baseline: TimingStats, candidate: TimingStats, seed: int = 0, resamples: int = 2000) -> PairedStats:
    """Speedup of medians plus a bootstrap 95% CI of the median per-sample ratio.

    Samples with the same index were taken adjacently in time (randomized order), so their
    ratio cancels slow drift.
    """
    ratios = [b / c for b, c in zip(baseline.samples_ms, candidate.samples_ms) if c > 0]
    rng = random.Random(seed)
    meds = []
    for _ in range(resamples):
        draw = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        meds.append(statistics.median(draw))
    meds.sort()
    return PairedStats(baseline_id=baseline.impl_id, scope=candidate.scope,
                       speedup_of_medians=baseline.median_ms / candidate.median_ms,
                       median_paired_ratio=statistics.median(ratios),
                       ci95_low=_percentile(meds, 0.025), ci95_high=_percentile(meds, 0.975),
                       n_pairs=len(ratios))


def _gpu_telemetry(gpu_index: int) -> dict[str, Any]:
    info = nvidia_smi_query(gpu_index)
    out: dict[str, Any] = {"available": info.get("available", False)}
    for key in ("utilization.gpu", "clocks.sm", "clocks.max.sm", "temperature.gpu", "memory.used", "display_active"):
        if key in info:
            out[key] = info[key]
    return out


def _num(d: dict, key: str) -> float | None:
    try:
        return float(d[key])
    except (KeyError, TypeError, ValueError):
        return None


def dedupe_flags(flags) -> list[str]:
    """One entry per flag name (the text before '('), keeping the first occurrence."""
    seen: dict[str, str] = {}
    for f in flags:
        seen.setdefault(f.split("(")[0], f)
    return sorted(seen.values())


def capture_graph(fn: Callable[[], Any], inner_calls: int):
    """Capture `inner_calls` back-to-back calls in one CUDA graph (outputs come from the graph pool)."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner_calls):
            fn()
    torch.cuda.synchronize()
    return graph


def contamination_flags(before: dict, after: dict, swaps: tuple | None, stats: list[TimingStats],
                        generator_resident: bool) -> list[str]:
    flags: list[str] = []
    if not before.get("available"):
        flags.append("gpu_telemetry_unavailable")
    util = _num(before, "utilization.gpu")
    if util is not None and util > 15:
        flags.append(f"concurrent_gpu_activity(util={util:.0f}%)")
    c0, c1 = _num(before, "clocks.sm"), _num(after, "clocks.sm")
    if c0 and c1 and abs(c1 - c0) / max(c0, c1) > 0.10:
        flags.append(f"sm_clock_drift({c0:.0f}->{c1:.0f}MHz)")
    t1 = _num(after, "temperature.gpu")
    if t1 is not None and t1 >= 85:
        flags.append(f"thermal({t1:.0f}C)")
    if str(before.get("display_active", "")).lower() == "enabled":
        flags.append("gpu_drives_display")
    if swaps is not None and sum(swaps) > 0:
        flags.append(f"host_swapping(pages={sum(swaps)})")
    unstable = [s for s in stats if s.cv > 0.15]
    if unstable:
        worst = max(unstable, key=lambda s: s.cv)
        flags.append(f"unstable_timing({worst.impl_id}/{worst.scope}: cv={worst.cv:.2f})")
    if generator_resident:
        flags.append("generator_or_trainer_resident")
    return dedupe_flags(flags)


def time_implementations(fns: dict[str, Callable[[], Any]], scope: str, settings: BenchmarkSettings,
                         gpu_index: int = 0, generator_resident: bool = False,
                         inner_calls: int = 1) -> tuple[list[TimingStats], list[str], dict]:
    """`inner_calls` > 1 when each fn() already performs that many operator calls (CUDA-graph replay)."""
    import torch

    names = list(fns)
    torch.cuda.synchronize()
    for name in names:
        for _ in range(settings.warmup_iterations):
            fns[name]()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def batch(fn: Callable[[], Any], calls: int) -> tuple[float, float]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        host = (time.perf_counter() - t0) * 1e3
        return start.elapsed_time(end), host

    # Symmetric calibration: the fastest implementation must fill min_batch_ms.
    calls = settings.calls_per_sample
    while calls < settings.max_calls_per_sample:
        fastest = min(batch(fns[n], calls)[0] for n in names)
        if fastest >= settings.min_batch_ms:
            break
        calls = min(calls * 2, settings.max_calls_per_sample)

    rng = random.Random(settings.seed)
    dev: dict[str, list[float]] = {n: [] for n in names}
    host: dict[str, list[float]] = {n: [] for n in names}
    before = _gpu_telemetry(gpu_index)
    swaps0 = _vmstat_swaps()
    per_sample = calls * inner_calls
    for _ in range(settings.samples):
        order = names[:]
        rng.shuffle(order)
        for n in order:
            d, h = batch(fns[n], calls)
            dev[n].append(d / per_sample)
            host[n].append(h / per_sample)
    swaps1 = _vmstat_swaps()
    after = _gpu_telemetry(gpu_index)
    swaps = (swaps1[0] - swaps0[0], swaps1[1] - swaps0[1]) if swaps0 and swaps1 else None
    stats = [summarize(n, scope, per_sample, dev[n], host[n]) for n in names]
    flags = contamination_flags(before, after, swaps, stats, generator_resident)
    telemetry = {"before": before, "after": after, "l2_flush": False, "order": "randomized_interleaved",
                 "inner_calls": inner_calls}
    return stats, flags, telemetry


def storage_bytes(tensors: dict[str, Any]) -> int:
    seen, total = set(), 0
    for t in tensors.values():
        if t is None:
            continue
        st = t.untyped_storage()
        if st.data_ptr() in seen:
            continue
        seen.add(st.data_ptr())
        total += st.nbytes()
    return total


def measure_memory(run: Callable[[], dict], inputs: dict, workload_id: str, impl_id: str) -> MemoryOutcome:
    """Peak bytes above the resident inputs during one call (outputs + temporaries)."""
    import torch

    out = run()  # warm call: compilation / lazy init excluded
    torch.cuda.synchronize()
    del out
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = run()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    out_bytes = storage_bytes(out)
    del out
    return MemoryOutcome(workload_id=workload_id, impl_id=impl_id, status="passed",
                         input_bytes=storage_bytes(inputs), output_bytes=out_bytes,
                         allocated_before_bytes=before, peak_allocated_bytes=peak,
                         incremental_peak_bytes=peak - before, reserved_peak_bytes=reserved)
