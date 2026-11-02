"""Versioned data contracts shared by the controller and the GPU worker.

Nothing in this module imports torch or triton: the controller validates task and
result bundles on a CPU-only machine, and the worker re-validates them on its side.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

Operator = Literal["layernorm", "mrope"]
DType = Literal["float32", "float16", "bfloat16"]
Split = Literal["smoke", "search", "dev", "heldout"]
MeasurementKind = Literal["clean", "in_training", "fixture"]
FailureStage = Literal[
    "validation", "compile", "execution", "correctness", "benchmark", "memory",
    "timeout", "worker", "budget", "unsupported",
]
RecordStatus = Literal["passed", "failed", "error", "skipped", "not_run"]

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}
OPERATORS: tuple[str, ...] = ("layernorm", "mrope")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Correctness tolerances. Declared before any search and recorded with every
# evaluation. They must not be loosened to accept a candidate.
# ---------------------------------------------------------------------------

class Tolerance(BaseModel):
    model_config = ConfigDict(frozen=True)
    atol: float
    rtol: float


# An element passes when |out - ref| <= atol + rtol * |ref| (same rule as torch.allclose),
# with `ref` the float64 CPU reference evaluated on the exact (dtype-rounded) inputs.
TOLERANCES: dict[str, dict[str, Tolerance]] = {
    # FP32 reductions over up to 8192 elements, amplified by rstd <= 1/sqrt(eps).
    "layernorm": {
        "float32": Tolerance(atol=1e-4, rtol=1e-4),
        "float16": Tolerance(atol=2e-3, rtol=2e-3),
        "bfloat16": Tolerance(atol=1e-2, rtol=1e-2),
    },
    # theta is the FP32 product fp32(position) * inv_freq[i]; trig is accurate on that value.
    "mrope": {
        "float32": Tolerance(atol=1e-5, rtol=1e-5),
        "float16": Tolerance(atol=2e-3, rtol=2e-3),
        "bfloat16": Tolerance(atol=1e-2, rtol=1e-2),
    },
}
# Relative error is reported as max(|out - ref| / max(|ref|, REL_ERROR_FLOOR)).
REL_ERROR_FLOOR = 1e-3


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

class LayerNormCase(BaseModel):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta over the last dimension.

    Population variance, FP32 accumulation, output in the input dtype. Forward only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    operator: Literal["layernorm"] = "layernorm"
    workload_id: str
    M: int = Field(ge=1, le=1 << 20)
    N: int = Field(ge=1, le=1 << 16)
    dtype: DType
    eps: float = Field(default=1e-5, gt=0)
    seed: int
    input_kind: Literal["normal", "constant_rows", "low_variance", "negative", "large_offset"] = "normal"

    @property
    def selector(self) -> int:
        return self.N

    def shapes(self) -> dict[str, list[int]]:
        return {"x": [self.M, self.N], "gamma": [self.N], "beta": [self.N], "y": [self.M, self.N]}

    def input_bytes(self) -> int:
        e = DTYPE_BYTES[self.dtype]
        return (self.M * self.N + 2 * self.N) * e

    def output_bytes(self) -> int:
        return self.M * self.N * DTYPE_BYTES[self.dtype]

    def estimated_live_bytes(self) -> int:
        # inputs + a preservation clone + two outputs + FP32 temporaries of an eager baseline
        return 2 * self.input_bytes() + 2 * self.output_bytes() + 4 * self.M * self.N * 4

    def describe(self) -> str:
        return f"M={self.M} N={self.N} {self.dtype} eps={self.eps:g} {self.input_kind}"


class MRoPECase(BaseModel):
    """Qwen2-VL-style multimodal rotary embedding (temporal / height / width sections).

    q: [B, Hq, S, D], k: [B, Hkv, S, D], position_ids: [3, B, S] int64, inv_freq: [D/2] fp32.
    Pair i in [0, D/2) uses axis 0 for i < nt, axis 1 for i < nt + nh, axis 2 otherwise.
    theta = fp32(position_ids[axis, b, s]) * inv_freq[i] (an FP32 product), split-half rotation:
      out[i]       = x[i] * cos(theta) - x[i + D/2] * sin(theta)
      out[i + D/2] = x[i] * sin(theta) + x[i + D/2] * cos(theta)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    operator: Literal["mrope"] = "mrope"
    workload_id: str
    B: int = Field(ge=1, le=64)
    Hq: int = Field(ge=1, le=256)
    Hkv: int = Field(ge=1, le=256)
    S: int = Field(ge=1, le=1 << 17)
    D: int = Field(ge=2, le=512)
    sections: tuple[int, int, int]
    dtype: DType
    seed: int
    position_kind: Literal["text", "image", "video"] = "text"
    position_offset: int = Field(default=0, ge=0, le=(1 << 24) - (1 << 18))
    rope_theta: float = Field(default=1_000_000.0, gt=1.0)
    contract: Literal["positions_to_output", "precomputed_trig"] = "positions_to_output"

    @model_validator(mode="after")
    def _check(self) -> "MRoPECase":
        if self.D % 2:
            raise ValueError("D must be even")
        if sum(self.sections) != self.D // 2 or min(self.sections) < 0:
            raise ValueError(f"sections {self.sections} must be non-negative and sum to D/2={self.D // 2}")
        return self

    @property
    def selector(self) -> int:
        return self.S

    def shapes(self) -> dict[str, list[int]]:
        s = {"q": [self.B, self.Hq, self.S, self.D], "k": [self.B, self.Hkv, self.S, self.D]}
        if self.contract == "precomputed_trig":
            s.update(cos=[self.B, self.S, self.D // 2], sin=[self.B, self.S, self.D // 2])
        else:
            s.update(position_ids=[3, self.B, self.S], inv_freq=[self.D // 2])
        return s

    def input_bytes(self) -> int:
        e = DTYPE_BYTES[self.dtype]
        qk = self.B * (self.Hq + self.Hkv) * self.S * self.D * e
        if self.contract == "precomputed_trig":
            return qk + 2 * self.B * self.S * (self.D // 2) * 4
        return qk + 3 * self.B * self.S * 8 + (self.D // 2) * 4

    def output_bytes(self) -> int:
        return self.B * (self.Hq + self.Hkv) * self.S * self.D * DTYPE_BYTES[self.dtype]

    def estimated_live_bytes(self) -> int:
        fp32_qk = self.B * (self.Hq + self.Hkv) * self.S * self.D * 4
        return 2 * self.input_bytes() + 2 * self.output_bytes() + 3 * fp32_qk

    def describe(self) -> str:
        return (f"B={self.B} Hq={self.Hq} Hkv={self.Hkv} S={self.S} D={self.D} {self.dtype} "
                f"{self.position_kind}+{self.position_offset} {self.contract}")


Workload = Annotated[Union[LayerNormCase, MRoPECase], Field(discriminator="operator")]


class Manifest(BaseModel):
    """A declared workload matrix. Held-out manifests are frozen before final evaluation."""

    model_config = ConfigDict(extra="forbid")
    name: str
    operator: Operator
    split: Split
    cases: list[Workload]
    description: str = ""

    @property
    def manifest_hash(self) -> str:
        return sha256_text(canonical_json([c.model_dump() for c in self.cases]))


# ---------------------------------------------------------------------------
# Candidates and launch policies
# ---------------------------------------------------------------------------

POW2 = [1 << i for i in range(0, 17)]

# Bounded launch metadata the trusted host wrapper accepts per operator.
# For layernorm, BLOCK_SIZE == 0 means "next power of two >= N".
LAUNCH_SPACE: dict[str, dict[str, list[int]]] = {
    "layernorm": {
        "BLOCK_SIZE": [0] + [p for p in POW2 if 16 <= p <= 65536],
        "ROWS_PER_PROGRAM": [1, 2, 4, 8, 16, 32, 64],
        "num_warps": [1, 2, 4, 8, 16],
        "num_stages": [1, 2, 3, 4],
    },
    "mrope": {
        "BLOCK_S": [p for p in POW2 if 1 <= p <= 256],
        "HEAD_SPLITS": [1, 2, 4, 8, 16],
        "num_warps": [1, 2, 4, 8, 16],
        "num_stages": [1, 2, 3, 4],
    },
}
LAUNCH_DEFAULTS: dict[str, dict[str, int]] = {
    "layernorm": {"BLOCK_SIZE": 0, "ROWS_PER_PROGRAM": 1, "num_warps": 4, "num_stages": 1},
    "mrope": {"BLOCK_S": 16, "HEAD_SPLITS": 1, "num_warps": 4, "num_stages": 1},
}
SELECTOR_NAME = {"layernorm": "N", "mrope": "S"}
MAX_SOURCE_CHARS = 20_000
MAX_SOURCE_LINES = 400


class LaunchTier(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    upto: int | None = Field(default=None, ge=1, description="applies when selector <= upto; null = catch-all")
    params: dict[str, int]


class LaunchPolicy(BaseModel):
    """Ordered tiers keyed on the operator selector (layernorm: N, mrope: S)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    tiers: list[LaunchTier] = Field(min_length=1, max_length=8)

    def select(self, selector: int) -> dict[str, int]:
        for tier in self.tiers:
            if tier.upto is None or selector <= tier.upto:
                return dict(tier.params)
        raise ValueError(f"no launch tier covers selector={selector}")

    def validate_for(self, operator: str) -> list[str]:
        """Return a list of problems (empty when the policy is inside the bounded space)."""
        space = LAUNCH_SPACE[operator]
        problems: list[str] = []
        if self.tiers[-1].upto is not None:
            problems.append("last tier must be a catch-all (upto: null)")
        last = 0
        for i, tier in enumerate(self.tiers):
            if tier.upto is not None:
                if tier.upto <= last:
                    problems.append(f"tier {i}: upto must increase")
                last = tier.upto
            unknown = set(tier.params) - set(space)
            if unknown:
                problems.append(f"tier {i}: unknown launch params {sorted(unknown)}")
            for name, allowed in space.items():
                value = tier.params.get(name, LAUNCH_DEFAULTS[operator][name])
                if value not in allowed:
                    problems.append(f"tier {i}: {name}={value} not in {allowed}")
        return problems

    @classmethod
    def single(cls, **params: int) -> "LaunchPolicy":
        return cls(tiers=[LaunchTier(upto=None, params=params)])


def canonical_launch(operator: str, launch: dict) -> dict:
    """Semantics-preserving normal form: defaults filled in (exactly as the trusted wrapper applies them)
    and adjacent tiers with identical params merged. Unknown params are kept so validation still sees them."""
    defaults = LAUNCH_DEFAULTS.get(operator, {})
    tiers = [{"upto": t.get("upto"), "params": {**defaults, **(t.get("params") or {})}}
             for t in launch.get("tiers", [])]
    merged: list[dict] = []
    for tier in tiers:
        if merged and merged[-1]["params"] == tier["params"]:
            merged[-1] = tier  # the later tier covers a superset of selectors
        else:
            merged.append(tier)
    return {"tiers": merged}


CandidateOrigin = Literal["manual", "baseline_triton", "generated", "canary", "scripted", "tuning"]


class CandidateSpec(BaseModel):
    """The narrow candidate contract: Triton JIT source + entry point + bounded launch policy.

    Trusted host wrappers own allocation, input creation, argument binding and result collection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    operator: Operator
    kernel_source: str = Field(max_length=MAX_SOURCE_CHARS)
    entry_point: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    launch: LaunchPolicy
    explanation: str = Field(default="", max_length=4000)
    origin: CandidateOrigin = "generated"
    search_mode: Literal["full", "launch_only", "manual"] = "full"
    label: str = ""

    @model_validator(mode="before")
    @classmethod
    def _canonical_launch(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("operator") in LAUNCH_DEFAULTS and data.get("launch") is not None:
            launch = data["launch"]
            launch = launch.model_dump() if isinstance(launch, LaunchPolicy) else launch
            if isinstance(launch, dict) and isinstance(launch.get("tiers"), list):
                data = {**data, "launch": canonical_launch(data["operator"], launch)}
        return data

    @property
    def source_hash(self) -> str:
        """Content address over source, entry point and the canonical launch policy, so a policy that
        only restates defaults is a duplicate of its parent."""
        return sha256_text(canonical_json({
            "operator": self.operator,
            "kernel_source": self.kernel_source,
            "entry_point": self.entry_point,
            "launch": canonical_launch(self.operator, self.launch.model_dump()),
        }))

    @property
    def short_hash(self) -> str:
        return self.source_hash[:12]

    @property
    def semantic_hash(self) -> str:
        """Dedup key: like source_hash, but over the parsed AST so whitespace/comment-only edits of a
        kernel count as duplicates. source_hash stays the recorded identity of the exact bytes."""
        try:
            src = ast.dump(ast.parse(self.kernel_source), include_attributes=False)
        except SyntaxError:
            src = self.kernel_source
        return sha256_text(canonical_json({
            "operator": self.operator, "ast": src, "entry_point": self.entry_point,
            "launch": canonical_launch(self.operator, self.launch.model_dump()),
        }))


# ---------------------------------------------------------------------------
# Task and result bundles exchanged with the worker
# ---------------------------------------------------------------------------

class BenchmarkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warmup_iterations: int = Field(default=25, ge=1, le=10_000)
    samples: int = Field(default=50, ge=5, le=10_000)
    calls_per_sample: int = Field(default=20, ge=1, le=100_000)
    # Batches shorter than this are lengthened (calls doubled) to stay clear of timer/launch overhead.
    min_batch_ms: float = Field(default=0.5, ge=0.0)
    max_calls_per_sample: int = Field(default=2000, ge=1)
    timing_scopes: list[Literal["operator", "kernel", "graph"]] = ["operator", "kernel", "graph"]
    seed: int = 0


TaskKind = Literal["probe", "smoke", "selftest", "correctness", "timing", "memory"]


class TaskBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    task_id: str
    experiment_id: str
    kind: TaskKind
    candidate: CandidateSpec | None = None
    # Implementations compared against the candidate (timing) or measured alone (memory).
    baselines: list[str] = []
    implementation: str = "candidate"
    workloads: list[Workload] = []
    split: Split = "smoke"
    benchmark: BenchmarkSettings | None = None
    max_tensor_bytes: int = Field(default=1 << 30, ge=1 << 20)
    measurement_kind: MeasurementKind = "clean"
    session_index: int = 0
    generator_resident: bool = False


class CorrectnessOutcome(BaseModel):
    workload_id: str
    status: RecordStatus
    failure_stage: FailureStage | None = None
    message: str | None = None
    passed: bool | None = None
    checks: dict[str, bool] = {}
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    mismatch_count: int | None = None
    element_count: int | None = None
    tolerance: Tolerance | None = None


TimingScope = Literal["operator", "kernel", "graph"]


class TimingStats(BaseModel):
    impl_id: str
    scope: TimingScope
    calls_per_sample: int
    samples_ms: list[float]
    host_samples_ms: list[float]
    median_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    cv: float
    host_median_ms: float


class PairedStats(BaseModel):
    baseline_id: str
    scope: TimingScope
    speedup_of_medians: float
    median_paired_ratio: float
    ci95_low: float
    ci95_high: float
    n_pairs: int


class TimingOutcome(BaseModel):
    workload_id: str
    status: RecordStatus
    failure_stage: FailureStage | None = None
    message: str | None = None
    stats: list[TimingStats] = []
    paired: list[PairedStats] = []
    contamination_flags: list[str] = []
    telemetry: dict[str, Any] = {}


class MemoryOutcome(BaseModel):
    workload_id: str
    impl_id: str
    status: RecordStatus
    failure_stage: FailureStage | None = None
    message: str | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None
    allocated_before_bytes: int | None = None
    peak_allocated_bytes: int | None = None
    incremental_peak_bytes: int | None = None
    reserved_peak_bytes: int | None = None
    resolution_bytes: int = 512


class ResultBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    task_id: str
    kind: TaskKind
    status: Literal["ok", "failed", "error", "quarantined", "timeout"]
    failure_stage: FailureStage | None = None
    message: str | None = None
    hardware: dict[str, Any] = {}
    software: dict[str, Any] = {}
    isolation: dict[str, Any] = {}
    correctness: list[CorrectnessOutcome] = []
    timing: list[TimingOutcome] = []
    memory: list[MemoryOutcome] = []
    probe: dict[str, Any] | None = None
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None


# ---------------------------------------------------------------------------
# The persisted evaluation record (one per candidate x workload x split)
# ---------------------------------------------------------------------------

MEMORY_METRIC = "incremental_peak_allocated_bytes_including_output"
MEMORY_METRIC_DEFINITION = (
    "Peak torch.cuda.max_memory_allocated() during one operator call minus memory allocated "
    "immediately before the call (inputs resident), measured in a fresh process per implementation. "
    "Includes the returned outputs and all temporaries; excludes inputs and allocator reserve. "
    "Resolution: 512-byte allocator blocks."
)


class EvaluationRecord(BaseModel):
    """One versioned evidence record. Unmeasured metrics are null, never zero."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    experiment_id: str
    workload_id: str
    candidate_hash: str
    operator: Operator
    evaluation_split: Split
    status: RecordStatus
    failure_stage: FailureStage | None = None
    failure_message: str | None = None
    hardware: dict[str, Any] = {}
    software_versions: dict[str, Any] = {}
    dtype: DType
    shapes: dict[str, list[int]]
    layout: str = "contiguous"
    contract: str
    baseline_id: str
    launch_config: dict[str, int] = {}
    correctness_pass: bool | None = None
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    tolerance: Tolerance | None = None
    raw_timing_samples: dict[str, list[float]] | None = None
    timing_scope: TimingScope | None = None
    calls_per_sample: int | None = None
    baseline_median_ms: float | None = None
    candidate_median_ms: float | None = None
    baseline_p10_p90_ms: list[float] | None = None
    candidate_p10_p90_ms: list[float] | None = None
    speedup: float | None = None
    speedup_ci95: list[float] | None = None
    sessions: int | None = None
    memory_metric: str | None = None
    baseline_peak_bytes: int | None = None
    candidate_peak_bytes: int | None = None
    memory_reduction_percent: float | None = None
    memory_detail: dict[str, Any] | None = None
    contamination_flags: list[str] = []
    measurement_kind: MeasurementKind = "clean"
    workload: dict[str, Any] = {}
    timestamp: str = Field(default_factory=utc_now)

    @field_validator("speedup", "baseline_median_ms", "candidate_median_ms")
    @classmethod
    def _positive_or_null(cls, v: float | None) -> float | None:
        if v is not None and not (v > 0 and math.isfinite(v)):
            raise ValueError("timing metrics must be positive and finite, or null when unmeasured")
        return v


# ---------------------------------------------------------------------------
# Generator and critic exchanges
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class CritiqueResult(BaseModel):
    arm: str
    provider: str
    model: str | None
    status: Literal["completed", "unavailable", "error", "refused", "not_run"]
    reason: str | None = None
    bottlenecks: list[str] = []
    correctness_risks: list[str] = []
    changes: list[str] = []
    usage: TokenUsage = TokenUsage()
    duration_s: float | None = None


# ---------------------------------------------------------------------------
# Metric helpers (shared by the evaluator and the report)
# ---------------------------------------------------------------------------

def speedup(baseline_ms: float | None, candidate_ms: float | None) -> float | None:
    if not baseline_ms or not candidate_ms:
        return None
    return baseline_ms / candidate_ms


def latency_reduction_percent(baseline_ms: float | None, candidate_ms: float | None) -> float | None:
    if not baseline_ms or not candidate_ms:
        return None
    return 100.0 * (1.0 - candidate_ms / baseline_ms)


def memory_reduction_percent(baseline_bytes: int | None, candidate_bytes: int | None) -> float | None:
    if baseline_bytes is None or candidate_bytes is None or baseline_bytes <= 0:
        return None
    return 100.0 * (1.0 - candidate_bytes / baseline_bytes)


def geomean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and v > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def speedup_to_latency_reduction(speedup_factor: float) -> float:
    """1.22x faster == 18.03% lower latency (not 22%)."""
    return 100.0 * (1.0 - 1.0 / speedup_factor)
