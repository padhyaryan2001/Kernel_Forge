# KernelForge: the full project story

**What it is:** a system on one home PC in which a local AI model writes GPU kernels, a trusted judge checks and
times every answer, and the measured results go back to the model so it can try again.

**Where it ran:** one NVIDIA RTX 5060 Ti with 8 GB of memory, under Windows 11 + WSL2 Ubuntu, on 2026-11-01.

**Kernel-writing model:** Qwen2.5-Coder-7B-Instruct, 4-bit (Q4_K_M), run locally with llama.cpp.

Every number below comes from an evidence record under `runs/` (raw timing samples, source hashes, hardware and
software versions). Each one can be regenerated with `python -m kernelforge summarize --experiment <id>`. The
experiment IDs are listed in Appendix A.

---

## The short version

1. **The plumbing works.** The 7B model ran on the GPU at about 4.7 GB of memory and 78 tokens/s. It was started
   and stopped 16 times during the search and critic runs, and every time its GPU memory came back before any kernel was
   timed.
2. **The hand-written kernels are real wins in some measurements.**
   * On unseen shapes, the fused MRoPE kernel is **10.9× faster than eager PyTorch** and uses **77–83% less
     operator memory** for fp16/bf16.
   * The LayerNorm kernel is **1.69× faster than PyTorch's `F.layer_norm`** when both are replayed from a CUDA
     graph. In ordinary eager calls it is slower (0.72×), because launching a Triton kernel from Python costs
     more time than this small operation itself.
3. **The 7B model did not find a faster kernel.** It wrote 32 kernels: 22 were repeats, broken, or wrong. The other
   10 passed every test, but each was the starting kernel with a cosmetic change, and none was faster. It never
   changed the launch settings. It tried the one change that matters for MRoPE (computing the trig once for all
   heads) exactly once, and that attempt broke the outputs.
4. **The judge did its job.**
   * It rejected all 11 broken or wrong kernels before timing them.
   * When the search "selected" a renamed copy of the MRoPE kernel because of a 1.4% timing difference, timing
     the copy and the original side by side on unseen shapes showed it at 0.99–1.00×, i.e. the difference was
     noise.
   * That exposed a flaw in how the search picks winners (section 15).

| Goal from the brief | Measured | Met? |
| --- | --- | --- |
| LayerNorm 1.22× faster than PyTorch | 1.687× (CUDA-graph timing), 0.718× (eager calls) | only in CUDA-graph timing |
| MRoPE 60% lower operator memory | 76.5–83.3% for fp16/bf16, 37–40% for fp32 (vs eager PyTorch) | only for 16-bit types vs eager PyTorch |
| AI-generated kernel 2.2× faster | every passing 7B kernel ≈1.00× its starting kernel | **no** |

---

## 1. The goal and the ground rules

The project brief (`KernelForge_RTX5060Ti_Implementation_Plan.md`) asked for six things on this one PC:

1. An optimized **LayerNorm** forward kernel in Triton.
2. A fused **multimodal rotary position embedding (MRoPE)** kernel in Triton, the Qwen2-VL
   temporal/height/width variant.
3. A **self-hosted, quantized Qwen coder agent**: it proposes kernels, receives execution feedback and revises
   them.
4. A **controlled comparison of AI code critics**: no critic, self-critique, OpenAI, Claude and Gemini.
5. **Automated tests, benchmarks and a static HTML report**, publishable to GitHub Pages.
6. A separate, optional **reinforcement-learning** extension with real parameter updates.

The brief set strict rules, and they shaped every design decision:

* **Targets are not results.** The three headline numbers (22% faster LayerNorm, 60% less MRoPE memory, 2.2×
  faster generated kernels) live only in a `targets:` block of the config. They are never written into a
  result.
* **Unmeasured means `null`, never zero.** A kernel that never ran must not show up as "0 ms".
* **Negative results get reported.** "The model didn't beat the baseline" is a valid outcome.
* **The judge is trusted; the model's code is not.** Generated code never runs inside the controller. The model
  cannot edit tests or the benchmark. The evaluator never sees API keys.
* **One GPU job at a time.** The model server must be shut down, and its memory confirmed free, before any kernel
  is timed.
* **Nothing outward-facing happens automatically.** No paid compute, no billing, no publishing a repository, no
  deploying Pages.
* **Never claim GPU success from a CPU run.** Skipped GPU tests are reported as skipped.

---

## 2. The machine

| Part | Details |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti: Blackwell, compute capability 12.0 (`sm_120`), 36 SMs, 8 GB, 32 MB L2 |
| Driver | Windows driver 616.56 (CUDA 13.4) |
| OS | Windows 11 + WSL2, Ubuntu 26.04.1 (kernel 6.18); WSL sees 7.3 GiB of RAM |
| Python stack | Python 3.12.16, PyTorch 2.10.0+cu128, Triton 3.6.0 |
| Model server | llama.cpp release b12024 (CUDA 13.4 build) |
| Kernel-writing model | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`, file `qwen2.5-coder-7b-instruct-q4_k_m.gguf`, revision `13fb94bf`, SHA-256 `509287f7…4d3c` |

This GPU also drives the Windows desktop: Chrome and an animated wallpaper were running during the runs. That
matters for timing noise, so every timing record carries a `gpu_drives_display` flag and, usually, a
`concurrent_gpu_activity` flag.

---

## 3. Step 1: Getting the GPU working

The RTX 5060 Ti is a new architecture (`sm_120`), and only recent software supports it. The brief warned that
`import torch` succeeding, or `torch.cuda.is_available()` returning True, proves nothing. Real computation has to
run on the card.

What was set up (`scripts/setup_wsl.sh`):

* **WSL2 Ubuntu, using the Windows driver projected into Linux.** No Linux NVIDIA driver was installed inside WSL,
  which would break things.
* **A Python 3.12 virtual environment created with `uv`.** It holds `torch==2.10.0` from the official CUDA 12.8
  wheel index, plus Triton 3.6.0, the version that PyTorch wheel pins.
* **No separate CUDA toolkit.** The PyTorch wheel bundles its CUDA runtime, Triton bundles its own compiler
  (`ptxas`), and llama.cpp ships a prebuilt CUDA binary.
* **Exact versions are locked** in `requirements-gpu.lock`.

**Smoke test** (`python -m kernelforge smoke`, `scripts/cuda_smoke.py`): a CUDA matrix multiply and a small Triton
vector-add kernel, both run on the GPU and compared with PyTorch. Both passed on `sm_120`.

**Health check** (`python -m kernelforge doctor`) reports:

* WSL vs native Linux, the distribution, and visible RAM (it notes that WSL shows 7.3 GiB, not the full 16 GB),
  swap, and disk space;
* the GPU's model, UUID, total and free memory, driver, CUDA version and compute capability;
* PyTorch and Triton versions, and the model files on disk;
* whether API keys are present. It never prints their values.

---

## 4. Step 2: Writing down the math precisely

Before writing any GPU code, both operators were defined exactly. An independent float64 reference was written on
the CPU for each.

**LayerNorm (forward only):**

* Input `x[M, N]`; `gamma` and `beta` of size `[N]`.
* `mean = Σx/N` and `var = Σ(x−mean)²/N` (population variance).
* `y = (x−mean)/√(var+eps)·γ + β`.
* Arithmetic is accumulated in float32, and the output keeps the input's dtype (float32, float16 or bfloat16).
* Non-contiguous inputs are rejected, so no hidden copies are made.

**MRoPE (Qwen2-VL style):**

* Inputs:
  * `q[B, Hq, S, D]` and `k[B, Hkv, S, D]` (query and key heads may differ in number);
  * integer `position_ids[3, B, S]`, holding a temporal, a height and a width position for each token;
  * `inv_freq[D/2]`;
  * `sections = [nt, nh, nw]`, which split the D/2 rotation pairs among the three axes: `[16, 24, 24]` for
    D=128 and `[8, 12, 12]` for D=64.
* Pair `i` takes its angle from the temporal, height or width position, depending on which section it falls in.
* The rotation is "split-half": element `i` is rotated together with element `i + D/2`.
* The convention was checked against the actual Qwen2-VL code in Hugging Face `transformers` (pinned revision
  `f90e93c4`). A test also shows that the other common convention (interleaved even/odd pairs) gives different
  answers, so the two can't be mixed up silently.
* Two benchmark contracts are kept apart:
  * **positions→output:** both sides start from positions and compute the trig themselves;
  * **precomputed-trig→output:** both sides receive the same precomputed cos/sin.

**Result:** `python -m kernelforge test-reference --device cpu` passed 20/20 checks.

---

## 5. Step 3: The test cases (workloads)

`kernelforge/workloads.py` defines four fixed sets of inputs:

| Set | Purpose | LayerNorm | MRoPE |
| --- | --- | --- | --- |
| `smoke` | quick sanity check | a few shapes | a few shapes |
| `search` | what the agent is timed on each round | 6 shapes (e.g. M=2048, N=768 fp16) | a small subset |
| `dev` | full correctness gate during development and search | **83 workloads** | **33 workloads** |
| `heldout` | frozen final exam, run once per selected kernel | **23 workloads** | **8 workloads** |

The **dev** sets cover these cases:

* **LayerNorm:** M ∈ {1, 32, 256, 2048} × N ∈ {128, 768, 1024, 3072, 4096, 8192} × 3 dtypes. On top of that: sizes
  that are not powers of two, constant rows, low-variance rows, negative values, large offsets and several
  epsilons.
* **MRoPE:** S ∈ {128, 1024, 4096}, D ∈ {64, 128}, B ∈ {1, 2}, Hq ∈ {8, 16} with fewer key heads, and text-, image-
  and video-style positions. Both contracts are included, with positions up to 500,000 to catch trigonometry
  precision problems.

The **held-out** set uses shapes the search never sees (N = 1536, 5120, 640; S = 768, 2048, 3000, 333) and random
seeds ≥ 9000. These never appear in a model prompt. Each set's manifest is hashed and the hash is stored with
every experiment.

---

## 6. Step 4: The trusted judge (evaluator)

The architecture has four roles, and each stays in its lane:

```
Controller (CPU, SQLite)  →  starts model server  →  model writes candidate kernels
        ↓                                                 ↓
stops the model server, confirms GPU memory is free   ←───┘
        ↓
Fresh worker process (the only place candidate code ever runs)
        → correct on every dev workload?  no → compiler/numerical feedback
                                          yes → timing (3 scopes) + memory
        → evidence record → feedback to the model → next round, until the budget runs out
        → freeze the best kernel → held-out exam (once) → static report
```

* **Controller** (`kernelforge/__main__.py`, `agent.py`, `store.py`): handles the command line, the SQLite
  experiment state, starting and stopping the model, and the reports. **It never creates a CUDA context.** Its GPU
  facts come from `nvidia-smi` and from worker probes.
* **Worker** (`kernelforge/worker.py`, `evaluate.py`, `benchmark.py`): a fresh process for each task. It is the only
  place a candidate is ever imported. It generates the inputs, decides correctness, times the kernel, measures
  memory, and writes a schema-validated result.
* **Model server**: llama.cpp, bound to `127.0.0.1` only.
* **Critics**: return structured JSON advice only. They never run code or decide whether a kernel is correct.

### Correctness gates (`evaluate.py`)

* Outputs are compared with the float64 CPU reference, run on the exact same (dtype-rounded) inputs.
* The tolerances were fixed in `contracts.TOLERANCES` before any search and are recorded in every result. They are
  never loosened to let a candidate pass.
* Output buffers are filled with NaN before the call, so a kernel that writes nothing fails.
* Inputs are compared bit-for-bit after the call, so a kernel that modifies its input fails.
* Outputs must not alias inputs; shapes, dtypes and devices are checked, and any non-finite output fails.
* A candidate must pass **every** dev workload before it is timed at all.

### Proving the judge works: canaries

`python -m kernelforge selftest-evaluator` feeds the judge 9 deliberately bad kernels:

1. variance divided by N−1 (unbiased) instead of N;
2. tail handling that only works when N is a power of two;
3. a kernel that does nothing;
4. a kernel that writes hard-coded zeros;
5. a kernel that modifies its input;
6. a kernel that fails to compile;
7. a kernel that sneaks in host Python code;
8. MRoPE using the wrong (interleaved) rotation;
9. MRoPE using only one position axis.

**All 9 were rejected, each at the stage where it should be.**

### Isolation (`executor.py`, `source_policy.py`)

* **Source filter.** Before any code runs, an allowlist check of the code's syntax tree rejects:
  * imports other than `triton`, `triton.language`, `libdevice` and `math`;
  * any host (non-`@triton.jit`) function;
  * file, OS or network access;
  * dunder attribute access, inline assembly, prints, lambdas and classes.
* **Fresh process per task.** Each task runs with:
  * a scrubbed environment (no API keys) and a private HOME and TMPDIR;
  * resource limits and a wall-clock kill;
  * quarantine on a CUDA fault: the next task starts only after a clean probe.
* **Container mode** (`Dockerfile.worker`): adds no network, a read-only root, dropped capabilities and a non-root
  user. It is implemented but was not run, because Docker Desktop's WSL integration is not enabled on this PC.
* **Honest limit:** candidates still share the GPU driver with the host. This is a development evaluator, not a
  hardened service for arbitrary public code.

---

## 7. Step 5: Measuring speed and memory honestly (`benchmark.py`)

### Timing

* **Warmup is never timed.** Compilation and autotuning happen before measurement.
* **Batch length adapts:** calls per sample are doubled until the fastest implementation's batch takes at least
  0.5 ms, so the timer's resolution doesn't dominate.
* **50 samples per implementation**, with the baseline and candidate **interleaved in random order** to cancel
  drift.
* **Statistics:** median, p10/p90, mean, coefficient of variation, and host wall time are recorded. A 95% bootstrap
  confidence interval (2,000 resamples) is computed on the paired per-sample ratios.
* **Repeats:** baselines and held-out exams run in **three independent worker sessions**.

### Three separately labeled timing scopes

| Scope | What it measures | Why it matters |
| --- | --- | --- |
| `operator` | a normal eager call: output allocation + Python dispatch + kernel | what a PyTorch user actually sees |
| `kernel` | only the Triton launch into pre-allocated outputs | the kernel itself (Triton implementations only) |
| `graph` | the same operator calls captured in a CUDA graph and replayed | pure GPU device time, no Python overhead; symmetric for every implementation |

The agent ranks candidates by **graph** scope. During development, a kernel timed against an identical copy of
itself read **0.93× in operator scope but 1.00× in kernel and graph scope**: small eager calls are dominated by
Python dispatch and are noisy under WSL. All three scopes are always recorded and reported.

### Memory

* Each implementation is measured in its own fresh process with the same inputs.
* The procedure is: synchronize, clear the cache, reset the peak counter, make one call.
* The metric is **incremental peak allocated bytes, including the output**: peak memory during the call minus
  memory before it, at 512-byte resolution.
* `nvidia-smi` process totals are never used as operator memory.

### Contamination flags

Every timing record notes any of the following:

* other GPU activity before timing;
* clock drift or thermal changes;
* that the GPU drives a display;
* host swapping;
* unstable timing (coefficient of variation above 0.15);
* a model or trainer still resident on the GPU.

### Reporting rules

* Speedups are combined with a **geometric mean** ("geomean"), the right average for ratios.
* A geomean is only taken within one candidate × split × baseline × scope.
* Failures and regressions stay in the tables.
* Results from different GPUs, dtypes or contracts are never merged.

### A bug the evidence caught

The first attempt at the LayerNorm dev-set baseline with `torch.compile` failed every timing task. The worker had inherited WSL's
Windows `/mnt/c/...` PATH entries, and PyTorch's compiler crashed with `PermissionError` while probing for `nvcc`.

The fix was to restrict the worker's PATH to Linux directories. The failed run is kept in the database with status
`aborted`; it was not deleted. Then the run was repeated.

---

## 8. Step 6: The hand-written kernels

These were written by hand before any AI model was involved. They set the bar the model would try to beat.

### LayerNorm (`kernelforge/kernels/layernorm.py`)

* **Untuned Triton baseline:** follows the structure of the official Triton tutorial.
  * One row per program, with the whole row in one power-of-two block and masked loads.
  * FP32 sums, with the scale and shift fused into the same kernel.
  * Always 4 warps.
* **Tuned kernel:** treats several rows as one 2D tile, so short rows share a program.
  * Gamma and beta are loaded once per tile.
  * Row offsets are 64-bit, and the variance is computed from registers around the mean.
  * `rsqrt` replaces `1/sqrt`.
  * Launch settings are chosen per row length.
* **Tuning** (`kernelforge tune --operator layernorm`): a bounded sweep of rows-per-program {1, 2, 4, 8, 16} ×
  warps {1, 2, 4, 8, 16} for each of four row-length tiers. It was run on fp16 dev shapes and never on held-out
  shapes. The frozen policy:

| Row length N | rows per program | warps |
| --- | --- | --- |
| ≤ 256 | 4 | 2 |
| ≤ 1024 | 2 | 2 |
| ≤ 4096 | 2 | 8 |
| larger | 1 | 8 |

### MRoPE (`kernelforge/kernels/mrope.py`)

* **Untuned Triton baseline:** one program walks every head, and it recomputes positions, angles, cos and sin for
  **every head**.
* **Fused kernel:**
  * It loads the three position rows **once per tile of tokens**, picks the right axis for each rotation pair, and
    computes cos and sin once.
  * It then **reuses them for every query and key head**.
  * Heads are split across programs (`HEAD_SPLITS`), with separate loops for query and key heads, which handles
    Hq ≠ Hkv.
  * It writes only the two outputs: no expanded trig tensors, no `rotate_half` concatenation, no FP32 copies of q
    and k.
* **Tuning:** tokens per block {8, 16, 32, 64} × head splits {1, 2, 4, 8} × warps {2, 4, 8} for three
  sequence-length tiers.

### Results on the held-out shapes (3 independent sessions)

| Kernel | Compared with | Scope | Correct | Geomean speedup | Range |
| --- | --- | --- | --- | --- | --- |
| LayerNorm tuned | `F.layer_norm` | graph | 23/23 | **1.687×** | 1.01–2.43× |
| LayerNorm tuned | `F.layer_norm` | operator | 23/23 | 0.718× | 0.50–1.28× |
| LayerNorm tuned | untuned Triton | graph / kernel / operator | 23/23 | 1.020× / 1.005× / 1.001× | |
| MRoPE fused | eager PyTorch | operator | 8/8 | **10.86×** | 3.60–21.94× |
| MRoPE fused | eager PyTorch | graph | 8/8 | 17.43× | 2.92–45.65× |
| MRoPE fused | untuned Triton | operator / kernel / graph | 8/8 | 1.096× / 1.080× / **1.777×** | |

**MRoPE operator memory** vs eager PyTorch (held-out):

| dtype | workloads | memory reduction |
| --- | --- | --- |
| float16 / bfloat16 | 6 | **76.5–83.3%** (median 79.9%) |
| float32 | 2 | 37.3–40.3% |

LayerNorm memory vs `F.layer_norm`: 0.2% median difference. Both allocate only the output; PyTorch also returns
the mean and rstd.

**What these numbers mean:**

* **LayerNorm.** The tuned kernel's 1.69× graph-scope win over PyTorch comes mostly from Triton itself: the
  untuned Triton kernel reaches about 1.67× too, so the tuning adds only about 2%. In normal eager calls, Triton's
  roughly 15 µs Python launch cost makes it slower than PyTorch (0.72×).
* **MRoPE.** Here the fusion really matters. Computing the trig once per token tile instead of once per head makes
  the fused kernel **1.78× faster than the untuned Triton kernel** in graph scope.

### Against `torch.compile` (dev shapes, 1 session, all correct)

| Kernel | vs | operator | graph | memory reduction (median) |
| --- | --- | --- | --- | --- |
| LayerNorm tuned | `F.layer_norm` | 0.662× | 1.780× | 0.5% |
| LayerNorm tuned | `torch.compile` | 1.401× | 1.031× (0.61–2.56) | 0.0% |
| MRoPE fused | eager PyTorch | 9.008× | 12.653× | 16-bit 79.2%, fp32 37.5% |
| MRoPE fused | `torch.compile` | 1.800× | 2.192× (1.02–6.55) | 16-bit 4.7%, fp32 2.0% |

`torch.compile` also fuses away MRoPE's temporary tensors, so the memory advantage mostly disappears against it.
The speed advantage (about 1.8–2.2×) remains, because of the shared-trig design.

### Proving the loop end to end (demo)

Before trusting a real model, `scripts/demo.sh` ran the whole loop with a **scripted stand-in generator**, clearly
labeled `scripted-demo`, against the real GPU judge:

1. Round 1 submitted a kernel that divides by the padded block size instead of N. It was rejected
   (`ln-dev-003-M1-N768-float32: failed checks: y_numeric`).
2. The feedback round submitted the tuned kernel. It was correct on 83/83 dev workloads and measured 2.013× vs
   `F.layer_norm`, against 1.855× for the starting kernel.
3. The report was generated.

---

## 9. Step 7: Bringing in the 7B model

### Why this model

The brief required a quantized Qwen coder running locally on the 8 GB card, with the generator stopped before any
GPU timing.

* **Qwen2.5-Coder-7B-Instruct** is code-specialized, and Qwen publishes an official GGUF build of it.
* Its 4-bit Q4_K_M file is 4.68 GB. That leaves room on an 8 GB card that is also drawing the Windows desktop.
* The 30B Qwen3-Coder from the original screenshot does not fit 8 GB of VRAM / 16 GB of RAM at useful precision.
  It is never downloaded; it is only supported as an optional remote endpoint.

### Download and verification

* Downloaded only the Q4_K_M file, at the pinned revision `13fb94bfda8c8cf22497dc57b78f391a9acb426a`:
  4,683,073,536 bytes in 186 s.
* Checked its SHA-256 against the published value (`509287f78cb4…4d3c`) before use.
* The revision, filename and hash are recorded in `config.yaml` and in every experiment.

### Does it fit? (`python -m kernelforge generator-check`)

The controller starts the official llama.cpp server with fixed flags:

```
llama-server -m ~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf --alias local-coder
             --host 127.0.0.1 --port 8000 -c 4096 -ngl 99 --parallel 1 --cache-ram 0 --no-webui
```

These flags mean:

* loopback only;
* a 4,096-token context;
* all layers on the GPU;
* one request at a time;
* no growing prompt cache.

| Measurement | Value |
| --- | --- |
| Load time | 4.2 s |
| Free GPU memory before → loaded | 6,771 MiB → 2,090 MiB |
| Server GPU memory | **≈ 4,681 MiB** |
| Host RAM used by the server | 777 MiB |
| Test request | 63 prompt tokens → 139 output tokens in 1.77 s = **78.5 tokens/s** |
| Shutdown | 0.54 s, clean exit |
| Free GPU memory after stop | 6,849 MiB, **released** |
| Evaluator probe after stop | allocation OK |

The model's answer to the simple test prompt is telling. It wrote a "Triton" vector add using
`triton.blockIdx.x` and `triton.threadIdx.x`. Those are CUDA concepts, and Triton has no such functions. It was a
first hint that this model knows CUDA better than Triton.

### The lifecycle during the real runs

The model server was started and stopped **16 times** across the search and critic runs.

| Measurement | Value |
| --- | --- |
| Load time | 1.6–3.1 s |
| Server GPU memory | 4,681–4,682 MiB |
| Shutdown time | 0.26–0.32 s |
| Free GPU memory before timing | back to 6,865–6,866 MiB every time |

**All 16 stops confirmed `vram_released=True` before the judge ran.**

---

## 10. Step 8: How the agent loop works (`kernelforge/agent.py`)

This is **execution-guided search** (iterative refinement), not reinforcement learning: the model's weights never
change.

### States

Each candidate moves through:

```
queued → generated → validated → compiled → correctness_checked → benchmarked → critiqued → revised → selected → heldout_evaluated
```

The failure states are `invalid`, `duplicate`, `compile_failed`, `incorrect`, `benchmark_failed`, `timed_out` and
`worker_error`. The last two are retried at most twice. Every state change is a database transaction, so an
interrupted experiment can resume, and a failed candidate never destroys an earlier good one.

### One round

1. Start the 7B server and generate **2 candidates**. Generation uses temperature 0.7, a recorded sampling seed,
   and at most 1,536 output tokens.
2. Stop the server and confirm that GPU memory is free.
3. Check each answer against the source policy and the required function signature. Truncated answers are
   rejected, never run.
4. **Deduplicate.** The code is normalized and compared on its syntax tree, and the launch settings are put in a
   standard form. A candidate that is only the parent with different whitespace, comments or an empty launch
   setting would otherwise get a new hash and could "win" through timing noise alone.
5. Run correctness on **all** dev workloads, then time the kernels that pass on the `search` workloads.
6. Turn the results into short feedback for the next round, plus an optional critique.

### What the model sees

The prompt is about 945–1,860 tokens and has these parts:

* **System message:** "You are an expert Triton GPU kernel engineer." It requires a reply with a Python code
  block, a JSON launch policy and a one- or two-sentence explanation.
* **The operator spec:** the math, the dtypes and the masking rules.
* **The exact required function signature.**
* **The source rules:** allowed imports, `@triton.jit` only, never write to inputs.
* **The GPU** (RTX 5060 Ti, compute capability 12.0, 36 SMs).
* **The launch-policy format and allowed values.**
  * LayerNorm: rows-per-program 1–64 and warps 1–16.
  * MRoPE: tokens per block, head splits and warps.
* **The current best correct kernel**, with its launch policy and measured score, e.g. "geomean speedup 1.888x vs
  torch_eager (graph timing)".
* **Last round's feedback.** Examples:
  * "candidate e96f92af62a5: incorrect at correctness — ln-dev-003-M1-N768-float32: failed checks: y_numeric";
  * "candidate f749f1e7ea4b: benchmarked; geomean speedup 1.814x (M2048-N768-float16: 2.04x, …)".
* **The task:** "Write a faster kernel that stays exactly correct. Make one or two focused changes."

### Budgets and selection

* **Budgets:** 4 rounds × 2 candidates per operator; 200,000 generator tokens; 180 minutes per experiment;
  120 s per compile; 30 s per execution.
* **Starting point:** the **untuned Triton kernel**, for both operators.
* **Selection:** the correct candidate with the highest dev geomean (graph scope vs eager PyTorch) is frozen. The
  starting kernel is also eligible.
* **Held-out exam:** runs **once** (a second run requires `--force` and is flagged).

---

## 11. The 7B search on LayerNorm

**Experiment:** `optimize-layernorm-20261101-185026-9d60`. The starting kernel `589441de2e46` (the untuned
Triton kernel) scored **1.888×** vs eager PyTorch on the search shapes.

| Round | Answer | What the 7B changed | Judge's verdict |
| --- | --- | --- | --- |
| 1 | `e96f92af62a5` | `xc = x - mean`: **dropped the mask** on the centered values | **Incorrect** on N = 768 and 3072. Padded lanes held 0, so `0 − mean` added `mean²` to the variance. |
| 1 | `f749f1e7ea4b` | same subtraction, then masked on a second line | correct; **1.814×** (slower than the start) |
| 2 | `5e3448cbf978` | stored the two sums in temporary variables first | correct; **1.855×** (slower) |
| 2 | (repeat) | returned the starting kernel unchanged: "already efficient and correct" | duplicate, not re-run |
| 3 | (repeat) | repeated `f749f1e7ea4b` | duplicate |
| 3 | (repeat) | returned the starting kernel unchanged | duplicate |
| 4 | (repeat) | repeated the already-rejected `e96f92af62a5` | duplicate |
| 4 | `9ff178b1d204` | **added two unused imports** (`libdevice`, `math`) | correct; **1.729×** (slower) |

* **Totals:** 8 answers, 4 repeats, 1 incorrect, 3 correct (all cosmetic), 0 faster.
* **Selected:** the starting kernel itself.
* **Cost:** 11,591 tokens (8,540 prompt + 3,051 output) and 38.9 s of generation in a 3 min 13 s experiment.

Round 3 produced only repeats, so round 4 got no new feedback, and its prompt was the same as round 1's. The model
then re-proposed the mask removal that had already failed.

**Held-out exam** (23 unseen shapes, 3 sessions): 23/23 correct.

* 1.657× vs `F.layer_norm` in graph scope, 0.723× in operator scope.
* 0.995× / 0.997× / 0.984× of the hand-tuned kernel (operator / kernel / graph).

---

## 12. The 7B search on MRoPE

**Experiment:** `optimize-mrope-20261101-185340-2504`. The starting kernel `63a6c0f75c1d` (the untuned Triton
MRoPE, which recomputes the trig for every head) scored **6.738×**.

| Round | Answer | What the 7B changed | Judge's verdict |
| --- | --- | --- | --- |
| 1 | `ef36936d7a9e` | removed the query/key branch, so every head index was also written through the query path | **Incorrect** (`q_out_numeric`) |
| 1 | `7f3294b55910` | **moved the cos/sin computation out of the head loop**, the right idea and the one the hand-written kernel uses, but then wrote every head, keys included, into the query output | **Incorrect** (`q_out`, `k_out`, non-finite values): the key output was never written |
| 2 | `c0aeab5ca794` | gave two intermediate results names: `x_out_real = x1*cos - x2*sin` | correct; **6.830×**; becomes the new "best" |
| 2 | `f4b0a587e07e` | replaced stores with `tl.atomic_add` "to improve correctness" | **Invalid**: syntax error, unmatched `)` |
| 3 | (repeat) | returned `c0aeab5ca794` unchanged: "already optimized" | duplicate |
| 3 | `1d1906b40f3a` | merged the query and key paths, but always read from and wrote to the query tensors | **Incorrect** (`k_out_numeric`, non-finite): the key output was never written |
| 4 | (repeat) | returned `c0aeab5ca794` unchanged | duplicate |
| 4 | `7854352f5ac2` | loaded every head from the query tensor, keys included | **Incorrect** (`k_out_numeric`): the key output was computed from query data |

* **Totals:** 8 answers, 2 repeats, 1 invalid, 4 incorrect, 1 correct (cosmetic), 0 really faster.
* **Selected:** `c0aeab5ca794`, a "generated" kernel. Its launch settings and math are identical to the starting
  kernel's.
* **Cost:** 20,628 tokens (13,972 + 6,656) and 85.0 s of generation in a 3 min 44 s experiment. MRoPE answers are
  longer, at about 8–12 s each.

**The "win" was noise.** The selected copy scored 6.830× against the original's 6.738×, a 1.4% gap measured in
separate timing tasks. On the held-out exam, where both were timed side by side in the same task, the copy came out
at **0.987× / 0.991× / 0.999×** of the original (operator / kernel / graph).

**Held-out exam** (8 unseen shapes, 3 sessions): 8/8 correct.

* 8.744× (operator) / 8.819× (graph) vs eager PyTorch.
* Median 79.2% less operator memory than eager PyTorch.
* 0.946× / 0.905× / **0.552×** of the hand-written fused kernel. The 7B kernel is about half as fast in device
  time, because it still recomputes the trig for every head.

---

## 13. The critic comparison

**Question:** does adding a critic, which reviews the code and suggests changes before the model revises it, help
the 7B?

**Setup** (`compare-critics`, LayerNorm, experiment `critics-layernorm-20261101-190144-d62e`):

* The generator stays fixed (the 7B), as do the starting kernel, workloads and budgets.
* 2 seeds per arm, 2 rounds × 2 candidates, and a held-out exam for every run.
* Every critic receives exactly the same feedback fields.
* Critics return structured JSON: bottleneck hypotheses, correctness risks and concrete changes. They never decide
  correctness.

| Arm | Runs | Answers (repeats) | New code | Passed | Compile fail / incorrect | Dev geomean of selection | Held-out geomean | Tokens: generator / critic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none (control) | 2 | 8 (3) | 5 | 4 | 0 / 1 | 1.892× | 1.621× | 11,490 / 0 |
| self (the 7B critiques its own code) | 2 | 8 (2) | 6 | 2 | 3 / 1 | 1.820× | 1.628× | 12,724 / 3,744 |
| OpenAI | — | — | — | — | — | not run | not run | no model ID configured |
| Claude | — | — | — | — | — | not run | not run | no model ID configured |
| Gemini | — | — | — | — | — | not run | not run | no model ID configured |

**Run by run:**

| Run | Starting kernel's dev score | Answers | Selected | Held-out |
| --- | --- | --- | --- | --- |
| none, seed 0 | 1.885× | mask removal (incorrect), two-line re-mask 1.725×, temporaries 1.875×, 1 repeat | starting kernel | 1.606× |
| none, seed 1 | 1.900× | 2 repeats, two-line re-mask 1.809×, unused import 1.846× | starting kernel | 1.637× |
| self, seed 0 | 1.759× | 2 repeats, re-mask with a renamed variable 1.721×, `tl.reduce` compile failure | starting kernel | 1.658× |
| self, seed 1 | 1.883× | mask removal (incorrect), two-line re-mask 1.745×, 2 `tl.reduce` compile failures | starting kernel | 1.598× |

**What self-critique did:**

* The critiques were short: 650–940 tokens in, 120–220 tokens out, about 2 s each.
* They pushed the 7B toward "use `tl.reduce` for mean and variance". The model then invented signatures that
  Triton doesn't have, so the code failed to compile:
  * `tl.reduce(x, axis=0, fn=tl.sum)` → `reduce() got an unexpected keyword argument 'fn'`
  * `tl.reduce(x, tl.sum, axis=0, mask=mask)` → `reduce() got multiple values for argument 'axis'`
* Those account for all 3 of the arm's compile failures.

**Verdict:**

* **All four runs selected the starting kernel**, and every kernel that passed was a cosmetic rewrite that measured
  slower than its parent.
* Self-critique made the 7B attempt bolder changes, which mostly broke.
* Two seeds per arm is far too few for a statistical claim, and none is made.
* The hosted critics (OpenAI, Claude, Gemini) are fully implemented but **not run**: no model IDs or API keys were
  configured. The report shows them as "not run", never as zero.

---

## 14. What the 7B actually did, across all 32 answers

| Outcome | Count |
| --- | --- |
| Repeat of code already seen (often the starting kernel returned unchanged with "already optimal") | 11 |
| Not valid Python (syntax error) | 1 |
| Did not compile (invented `tl.reduce` signatures) | 3 |
| Compiled but gave wrong answers | 7 |
| Passed every dev workload | 10 (only **6 distinct kernels**: the same ones kept recurring) |
| Faster than its starting kernel beyond noise | **0** |

**The six passing kernels, and what each changed:**

| Kernel | Change |
| --- | --- |
| `f749f1e7ea4b` | masks the centered values in two lines instead of one |
| `4f2741e295c3` | same as above, with the masked result in a renamed variable |
| `5e3448cbf978` | stores the sums in temporaries |
| `9ff178b1d204` | adds 2 unused imports |
| `b6cb9b42e796` | adds 1 unused import |
| `c0aeab5ca794` | names two MRoPE intermediates |

All six compute exactly the same thing, in the same order, as their starting kernel. Only the way the Python
source is written differs.

### Key observations

1. **It never touched the launch settings.** All 32 answers kept the starting launch policy exactly: for
   LayerNorm 1 row per program and 4 warps; for MRoPE 16 tokens per block, 1 head split and 4 warps. The prompt
   listed the allowed values. The hand tuning showed this was where speed was available, especially for MRoPE.
2. **It found the right idea once and broke it.** The hand-written MRoPE kernel is 1.78× faster than the
   starting kernel in device time, because it computes cos/sin once per token tile, reuses them for every head,
   and splits heads across programs.
   * In round 1 of the MRoPE search, the 7B did move the cos/sin computation out of the head loop
     (`7f3294b55910`). It then sent the key heads' results to the query output, and the judge rejected it.
   * The next round's feedback showed only the failed checks, not the reason. The model never tried that idea
     again.
   * For LayerNorm the available headroom was small (about 2%), so finding nothing there is less surprising.
3. **Its explanations didn't match its code.** It claimed speedups for changes that cannot change the compiled
   kernel, such as "reduces the number of memory accesses" for splitting one expression into two lines. Many
   answers said "the kernel is already optimal; no changes made".
4. **Its real algorithmic edits were wrong.** Examples:
   * dropping a needed mask;
   * mixing up which heads go to the query output and which to the key output (all 4 wrong MRoPE answers);
   * using `atomic_add` instead of a store.
5. **It produces the same answers repeatedly.** Sampling is seeded, so identical prompts give identical kernels.
   `f749f1e7ea4b` came up in 4 different runs, and the seed-0 critic control reproduced the first two rounds of
   the main LayerNorm search.
6. **It was cheap and fast.**
   * About 60,000 tokens in total: 56,433 generator + 3,744 critic.
   * About 3.4 minutes of generation for all 32 answers.
   * The full 7B pipeline, from health check to report, took **30 minutes** (18:50–19:20 UTC). Most of that was
     the judge's correctness checks and timing, not the model.

---

## 15. The measurement-noise discovery

The 7B runs gave many chances to time *the same code* repeatedly:

| Same code | Measured in separate runs | Spread |
| --- | --- | --- |
| Starting LayerNorm kernel `589441de2e46`, search shapes | 1.888, 1.885, 1.900, 1.759, 1.883 | 1.759–1.900× (8%) |
| Cosmetic copy `f749f1e7ea4b`, search shapes | 1.814, 1.725, 1.809, 1.745 | 1.725–1.814× (5%) |
| Starting LayerNorm kernel, held-out exam | 1.657, 1.606, 1.637, 1.658, 1.598 | 1.598–1.658× (4%) |
| MRoPE copy vs original, **timed side by side** | 0.987 / 0.991 / 0.999 | within about 1.3% |

**The lesson:**

* Timing two kernels side by side in the same task is accurate to about 1%.
* Comparing numbers from *separate* tasks can be off by up to about 8%. The GPU also drives the desktop, and WSL
  adds jitter.
* The search currently picks the highest separately-timed score. That is how a renamed copy of the MRoPE kernel
  got "selected".

**The fix (designed, not yet implemented):**

* A new kernel should replace its parent only if it beats the parent **in a paired, same-task timing by more than
  a noise margin**.
* Cosmetic copies should be caught by comparing the **compiled GPU code** instead of the Python source.

Until then, any "generated" selection within about 10% of its parent should be treated as unproven.

---

## 16. The reinforcement-learning extension (separate and optional)

The brief asked for *genuine* RL with real weight updates. It was kept separate from the agent above.

* **Policy:** a small trainable model, `Qwen/Qwen2.5-Coder-0.5B-Instruct` (revision `ea3f2471`). This is a
  different model from the 7B generator, and it is never presented as such.
* **Method:** LoRA rank 8 on the q/k/v/o projections, trained with GRPO from TRL 1.17.1 (PEFT 0.21.2,
  transformers 5.22.1) in its own virtual environment.
* **Settings:** 2 completions per prompt, KL β = 0.04, gradient checkpointing, bf16, no vLLM.
* **Budget check before training:** done with the real tokenizer.
  * LayerNorm prompts are 632–639 tokens.
  * A complete tuned LayerNorm answer is 569 tokens, so the completion budget was raised from 512 to 640 instead of
    cutting off valid kernels.
  * MRoPE (about 1,233-token prompts and 931-token answers) was excluded from RL.
* **Reward:** comes only from the trusted judge.
  * −1 for anything invalid, failing to compile, timing out or incorrect.
  * Otherwise `clip(log2(baseline_ms / candidate_ms), −0.5, 2.0)`.
  * Because the policy stays on the GPU during reward timing, these are labeled **in-training** measurements.

| Run | What happened | Held-out prompts correct | Median speedup of correct samples |
| --- | --- | --- | --- |
| Smoke: 2 GRPO steps, cold start | All training answers failed, so every group had the same reward. Loss and gradient were 0: **no learning signal**. | base 0/4, adapter 0/4 | — |
| 24-step supervised warm-start (labeled *not RL*) + 4 GRPO steps | Steps 1–3: all failed. Step 4: one correct answer, which gave one real policy-gradient update. | base 0/8, adapter 5/8 | 0.859× |
| Same supervised adapter, no GRPO (attribution check) | evaluation only | base 0/8, adapter 5/8 | 0.782× |

**Conclusion:** real optimizer updates happened, and adapters were saved and evaluated, but **RL produced no
measurable improvement**. The correctness gain came entirely from the supervised warm-start: 5/8 with or without
GRPO. With one informative GRPO step and only 5 correct samples, 0.86× vs 0.78× is within noise.

---

## 17. Report, evidence, automation and tests

* **Database and artifacts:**
  * SQLite (`runs/kernelforge.sqlite3`) holds 6 tables: experiments, candidates, tasks, evaluations, critic
    trials and state transitions.
  * Each experiment folder holds immutable candidate source, prompts, raw responses, manifests, raw timing samples
    and logs.
* **Evidence records:** each one is a versioned JSON file containing:
  * the GPU, driver, PyTorch, Triton and evaluator SHA-256;
  * shapes, dtypes, the launch configuration and the tolerances;
  * maximum absolute and relative error;
  * raw samples, medians, speedups and memory bytes;
  * contamination flags.
* **Static report:** `python -m kernelforge report` builds a self-contained HTML dashboard,
  `runs/report/index.html`. It includes filters, correctness coverage, latency distributions, per-workload
  speedups, memory, candidate history, critic tables grouped by generator, and the generator lifecycle. It opens
  with no server, shows "No GPU measurements yet" when empty, and keeps targets in a separate panel.
* **Evidence export:** `export-evidence` zips each experiment.
* **GitHub Actions** (`.github/workflows/`), written but not run:
  * **CPU checks:** Python 3.11 and 3.12; GPU tests are explicitly skipped.
  * **Manual GPU benchmark workflow:** for this PC as a self-hosted runner; trusted revisions only, never fork pull
    requests.
  * **Opt-in Pages workflow.**
  * They have not been run because no GitHub repository or runner is configured, and publishing was deliberately
    left to the owner.
* **Tests:**

| Where | Result |
| --- | --- |
| WSL2 + RTX 5060 Ti | **93 passed, 0 skipped**, including GPU smoke, both kernels on the full dev sets, the 9 canaries, and timing/memory tasks |
| Windows, CPU-only PyTorch, no Triton | **76 passed, 17 skipped**: 16 GPU tests with no GPU, and 1 upstream Qwen2-VL check because `transformers` isn't installed there. All reported as skipped, never as passed. |

---

## 18. What is not done

| Item | Status | What it needs |
| --- | --- | --- |
| Hosted critics (OpenAI, Claude, Gemini) | implemented, not run | exact model IDs in `config.yaml`, `pip install '.[critics]'`, API keys exported |
| Container isolation | implemented, not run | Docker Desktop → WSL integration + GPU support, then `docker build -f Dockerfile.worker …` |
| CI / Pages on GitHub | implemented, not run | a GitHub repository, this PC registered as a self-hosted runner, Pages explicitly enabled |
| SSH remote worker | implemented (optional), not run | a remote host |
| Better selection rule | designed, not implemented | paired timing vs parent with a noise margin + compiled-code dedup |
| Cleaner timing | — | close GPU-using desktop apps (animated wallpaper, Chrome) before benchmark runs |

---

## 19. Lessons learned

1. **The judge is the product.** The most valuable part turned out to be the trusted evaluator, not the AI. It
   caught every broken kernel and exposed a fake win.
2. **Pick the timing scope before you look at results.** The same LayerNorm kernel is 1.69× faster *or* 0.72×
   slower than PyTorch, depending on whether Python launch overhead is counted. Both are true, and both are
   reported.
3. **Compare side by side or don't compare.** Separate runs drift by up to about 8% on a desktop GPU, while paired
   timing is accurate to about 1%.
4. **A 7B 4-bit coder is not yet a Triton optimizer.** It writes syntactically plausible code, copies a lot,
   invents APIs, misdescribes its own changes, and ignores tuning settings, even with precise feedback.
5. **Most of the speed came from human design decisions:** fusing MRoPE's trig across heads, multi-row LayerNorm
   tiles, and launch tuning.
6. **"No improvement" is a result.** It was recorded with full evidence, not hidden or dressed up.

---

## 20. What you can honestly claim

**Supported by the evidence:**

* Built a reproducible Triton kernel-optimization system with a trusted evaluator: correctness gates, 9 evaluator
  canaries, three timing scopes, clean-process memory measurement, and full evidence records.
* Hand-written fused MRoPE kernel: **10.9× faster than eager PyTorch** (operator scope) and **77–83% less operator
  memory for fp16/bf16** on held-out shapes, on an RTX 5060 Ti.
* Hand-written LayerNorm kernel: **1.69× faster than `F.layer_norm` in CUDA-graph (device-time) timing** on
  held-out shapes. Always add that it is 0.72× in eager calls.
* Ran a **local Qwen2.5-Coder-7B-Instruct (Q4_K_M)** execution-feedback agent on an 8 GB GPU, with a managed
  lifecycle that freed GPU memory before every timing phase (16/16).
* The evaluator rejected every incorrect generated kernel and showed that an apparent generated speedup was
  measurement noise.

**Not supported; don't claim:**

* That AI-generated kernels were faster: none were, and the 2.2× target was not met.
* "22% faster LayerNorm" without saying it is CUDA-graph timing.
* That RL improved kernel generation.
* That GPT, Claude or Gemini critics were compared: those arms were not run.

---

## Appendix A: experiment IDs

| What | Experiment ID |
| --- | --- |
| LayerNorm tuning (hand-written kernel) | `tune-layernorm-20261101-062214-3c06` |
| MRoPE tuning (hand-written kernel) | `tune-mrope-20261101-062918-35dc` |
| Dev baselines incl. `torch.compile` | `baseline-layernorm-20261101-072658-fe73`, `baseline-mrope-20261101-073432-b143` |
| Aborted baseline (PATH bug, kept) | `baseline-layernorm-20261101-072324-71ec` |
| Scripted end-to-end demo | `demo-layernorm-20261101-065349-76c7` |
| 7B generator check | `runs/_system/generator/generator-check-20261101-135017.json` |
| 7B LayerNorm search + held-out | `optimize-layernorm-20261101-185026-9d60` |
| 7B MRoPE search + held-out | `optimize-mrope-20261101-185340-2504` |
| 7B critic comparison | `critics-layernorm-20261101-190144-d62e` (runs `-none-s0`, `-none-s1`, `-self-s0`, `-self-s1`) |
| RL runs | `runs/rl/20261101-024005`, `runs/rl/20261101-024707`, `runs/rl/20261101-025153` |

## Appendix B: the 7B pipeline, step by step (`bash scripts/run_7b.sh`, log `runs/run_7b.log`)

| Time (UTC) | Step | Duration |
| --- | --- | --- |
| 18:50:12 | `doctor` (health check) | 5 s |
| 18:50:17 | `generator-check` (fit + release test) | 9 s |
| 18:50:26 | `optimize --operator layernorm` | 3 min 13 s |
| 18:53:39 | `optimize --operator mrope` | 3 min 44 s |
| 18:57:23 | `evaluate-heldout` LayerNorm | 2 min 28 s |
| 18:59:51 | `evaluate-heldout` MRoPE | 1 min 52 s |
| 19:01:43 | `compare-critics --operator layernorm --seeds 0,1 --rounds 2 --heldout` | 18 min |
| 19:19:43 | `report` | 31 s |

Every step exited with code 0.

## Appendix C: commands

```bash
python -m kernelforge doctor
python -m kernelforge smoke
python -m kernelforge test-reference --device cpu
python -m kernelforge selftest-evaluator
python -m kernelforge baseline --operator layernorm --suite dev --sessions 1 --with-compile
python -m kernelforge tune --operator layernorm
python -m kernelforge generator-check
python -m kernelforge optimize --operator layernorm
python -m kernelforge optimize --operator mrope
python -m kernelforge evaluate-heldout --experiment EXPERIMENT_ID
python -m kernelforge compare-critics --operator layernorm --seeds 0,1 --rounds 2 --heldout
python -m kernelforge report
python -m kernelforge summarize --experiment EXPERIMENT_ID
python -m kernelforge export-evidence --experiment EXPERIMENT_ID
~/venvs/kernelforge-rl/bin/python -m kernelforge train-rl --config rl.yaml --kf-config config.yaml
```
