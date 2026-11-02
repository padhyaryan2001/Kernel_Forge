# KernelForge — what was executed on this PC (2026-11-01)

Every number below comes from an evidence record under `runs/<experiment_id>/evaluations/…` (raw samples,
source hashes, tolerances, GPU/driver/torch/triton versions, evaluator SHA-256, contamination flags) and can be
regenerated with `python -m kernelforge summarize --experiment <id>` or viewed in `runs/report/index.html`.
The brief's targets are not results; they are compared against measurements only in the "Targets" section.

**Hardware / software:** NVIDIA GeForce RTX 5060 Ti (GPU-3bca2510-…, cc 12.0, 36 SMs, 8 GB, 32 MB L2), Windows
driver 616.56 (CUDA 13.4), WSL2 Ubuntu 26.04.1 (kernel 6.18, 7.3 GiB visible RAM), Python 3.12.16,
torch 2.10.0+cu128, Triton 3.6.0, llama.cpp b12024 (CUDA 13.4 build), Qwen2.5-Coder-3B-Instruct GGUF Q4_K_M
(`f74adce6`, sha256 `724fb256…30b7`) and, in the "7B generator" section only, Qwen2.5-Coder-7B-Instruct GGUF Q4_K_M
(`13fb94bf`, sha256 `509287f7…4d3c`). The GPU also drives the Windows desktop (Chrome and an animated wallpaper were
running); every timing record carries `gpu_drives_display` and usually `concurrent_gpu_activity` flags.

## Implemented vs executed

| Capability | Implemented | Executed here | Evidence |
| --- | --- | --- | --- |
| Blackwell smoke (CUDA matmul + Triton kernel on `sm_120`) | yes | yes, pass | `kernelforge smoke`, `scripts/cuda_smoke.py` |
| CPU references + upstream Qwen2-VL convention check | yes | yes, 20/20 checks; pytest pass | `test-reference --device cpu` |
| LayerNorm Triton (untuned + tuned) | yes | yes, correct on 83 dev + 23 held-out workloads | `tune-layernorm-20261101-062214-3c06` |
| Fused MRoPE Triton (+ untuned Triton) | yes | yes, correct on 33 dev + 8 held-out workloads | `tune-mrope-20261101-062918-35dc` |
| Evaluator canaries rejected | yes | yes, 9/9 at the expected stage | `selftest-evaluator` |
| Benchmark harness (3 scopes, 3 sessions, memory) | yes | yes | all experiments |
| Local 3B generator, managed lifecycle | yes | yes: one request inside budget, VRAM released | `runs/_system/generator/` |
| Local 7B generator (same lifecycle) | yes (config only) | yes: fits 8 GB, all 16 stops released VRAM | `generator-check-20261101-135017.json` |
| Execution-guided agent (bounded, dedup, resume) | yes | yes, 2 operators × 4 rounds, with the 3B and with the 7B; resume covered by tests | `optimize-*` |
| Demo: baseline → failure → feedback → correct revision → measurement → report | yes | yes (scripted generator, real evaluator) | `demo-layernorm-20261101-065349-76c7` |
| Critic comparison | yes | none + self arms × 2 seeds, per generator (3B, 7B); OpenAI/Claude/Gemini **not run** (no model IDs / keys) | `critics-layernorm-20261101-070805-9d79`, `critics-layernorm-20261101-190144-d62e` |
| Held-out evaluation (frozen, once) | yes | yes for 8 frozen candidates | `evaluate-heldout` |
| Static report + evidence export | yes | yes | `runs/report/index.html`, `runs/*/evidence-*.zip` |
| CPU CI workflow | yes | equivalent run locally (Windows CPU-only + WSL); not run on GitHub | `.github/workflows/checks.yml` |
| GPU workflow / Pages | yes | **not run** (no GitHub repository or self-hosted runner configured) | `.github/workflows/` |
| Container isolation (`local_container`) | yes | **not run** (Docker Desktop WSL integration not enabled) | `Dockerfile.worker` |
| SSH remote adapter | yes (optional) | not run (no host supplied) | `executor._remote` |
| RL extension (GRPO + LoRA) | yes | yes: 3 runs, real updates + saved adapters + base-vs-adapter evaluation; **no measurable RL improvement** | `runs/rl/` |

## Tests

| Where | Command | Result |
| --- | --- | --- |
| Windows 11, CPU-only PyTorch, no Triton | `python -m pytest` | 76 passed, 17 skipped (16 CUDA tests with no CUDA device, plus the upstream Qwen2-VL check because `transformers` is not installed there; all reported as skipped) |
| WSL2 + RTX 5060 Ti | `python -m pytest -rs` | 93 passed, 0 skipped (CUDA tests included: smoke, trusted kernels on the full dev manifests, 9 canaries, timing/memory tasks) |

## Environment and generator

* `generator-check`: llama-server loaded the 3B Q4_K_M model in 3.1 s; free VRAM 5,787 → 3,576 MiB (server ≈ 2,211 MiB
  at 4,096-token context, 1 slot, `--cache-ram 0`), host RSS 641 MiB, 140 tokens/s; graceful shutdown 0.47 s, free
  VRAM back to 5,787 MiB, evaluator allocation probe OK afterwards.
* Across the agent and critic runs the managed server was started 17 times (1.09–2.12 s load); all 17 stops reported
  `vram_released=True` before the evaluation phase began (report → "Generator lifecycle").
* 7B `generator-check`: llama-server loaded the 7B Q4_K_M model in 4.2 s; free VRAM 6,771 → 2,090 MiB (server
  ≈ 4,681 MiB, same flags), host RSS 777 MiB, 78.5 tokens/s; shutdown 0.54 s, free VRAM back to 6,849 MiB, evaluator
  probe OK. The 7B runs started it 16 more times (1.6–3.1 s load); all 16 stops reported `vram_released=True`.

## Manual kernels — frozen held-out results (3 independent sessions, clean-process memory)

Scopes: **operator** = eager calls incl. output allocation and host dispatch; **kernel** = Triton launch into
pre-allocated outputs; **graph** = the same operator calls replayed from a CUDA graph (device time). Geomeans are over
the held-out workloads of one candidate × baseline × scope; ranges are min–max per-workload speedups.

| Candidate | vs | scope | correct | geomean | range |
| --- | --- | --- | --- | --- | --- |
| LayerNorm tuned `37ebb0b1ae32` | `F.layer_norm` | graph | 23/23 | **1.687×** | 1.01–2.43 |
| | `F.layer_norm` | operator | 23/23 | 0.718× | 0.50–1.28 |
| | untuned Triton | graph / kernel / operator | 23/23 | 1.020× / 1.005× / 1.001× | |
| MRoPE fused `8a89bbae67d5` | eager PyTorch | operator | 8/8 | **10.86×** | 3.60–21.94 |
| | eager PyTorch | graph | 8/8 | 17.43× | 2.92–45.65 |
| | untuned Triton MRoPE | operator / kernel / graph | 8/8 | 1.096× / 1.080× / 1.777× | |

MRoPE operator memory (incremental peak incl. outputs, vs eager PyTorch, held-out):

| dtype group | workloads | memory reduction |
| --- | --- | --- |
| float16 / bfloat16 | 6 | 76.5% – 83.3% (median 79.9%) |
| float32 | 2 | 37.3% – 40.3% |

LayerNorm memory vs `F.layer_norm`: 0.2% median (both allocate only the output; PyTorch also returns mean/rstd).

## Development-matrix baselines including `torch.compile` (1 session; all workloads correct)

`baseline-layernorm-20261101-072658-fe73` (83 workloads) and `baseline-mrope-20261101-073432-b143` (33 workloads).
`torch.compile` = default Inductor mode, `dynamic=False`, recompile limit raised so no shape silently falls back to
eager; compilation excluded from timing.

| Kernel | vs | operator | graph | memory reduction (median) |
| --- | --- | --- | --- | --- |
| LayerNorm tuned | `F.layer_norm` | 0.662× | 1.780× | 0.5% |
| LayerNorm tuned | `torch.compile` | 1.401× | 1.031× (0.61–2.56) | 0.0% |
| LayerNorm tuned | untuned Triton | 1.004× | 1.057× | 0.0% |
| MRoPE fused | eager PyTorch | 9.008× | 12.653× | 16-bit: 79.2% (72.9–83.3, n=20); fp32: 37.5% (31.4–47.4, n=13) |
| MRoPE fused | `torch.compile` | 1.800× | 2.192× (1.02–6.55) | 16-bit: 4.7% (0–17.1); fp32: 2.0% (0–4.8) |
| MRoPE fused | untuned Triton MRoPE | 1.054× | 2.086× | 0.0% |

Against a compiled baseline the MRoPE memory advantage largely disappears (Inductor also fuses away the
intermediates); the latency advantage remains (≈1.8–2.2×), mainly from computing cos/sin once per token tile and
reusing it across all q and k heads.

An earlier attempt of the LayerNorm dev baseline (`baseline-layernorm-20261101-072324-71ec`, status `aborted`) failed
every timing task because the worker inherited WSL's Windows `/mnt/c/…` PATH entries and Inductor's `nvcc` probe
raised `PermissionError`; the worker PATH is now restricted to Linux directories and the run was repeated.

## Agent (execution-guided search, Qwen2.5-Coder-3B Q4_K_M, 4 rounds × 2 candidates)

| Experiment | generations | duplicates of parent | invalid | compile fail | incorrect | valid & benchmarked | selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `optimize-layernorm-20261101-065514-7ba0` | 8 | 5 | 0 | 0 | 1 | 2 (both semantic copies of the parent) | `51eda03357a0` (generated) |
| `optimize-mrope-20261101-065741-f793` | 8 | 4 | 2 | 0 | 1 | 1 | `63a6c0f75c1d` (the untuned starting kernel) |

Held-out for the selections: LayerNorm `51eda03357a0` is 1.004× / 0.992× / 0.986× vs the manual kernel
(operator / kernel / graph) and 1.655× vs `F.layer_norm` (graph) — but its source differs from the starting kernel
only by recomputing the mask and moving a cast, i.e. it is the starting kernel. MRoPE kept the starting kernel
(9.53× operator / 9.72× graph vs eager on held-out; 0.97× / 0.56× vs the fused manual kernel). **The local 3B
generator did not produce a meaningful optimization in these runs.** Its non-duplicate edits were mostly wrong
(e.g. `Σ(x−mean)` instead of `Σ(x−mean)²` for the variance; renamed entry points).

Pilot finding that changed the methodology before these runs (`optimize-layernorm-20261101-064550-8fc8`): the
generator returned the starting kernel with an empty launch policy, which hashed differently and "improved"
0.855× → 0.948× in operator scope. Fixes: launch policies are canonicalized and candidates are deduplicated on their
AST; the agent's selection score is the graph scope (in the pilot the identical kernel timed against itself read
0.93× in operator scope and 1.00× in kernel/graph scope).

Demo (`demo-layernorm-20261101-065349-76c7`, scripted generator, real evaluator): round 1 normalised by the padded
block size → rejected (`ln-dev-003-M1-N768-float32: failed checks: y_numeric`, and other non-power-of-two N);
the feedback round returned the tuned kernel → correct on 83/83 dev workloads, 2.013× vs `F.layer_norm` (graph,
search suite) against 1.855× for the starting kernel.

## Critic comparison (LayerNorm, generator fixed, seeds 0 and 1, 2 rounds × 2 candidates)

| Arm | runs | generations (duplicates) | novel | valid | compile / incorrect | dev geomean of selection | held-out geomean | generator / critic tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none (control) | 2 | 8 (7) | 1 | 0 | 0 / 1 | 1.859× | 1.666× | 10,664 / 0 |
| self (local Qwen) | 2 | 8 (3) | 5 | 1 | 2 / 2 | 1.855× | 1.670× | 13,399 / 3,952 |
| OpenAI | — | — | — | — | — | not run | not run | `critics.openai_model` not configured |
| Claude | — | — | — | — | — | not run | not run | `critics.anthropic_model` not configured |
| Gemini | — | — | — | — | — | not run | not run | `critics.google_model` not configured |

Self-critique made the 3B coder change code far more often (5 novel kernels vs 1) but 4 of 5 changes failed and the
one valid change was within measurement noise of its parent. Two seeds per arm is too few for a statistical claim.

## 7B generator (Qwen2.5-Coder-7B-Instruct Q4_K_M, same settings as the 3B runs)

`bash scripts/run_7b.sh` (log: `runs/run_7b.log`) repeated the agent runs, held-out evaluation and critic comparison
with only `generator.*` changed. The evaluator, starting kernels, workloads, rounds, budgets and seeds were identical.

| Experiment | generations | duplicates of parent | invalid | compile fail | incorrect | valid & benchmarked | selected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `optimize-layernorm-20261101-185026-9d60` | 8 | 4 | 0 | 0 | 1 | 3 (all cosmetic copies of the parent) | `589441de2e46` (the starting kernel) |
| `optimize-mrope-20261101-185340-2504` | 8 | 2 | 1 (syntax error) | 0 | 4 | 1 (cosmetic copy) | `c0aeab5ca794` (generated; cosmetic copy) |

* The 3 "valid" LayerNorm kernels split an expression into temporaries (`x_sum = tl.sum(x); mean = x_sum / N`) or add
  unused imports; the only real edit (`xc = x - mean` without the mask) lets padded lanes add `mean²` to the variance
  and was rejected on `ln-dev-003-M1-N768-float32` (`y_numeric`). Held-out: 23/23 correct, 1.657× vs `F.layer_norm`
  (graph) / 0.723× (operator), 0.995× / 0.997× / 0.984× vs the tuned manual kernel (operator / kernel / graph).
* The selected MRoPE kernel only names two intermediate results (`x_out_real = x1*cos - x2*sin`); launch policy and math
  are identical to the starting kernel. Its dev "win" (6.830× vs 6.738×) is noise: on held-out, timed in the same task
  as the starting kernel, it is **0.987× / 0.991× / 0.999×** of it (operator / kernel / graph). Held-out: 8/8 correct,
  8.744× / 8.819× vs eager (operator / graph), median memory reduction 79.2% vs eager, 0.946× / 0.905× / 0.552× vs the
  fused manual kernel. The other MRoPE edits failed `q_out_numeric` / `k_out_numeric` / `finite` or did not parse.

Critic comparison with the 7B (`critics-layernorm-20261101-190144-d62e`, LayerNorm, seeds 0 and 1, 2 rounds × 2):

| Arm | runs | generations (duplicates) | novel | valid | compile / incorrect | dev geomean of selection | held-out geomean | generator / critic tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none (control) | 2 | 8 (3) | 5 | 4 | 0 / 1 | 1.892× | 1.621× | 11,490 / 0 |
| self (local 7B) | 2 | 8 (2) | 6 | 2 | 3 / 1 | 1.820× | 1.628× | 12,724 / 3,744 |

All four runs selected the starting kernel; every valid generated kernel was a cosmetic rewrite (or an equivalent
re-masking) and measured slower than its parent. Self-critique led the 7B to call `tl.reduce` with invented
signatures (e.g. `tl.reduce(x, tl.sum, axis=0, mask=mask)`), which is where its 3 compile failures came from.

3B vs 7B, over the same 32 generations each (agent runs + critic comparison):

| | 3B | 7B |
| --- | --- | --- |
| Server VRAM at 4,096 context / generation speed | ≈2,211 MiB / ~140 tokens/s | ≈4,681 MiB / ~78 tokens/s |
| Exact or semantic duplicates of the parent | 19 | 11 |
| Invalid / compile fail / incorrect | 2 / 2 / 5 | 1 / 3 / 7 |
| Novel kernels that passed every dev workload | 4 | 10 |
| Kernels that were faster than the starting kernel beyond noise | 0 | 0 |

The 7B copies less and produces more kernels that pass, but every one that passed is the starting kernel with
cosmetic edits, and every real algorithmic edit was wrong. **The 7B did not produce a meaningful optimization either.**

Measurement noise exposed by these runs: identical device code timed in separate evaluations varies far more than
when timed side by side. The starting LayerNorm kernel's search-suite geomean (graph scope vs eager) ranged
1.759–1.900× across its 11 graph-scope agent/demo evaluations today; the cosmetic copy `f749f1e7ea4b` ranged 1.725–1.814× across four; the MRoPE starting kernel read
7.46× in the 3B run and 6.74× in the 7B run. Paired in one task, the MRoPE copy was within 1.3% of the starting kernel
in every scope. Selection still takes the highest unpaired dev geomean (kept unchanged so the 3B and 7B runs use the
same rule), which is how cosmetic copies get "selected"; see the next steps.

## RL extension (GRPO + LoRA, separate environment)

Policy: `Qwen/Qwen2.5-Coder-0.5B-Instruct` @ `ea3f2471` with LoRA r=8 on q/k/v/o (not the 3B GGUF generator, not
Qwen3-Coder). trl 1.17.1, peft 0.21.2, transformers 5.22.1; 2 completions per prompt, β=0.04 KL, gradient
checkpointing, bf16, no vLLM. Budgets were validated before training with the real tokenizer: LayerNorm prompts are
632–639 tokens (≤ 768), but a complete tuned LayerNorm answer is 569 tokens, so the completion budget was raised from
512 to 640 instead of truncating valid kernels. MRoPE (≈1,233-token prompts, 931-token answers) is excluded from RL.
Rewards came only from the trusted evaluator (smoke correctness + 2 search workloads, operator-scope timing) with the
policy resident, i.e. **in-training** measurements; base vs adapter was evaluated afterwards with the model unloaded,
on 2 held-out instruction variants × 4 samples, equal generation budgets.

| Run | What happened | Held-out prompts: correct | Median speedup of correct samples vs `F.layer_norm` |
| --- | --- | --- | --- |
| `runs/rl/20261101-024005` (`--smoke`, 2 GRPO steps, no warm-start) | all 4 training completions failed (3 unparseable, 1 invalid policy) → every group had zero reward variance → loss 0, grad-norm 0: **no learning signal; the adapter equals its initialization** | base 0/4, adapter 0/4 | — |
| `runs/rl/20261101-024707` (24-step supervised warm-start, labeled *not RL*, then 4 GRPO steps) | steps 1–3: all completions failed (only the KL term produced gradients); step 4: one correct completion (reward −0.21, 0.864×) → one real policy-gradient update (grad-norm 0.23) | base 0/8, SFT + GRPO adapter 5/8 | 0.859× |
| `runs/rl/20261101-025153` (`--evaluate-only` on the SFT-only adapter, same prompts and seeds) | attribution check | base 0/8, SFT-only adapter 5/8 | 0.782× |

Conclusion: real optimizer updates and adapters were saved and evaluated, but **RL produced no measurable
improvement**. The correctness gain is explained by the supervised warm-start (identical 5/8 with or without the GRPO
steps); with one informative GRPO step and n=5 correct samples the speed difference (0.86× vs 0.78×) is within
noise. The first run's `status.json` predates the end-of-run "no learning signal" diagnosis that was added afterwards.

## Targets (aspirational) vs measured

| Target | Measured | Status |
| --- | --- | --- |
| LayerNorm 1.22× (≈18.0% lower latency) vs `F.layer_norm` | held-out geomean 1.687× in graph (device-time) scope, 0.718× in eager operator scope | met only in graph scope; in eager use Triton's ~15 µs Python launch cost dominates. The untuned Triton kernel reaches ~1.67× too, so tuning itself adds ≈2%. |
| MRoPE 60% lower operator memory | held-out: 76.5–83.3% for fp16/bf16 vs eager PyTorch, 37–40% for fp32; dev: 4.7% (16-bit) / 2.0% (fp32) median vs `torch.compile` | met only for 16-bit dtypes against the eager baseline; not for fp32, and not against `torch.compile` |
| 2.2× generated-kernel speedup | 3B and 7B: every selected or valid generated kernel is the starting kernel or a cosmetic copy of it (≈1.00× vs it when timed side by side) | **not met** by generation (3B or 7B) |

## Unresolved dependencies and next commands

* Hosted critics: set exact model IDs in `config.yaml` (`critics.openai_model`, `critics.anthropic_model`,
  `critics.google_model`), install `pip install '.[critics]'`, export the keys, then
  `python -m kernelforge compare-critics --config config.yaml --seeds 0,1,2 --heldout`.
* Container isolation: enable Docker Desktop → Settings → Resources → WSL integration (Ubuntu) and GPU support, then
  `docker build -f Dockerfile.worker -t kernelforge-worker:latest .` and set `worker.transport: local_container`.
* CI / Pages: `git init`, push to GitHub, register this PC as a self-hosted runner with labels
  `self-hosted, linux, gpu, rtx5060ti`, set `KERNELFORGE_PAGES_ENABLED=true` to allow the Pages workflow.
* Selection rule: a generated candidate should replace its parent only when a paired, same-task timing against the
  parent clears a noise margin, and cosmetic copies should be caught by comparing compiled code rather than the AST.
  Neither is implemented yet; until then, treat any "generated" selection within ~10% of its parent as unproven.
* Generator: `config.yaml` now points at the 7B; the 3B pin is kept in a comment there and in `config.example.yaml`.
* Cleaner timing: close GPU-using desktop apps (the animated wallpaper's `mpv.exe`, Chrome) before `run_all.sh`.
