"""Optional genuine RL extension: GRPO + LoRA on a small trainable coder, rewarded only by the
trusted KernelForge evaluator. Runs in its own environment (TRL/PEFT/transformers), never
required by the core project, and never makes claims without saved updates + evaluation.

Phases (strictly sequential on the shared 8 GB GPU):
  1. preflight: free VRAM / host RAM; prompt and reference-kernel token budgets
  2. optional supervised warm-start (separately labeled) on verified kernels
  3. GRPO: generate -> synchronize -> reward via evaluator subprocesses -> optimizer step.
     The policy stays resident during reward timing, so those are *in-training* measurements.
  4. save adapter, unload trainer
  5. base vs adapter on held-out prompt variants with equal generation budgets; generation
     finishes and the model is unloaded before the clean evaluator run.

Reward: -1 for invalid / compile / timeout / correctness failure;
        clip(log2(baseline_ms / candidate_ms), -0.5, 2.0) for a correct candidate.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from .config import load_config
from .contracts import utc_now
from .kernels import module_for

INSTRUCTIONS = [  # train uses the first four, evaluation the last two (held-out prompt variants)
    "Write a faster version of this kernel that stays exactly correct.",
    "Improve memory access or launch efficiency of this kernel without changing its results.",
    "Rewrite this kernel to reduce redundant work while keeping the exact contract.",
    "Tune this kernel's tiling and launch policy for the GPU; keep it correct.",
    "Produce an optimized, correct kernel for this operator.",
    "Make this kernel faster on this GPU while preserving numerical behavior.",
]


class RLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    model_revision: str | None = None
    operators: list[str] = ["layernorm"]
    output_dir: str = "runs/rl"
    seed: int = 0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    learning_rate: float = 1e-5
    max_steps: int = 4
    num_generations: int = 2
    prompts_per_step: int = 1
    max_prompt_tokens: int = 768
    max_completion_tokens: int = 640
    temperature: float = 0.9
    beta: float = 0.04              # KL coefficient against the reference policy (0 disables the ref model)
    gradient_checkpointing: bool = True
    bf16: bool = True
    save_steps: int = 2
    min_free_vram_gib: float = 3.5
    min_free_ram_gib: float = 3.0
    reward_timing_samples: int = 10
    stop_if_no_signal_steps: int = 4  # all groups identical for this many steps -> stop with a diagnosis
    warm_start_sft_steps: int = 0     # >0 enables the separately labeled supervised warm-start
    eval_samples_per_prompt: int = 4
    eval_max_new_tokens: int = 640


def load_rl_config(path: str) -> RLConfig:
    return RLConfig(**(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}))


# ---------------------------------------------------------------------------

def build_prompt(operator: str, instruction: str) -> list[dict]:
    mod = module_for(operator)
    parent = mod.BASIC_SPEC
    user = (f"{mod.SPEC_DOC}\n\nEntry point (exact):\n{mod.SIGNATURE_DOC}\n\nOnly `import triton` and "
            f"`import triton.language as tl`; every function @triton.jit; no host code.\n\nCurrent kernel:\n"
            f"```python\n{parent.kernel_source}```\nLaunch policy JSON: {json.dumps(parent.launch.model_dump())}\n\n"
            f"{instruction} Reply with one ```python block, one ```json launch policy block, then EXPLANATION:.")
    return [{"role": "system", "content": "You are an expert Triton GPU kernel engineer."},
            {"role": "user", "content": user}]


def prompt_token_count(tok, msgs: list[dict]) -> int:
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out  # transformers 5 returns a BatchEncoding
    return len(ids)


def reference_completion(operator: str) -> str:
    mod = module_for(operator)
    spec = mod.MANUAL_SPEC
    return (f"```python\n{spec.kernel_source}```\n```json\n{json.dumps(spec.launch.model_dump())}\n```\n"
            f"EXPLANATION: {spec.explanation}")


class EvaluatorReward:
    """Callable reward for TRL. Every completion goes through the same trusted evaluator."""

    __name__ = "kernelforge_evaluator_reward"

    def __init__(self, kf_cfg, rl: RLConfig, log_path: Path, in_training: bool = True):
        from .executor import Executor
        from .store import Store

        self.kf_cfg = kf_cfg
        self.rl = rl
        self.store = Store(kf_cfg.runs_path())
        self.executor = Executor(kf_cfg, self.store)
        self.log_path = log_path
        self.in_training = in_training
        self.experiment_id = f"rl-reward-{time.strftime('%Y%m%d-%H%M%S')}"
        self.history: list[list[float]] = []

    def score(self, operator: str, text: str) -> tuple[float, dict]:
        from .providers import parse_generation
        from .source_policy import check_source
        from .workloads import manifest

        mod = module_for(operator)
        spec, problems, _ = parse_generation(text, operator, "full", mod.BASIC_SPEC, truncated=False)
        if spec is None:
            return -1.0, {"stage": "parse", "problems": problems[:2]}
        rep = check_source(spec.kernel_source, operator, spec.entry_point)
        bad = rep.problems + spec.launch.validate_for(operator)
        if bad:
            return -1.0, {"stage": "validation", "problems": bad[:2]}
        smoke = manifest(operator, "smoke").cases
        corr = self.executor.run_task(self.executor.make_task(self.experiment_id, "correctness", candidate=spec,
                                                              workloads=smoke, split="smoke"), 300)
        if corr.status != "ok" or any(c.status != "passed" for c in corr.correctness if c.status != "skipped"):
            stage = corr.failure_stage or "correctness"
            return -1.0, {"stage": stage, "hash": spec.short_hash}
        timing_cases = manifest(operator, "search").cases[:2]
        settings = self.kf_cfg.benchmark_settings(warmup_iterations=5, samples=self.rl.reward_timing_samples,
                                                  calls_per_sample=10, timing_scopes=["operator"])
        res = self.executor.run_task(self.executor.make_task(
            self.experiment_id, "timing", candidate=spec, workloads=timing_cases, split="search",
            baselines=[mod.PRIMARY_BASELINE], benchmark=settings, generator_resident=self.in_training,
            measurement_kind="in_training" if self.in_training else "clean"), 600)
        ratios = []
        for t in res.timing:
            for p in t.paired:
                if p.baseline_id == mod.PRIMARY_BASELINE:
                    ratios.append(p.speedup_of_medians)
        if len(ratios) != len(timing_cases):
            return -1.0, {"stage": "benchmark", "hash": spec.short_hash}
        g = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        return max(-0.5, min(2.0, math.log2(g))), {"stage": "ok", "hash": spec.short_hash, "speedup": g}

    def __call__(self, prompts=None, completions=None, operator=None, **kwargs) -> list[float]:
        import torch

        torch.cuda.synchronize()  # no training kernels in flight while the evaluator times
        rewards, details = [], []
        ops = operator or ["layernorm"] * len(completions)
        for comp, op in zip(completions, ops):
            text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
            r, d = self.score(op, text)
            rewards.append(r)
            details.append(d)
        n = self.rl.num_generations
        groups = [rewards[i:i + n] for i in range(0, len(rewards), n)]
        self.history.append([statistics.pstdev(g) if len(g) > 1 else 0.0 for g in groups])
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": utc_now(), "rewards": rewards, "details": details,
                                 "measurement_kind": "in_training" if self.in_training else "clean"}) + "\n")
        return rewards


def preflight(rl: RLConfig) -> dict:
    import psutil
    import torch

    out: dict[str, Any] = {"cuda": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        out["supported"] = False
        out["reason"] = "CUDA unavailable"
        return out
    free, total = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory().available
    out.update(free_vram_gib=round(free / 2**30, 2), total_vram_gib=round(total / 2**30, 2),
               free_ram_gib=round(ram / 2**30, 2))
    reasons = []
    if free / 2**30 < rl.min_free_vram_gib:
        reasons.append(f"free VRAM {free / 2**30:.2f} GiB < required {rl.min_free_vram_gib} GiB")
    if ram / 2**30 < rl.min_free_ram_gib:
        reasons.append(f"available host RAM {ram / 2**30:.2f} GiB < required {rl.min_free_ram_gib} GiB")
    out["supported"] = not reasons
    out["reason"] = "; ".join(reasons) or None
    return out


def _filter_fields(cls, values: dict) -> tuple[dict, list[str]]:
    names = {f.name for f in dataclasses.fields(cls)}
    kept = {k: v for k, v in values.items() if k in names}
    return kept, sorted(set(values) - names)


def generate_samples(model, tokenizer, prompts: list[list[dict]], n: int, max_new_tokens: int, temperature: float,
                     seed: int) -> list[list[str]]:
    import torch

    outs = []
    for i, msgs in enumerate(prompts):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        torch.manual_seed(seed + i)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens,
                                 num_return_sequences=n, pad_token_id=tokenizer.eos_token_id)
        outs.append([tokenizer.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True) for g in gen])
    return outs


def evaluate_base_vs_adapter(rl: RLConfig, kf_cfg, adapter_dir: Path | None, out_dir: Path) -> dict:
    """Identical held-out prompt variants and generation budgets for both models; samples are saved and
    the model is unloaded before the clean evaluator run."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [(op, build_prompt(op, ins)) for op in rl.operators for ins in INSTRUCTIONS[4:]]
    samples: dict[str, list] = {}
    for label in ("base", "adapter"):
        if label == "adapter" and adapter_dir is None:
            continue
        tok = AutoTokenizer.from_pretrained(rl.model, revision=rl.model_revision)
        model = AutoModelForCausalLM.from_pretrained(rl.model, revision=rl.model_revision, dtype=torch.bfloat16,
                                                     device_map="cuda")
        if label == "adapter":
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
        texts = generate_samples(model, tok, [p for _, p in prompts], rl.eval_samples_per_prompt,
                                 rl.eval_max_new_tokens, rl.temperature, rl.seed + 1000)
        samples[label] = [{"operator": op, "texts": t} for (op, _), t in zip(prompts, texts)]
        del model
        gc.collect()
        torch.cuda.empty_cache()
    (out_dir / "eval_samples.json").write_text(json.dumps(samples, indent=1), encoding="utf-8")
    reward = EvaluatorReward(kf_cfg, rl, out_dir / "eval_rewards.jsonl", in_training=False)
    result: dict[str, Any] = {"prompts": "held-out instruction variants", "samples_per_prompt": rl.eval_samples_per_prompt,
                              "max_new_tokens": rl.eval_max_new_tokens}
    for label, items in samples.items():
        scores, speedups = [], []
        for it in items:
            for text in it["texts"]:
                r, d = reward.score(it["operator"], text)
                scores.append(r)
                if d.get("speedup"):
                    speedups.append(d["speedup"])
        result[label] = {"n": len(scores), "correct_rate": sum(s > -1 for s in scores) / max(len(scores), 1),
                         "mean_reward": statistics.fmean(scores) if scores else None,
                         "median_speedup_correct": statistics.median(speedups) if speedups else None,
                         "speedups": speedups}
    return result


def main(args) -> int:
    import torch

    rl = load_rl_config(args.config)
    if args.smoke:
        rl.max_steps = min(rl.max_steps, 2)
        rl.eval_samples_per_prompt = min(rl.eval_samples_per_prompt, 2)
    kf_cfg = load_config(args.kf_config)
    run_dir = Path(rl.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"started": utc_now(), "model": rl.model, "model_revision": rl.model_revision,
                              "note": "RL policy is this small model with a LoRA adapter, not the 3B GGUF generator "
                                      "and not Qwen3-Coder.", "config": rl.model_dump(), "phase": "preflight"}

    def save() -> None:
        (run_dir / "status.json").write_text(json.dumps(status, indent=1, default=str), encoding="utf-8")

    if args.evaluate_only:
        status["phase"] = "evaluate"
        status["evaluation"] = evaluate_base_vs_adapter(rl, kf_cfg, Path(args.evaluate_only), run_dir)
        save()
        print(json.dumps(status["evaluation"], indent=1, default=str))
        return 0

    pf = preflight(rl)
    status["preflight"] = pf
    if not pf["supported"]:
        status.update(phase="unsupported", result="not run", reason=pf["reason"])
        save()
        print(f"RL run not supported on this machine right now: {pf['reason']} (see {run_dir / 'status.json'})")
        return 2

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import trl

    status["versions"] = {"trl": trl.__version__, "torch": torch.__version__}
    tok = AutoTokenizer.from_pretrained(rl.model, revision=rl.model_revision)
    # Budget validation before any training: prompts must fit, and a complete reference kernel must fit
    # the completion budget (complexity is reduced instead of silently truncating valid kernels).
    rows = []
    for op in rl.operators:
        ref_tokens = len(tok(reference_completion(op))["input_ids"])
        if ref_tokens > rl.max_completion_tokens:
            status.update(phase="unsupported", reason=f"{op}: reference kernel needs {ref_tokens} tokens > "
                                                      f"max_completion_tokens={rl.max_completion_tokens}")
            save()
            print(status["reason"])
            return 2
        for ins in INSTRUCTIONS[:4]:
            msgs = build_prompt(op, ins)
            n = prompt_token_count(tok, msgs)
            if n > rl.max_prompt_tokens:
                status.update(phase="unsupported", reason=f"{op}: prompt needs {n} tokens > {rl.max_prompt_tokens}")
                save()
                print(status["reason"])
                return 2
            rows.append({"prompt": msgs, "operator": op})
    status["budgets"] = {"train_prompts": len(rows)}
    dataset = Dataset.from_list(rows)
    lora = LoraConfig(r=rl.lora_rank, lora_alpha=rl.lora_alpha, lora_dropout=rl.lora_dropout,
                      target_modules=rl.lora_target_modules, task_type="CAUSAL_LM")
    from transformers import AutoModelForCausalLM

    # Loaded explicitly so the pinned revision and dtype do not depend on trainer kwarg names.
    model_ref: Any = AutoModelForCausalLM.from_pretrained(
        rl.model, revision=rl.model_revision, dtype=torch.bfloat16 if rl.bf16 else torch.float32)

    if rl.warm_start_sft_steps > 0:
        status["phase"] = "sft_warm_start"
        save()
        from trl import SFTConfig, SFTTrainer

        sft_rows = [{"prompt": r["prompt"], "completion": [{"role": "assistant",
                                                            "content": reference_completion(r["operator"])}]}
                    for r in rows]
        sft_args, dropped = _filter_fields(SFTConfig, dict(
            output_dir=str(run_dir / "sft"), max_steps=rl.warm_start_sft_steps, per_device_train_batch_size=1,
            learning_rate=rl.learning_rate * 10, logging_steps=1, report_to=[], bf16=rl.bf16, seed=rl.seed,
            gradient_checkpointing=rl.gradient_checkpointing, save_strategy="no"))
        sft = SFTTrainer(model=model_ref, args=SFTConfig(**sft_args), train_dataset=Dataset.from_list(sft_rows),
                         processing_class=tok, peft_config=lora)
        sft.train()
        sft.save_model(str(run_dir / "sft_adapter"))
        model_ref = sft.model
        status["sft"] = {"steps": rl.warm_start_sft_steps, "label": "supervised warm-start (not RL)",
                         "dropped_args": dropped}
        lora = None

    status["phase"] = "grpo"
    save()
    reward = EvaluatorReward(kf_cfg, rl, run_dir / "rewards.jsonl", in_training=True)
    grpo_values = dict(
        output_dir=str(run_dir / "grpo"), learning_rate=rl.learning_rate, max_steps=rl.max_steps,
        per_device_train_batch_size=rl.prompts_per_step * rl.num_generations, gradient_accumulation_steps=1,
        num_generations=rl.num_generations, max_prompt_length=rl.max_prompt_tokens,
        max_completion_length=rl.max_completion_tokens, temperature=rl.temperature, beta=rl.beta,
        gradient_checkpointing=rl.gradient_checkpointing, bf16=rl.bf16, logging_steps=1, save_steps=rl.save_steps,
        seed=rl.seed, report_to=[], use_vllm=False, log_completions=False,
        mask_truncated_completions=True)  # truncated kernels are never trained on as if complete
    grpo_args, dropped = _filter_fields(GRPOConfig, grpo_values)
    status["grpo_args_dropped_by_installed_trl"] = dropped

    from transformers import TrainerCallback

    class SignalGuard(TrainerCallback):
        def on_step_end(self, args_, state, control, **kw):
            k = rl.stop_if_no_signal_steps
            recent = reward.history[-k:]
            if len(reward.history) >= k and all(all(s == 0.0 for s in step) for step in recent):
                status["diagnosis"] = (f"no learning signal: every reward group was identical for {k} steps "
                                       "(all-invalid completions give no advantage). Consider warm_start_sft_steps>0 "
                                       "or simpler tasks.")
                control.should_training_stop = True
            return control

    trainer = GRPOTrainer(model=model_ref, reward_funcs=[reward], args=GRPOConfig(**grpo_args), train_dataset=dataset,
                          peft_config=lora, processing_class=tok, callbacks=[SignalGuard()])
    t0 = time.monotonic()
    trainer.train()
    adapter_dir = run_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    steps_with_signal = sum(1 for step in reward.history if any(s > 0 for s in step))
    grad_norms = [h["grad_norm"] for h in trainer.state.log_history if "grad_norm" in h]
    status["training"] = {"global_steps": trainer.state.global_step, "seconds": round(time.monotonic() - t0, 1),
                          "log_history": trainer.state.log_history, "adapter_dir": str(adapter_dir),
                          "reward_steps": len(reward.history), "reward_steps_with_signal": steps_with_signal,
                          "nonzero_grad_steps": sum(1 for g in grad_norms if g and g > 0),
                          "reward_measurements": "in_training (policy resident during reward timing)"}
    if steps_with_signal == 0 and not status.get("diagnosis"):
        status["diagnosis"] = ("no learning signal: every reward group had identical rewards, so GRPO advantages and "
                               "gradients were zero and the saved adapter equals its initialization. Consider "
                               "warm_start_sft_steps > 0 (separately labeled) or simpler tasks.")
    save()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    status["phase"] = "evaluation"
    save()
    status["evaluation"] = evaluate_base_vs_adapter(rl, kf_cfg, adapter_dir, run_dir)
    status["phase"] = "done"
    status["finished"] = utc_now()
    save()
    print(json.dumps({"run_dir": str(run_dir), "steps": status["training"]["global_steps"],
                      "evaluation": {k: {kk: vv for kk, vv in v.items() if kk != "speedups"}
                                     for k, v in status["evaluation"].items() if isinstance(v, dict)},
                      "diagnosis": status.get("diagnosis")}, indent=1, default=str))
    return 0
