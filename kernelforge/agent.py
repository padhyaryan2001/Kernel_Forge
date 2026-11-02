"""Execution-guided search (iterative refinement) and the experiment drivers built on it.

This is not reinforcement learning: the generator's parameters never change here. Each
round: start the managed coder, generate a small population, stop it and confirm VRAM is
released, validate + evaluate every candidate with the trusted worker, feed compiler /
numerical / timing results (and optionally a critic's structured notes) into the next
prompt. The best valid development candidate is kept even if later revisions fail, and the
experiment can resume from SQLite after an interruption.
"""
from __future__ import annotations

import itertools
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .config import KernelForgeConfig
from .contracts import (CandidateSpec, CritiqueResult, LaunchPolicy, LaunchTier, Manifest, canonical_json,
                        geomean, sha256_text, utc_now)
from .executor import CandidateEvaluator, Executor
from .kernels import module_for
from .providers import (GeneratorError, LlamaServer, OpenAICompatibleClient, build_generation_messages,
                        critique_payload, parse_generation, run_critic)
from .source_policy import check_source
from .store import RETRYABLE, Store, new_id
from .workloads import manifest

GOOD_STATES = ("benchmarked", "critiqued", "revised", "selected", "heldout_evaluated")


def freeze_manifests(store: Store, experiment_id: str, operator: str) -> dict[str, str]:
    out = {}
    for split in ("smoke", "search", "dev", "heldout"):
        m = manifest(operator, split)
        store.write_json(store.exp_dir(experiment_id) / "manifests" / f"{split}.json",
                         {"manifest": m.model_dump(), "manifest_hash": m.manifest_hash, "frozen_at": utc_now()})
        out[split] = m.manifest_hash
    return out


def load_frozen_manifest(store: Store, experiment_id: str, split: str) -> Manifest:
    path = store.exp_dir(experiment_id) / "manifests" / f"{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    m = Manifest.model_validate(data["manifest"])
    if m.manifest_hash != data["manifest_hash"]:
        raise RuntimeError(f"frozen {split} manifest hash mismatch: the file was modified")
    return m


def gpu_description(executor: Executor) -> str:
    probe = executor.probe()
    hw = probe.get("hardware") or {}
    if not hw.get("gpu"):
        return "NVIDIA GPU (details unavailable)"
    return (f"{hw['gpu']} (compute capability {hw.get('compute_capability')}, {hw.get('sm_count')} SMs, "
            f"{(hw.get('total_memory_bytes') or 0) / 2**30:.1f} GiB, L2 {(hw.get('l2_cache_bytes') or 0) / 2**20:.0f} MiB)")


def summary_line(summary: dict, baseline: str) -> str:
    if summary.get("geomean_speedup"):
        per = ", ".join(f"{w.split('-', 3)[-1]}: {s:.2f}x" for w, s in list(summary["speedups"].items())[:6])
        return (f"correct on all dev workloads; geomean speedup vs {baseline} {summary['geomean_speedup']:.3f}x "
                f"({per})")
    return "correct" if summary.get("correct_all") else "not correct"


class SearchAgent:
    def __init__(self, cfg: KernelForgeConfig, store: Store, operator: str, experiment_id: str | None = None,
                 arm: str = "none", seed: int = 0, client=None, kind: str = "optimize",
                 parent_id: str | None = None, log=print):
        self.cfg, self.store, self.operator = cfg, store, operator
        self.arm, self.seed, self.kind, self.log = arm, seed, kind, log
        self.executor = Executor(cfg, store)
        self.evaluator = CandidateEvaluator(cfg, store, self.executor)
        self.client = client or OpenAICompatibleClient(cfg.generator.base_url, cfg.generator.model,
                                                       cfg.generator.request_timeout_s)
        self.managed = (cfg.generator.managed_process and cfg.generator.runtime == "llama_cpp"
                        and getattr(self.client, "kind", "") == "openai_compatible")
        self.generator_resident = (not self.managed and getattr(self.client, "kind", "") == "openai_compatible"
                                   and any(h in cfg.generator.base_url for h in ("127.0.0.1", "localhost")))
        self.mod = module_for(operator)
        self.baselines = [cfg.search.primary_baseline, "triton_basic"]
        if experiment_id and store.experiment(experiment_id):
            self.experiment_id = experiment_id
            exp = store.experiment(experiment_id)
            self.arm, self.seed = exp["arm"] or arm, exp["seed"] if exp["seed"] is not None else seed
            self.log(f"[resume] {experiment_id} at round {exp['progress'].get('round', 0)}")
        else:
            self.experiment_id = experiment_id or new_id(f"{kind}-{operator}")
            store.create_experiment(self.experiment_id, operator, kind, self._config_snapshot(), arm=arm, seed=seed,
                                    parent_id=parent_id)
            hashes = freeze_manifests(store, self.experiment_id, operator)
            self._save_progress(round=0, manifest_hashes=hashes, tokens=0, started=time.time(),
                                generator=self._generator_identity())

    # -- bookkeeping ----------------------------------------------------------
    def _config_snapshot(self) -> dict:
        snap = self.cfg.model_dump()
        snap["critics"].pop("pricing", None)
        return snap

    def _generator_identity(self) -> dict:
        g = self.cfg.generator
        return {"client": getattr(self.client, "kind", "unknown"), "serving_alias": g.model,
                "source_model": g.source_model if getattr(self.client, "kind", "") == "openai_compatible"
                else getattr(self.client, "model", "scripted"),
                "source_file": g.source_file, "source_revision": g.source_revision, "sha256": g.source_sha256,
                "quantization": g.quantization, "context_tokens": g.context_tokens,
                "max_output_tokens": g.max_output_tokens, "temperature": g.temperature,
                "search_mode": self.cfg.search.mode}

    @property
    def progress(self) -> dict:
        return self.store.experiment(self.experiment_id)["progress"]

    def _save_progress(self, **updates: Any) -> None:
        p = self.store.experiment(self.experiment_id)["progress"] if self.store.experiment(self.experiment_id) else {}
        p.update(updates)
        self.store.update_experiment(self.experiment_id, progress=p)

    def _event(self, name: str, **data: Any) -> None:
        path = self.store.exp_dir(self.experiment_id) / "events.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": utc_now(), "event": name, **data}, default=str) + "\n")

    # -- generator lifecycle -----------------------------------------------------
    @contextmanager
    def generator_session(self) -> Iterator[None]:
        if not self.managed:
            yield
            return
        server = LlamaServer(self.cfg.generator, self.store.exp_dir(self.experiment_id) / "generator" / "server.log",
                             self.cfg.worker.gpu_index)
        metrics = server.start()
        self._event("generator_started", **metrics)
        self.log(f"[generator] started pid={metrics['pid']} in {metrics['load_seconds']}s, "
                 f"server VRAM ~{metrics['server_vram_mib']} MiB")
        try:
            yield
        finally:
            stopped = server.stop()
            self._event("generator_stopped", **stopped)
            self.log(f"[generator] stopped in {stopped['shutdown_seconds']}s; VRAM released={stopped['vram_released']}")
            if self.cfg.generator.stop_before_gpu_evaluation and stopped.get("vram_released") is False:
                raise GeneratorError("GPU memory did not recover after stopping the generator; refusing to benchmark")

    # -- candidates ------------------------------------------------------------
    def seed_candidate(self) -> str:
        """Round 0: the untuned Triton baseline is the common starting candidate for every arm."""
        spec = self.mod.BASIC_SPEC
        cid, new = self.store.add_candidate(self.experiment_id, spec, 0, None, {"role": "starting_candidate"})
        if new:
            self.store.transition(self.experiment_id, cid, "generated", "starting candidate")
        return cid

    def best(self) -> dict | None:
        good = [c for c in self.store.candidates(self.experiment_id)
                if c["state"] in GOOD_STATES and c["dev_score"] is not None]
        return max(good, key=lambda c: c["dev_score"]) if good else None

    def evaluate_candidate(self, cid: str) -> dict:
        st = self.store
        cand = st.candidate(self.experiment_id, cid)
        spec = st.load_spec(self.experiment_id, cid)
        if cand["state"] in RETRYABLE:
            st.transition(self.experiment_id, cid, "validated", "retry")
            st.clear_evaluations(self.experiment_id, cid)
        elif cand["state"] == "validated":  # interrupted mid-evaluation
            st.clear_evaluations(self.experiment_id, cid)
        elif cand["state"] == "generated":
            report = check_source(spec.kernel_source, spec.operator, spec.entry_point)
            problems = report.problems + spec.launch.validate_for(spec.operator)
            if problems:
                st.transition(self.experiment_id, cid, "invalid", "; ".join(problems), failure_stage="validation")
                return {"candidate": cid, "status": "invalid", "stage": "validation", "details": problems[:3]}
            st.transition(self.experiment_id, cid, "validated")
        else:
            return {"candidate": cid, "status": cand["state"]}
        dev = load_frozen_manifest(st, self.experiment_id, self.cfg.search.correctness_suite)
        timing = load_frozen_manifest(st, self.experiment_id, self.cfg.search.benchmark_suite).cases
        self.log(f"[eval] {cid} ({spec.origin}, round {cand['round']}): correctness on {len(dev.cases)} dev cases, "
                 f"timing on {len(timing)} {self.cfg.search.benchmark_suite} cases")
        s = self.evaluator.evaluate(self.experiment_id, cid, spec, dev, timing, self.baselines, sessions=1,
                                    memory=False, generator_resident=self.generator_resident)
        details = [f"{f['workload_id']}: {f['stage']}: {f['message']}" for f in s["failures"][:3]]
        stage = s["stage"]
        if stage == "validation":
            st.transition(self.experiment_id, cid, "invalid", "; ".join(details), failure_stage="validation")
            status = "invalid"
        elif stage == "compile":
            st.transition(self.experiment_id, cid, "compile_failed", "; ".join(details), failure_stage="compile")
            status = "compile_failed"
        elif stage == "timeout":
            st.transition(self.experiment_id, cid, "timed_out", "; ".join(details), failure_stage="timeout")
            status = "timed_out"
        elif stage == "worker":
            st.transition(self.experiment_id, cid, "worker_error", "; ".join(details), failure_stage="worker")
            status = "worker_error"
        elif not s["correct_all"]:
            st.transition(self.experiment_id, cid, "compiled")
            st.transition(self.experiment_id, cid, "incorrect", "; ".join(details), failure_stage=stage or "correctness")
            status = "incorrect"
        else:
            st.transition(self.experiment_id, cid, "compiled")
            st.transition(self.experiment_id, cid, "correctness_checked")
            if s.get("benchmark_failed") or s["geomean_speedup"] is None:
                st.transition(self.experiment_id, cid, "benchmark_failed", "; ".join(details) or "no timing",
                              failure_stage="benchmark")
                status = "benchmark_failed"
            else:
                st.transition(self.experiment_id, cid, "benchmarked", dev_score=s["geomean_speedup"])
                status = "benchmarked"
                details = [summary_line(s, self.cfg.search.primary_baseline)]
        self.log(f"[eval] {cid}: {status}" + (f" geomean={s['geomean_speedup']:.3f}x" if s.get("geomean_speedup") else ""))
        return {"candidate": cid, "status": status, "stage": None if status == "benchmarked" else stage,
                "geomean_speedup": s.get("geomean_speedup"), "baseline": self.cfg.search.primary_baseline,
                "details": details}

    # -- the search loop -------------------------------------------------------
    def _budget_exhausted(self) -> str | None:
        p = self.progress
        if p.get("tokens", 0) >= self.cfg.search.max_total_generator_tokens:
            return "token budget"
        if (time.time() - p.get("started", time.time())) / 60 > self.cfg.search.max_experiment_minutes:
            return "time budget"
        return None

    def _critique(self, round_: int, feedback: list[dict], gpu: str) -> CritiqueResult | None:
        if self.arm == "none":
            return None
        best = self.best()
        if best is None:
            return None
        spec = self.store.load_spec(self.experiment_id, best["id"])
        payload = critique_payload(self.operator, spec, f"geomean speedup {best['dev_score']:.3f}x vs "
                                   f"{self.cfg.search.primary_baseline}", feedback, gpu)
        result = run_critic(self.arm, self.cfg, payload, local_client=self.client)
        path = self.store.write_json(self.store.exp_dir(self.experiment_id) / "critiques" / f"round{round_}.json",
                                     {"payload_sha256": sha256_text(payload), "payload": payload,
                                      "result": result.model_dump()})
        self.store.add_critic_trial(self.experiment_id, self.arm, self.seed, round_, best["id"], result, path)
        if result.status == "completed":
            if self.store.candidate(self.experiment_id, best["id"])["state"] == "benchmarked":
                self.store.transition(self.experiment_id, best["id"], "critiqued", f"{self.arm} critique")
        self.log(f"[critic] arm={self.arm} status={result.status} {result.reason or ''}")
        return result

    def run(self) -> dict:
        start_round = self.progress.get("round", 0)
        if start_round == 0:
            seed = self.seed_candidate()
            self.evaluate_candidate(seed)
            self._save_progress(round=0, round0_done=True)
        # Resume: finish any candidate whose evaluation was interrupted.
        for c in self.store.candidates(self.experiment_id):
            if c["state"] in ("generated", "validated") or (c["state"] in RETRYABLE and c["retry_count"] < 2):
                self.evaluate_candidate(c["id"])
        gpu = self.progress.get("gpu") or gpu_description(self.executor)
        self._save_progress(gpu=gpu)
        last_feedback: list[dict] = self.progress.get("last_feedback", [])
        stop_reason = None
        for round_ in range(self.progress.get("round", 0) + 1, self.cfg.search.max_rounds + 1):
            stop_reason = self._budget_exhausted()
            if stop_reason:
                break
            best = self.best()
            if best is None:
                stop_reason = "no correct starting candidate"
                break
            parent = self.store.load_spec(self.experiment_id, best["id"])
            parent_result = (f"geomean speedup {best['dev_score']:.3f}x vs {self.cfg.search.primary_baseline} "
                             f"({self.cfg.search.score_scope} timing)")
            new_ids: list[str] = []
            with self.generator_session():
                critique = self._critique(round_, last_feedback, gpu)
                for i in range(self.cfg.search.candidates_per_round):
                    cid = self._generate_one(round_, i, parent, best["id"], parent_result, last_feedback, critique, gpu)
                    if cid:
                        new_ids.append(cid)
            if self.store.candidate(self.experiment_id, best["id"])["state"] in ("benchmarked", "critiqued"):
                self.store.transition(self.experiment_id, best["id"], "revised", f"parent of round {round_}")
            feedback = [self.evaluate_candidate(cid) for cid in new_ids]
            invalid = [c for c in self.store.candidates(self.experiment_id)
                       if c["round"] == round_ and c["id"].startswith("invalid-")]
            feedback += [{"candidate": c["id"], "status": "invalid", "stage": "validation",
                          "details": [c["failure_message"] or ""]} for c in invalid]
            last_feedback = feedback
            self._save_progress(round=round_, last_feedback=feedback)
        result = self.finish(stop_reason)
        return result

    def _generate_one(self, round_: int, i: int, parent: CandidateSpec, parent_id: str, parent_result: str,
                      feedback: list[dict], critique: CritiqueResult | None, gpu: str) -> str | None:
        g = self.cfg.generator
        max_prompt = g.context_tokens - g.max_output_tokens - 64
        counter = getattr(self.client, "count_tokens", None)
        messages, pmeta = build_generation_messages(self.operator, self.cfg.search.mode, parent, parent_result,
                                                    feedback, critique, gpu, max_prompt, counter)
        sample_seed = self.seed * 100_000 + round_ * 100 + i
        gen_id = f"r{round_}-{i}"
        gdir = self.store.exp_dir(self.experiment_id) / "generations" / gen_id
        self.store.write_json(gdir / "prompt.json", {"messages": messages, "meta": pmeta, "seed": sample_seed})
        try:
            r = self.client.chat(messages, g.max_output_tokens, g.temperature, seed=sample_seed)
        except GeneratorError as exc:
            self._event("generation_failed", generation=gen_id, error=str(exc))
            self.store.add_invalid_generation(self.experiment_id, gen_id, sha256_text(str(exc)), self.operator, round_,
                                              parent_id, [f"generator error: {exc}"], {"search_mode": self.cfg.search.mode})
            return None
        self.store.write_once(gdir / "response.txt", r.text)
        meta = {"generation": gen_id, "model": r.model, "finish_reason": r.finish_reason,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "duration_s": round(r.duration_s, 2),
                "sampling": {**r.request, "seed": sample_seed}, "prompt": pmeta, "search_mode": self.cfg.search.mode,
                "critic_arm": self.arm, "critique_status": critique.status if critique else None,
                "generator": self._generator_identity()}
        self.store.write_json(gdir / "meta.json", meta)
        self._save_progress(tokens=self.progress.get("tokens", 0) + (r.input_tokens or 0) + (r.output_tokens or 0))
        spec, problems, pmeta2 = parse_generation(r.text, self.operator, self.cfg.search.mode, parent, r.truncated)
        meta.update(pmeta2)
        if spec is None:
            self.store.add_invalid_generation(self.experiment_id, gen_id, sha256_text(r.text), self.operator, round_,
                                              parent_id, problems, meta)
            self.log(f"[gen] {gen_id}: invalid output ({problems[0][:120]})")
            return None
        cid, is_new = self.store.add_candidate(self.experiment_id, spec, round_, parent_id, meta,
                                               {"response.txt": r.text})
        if not is_new:
            self._event("duplicate_generation", generation=gen_id, candidate=cid)
            self.log(f"[gen] {gen_id}: duplicate of {cid}; not re-evaluated")
            return None
        self.store.transition(self.experiment_id, cid, "generated", gen_id)
        self.log(f"[gen] {gen_id}: candidate {cid} ({r.output_tokens} tokens, {r.duration_s:.1f}s)")
        return cid

    def finish(self, stop_reason: str | None) -> dict:
        best = self.best()
        selection: dict[str, Any] = {"stop_reason": stop_reason or "round budget", "selected": None}
        if best is not None:
            if best["state"] != "selected":
                self.store.transition(self.experiment_id, best["id"], "selected", "best development geomean speedup")
            spec = self.store.load_spec(self.experiment_id, best["id"])
            selection.update(selected=best["id"], origin=spec.origin, dev_geomean_speedup=best["dev_score"],
                             source_hash=spec.source_hash, baseline=self.cfg.search.primary_baseline)
            self.store.write_json(self.store.exp_dir(self.experiment_id) / "frozen" / "candidate.json", spec)
        generated = [c for c in self.store.candidates(self.experiment_id)
                     if c["origin"] == "generated" and c["state"] in GOOD_STATES and c["dev_score"]]
        if generated:
            bg = max(generated, key=lambda c: c["dev_score"])
            selection["best_generated"] = {"candidate": bg["id"], "dev_geomean_speedup": bg["dev_score"]}
        self.store.write_json(self.store.exp_dir(self.experiment_id) / "frozen" / "selection.json", selection)
        self._save_progress(selection=selection, finished=utc_now())
        self.store.update_experiment(self.experiment_id, status="searched")
        self.log(f"[done] {self.experiment_id}: {json.dumps(selection, default=str)}")
        return selection


# ---------------------------------------------------------------------------
# Manual tuning, baselines, held-out evaluation, critic comparison
# ---------------------------------------------------------------------------

def tuning_cases(operator: str) -> list:
    dev = manifest(operator, "dev").cases
    if operator == "layernorm":
        return [c for c in dev if c.dtype == "float16" and c.M in (256, 2048) and c.input_kind == "normal"
                and c.eps == 1e-5]
    return [c for c in dev if c.dtype in ("float16", "bfloat16") and c.contract == "positions_to_output"]


def tune_manual(cfg: KernelForgeConfig, store: Store, operator: str, log=print) -> dict:
    """Bounded launch-configuration sweep for the manual kernel on development shapes only."""
    mod = module_for(operator)
    exp = new_id(f"tune-{operator}")
    store.create_experiment(exp, operator, "tune", cfg.model_dump())
    freeze_manifests(store, exp, operator)
    executor = Executor(cfg, store)
    evaluator = CandidateEvaluator(cfg, store, executor)
    cases = tuning_cases(operator)
    # Launch parameters change device work, so tune on CUDA-graph (device) time; dispatch cost is identical.
    quick = cfg.benchmark_settings(warmup_iterations=5, samples=15, calls_per_sample=10, timing_scopes=["graph"])
    names = list(mod.TUNING_GRID)
    results: list[dict] = []
    for values in itertools.product(*mod.TUNING_GRID.values()):
        params = {**dict(zip(names, values)), "num_stages": 1}
        if operator == "layernorm":
            params["BLOCK_SIZE"] = 0
        spec = mod.MANUAL_SPEC.model_copy(update={"launch": LaunchPolicy.single(**params), "origin": "tuning",
                                                  "label": "tune:" + canonical_json(params)})
        task = executor.make_task(exp, "timing", candidate=spec, workloads=cases, split="dev",
                                  baselines=[mod.PRIMARY_BASELINE], benchmark=quick)
        res = executor.run_task(task, timeout_s=cfg.search.max_compile_seconds + 20 * len(cases) + 120)
        per = {}
        for t in res.timing:
            if t.status == "passed":
                p = next(p for p in t.paired if p.baseline_id == mod.PRIMARY_BASELINE)
                per[t.workload_id] = p.speedup_of_medians
        results.append({"params": params, "status": res.status, "speedups": per})
        log(f"[tune] {params}: {len(per)}/{len(cases)} ok, geomean={geomean(list(per.values())) or 0:.3f}")
    # Choose the best configuration per selector tier.
    tiers = []
    lo = 0
    for upto in mod.TUNING_TIERS:
        in_tier = [c.workload_id for c in cases if lo < mod.launch_selector(c) <= (upto or 1 << 30)]
        lo = upto or lo
        if not in_tier:
            continue
        scored = []
        for r in results:
            vals = [r["speedups"].get(w) for w in in_tier]
            if all(v is not None for v in vals):
                scored.append((geomean(vals), r["params"]))
        if scored:
            score, params = max(scored, key=lambda x: x[0])
            tiers.append({"upto": upto, "params": params, "tier_geomean": score, "workloads": in_tier})
    if not tiers:
        raise RuntimeError("no launch configuration passed on every tuning workload")
    tiers[-1]["upto"] = None
    policy = LaunchPolicy(tiers=[LaunchTier(upto=t["upto"], params=t["params"]) for t in tiers])
    tuned = mod.MANUAL_SPEC.model_copy(update={"launch": policy, "label": "manual-tuned", "origin": "manual"})
    store.write_json(store.exp_dir(exp) / "tuning_results.json", {"cases": [c.workload_id for c in cases],
                                                                  "results": results, "tiers": tiers})
    cid, _ = store.add_candidate(exp, tuned, 1, None, {"role": "tuned_manual", "tiers": tiers})
    store.transition(exp, cid, "generated", "tuned manual candidate")
    store.transition(exp, cid, "validated")
    dev = load_frozen_manifest(store, exp, "dev")
    s = evaluator.evaluate(exp, cid, tuned, dev, None, [mod.PRIMARY_BASELINE, "triton_basic"],
                           sessions=1, memory=False)
    if s["correct_all"] and s["geomean_speedup"]:
        store.transition(exp, cid, "compiled")
        store.transition(exp, cid, "correctness_checked")
        store.transition(exp, cid, "benchmarked", dev_score=s["geomean_speedup"])
        store.transition(exp, cid, "selected", "tuned manual kernel frozen")
        store.write_json(store.exp_dir(exp) / "frozen" / "candidate.json", tuned)
    else:
        store.transition(exp, cid, "compiled")
        store.transition(exp, cid, "incorrect", json.dumps(s["failures"][:3], default=str))
    store.update_experiment(exp, status="searched", progress={"tiers": tiers, "dev_geomean": s["geomean_speedup"],
                                                               "correct_all": s["correct_all"]})
    log(f"[tune] {exp}: policy={json.dumps(policy.model_dump())} dev geomean={s['geomean_speedup']}")
    return {"experiment_id": exp, "policy": policy.model_dump(), "dev_geomean_speedup": s["geomean_speedup"],
            "correct_all": s["correct_all"]}


def run_baseline(cfg: KernelForgeConfig, store: Store, operator: str, suite: str, sessions: int,
                 with_compile: bool = False, spec: CandidateSpec | None = None, log=print) -> dict:
    """Measure the manual kernel and the untuned Triton kernel against the PyTorch baselines."""
    mod = module_for(operator)
    exp = new_id(f"baseline-{operator}")
    store.create_experiment(exp, operator, "baseline", cfg.model_dump())
    freeze_manifests(store, exp, operator)
    evaluator = CandidateEvaluator(cfg, store, Executor(cfg, store))
    m = load_frozen_manifest(store, exp, suite)
    extra = ["torch_compile"] if with_compile else []
    out = {"experiment_id": exp, "suite": suite, "candidates": {}}
    plan = [(spec or mod.MANUAL_SPEC, [mod.PRIMARY_BASELINE, "triton_basic"] + extra),
            (mod.BASIC_SPEC, [mod.PRIMARY_BASELINE] + extra)]
    for cand_spec, baselines in plan:
        cid, _ = store.add_candidate(exp, cand_spec, 0, None, {"role": cand_spec.label})
        store.transition(exp, cid, "generated")
        store.transition(exp, cid, "validated")
        log(f"[baseline] {cand_spec.label} ({cid}) vs {baselines} on {suite} ({len(m.cases)} cases, {sessions} sessions)")
        s = evaluator.evaluate(exp, cid, cand_spec, m, None, baselines, sessions=sessions, memory=True)
        store.transition(exp, cid, "compiled" if s["stage"] != "compile" else "compile_failed")
        if s["correct_all"]:
            store.transition(exp, cid, "correctness_checked")
            if s["geomean_speedup"] is not None:
                store.transition(exp, cid, "benchmarked", dev_score=s["geomean_speedup"])
        elif s["stage"] != "compile":
            store.transition(exp, cid, "incorrect", json.dumps(s["failures"][:3], default=str))
        out["candidates"][cand_spec.label] = {"candidate": cid, "correct_all": s["correct_all"],
                                              "geomean_speedup_vs_primary": s["geomean_speedup"],
                                              "failures": s["failures"][:5]}
        log(f"[baseline] {cand_spec.label}: correct_all={s['correct_all']} geomean vs {mod.PRIMARY_BASELINE}="
            f"{s['geomean_speedup']}")
    store.update_experiment(exp, status="completed", progress=out)
    return out


def evaluate_heldout(cfg: KernelForgeConfig, store: Store, experiment_id: str, force: bool = False,
                     sessions: int | None = None, log=print) -> dict:
    """Single evaluation of the frozen candidate on the frozen held-out manifest. The generator is
    not running (this command never starts it) and the result is never fed back into a prompt."""
    exp = store.experiment(experiment_id)
    if exp is None:
        raise SystemExit(f"unknown experiment {experiment_id}")
    progress = exp["progress"]
    if progress.get("heldout_done") and not force:
        raise SystemExit("held-out evaluation already ran for this experiment; re-running would turn the held-out "
                         "set into tuning data. Use --force to re-run (recorded as a contamination flag).")
    frozen = store.exp_dir(experiment_id) / "frozen" / "candidate.json"
    if not frozen.exists():
        raise SystemExit("experiment has no frozen candidate (run optimize/tune to completion first)")
    spec = CandidateSpec.model_validate_json(frozen.read_text(encoding="utf-8"))
    mod = module_for(spec.operator)
    cid = spec.short_hash
    heldout = load_frozen_manifest(store, experiment_id, "heldout")
    executor = Executor(cfg, store)
    probe = executor.probe(experiment_id)
    log(f"[heldout] free VRAM before evaluation: {(probe.get('probe') or {}).get('free_bytes', 0) / 2**30:.2f} GiB")
    baselines = [mod.PRIMARY_BASELINE, "triton_basic"] + (["manual"] if spec.origin != "manual" else [])
    n = sessions or cfg.independent_sessions
    s = CandidateEvaluator(cfg, store, executor).evaluate(experiment_id, cid, spec, heldout, None, baselines,
                                                          sessions=n, memory=True)
    cand = store.candidate(experiment_id, cid)
    if cand and cand["state"] == "selected":
        store.transition(experiment_id, cid, "heldout_evaluated", "frozen held-out evaluation")
    progress.update(heldout_done=True, heldout_runs=progress.get("heldout_runs", 0) + 1,
                    heldout={"correct_all": s["correct_all"], "n_failed": s["n_failed"],
                             "geomean_speedup": s["geomean_speedup"], "sessions": n, "at": utc_now()})
    if progress["heldout_runs"] > 1:
        progress["contamination"] = "held-out evaluated more than once"
    store.update_experiment(experiment_id, status="heldout_evaluated", progress=progress)
    log(f"[heldout] {experiment_id}: correct_all={s['correct_all']} geomean vs {mod.PRIMARY_BASELINE}="
        f"{s['geomean_speedup']}")
    return progress["heldout"]


def compare_critics(cfg: KernelForgeConfig, store: Store, operator: str, seeds: list[int], arms: list[str],
                    heldout: bool = False, client_factory=None, log=print) -> dict:
    """Same generator, starting candidate, workloads, rounds and budgets for every arm; only the critic varies."""
    from .providers import CRITIC_ARMS, critic_availability

    parent = new_id(f"critics-{operator}")
    store.create_experiment(parent, operator, "critic_comparison", cfg.model_dump())
    summary: dict[str, Any] = {"experiment_id": parent, "arms": {}}
    for arm in arms:
        if arm not in CRITIC_ARMS:
            raise ValueError(f"unknown arm {arm}")
        available, model, reason = critic_availability(arm, cfg)
        if arm != "none" and not available:
            summary["arms"][arm] = {"status": "unavailable", "model": model, "reason": reason}
            for seed in seeds:
                store.add_critic_trial(parent, arm, seed, 0, None,
                                       CritiqueResult(arm=arm, provider=arm, model=model, status="unavailable",
                                                      reason=reason), None)
            log(f"[critics] arm {arm}: unavailable ({reason}) - recorded, not run")
            continue
        runs = []
        for seed in seeds:
            child_cfg = cfg.model_copy(deep=True)
            child_cfg.search.seed = seed
            client = client_factory(arm, seed) if client_factory else None
            agent = SearchAgent(child_cfg, store, operator, arm=arm, seed=seed, client=client, kind="optimize",
                                parent_id=parent, experiment_id=f"{parent}-{arm}-s{seed}", log=log)
            sel = agent.run()
            if heldout and sel.get("selected"):
                sel["heldout"] = evaluate_heldout(child_cfg, store, agent.experiment_id, log=log)
            runs.append({"experiment_id": agent.experiment_id, "seed": seed, "selection": sel})
        summary["arms"][arm] = {"status": "completed", "model": model, "runs": runs}
    store.update_experiment(parent, status="completed", progress=summary)
    store.write_json(store.exp_dir(parent) / "summary.json", summary)
    return summary
