"""KernelForge command-line interface: `python -m kernelforge <command> --help`."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import KernelForgeConfig, doctor, load_config


def _cfg(args) -> KernelForgeConfig:
    cfg = load_config(getattr(args, "config", None))
    if getattr(args, "mode", None):
        cfg.mode = args.mode
    return cfg


def _store(cfg: KernelForgeConfig):
    from .store import Store

    return Store(cfg.runs_path())


def _require_cuda_mode(cfg: KernelForgeConfig, what: str) -> None:
    if cfg.mode != "cuda":
        raise SystemExit(f"{what} needs the CUDA worker (mode=cuda); mode={cfg.mode} only runs CPU reference checks. "
                         "Nothing was measured.")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------

def cmd_doctor(args) -> int:
    cfg = _cfg(args)
    report = doctor(cfg, probe_gpu=not args.no_probe, hash_model=args.hash_model)
    if args.json:
        _print(report)
        return 0
    h, smi, probe = report["host"], report["nvidia_smi"], report.get("evaluator_probe") or {}
    print(f"Platform     : {h['platform']} ({h['distribution']}, kernel {h['kernel']}, {h['architecture']})")
    print(f"Python       : {h['python']} ({h['python_executable']})")
    print(f"RAM          : {h['ram_total_gib']} GiB visible, {h['ram_available_gib']} GiB available; "
          f"swap {h['swap_used_gib']}/{h['swap_total_gib']} GiB; swapping now: {h['swap_activity'].get('active')}")
    if h.get("wsl_note"):
        print(f"               {h['wsl_note']}")
    print(f"Disk free    : {h['disk_free_gib']} GiB")
    print(f"Packages     : {h['packages']}")
    if smi.get("available"):
        print(f"GPU (smi)    : {smi.get('name')} {smi.get('uuid')} driver {smi.get('driver_version')} "
              f"(driver CUDA {smi.get('driver_cuda_version')}), cc {smi.get('compute_cap')}, "
              f"free {smi.get('memory.free')}/{smi.get('memory.total')} MiB, display {smi.get('display_active')}")
    else:
        print(f"GPU (smi)    : unavailable ({smi.get('reason')})")
    if probe.get("status") == "ok":
        hw, sw, pr = probe["hardware"], probe["software"], probe["probe"]
        print(f"Evaluator    : {hw.get('gpu')} cc {hw.get('compute_capability')} {hw.get('sm_count')} SMs, "
              f"torch {sw.get('torch')} (CUDA {sw.get('torch_cuda')}), triton {sw.get('triton')}, "
              f"free {pr['free_bytes'] / 2**30:.2f}/{pr['total_bytes'] / 2**30:.2f} GiB, "
              f"allocation probe {'ok' if pr.get('allocation_probe_ok') else 'FAILED'}")
    else:
        print(f"Evaluator    : {probe.get('status')} {probe.get('message') or probe.get('reason') or ''}")
    g = report["generator"]
    print(f"Generator    : {g['source_model']} / {g['source_file']} ({g['quantization']}) as '{g['serving_alias']}' at "
          f"{g['base_url']} loopback={g['loopback_only']}; binary exists={g.get('server_binary_exists')}, "
          f"model exists={g.get('model_file_exists')}" + (f", sha256 ok={g.get('sha256_matches_declared')}"
                                                          if "sha256_matches_declared" in g else ""))
    print(f"API keys     : {report['api_keys']}")
    return 0


def cmd_smoke(args) -> int:
    from .executor import Executor

    cfg = _cfg(args)
    _require_cuda_mode(cfg, "smoke")
    res = Executor(cfg, _store(cfg)).smoke()
    _print({"status": res.status, "message": res.message, "checks": res.probe, "software": res.software,
            "hardware": res.hardware})
    return 0 if res.status == "ok" else 1


def cmd_test_reference(args) -> int:
    if args.device == "cuda":
        from .executor import Executor
        from .workloads import manifest

        cfg = _cfg(args)
        _require_cuda_mode(cfg, "test-reference --device cuda")
        ex = Executor(cfg, _store(cfg))
        ok = True
        for op in ("layernorm", "mrope"):
            task = ex.make_task("_system", "selftest", workloads=manifest(op, "smoke").cases, split="smoke")
            res = ex.run_task(task, 600)
            for c in res.correctness:
                print(f"{op:9s} {c.workload_id:60s} {c.status:7s} max_abs={c.max_abs_error}")
            ok &= res.status == "ok"
        return 0 if ok else 1
    from .selfcheck import run_reference_checks

    results = run_reference_checks()
    for r in results:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']}: {r['detail']}")
    print(f"{sum(r['passed'] for r in results)}/{len(results)} reference checks passed (device=cpu)")
    return 0 if all(r["passed"] for r in results) else 1


def cmd_selftest_evaluator(args) -> int:
    from .executor import Executor
    from .kernels.canaries import canaries
    from .workloads import manifest

    cfg = _cfg(args)
    _require_cuda_mode(cfg, "selftest-evaluator")
    ex = Executor(cfg, _store(cfg))
    ok = True
    for name, (spec, expected) in canaries().items():
        m = manifest(spec.operator, "dev")
        res = ex.run_task(ex.make_task("_system", "correctness", candidate=spec, workloads=m.cases, split="dev"), 900)
        stages = sorted({c.failure_stage for c in res.correctness if c.status != "passed"})
        rejected = any(c.status != "passed" for c in res.correctness)
        good = rejected and expected in stages
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name:22s} expected={expected:11s} rejected={rejected} stages={stages}")
    return 0 if ok else 1


def cmd_baseline(args) -> int:
    from .agent import run_baseline
    from .contracts import CandidateSpec

    cfg = _cfg(args)
    _require_cuda_mode(cfg, "baseline")
    spec = CandidateSpec.model_validate_json(Path(args.spec).read_text()) if args.spec else None
    out = run_baseline(cfg, _store(cfg), args.operator, args.suite, args.sessions or cfg.independent_sessions,
                       args.with_compile, spec)
    _print(out)
    return 0


def cmd_tune(args) -> int:
    from .agent import tune_manual

    cfg = _cfg(args)
    _require_cuda_mode(cfg, "tune")
    _print(tune_manual(cfg, _store(cfg), args.operator))
    return 0


def cmd_generator_check(args) -> int:
    from .executor import Executor
    from .providers import LlamaServer, OpenAICompatibleClient

    cfg = _cfg(args)
    store = _store(cfg)
    out_dir = cfg.runs_path() / "_system" / "generator"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    server = LlamaServer(cfg.generator, out_dir / f"server-{stamp}.log", cfg.worker.gpu_index)
    result = {"generator": cfg.generator.model_dump(exclude={"server_extra_args"})}
    result["start"] = server.start()
    try:
        client = OpenAICompatibleClient(cfg.generator.base_url, cfg.generator.model, cfg.generator.request_timeout_s)
        r = client.chat([{"role": "user", "content": "Write a Triton kernel `add(X, Y, Z, N, BLOCK: tl.constexpr)` "
                                                     "that computes Z = X + Y with masking. Code only."}],
                        max_tokens=256, temperature=0.0, seed=0)
        result["request"] = {"finish_reason": r.finish_reason, "input_tokens": r.input_tokens,
                             "output_tokens": r.output_tokens, "duration_s": round(r.duration_s, 2),
                             "tokens_per_s": round((r.output_tokens or 0) / max(r.duration_s, 1e-6), 1),
                             "served_model": r.model, "text_head": r.text[:400]}
    finally:
        result["stop"] = server.stop()
    if cfg.mode == "cuda":
        result["evaluator_probe_after_stop"] = Executor(cfg, store).probe().get("probe")
    path = out_dir / f"generator-check-{stamp}.json"
    path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    _print(result)
    print(f"saved {path}")
    return 0 if result["stop"].get("vram_released") is not False else 1


def _scripted_client(operator: str):
    from .kernels import module_for
    from .kernels.canaries import canaries
    from .providers import ScriptedClient

    mod = module_for(operator)
    wrong = canaries()["ln_tail_mean" if operator == "layernorm" else "mr_single_axis"][0]
    policy = json.dumps(mod.MANUAL_SPEC.launch.model_dump())
    return ScriptedClient([
        f"```python\n{wrong.kernel_source}```\n```json\n{json.dumps(wrong.launch.model_dump())}\n```\n"
        "EXPLANATION: (scripted) normalize by the padded block size to drop a division.",
        f"```python\n{mod.MANUAL_SPEC.kernel_source}```\n```json\n{policy}\n```\n"
        "EXPLANATION: (scripted) multi-row tiles / reuse, corrected statistics.",
    ], model="scripted-demo")


def cmd_optimize(args) -> int:
    from .agent import SearchAgent

    cfg = _cfg(args)
    _require_cuda_mode(cfg, "optimize")
    if args.rounds:
        cfg.search.max_rounds = args.rounds
    if args.candidates:
        cfg.search.candidates_per_round = args.candidates
    if args.search_mode:
        cfg.search.mode = args.search_mode
    cfg.search.seed = args.seed
    client = None
    if args.scripted_demo:
        client = _scripted_client(args.operator)
        cfg.search.candidates_per_round = 1
    elif cfg.generator.backend == "fixture":
        raise SystemExit("generator.backend=fixture has no live model; use --scripted-demo or an "
                         "openai_compatible endpoint")
    agent = SearchAgent(cfg, _store(cfg), args.operator, experiment_id=args.experiment, arm=args.arm,
                        seed=args.seed, client=client, kind="demo" if args.scripted_demo else "optimize")
    _print({"experiment_id": agent.experiment_id, **agent.run()})
    return 0


def cmd_compare_critics(args) -> int:
    from .agent import compare_critics

    cfg = _cfg(args)
    _require_cuda_mode(cfg, "compare-critics")
    if args.rounds:
        cfg.search.max_rounds = args.rounds
    if args.candidates:
        cfg.search.candidates_per_round = args.candidates
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")
    _print(compare_critics(cfg, _store(cfg), args.operator, seeds, arms, heldout=args.heldout))
    return 0


def cmd_evaluate_heldout(args) -> int:
    from .agent import evaluate_heldout

    cfg = _cfg(args)
    _require_cuda_mode(cfg, "evaluate-heldout")
    _print(evaluate_heldout(cfg, _store(cfg), args.experiment, force=args.force, sessions=args.sessions))
    return 0


def cmd_report(args) -> int:
    from .report import write_report

    cfg = _cfg(args)
    store = _store(cfg)
    ids = args.experiment or None
    if args.output:
        out = Path(args.output)
    elif ids and len(ids) == 1:
        out = store.exp_dir(ids[0]) / "report" / "index.html"
    else:
        out = cfg.runs_path() / "report" / "index.html"
    print(write_report(store, cfg, out, ids))
    return 0


def cmd_summarize(args) -> int:
    from .report import markdown_summary

    cfg = _cfg(args)
    store = _store(cfg)
    ids = args.experiment or [e["id"] for e in store.experiments()]
    print(markdown_summary(store, ids))
    return 0


def cmd_export(args) -> int:
    from .report import export_evidence

    cfg = _cfg(args)
    print(export_evidence(_store(cfg), args.experiment, Path(args.output) if args.output else None))
    return 0


def cmd_list(args) -> int:
    cfg = _cfg(args)
    for e in _store(cfg).experiments():
        sel = (e["progress"] or {}).get("selection") or {}
        print(f"{e['id']:48s} {e['kind']:18s} {e['operator']:9s} {e['status']:18s} "
              f"arm={e['arm'] or '-':9s} selected={sel.get('selected', '-')} dev={sel.get('dev_geomean_speedup', '-')}")
    return 0


def cmd_train_rl(args) -> int:
    from .train_rl import main as rl_main

    return rl_main(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m kernelforge",
                                description="Execution-guided Triton kernel search with a trusted GPU evaluator.")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.add_argument("--config", default=None, help="YAML config (default: ./config.yaml if present)")
        sp.set_defaults(fn=fn)
        return sp

    sp = add("doctor", cmd_doctor, "Hardware/software preflight (GPU probe runs in a separate worker process).")
    sp.add_argument("--no-probe", action="store_true", help="skip the evaluator GPU probe")
    sp.add_argument("--hash-model", action="store_true", help="also SHA-256 the generator GGUF file")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--mode", choices=["reference", "cuda", "remote"])

    add("smoke", cmd_smoke, "Blackwell smoke check: CUDA matmul + Triton vector add on the device.")

    sp = add("test-reference", cmd_test_reference, "Check the independent references (cpu) or baselines vs references (cuda).")
    sp.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    add("selftest-evaluator", cmd_selftest_evaluator, "Run the canary kernels and confirm the evaluator rejects every one.")

    sp = add("baseline", cmd_baseline, "Measure manual and untuned Triton kernels against PyTorch baselines.")
    sp.add_argument("--operator", choices=["layernorm", "mrope"], required=True)
    sp.add_argument("--mode", choices=["reference", "cuda", "remote"], default="cuda")
    sp.add_argument("--suite", choices=["smoke", "search", "dev", "heldout"], default="smoke")
    sp.add_argument("--sessions", type=int, default=None, help="independent timing sessions (default: config)")
    sp.add_argument("--with-compile", action="store_true", help="also compare against torch.compile")
    sp.add_argument("--spec", default=None, help="candidate.json to measure instead of the built-in manual kernel")

    sp = add("tune", cmd_tune, "Bounded launch-configuration sweep of the manual kernel on development shapes.")
    sp.add_argument("--operator", choices=["layernorm", "mrope"], required=True)

    add("generator-check", cmd_generator_check, "Start the managed llama.cpp server, run one request, stop it and confirm VRAM release.")

    sp = add("optimize", cmd_optimize, "Execution-guided search with the local coder (bounded rounds, resumable).")
    sp.add_argument("--operator", choices=["layernorm", "mrope"], required=True)
    sp.add_argument("--experiment", default=None, help="resume this experiment id")
    sp.add_argument("--arm", choices=["none", "self", "openai", "anthropic", "google"], default="none")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--rounds", type=int, default=None)
    sp.add_argument("--candidates", type=int, default=None, help="candidates per round")
    sp.add_argument("--search-mode", choices=["full", "launch_only"], default=None)
    sp.add_argument("--scripted-demo", action="store_true",
                    help="replay a scripted wrong-then-correct generator (demo of the loop; real evaluator)")

    sp = add("compare-critics", cmd_compare_critics, "Controlled comparison of critic arms with a fixed generator.")
    sp.add_argument("--operator", choices=["layernorm", "mrope"], default="layernorm")
    sp.add_argument("--seeds", default="0,1,2")
    sp.add_argument("--arms", default="none,self,openai,anthropic,google")
    sp.add_argument("--rounds", type=int, default=None)
    sp.add_argument("--candidates", type=int, default=None)
    sp.add_argument("--heldout", action="store_true", help="also run the frozen held-out evaluation per run")

    sp = add("evaluate-heldout", cmd_evaluate_heldout, "Evaluate an experiment's frozen candidate once on the held-out manifest.")
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--sessions", type=int, default=None)

    sp = add("report", cmd_report, "Generate the static HTML report from saved results (no GPU or model needed).")
    sp.add_argument("--experiment", action="append", help="experiment id (repeatable); default: all")
    sp.add_argument("--output", default=None)

    sp = add("summarize", cmd_summarize, "Markdown tables (coverage, geomean speedups, memory) from evidence records.")
    sp.add_argument("--experiment", action="append", help="experiment id (repeatable); default: all")

    sp = add("export-evidence", cmd_export, "Zip an experiment's immutable evidence with a hash-linked index.")
    sp.add_argument("--experiment", required=True)
    sp.add_argument("--output", default=None)

    add("list", cmd_list, "List experiments.")

    sp = sub.add_parser("train-rl", help="Optional GRPO + LoRA RL extension (separate environment).",
                        description="Optional GRPO + LoRA RL extension (separate environment).")
    sp.add_argument("--config", required=True, help="rl.yaml")
    sp.add_argument("--kf-config", default=None, help="KernelForge config.yaml for the evaluator")
    sp.add_argument("--smoke", action="store_true", help="tiny integration run")
    sp.add_argument("--evaluate-only", default=None, help="adapter directory: run base-vs-adapter evaluation only")
    sp.set_defaults(fn=cmd_train_rl)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
