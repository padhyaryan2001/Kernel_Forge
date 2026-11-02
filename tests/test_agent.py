"""Orchestration tests in offline fixture mode: no GPU, no model. The fake evaluator returns
fixture summaries only; it never writes measured records."""
from __future__ import annotations

import json

import pytest

from kernelforge import agent as agent_mod
from kernelforge.contracts import CandidateSpec, LaunchPolicy
from kernelforge.kernels import layernorm as ln, mrope as mr
from kernelforge.kernels.canaries import canaries
from kernelforge.providers import (ScriptedClient, build_generation_messages, critic_availability, parse_generation,
                                   run_critic)
from kernelforge.source_policy import check_source
from kernelforge.store import StateError, Store


# ---------------------------------------------------------------------------
# Source policy
# ---------------------------------------------------------------------------

def test_policy_accepts_trusted_kernels():
    for spec in (ln.BASIC_SPEC, ln.MANUAL_SPEC, mr.BASIC_SPEC, mr.MANUAL_SPEC):
        rep = check_source(spec.kernel_source, spec.operator, spec.entry_point)
        assert rep.ok, rep.problems


@pytest.mark.parametrize("snippet,needle", [
    ("import os\n", "import of 'os'"),
    ("import subprocess\n", "import of 'subprocess'"),
    ("from pathlib import Path\n", "from-import"),
    ("def host():\n    return 1\n", "must be decorated"),
    ("X = open('f')\n", "only NAME"),
    ("class A:\n    pass\n", "ClassDef"),
])
def test_policy_rejects_host_code(snippet, needle):
    src = snippet + ln.MANUAL_SPEC.kernel_source
    rep = check_source(src, "layernorm", "layernorm_fwd")
    assert not rep.ok and any(needle in p for p in rep.problems), rep.problems


@pytest.mark.parametrize("body,needle", [
    ("    eval('1')\n", "eval"),
    ("    x = tl.inline_asm_elementwise('mov', '=r', [], dtype=tl.int32, is_pure=True, pack=1)\n", "inline_asm"),
    ("    y = M.__class__\n", "__class__"),
    ("    z = [i for i in range(3)]\n", "ListComp"),
    ("    print(M)\n", "print"),
    ("    q = undefined_name + 1\n", "unknown name"),
])
def test_policy_rejects_dangerous_kernel_bodies(body, needle):
    src = ln.BASIC_SOURCE.rstrip("\n") + "\n" + body
    rep = check_source(src, "layernorm", "layernorm_fwd")
    assert not rep.ok and any(needle in p for p in rep.problems), rep.problems


def test_policy_enforces_exact_signature_and_constexpr():
    src = ln.BASIC_SOURCE.replace("ROWS_PER_PROGRAM: tl.constexpr", "ROWS_PER_PROGRAM")
    assert any("constexpr" in p for p in check_source(src, "layernorm", "layernorm_fwd").problems)
    src = ln.BASIC_SOURCE.replace("M, N, eps", "N, M, eps")
    assert any("parameters must be exactly" in p for p in check_source(src, "layernorm", "layernorm_fwd").problems)
    assert any("not found" in p for p in check_source(ln.BASIC_SOURCE, "layernorm", "other").problems)


def test_canary_host_code_rejected_by_policy():
    spec, stage = canaries()["ln_host_code"]
    assert stage == "validation"
    assert not check_source(spec.kernel_source, "layernorm", spec.entry_point).ok


# ---------------------------------------------------------------------------
# Structured output parsing and prompts
# ---------------------------------------------------------------------------

def _response(src: str, launch: dict | None, expl: str = "faster") -> str:
    out = f"Here you go.\n```python\n{src}```\n"
    if launch is not None:
        out += f"```json\n{json.dumps(launch)}\n```\n"
    return out + f"EXPLANATION: {expl}"


def test_parse_valid_generation():
    text = _response(ln.MANUAL_SOURCE, {"tiers": [{"upto": None, "params": {"num_warps": 8}}]})
    spec, problems, meta = parse_generation(text, "layernorm", "full", ln.BASIC_SPEC)
    assert spec is not None and not problems
    assert spec.kernel_source.strip() == ln.MANUAL_SOURCE.strip()
    assert spec.launch.select(1000)["num_warps"] == 8 and spec.explanation == "faster"
    assert spec.launch.select(1000)["ROWS_PER_PROGRAM"] == 1  # defaults filled canonically


def test_parse_inherits_launch_when_missing_and_records_it():
    spec, _, meta = parse_generation(_response(ln.MANUAL_SOURCE, None), "layernorm", "full", ln.BASIC_SPEC)
    assert spec.launch == ln.BASIC_SPEC.launch and meta["launch_inherited"]


def test_restating_default_launch_params_is_a_duplicate():
    # pilot finding: a generated '{}' policy resolves to the parent's defaults and must dedup to the parent
    spec, _, _ = parse_generation(_response(ln.BASIC_SOURCE, {"tiers": [{"upto": None, "params": {}}]}),
                                  "layernorm", "full", ln.BASIC_SPEC)
    assert spec.source_hash == ln.BASIC_SPEC.source_hash
    two = {"tiers": [{"upto": 512, "params": {"num_warps": 4}}, {"upto": None, "params": {}}]}
    spec, _, _ = parse_generation(_response(ln.BASIC_SOURCE, two), "layernorm", "full", ln.BASIC_SPEC)
    assert len(spec.launch.tiers) == 1 and spec.source_hash == ln.BASIC_SPEC.source_hash


def test_parse_rejects_truncated_and_incomplete_output():
    spec, problems, _ = parse_generation(_response(ln.MANUAL_SOURCE, None), "layernorm", "full", ln.BASIC_SPEC,
                                         truncated=True)
    assert spec is None and "truncated" in problems[0]
    spec, problems, _ = parse_generation("```python\nimport triton\n@triton.jit\ndef f(", "layernorm", "full",
                                         ln.BASIC_SPEC)
    assert spec is None and "unterminated" in problems[0]
    spec, problems, _ = parse_generation(_response(ln.MANUAL_SOURCE, None).replace("```json", "```json\n{bad"),
                                         "layernorm", "full", ln.BASIC_SPEC)


def test_parse_launch_only_mode_keeps_verified_kernel():
    text = "```json\n{\"num_warps\": 2, \"ROWS_PER_PROGRAM\": 4}\n```\nEXPLANATION: smaller tiles"
    spec, problems, _ = parse_generation(text, "layernorm", "launch_only", ln.MANUAL_SPEC)
    assert spec.kernel_source == ln.MANUAL_SPEC.kernel_source and spec.search_mode == "launch_only"
    assert {k: spec.launch.select(1)[k] for k in ("num_warps", "ROWS_PER_PROGRAM")} == {"num_warps": 2, "ROWS_PER_PROGRAM": 4}


@pytest.mark.parametrize("op", ["layernorm", "mrope"])
def test_prompt_fits_context_budget_and_trims_feedback_first(op):
    mod = ln if op == "layernorm" else mr
    feedback = [{"candidate": f"c{i}", "status": "compile_failed", "stage": "compile", "details": ["x" * 400] * 3}
                for i in range(30)]
    msgs, meta = build_generation_messages(op, "full", mod.BASIC_SPEC, "1.0x", feedback, None, "RTX", 2496)
    assert meta["prompt_tokens_estimate"] <= 2496 and meta["feedback_items"] == 30
    _, bare = build_generation_messages(op, "full", mod.BASIC_SPEC, "1.0x", [], None, "RTX", 2496)
    tight = bare["prompt_tokens_estimate"] + 150   # room for a little feedback only
    msgs, meta = build_generation_messages(op, "full", mod.BASIC_SPEC, "1.0x", feedback, None, "RTX", tight)
    assert meta["prompt_tokens_estimate"] <= tight and 0 <= meta["feedback_items"] < 30
    assert mod.SPEC_DOC in msgs[1]["content"] and mod.SIGNATURE_DOC in msgs[1]["content"]
    with pytest.raises(Exception, match="budget"):
        build_generation_messages(op, "full", mod.BASIC_SPEC, "1.0x", feedback, None, "RTX", 50)


# ---------------------------------------------------------------------------
# Store and state machine
# ---------------------------------------------------------------------------

def test_state_machine_rejects_illegal_transitions(store: Store):
    store.create_experiment("e1", "layernorm", "optimize", {})
    cid, new = store.add_candidate("e1", ln.MANUAL_SPEC, 0, None, {})
    assert new and store.add_candidate("e1", ln.MANUAL_SPEC, 0, None, {}) == (cid, False)  # dedup by hash
    with pytest.raises(StateError):
        store.transition("e1", cid, "benchmarked")
    for s in ("generated", "validated", "timed_out"):
        store.transition("e1", cid, s, failure_stage="timeout" if s == "timed_out" else None)
    store.transition("e1", cid, "validated", "retry")
    assert store.candidate("e1", cid)["retry_count"] == 1
    for s in ("compiled", "correctness_checked", "benchmarked", "selected", "heldout_evaluated"):
        store.transition("e1", cid, s)
    assert [t["to_state"] for t in store.transitions("e1")][-1] == "heldout_evaluated"


def test_whitespace_only_edit_is_a_semantic_duplicate(store: Store):
    store.create_experiment("e2", "layernorm", "optimize", {})
    cid, _ = store.add_candidate("e2", ln.BASIC_SPEC, 0, None, {})
    edited_src = ln.BASIC_SOURCE.replace("\n\n\n@triton.jit", "\n\n@triton.jit").replace(
        "    mean =", "    # comment\n    mean =")
    edited = ln.BASIC_SPEC.model_copy(update={"kernel_source": edited_src})
    assert edited.source_hash != ln.BASIC_SPEC.source_hash          # exact bytes differ (recorded identity)
    assert store.add_candidate("e2", edited, 1, cid, {}) == (cid, False)  # but it is not re-evaluated
    changed = ln.BASIC_SPEC.model_copy(update={"kernel_source": ln.BASIC_SOURCE.replace("/ N", "/ (N + 0)")})
    assert store.add_candidate("e2", changed, 1, cid, {})[1] is True


def test_artifacts_are_write_once(store: Store, tmp_path):
    p = store.write_once(tmp_path / "a.txt", "one")
    assert store.write_once(p, "one") == p
    q = store.write_once(p, "two")
    assert q != p and p.read_text() == "one" and q.read_text() == "two"


# ---------------------------------------------------------------------------
# End-to-end search loop with a fake evaluator (fixture mode)
# ---------------------------------------------------------------------------

class FakeEvaluator:
    """Deterministic fixture verdicts keyed on source content. Produces no measurements."""

    calls: list[str] = []
    interrupt_on: str | None = None

    def __init__(self, *a, **k):
        pass

    def evaluate(self, experiment_id, candidate_id, spec, manifest, timing_cases, baselines, **kw):
        FakeEvaluator.calls.append(candidate_id)
        if FakeEvaluator.interrupt_on and FakeEvaluator.interrupt_on in spec.kernel_source:
            FakeEvaluator.interrupt_on = None
            raise KeyboardInterrupt("simulated interruption")
        base = dict(n_passed=0, n_failed=0, failures=[], speedups={}, geomean_speedup=None, records=[],
                    stage=None, quarantined=False)
        if "BLOCK_SIZE + 1" in spec.kernel_source:
            return {**base, "correct_all": False, "stage": "compile", "n_failed": 1,
                    "failures": [{"workload_id": "w", "stage": "compile", "message": "arange must be pow2"}]}
        if "/ BLOCK_SIZE" in spec.kernel_source:
            return {**base, "correct_all": False, "stage": "correctness", "n_failed": 3,
                    "failures": [{"workload_id": "ln-dev-x-N1000", "stage": "correctness", "message": "max err 0.4"}]}
        score = 1.3 if "rows64" in spec.kernel_source else 0.9
        return {**base, "correct_all": True, "n_passed": 5, "speedups": {"w": score}, "geomean_speedup": score}


@pytest.fixture
def fake_env(monkeypatch):
    FakeEvaluator.calls = []
    FakeEvaluator.interrupt_on = None
    monkeypatch.setattr(agent_mod, "CandidateEvaluator", FakeEvaluator)
    monkeypatch.setattr(agent_mod, "gpu_description", lambda ex: "fixture GPU")
    yield


def _gen(spec: CandidateSpec) -> str:
    return _response(spec.kernel_source, spec.launch.model_dump())


def test_search_loop_feedback_selection_and_failures_preserved(fake_env, tmp_cfg, store):
    wrong = canaries()["ln_tail_mean"][0]
    broken = canaries()["ln_compile_error"][0]
    tmp_cfg.search.max_rounds, tmp_cfg.search.candidates_per_round = 3, 1
    client = ScriptedClient([_gen(wrong), _gen(ln.MANUAL_SPEC), _gen(broken)])
    a = agent_mod.SearchAgent(tmp_cfg, store, "layernorm", client=client, seed=0, log=lambda *x: None)
    sel = a.run()
    cands = {c["id"]: c for c in store.candidates(a.experiment_id)}
    states = {c["state"] for c in cands.values()}
    assert {"incorrect", "compile_failed"} <= states
    assert sel["selected"] == ln.MANUAL_SPEC.short_hash and sel["dev_geomean_speedup"] == pytest.approx(1.3)
    assert cands[ln.MANUAL_SPEC.short_hash]["state"] == "selected"   # best kept despite a later failure
    frozen = CandidateSpec.model_validate_json(
        (store.exp_dir(a.experiment_id) / "frozen" / "candidate.json").read_text())
    assert frozen.source_hash == ln.MANUAL_SPEC.source_hash
    # round 2's prompt carried round 1's numerical failure back to the generator
    prompt2 = json.loads((store.exp_dir(a.experiment_id) / "generations" / "r2-0" / "prompt.json").read_text())
    assert "max err 0.4" in prompt2["messages"][1]["content"]
    assert store.evaluations(a.experiment_id) == []   # fixture mode never writes measured records


def test_duplicates_are_not_re_evaluated(fake_env, tmp_cfg, store):
    tmp_cfg.search.max_rounds, tmp_cfg.search.candidates_per_round = 2, 2
    client = ScriptedClient([_gen(ln.MANUAL_SPEC)])
    a = agent_mod.SearchAgent(tmp_cfg, store, "layernorm", client=client, log=lambda *x: None)
    a.run()
    assert FakeEvaluator.calls.count(ln.MANUAL_SPEC.short_hash) == 1


def test_invalid_generation_is_recorded(fake_env, tmp_cfg, store):
    tmp_cfg.search.max_rounds, tmp_cfg.search.candidates_per_round = 1, 1
    a = agent_mod.SearchAgent(tmp_cfg, store, "layernorm", client=ScriptedClient(["no code here"]),
                              log=lambda *x: None)
    a.run()
    inv = [c for c in store.candidates(a.experiment_id) if c["state"] == "invalid"]
    assert len(inv) == 1 and "python" in inv[0]["failure_message"]


def test_resume_after_interruption(fake_env, tmp_cfg, store):
    tmp_cfg.search.max_rounds, tmp_cfg.search.candidates_per_round = 2, 1
    FakeEvaluator.interrupt_on = "rows64"
    client = ScriptedClient([_gen(ln.MANUAL_SPEC), _gen(ln.MANUAL_SPEC.model_copy(
        update={"launch": LaunchPolicy.single(num_warps=8)}))])
    a = agent_mod.SearchAgent(tmp_cfg, store, "layernorm", client=client, log=lambda *x: None)
    with pytest.raises(KeyboardInterrupt):
        a.run()
    exp_id = a.experiment_id
    assert store.candidate(exp_id, ln.MANUAL_SPEC.short_hash)["state"] == "validated"  # interrupted mid-eval
    seed_evals = FakeEvaluator.calls.count(ln.BASIC_SPEC.short_hash)
    b = agent_mod.SearchAgent(tmp_cfg, store, "layernorm", experiment_id=exp_id, client=client, log=lambda *x: None)
    sel = b.run()
    assert FakeEvaluator.calls.count(ln.BASIC_SPEC.short_hash) == seed_evals   # finished work not repeated
    assert store.candidate(exp_id, ln.MANUAL_SPEC.short_hash)["state"] in ("selected", "revised", "benchmarked")
    assert sel["selected"] is not None


# ---------------------------------------------------------------------------
# Critics
# ---------------------------------------------------------------------------

def test_missing_critic_credentials_disable_only_that_arm(tmp_cfg, monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert run_critic("none", tmp_cfg, "p").status == "not_run"
    r = run_critic("openai", tmp_cfg, "p")
    assert r.status == "unavailable" and "not configured" in r.reason
    tmp_cfg.critics.openai_model = "some-exact-model-id"
    ok, model, reason = critic_availability("openai", tmp_cfg)
    assert not ok and model == "some-exact-model-id"


def test_self_critique_parses_structured_json(tmp_cfg):
    payload = json.dumps({"bottlenecks": ["launch overhead"], "correctness_risks": [], "changes": ["use 2D tiles"]})
    r = run_critic("self", tmp_cfg, "p", local_client=ScriptedClient([payload]))
    assert r.status == "completed" and r.changes == ["use 2D tiles"]
    r = run_critic("self", tmp_cfg, "p", local_client=ScriptedClient(["not json"]))
    assert r.status == "error"
