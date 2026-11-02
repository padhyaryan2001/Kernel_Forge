from __future__ import annotations

import json
import re

from kernelforge.contracts import EvaluationRecord
from kernelforge.kernels import layernorm as ln
from kernelforge.report import collect, export_evidence, render, write_report


def _embedded(html: str) -> dict:
    m = re.search(r'<script id="kf-data" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_empty_report_is_honest(store, tmp_cfg, tmp_path):
    out = write_report(store, tmp_cfg, tmp_path / "r" / "index.html")
    html = out.read_text()
    assert "No GPU measurements yet" in html
    data = _embedded(html)
    assert data["data"]["records"] == []
    assert [t["target"] for t in data["targets"]][0] == "LayerNorm forward latency"
    assert "aspirational, not results" in html


def test_fixture_records_are_not_measurements(store, tmp_cfg, tmp_path):
    store.create_experiment("e", "layernorm", "optimize", {})
    cid, _ = store.add_candidate("e", ln.MANUAL_SPEC, 0, None, {})
    rec = EvaluationRecord(experiment_id="e", workload_id="ln-dev-000-M1-N128-float16", candidate_hash="h",
                           operator="layernorm", evaluation_split="dev", status="passed", dtype="float16",
                           shapes={"x": [1, 128]}, contract="forward_inference", baseline_id="torch_eager",
                           correctness_pass=True, speedup=9.9, baseline_median_ms=1.0, candidate_median_ms=0.1,
                           timing_scope="operator", measurement_kind="fixture",
                           raw_timing_samples={"baseline": [1.0], "candidate": [0.1]})
    store.add_evaluation(rec, cid)
    html = render(collect(store), tmp_cfg)
    assert "No GPU measurements yet" in html   # the fixture speedup is never shown as a result
    assert _embedded(html)["data"]["records"][0]["measurement_kind"] == "fixture"


def test_report_has_no_secrets(store, tmp_cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-leak")
    store.create_experiment("e", "layernorm", "optimize", {"generator": {"model": "x"}})
    html = render(collect(store), tmp_cfg)
    assert "sk-test-should-not-leak" not in html


def test_evidence_export_indexes_records(store, tmp_cfg):
    store.create_experiment("e", "layernorm", "baseline", {"critics": {"x": 1}})
    cid, _ = store.add_candidate("e", ln.MANUAL_SPEC, 0, None, {})
    zp = export_evidence(store, "e")
    import zipfile

    with zipfile.ZipFile(zp) as zf:
        idx = json.loads(zf.read("evidence_index.json"))
        assert idx["candidates"][0]["source_hash"] == ln.MANUAL_SPEC.source_hash
        assert f"candidates/{cid}/kernel.py" in zf.namelist()
        assert "critics" not in idx["experiment"]["config"]


def test_critic_rows_never_pool_generators(store):
    from kernelforge.report import critic_summary

    for parent, model in (("c3", "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"), ("c7", "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF")):
        store.create_experiment(parent, "layernorm", "critic_comparison", {})
        store.create_experiment(f"{parent}-none-s0", "layernorm", "optimize", {}, arm="none", seed=0, parent_id=parent)
        store.update_experiment(f"{parent}-none-s0", progress={"generator": {"source_model": model}, "tokens": 10})
    rows = critic_summary(collect(store))
    assert [(r["generator"], r["arm"], r["runs"]) for r in rows] == [
        ("Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", "none", 1), ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "none", 1)]
