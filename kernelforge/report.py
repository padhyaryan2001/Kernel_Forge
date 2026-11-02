"""Static, self-contained HTML report and evidence export.

The report embeds sanitized result data (no API keys, no raw provider traffic, no tensors)
and renders with inline JS/SVG: it opens from disk and can be published to GitHub Pages.
Targets are shown in a separate panel and never mixed with measured values.
"""
from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path
from typing import Any

from .config import KernelForgeConfig
from .contracts import MEMORY_METRIC_DEFINITION, TOLERANCES, geomean, utc_now
from .store import Store


def _round(x: Any, sig: int = 5) -> Any:
    if isinstance(x, float):
        return float(f"{x:.{sig}g}")
    if isinstance(x, list):
        return [_round(v, sig) for v in x]
    if isinstance(x, dict):
        return {k: _round(v, sig) for k, v in x.items()}
    return x


def _thin(values: list, limit: int = 60) -> list:
    if len(values) <= limit:
        return values
    step = len(values) / limit
    return [values[int(i * step)] for i in range(limit)]


SAFE_PROGRESS_KEYS = ("round", "manifest_hashes", "tokens", "generator", "selection", "heldout", "heldout_runs",
                      "contamination", "gpu", "tiers", "dev_geomean", "correct_all", "finished", "suite")


def collect(store: Store, experiment_ids: list[str] | None = None) -> dict[str, Any]:
    exps = [store.experiment(e) for e in experiment_ids] if experiment_ids else store.experiments()
    exps = [e for e in exps if e]
    # include children of critic-comparison parents
    ids = {e["id"] for e in exps}
    for e in list(exps):
        if e["kind"] == "critic_comparison":
            for child in store.experiments(parent_id=e["id"]):
                if child["id"] not in ids:
                    exps.append(child)
                    ids.add(child["id"])
    data: dict[str, Any] = {"experiments": [], "candidates": [], "records": [], "critic_trials": [],
                            "events": []}
    for e in exps:
        prog = {k: v for k, v in (e.get("progress") or {}).items() if k in SAFE_PROGRESS_KEYS}
        data["experiments"].append({"id": e["id"], "operator": e["operator"], "kind": e["kind"], "status": e["status"],
                                    "arm": e["arm"], "seed": e["seed"], "parent_id": e["parent_id"],
                                    "created_at": e["created_at"], "progress": prog})
        for c in store.candidates(e["id"]):
            meta = c.get("meta") or {}
            data["candidates"].append({
                "experiment_id": e["id"], "id": c["id"], "origin": c["origin"], "search_mode": c["search_mode"],
                "round": c["round"], "parent_id": c["parent_id"], "state": c["state"], "retry_count": c["retry_count"],
                "dev_score": c["dev_score"], "failure_stage": c["failure_stage"],
                "failure_message": (c["failure_message"] or "")[:400], "source_hash": c["source_hash"],
                "tokens": (meta.get("input_tokens") or 0) + (meta.get("output_tokens") or 0),
                "model": meta.get("model"), "critic_arm": meta.get("critic_arm"),
            })
        for row in store.evaluations(e["id"]):
            try:
                rec = store.load_record(row["record_path"]).model_dump()
            except (OSError, ValueError):
                continue
            raw = rec.pop("raw_timing_samples", None) or {}
            # Display copy only: evenly thinned to <= 60 points per side; the evidence JSON keeps every sample.
            rec["samples"] = {"baseline": _thin(raw.get("baseline", [])), "candidate": _thin(raw.get("candidate", [])),
                              "session_speedups": raw.get("session_speedups", [])}
            rec["candidate_id"] = row["candidate_id"]
            rec["record_path"] = row["record_path"]
            rec.pop("memory_detail", None)
            hw = rec.pop("hardware", {}) or {}
            sw = rec.pop("software_versions", {}) or {}
            rec["gpu"] = hw.get("gpu")
            rec["gpu_uuid"] = hw.get("uuid")
            rec["driver"] = hw.get("driver_version")
            rec["torch"] = sw.get("torch")
            rec["triton"] = sw.get("triton")
            rec["evaluator_sha256"] = (sw.get("evaluator_sha256") or "")[:16]
            rec["transport"] = (sw.get("isolation") or {}).get("transport")
            data["records"].append(_round(rec))
        for t in store.critic_trials(e["id"]):
            data["critic_trials"].append({k: t[k] for k in ("experiment_id", "arm", "seed", "round", "provider",
                                                             "model", "status", "candidate_id", "input_tokens",
                                                             "output_tokens", "duration_s", "reason")})
        ev = store.exp_dir(e["id"]) / "events.jsonl"
        if ev.exists():
            for line in ev.read_text(encoding="utf-8").splitlines()[-200:]:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                item.pop("command", None)
                data["events"].append({"experiment_id": e["id"], **item})
    return data


def _generator_name(e: dict) -> str:
    return ((e.get("progress") or {}).get("generator") or {}).get("source_model") or "unknown generator"


def critic_summary(data: dict) -> list[dict]:
    """Per-(generator, arm) aggregates for the controlled critic comparison.

    Comparisons run with different generators are never pooled: the generator is held fixed within a row.
    """
    rows: dict[tuple[str, str], dict] = {}
    by_exp = {e["id"]: e for e in data["experiments"]}
    parent_gen = {e["parent_id"]: _generator_name(e) for e in data["experiments"] if e["parent_id"]}
    for t in data["critic_trials"]:
        if t["status"] in ("unavailable", "not_run") and t["round"] == 0:
            gen = parent_gen.get(t["experiment_id"], "unknown generator")
            rows.setdefault((gen, t["arm"]), {"arm": t["arm"], "generator": gen, "status": t["status"],
                                              "model": t["model"], "reason": t["reason"], "runs": 0})
    for e in data["experiments"]:
        if not e["parent_id"] or by_exp.get(e["parent_id"], {}).get("kind") != "critic_comparison":
            continue
        arm, gen = e["arm"] or "none", _generator_name(e)
        r = rows.setdefault((gen, arm), {"arm": arm, "generator": gen, "status": "completed", "runs": 0,
                                         "generations": 0, "duplicates": 0,
                                  "generated": 0, "valid": 0,
                                  "compile_failed": 0, "incorrect": 0, "invalid": 0, "dev_speedups": [],
                                  "heldout_speedups": [], "regressions": 0, "revisions": 0, "gen_tokens": 0,
                                  "critic_tokens": 0, "critic_status": {}})
        r["status"] = "completed"
        r["runs"] += 1
        cands = [c for c in data["candidates"] if c["experiment_id"] == e["id"] and c["origin"] == "generated"]
        scores = {c["id"]: c["dev_score"] for c in data["candidates"] if c["experiment_id"] == e["id"]}
        dups = sum(1 for ev in data["events"]
                   if ev["experiment_id"] == e["id"] and ev.get("event") == "duplicate_generation")
        r["duplicates"] += dups
        r["generations"] += len(cands) + dups
        r["generated"] += len(cands)
        r["revisions"] += len(cands)
        r["gen_tokens"] += e["progress"].get("tokens", 0) or 0  # every generation, duplicates included
        for c in cands:
            if c["state"] == "invalid":
                r["invalid"] += 1
            elif c["state"] == "compile_failed":
                r["compile_failed"] += 1
            elif c["state"] == "incorrect":
                r["incorrect"] += 1
            if c["dev_score"]:
                r["valid"] += 1
                parent_score = scores.get(c["parent_id"])
                if parent_score and c["dev_score"] < parent_score:
                    r["regressions"] += 1
        sel = (e["progress"].get("selection") or {})
        if sel.get("dev_geomean_speedup"):
            r["dev_speedups"].append(sel["dev_geomean_speedup"])
        ho = e["progress"].get("heldout") or {}
        if ho.get("geomean_speedup"):
            r["heldout_speedups"].append(ho["geomean_speedup"])
        for t in data["critic_trials"]:
            if t["experiment_id"] == e["id"]:
                r["critic_tokens"] += (t["input_tokens"] or 0) + (t["output_tokens"] or 0)
                r["critic_status"][t["status"]] = r["critic_status"].get(t["status"], 0) + 1
    out = []
    for r in rows.values():
        if "generated" in r:
            r["valid_rate"] = r["valid"] / r["generations"] if r["generations"] else None
            r["novel_rate"] = r["generated"] / r["generations"] if r["generations"] else None
            r["regression_rate"] = r["regressions"] / r["valid"] if r["valid"] else None
            r["dev_geomean"] = geomean(r.pop("dev_speedups"))
            r["heldout_geomean"] = geomean(r.pop("heldout_speedups"))
        out.append(r)
    order = ["none", "self", "openai", "anthropic", "google"]
    return sorted(out, key=lambda r: (r["generator"], order.index(r["arm"]) if r["arm"] in order else 9))


def render(data: dict, cfg: KernelForgeConfig, title: str = "KernelForge report") -> str:
    payload = {
        "generated_at": utc_now(),
        "data": data,
        "critics": critic_summary(data),
        "targets": cfg.targets.describe(),
        "memory_definition": MEMORY_METRIC_DEFINITION,
        "tolerances": {op: {dt: t.model_dump() for dt, t in d.items()} for op, d in TOLERANCES.items()},
    }
    blob = json.dumps(payload, default=str).replace("</", "<\\/")
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


def write_report(store: Store, cfg: KernelForgeConfig, out_path: Path, experiment_ids: list[str] | None = None) -> Path:
    data = collect(store, experiment_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(data, cfg), encoding="utf-8")
    return out_path


def markdown_summary(store: Store, experiment_ids: list[str]) -> str:
    """Per experiment x candidate x split x baseline x scope tables, computed from evidence records only."""
    import statistics

    lines: list[str] = []
    for eid in experiment_ids:
        exp = store.experiment(eid)
        if exp is None:
            continue
        rows = store.evaluations(eid)
        recs = []
        for r in rows:
            try:
                recs.append((r, store.load_record(r["record_path"])))
            except (OSError, ValueError):
                continue
        lines.append(f"### `{eid}` ({exp['kind']}, {exp['operator']})\n")
        cands = {c["id"]: c for c in store.candidates(eid)}
        groups: dict[tuple, list] = {}
        for row, rec in recs:
            key = (row["candidate_id"], rec.evaluation_split, rec.baseline_id, rec.timing_scope or "-")
            groups.setdefault(key, []).append(rec)
        lines.append("| candidate | origin | split | baseline | scope | correct / workloads | geomean speedup | "
                     "min-max | median memory reduction | sessions | flags |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for (cid, split, base, scope), rs in sorted(groups.items()):
            ws = {r.workload_id for r in rs}
            ok = {r.workload_id for r in rs if r.correctness_pass}
            sp = [r.speedup for r in rs if r.speedup]
            mem = [r.memory_reduction_percent for r in rs if r.memory_reduction_percent is not None]
            flags = sorted({f.split("(")[0] for r in rs for f in r.contamination_flags})
            g = geomean(sp)
            lines.append(f"| `{cid}` | {cands.get(cid, {}).get('origin', '?')} | {split} | {base} | {scope} | "
                         f"{len(ok)} / {len(ws)} | {f'{g:.3f}x' if g else '-'} | "
                         f"{f'{min(sp):.2f}-{max(sp):.2f}' if sp else '-'} | "
                         f"{f'{statistics.median(mem):.1f}%' if mem else '-'} | "
                         f"{max((r.sessions or 0) for r in rs) or '-'} | {', '.join(flags) or '-'} |")
        lines.append("")
    return "\n".join(lines)


def export_evidence(store: Store, experiment_id: str, out_path: Path | None = None) -> Path:
    """Zip of the experiment's immutable artifacts plus an index linking every record to its source hash."""
    exp_dir = store.exp_dir(experiment_id)
    out_path = out_path or exp_dir / f"evidence-{experiment_id}.zip"
    index = {"experiment": store.experiment(experiment_id), "records": [], "candidates": [],
             "exported_at": utc_now()}
    index["experiment"]["config"] = {k: v for k, v in index["experiment"]["config"].items() if k != "critics"}
    for row in store.evaluations(experiment_id):
        rec = store.load_record(row["record_path"])
        index["records"].append({"record_path": row["record_path"], "workload_id": rec.workload_id,
                                 "candidate_hash": rec.candidate_hash, "baseline_id": rec.baseline_id,
                                 "split": rec.evaluation_split, "scope": rec.timing_scope, "status": rec.status,
                                 "speedup": rec.speedup, "memory_reduction_percent": rec.memory_reduction_percent})
    for c in store.candidates(experiment_id):
        index["candidates"].append({k: c[k] for k in ("id", "source_hash", "origin", "state", "round", "dev_score",
                                                      "spec_path", "failure_stage")})
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evidence_index.json", json.dumps(index, indent=1, default=str))
        for path in sorted(exp_dir.rglob("*")):
            if not path.is_file() or path == out_path or path.suffix == ".zip":
                continue
            rel = path.relative_to(exp_dir)
            if rel.parts[0] in ("cache",) or "scratch" in rel.parts or "home" in rel.parts or "tmp" in rel.parts:
                continue
            zf.write(path, arcname=str(rel))
    return out_path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e2dc;--accent:#2f6fb3;--good:#2e7d4f;--bad:#b3412f;--warn:#a26b00;--base:#8a8a84}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecece6;--muted:#a3a39b;--line:#33332f;--accent:#6aa5e6;--good:#5cbf86;--bad:#e2725f;--warn:#e0a83a;--base:#8a8a84}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.sub{color:var(--muted)}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:10px 0}
.targets{border-style:dashed}.targets h2{margin-top:0}.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:12px;border:1px solid var(--line);color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;max-height:520px;overflow-y:auto}.pass{color:var(--good)}.fail{color:var(--bad)}.warn{color:var(--warn)}
.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:center}select{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
.empty{padding:40px;text-align:center;color:var(--muted);font-size:16px}code{font-size:12px}svg text{fill:var(--muted);font-size:11px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.kpi .v{font-size:22px;font-weight:600}.kpi .l{color:var(--muted);font-size:12px}
details summary{cursor:pointer;color:var(--muted)}
</style>
</head>
<body><main>
<h1>KernelForge report</h1>
<div class="sub" id="meta"></div>
<section class="panel targets"><h2>Targets <span class="tag">aspirational, not results</span></h2><div id="targets"></div></section>
<div id="content"></div>
</main>
<script id="kf-data" type="application/json">__DATA__</script>
<script>
const P = JSON.parse(document.getElementById('kf-data').textContent);
const D = P.data; const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = (x, d=3) => x == null ? '<span class="sub">—</span>' : (typeof x === 'number' ? x.toFixed(d) : esc(x));
const fmtB = (b) => b == null ? '—' : (b >= 2**20 ? (b/2**20).toFixed(2)+' MiB' : (b/1024).toFixed(1)+' KiB');
const gmean = (v) => { v = v.filter(x => x > 0); return v.length ? Math.exp(v.reduce((a,b)=>a+Math.log(b),0)/v.length) : null; };
const median = (v) => { if (!v.length) return null; const s=[...v].sort((a,b)=>a-b); const m=s.length>>1; return s.length%2?s[m]:(s[m-1]+s[m])/2; };
$('#meta').textContent = `Generated ${P.generated_at} · ${D.experiments.length} experiments · ${D.records.length} evidence records`;
$('#targets').innerHTML = '<table><tr><th>Target</th><th>Value</th><th>Definition</th></tr>' +
  P.targets.map(t => `<tr><td>${esc(t.target)}</td><td>${esc(t.value)}</td><td>${esc(t.definition)}</td></tr>`).join('') + '</table>';

const measured = D.records.filter(r => r.measurement_kind === 'clean' && r.status === 'passed' && r.speedup != null);
const content = $('#content');
if (!measured.length) {
  content.innerHTML = `<div class="panel empty">No GPU measurements yet.<br><span style="font-size:13px">Run <code>python -m kernelforge baseline --operator layernorm --mode cuda --suite smoke</code> on the CUDA worker, then regenerate this report.</span></div>`;
}
const uniq = (k) => [...new Set(D.records.map(r => r[k]).filter(x => x != null))].sort();
const state = {operator:'all', split:'all', baseline:'torch_eager', dtype:'all', scope:'operator', experiment:'all', kind:'clean'};
function opt(name, vals, label){ return `<label>${label} <select data-k="${name}">${['all',...vals].map(v=>`<option ${v===state[name]?'selected':''}>${esc(v)}</option>`).join('')}</select></label>`; }

function filtered(){
  return D.records.filter(r => (state.operator==='all'||r.operator===state.operator) && (state.split==='all'||r.evaluation_split===state.split)
    && (state.baseline==='all'||r.baseline_id===state.baseline) && (state.dtype==='all'||r.dtype===state.dtype)
    && (state.experiment==='all'||r.experiment_id===state.experiment) && (state.kind==='all'||r.measurement_kind===state.kind)
    && (r.timing_scope==null || state.scope==='all' || r.timing_scope===state.scope));
}
function labelOf(r){ const w=r.workload||{}; return r.operator==='layernorm' ? `M${w.M}×N${w.N} ${r.dtype}${w.input_kind&&w.input_kind!=='normal'?' '+w.input_kind:''}` : `B${w.B} Hq${w.Hq}/Hkv${w.Hkv} S${w.S} D${w.D} ${r.dtype} ${w.position_kind} ${w.contract==='precomputed_trig'?'pre':'pos'}`; }

function bars(rows){
  const W = 760, rowH = 18, H = rows.length*rowH + 30, max = Math.max(1.5, ...rows.map(r => r.speedup_ci95 ? r.speedup_ci95[1] : r.speedup));
  const x = (v) => 230 + (W-260) * (v/max);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="per-workload speedup">`;
  s += `<line x1="${x(1)}" y1="0" x2="${x(1)}" y2="${H-20}" stroke="var(--base)" stroke-dasharray="3 3"/><text x="${x(1)+3}" y="${H-6}">1.0× (baseline)</text>`;
  rows.forEach((r,i) => { const y = i*rowH+4; const c = r.speedup>=1?'var(--good)':'var(--bad)';
    s += `<text x="0" y="${y+12}">${esc(labelOf(r)).slice(0,38)}</text><rect x="230" y="${y+3}" width="${Math.max(1,x(r.speedup)-230)}" height="11" fill="${c}" opacity=".75"><title>${r.speedup.toFixed(3)}×</title></rect>`;
    if (r.speedup_ci95) s += `<line x1="${x(r.speedup_ci95[0])}" x2="${x(r.speedup_ci95[1])}" y1="${y+8.5}" y2="${y+8.5}" stroke="var(--ink)"/>`;
    s += `<text x="${x(r.speedup)+4}" y="${y+12}">${r.speedup.toFixed(2)}×</text>`; });
  return s + '</svg>';
}
function dist(r){
  const b = r.samples.baseline||[], c = r.samples.candidate||[]; if (!b.length) return '<span class="sub">no raw samples</span>';
  const all=[...b,...c], lo=Math.min(...all), hi=Math.max(...all), W=760, H=90, x=v=>20+(W-40)*((v-lo)/((hi-lo)||1));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="latency samples">`;
  [[b,'var(--base)',20,'baseline '+r.baseline_id],[c,'var(--accent)',55,'candidate']].forEach(([v,col,y,l]) => {
    v.forEach(t => s += `<circle cx="${x(t)}" cy="${y+ (Math.random()*8-4)}" r="2" fill="${col}" opacity=".6"/>`);
    const m = median(v); s += `<line x1="${x(m)}" x2="${x(m)}" y1="${y-9}" y2="${y+9}" stroke="${col}" stroke-width="2"/><text x="${x(m)+4}" y="${y-9}">${esc(l)} median ${m.toFixed(4)} ms</text>`; });
  return s + `<text x="20" y="${H-2}">${lo.toFixed(4)} ms</text><text x="${W-90}" y="${H-2}">${hi.toFixed(4)} ms</text></svg>`;
}

function draw(){
  const rows = filtered();
  const timed = rows.filter(r => r.status==='passed' && r.speedup!=null && r.timing_scope);
  const correctness = rows.filter(r => r.correctness_pass != null || r.status !== 'passed');
  const byCand = {};
  rows.forEach(r => { const k = r.experiment_id+'|'+r.candidate_id+'|'+r.evaluation_split; (byCand[k] ||= []).push(r); });
  let h = `<section class="panel"><div class="filters">${opt('experiment', uniq('experiment_id'), 'Experiment')}${opt('operator', uniq('operator'), 'Operator')}${opt('split', uniq('evaluation_split'), 'Split')}${opt('baseline', uniq('baseline_id'), 'Baseline')}${opt('dtype', uniq('dtype'), 'Dtype')}${opt('scope', ['operator','kernel','graph'], 'Timing scope')}${opt('kind', uniq('measurement_kind'), 'Measurement')}</div>
  <p class="sub">GPUs in selection: ${esc([...new Set(rows.map(r=>r.gpu).filter(Boolean))].join(', ')||'—')}. Results from different GPUs, dtypes or contracts are never merged into one number; geomeans below are per candidate × split × baseline × scope as filtered.</p></section>`;
  const groups = Object.entries(byCand).map(([k, rs]) => { const [e,c,s] = k.split('|'); const ws = new Set(rs.map(r=>r.workload_id));
    const ok = new Set(rs.filter(r=>r.correctness_pass===true).map(r=>r.workload_id)); const t = rs.filter(r=>r.speedup!=null);
    const mem = rs.filter(r=>r.memory_reduction_percent!=null).map(r=>r.memory_reduction_percent);
    const flags = new Set(rs.flatMap(r=>r.contamination_flags||[]).map(f=>f.split('(')[0]));
    return {e,c,s,n:ws.size,ok:ok.size,g:gmean(t.map(r=>r.speedup)),nt:t.length,mem:median(mem),flags:[...flags],kind:rs[0].measurement_kind}; });
  h += `<h2>Correctness coverage and aggregate speedup</h2><section class="panel scroll"><table><tr><th>Experiment</th><th>Candidate</th><th>Split</th><th class="n">Correct / workloads</th><th class="n">Geomean speedup (n)</th><th class="n">Median memory reduction</th><th>Flags</th></tr>` +
    groups.map(g => `<tr><td>${esc(g.e)}</td><td><code>${esc(g.c)}</code></td><td>${esc(g.s)}</td><td class="n ${g.ok===g.n?'pass':'fail'}">${g.ok} / ${g.n}</td><td class="n">${g.g?g.g.toFixed(3)+'× ('+g.nt+')':'—'}</td><td class="n">${g.mem!=null?g.mem.toFixed(1)+'%':'—'}</td><td class="sub">${esc(g.flags.join(', '))}</td></tr>`).join('') + '</table></section>';
  if (timed.length) h += `<h2>Per-workload speedup (${esc(state.baseline)} ÷ candidate, median of raw samples; bar = 95% bootstrap CI of paired ratio)</h2><section class="panel">${bars(timed.slice(0,60))}</section>`;
  h += `<h2>Workloads</h2><section class="panel scroll"><table><tr><th>Workload</th><th>Split</th><th>Status</th><th class="n">Max abs err</th><th class="n">Baseline ms (p10–p90)</th><th class="n">Candidate ms (p10–p90)</th><th class="n">Speedup [CI]</th><th class="n">Peak bytes base → cand</th><th class="n">Mem Δ</th><th>Flags</th></tr>` +
    rows.map((r,i) => `<tr data-i="${i}"><td>${esc(labelOf(r))}<br><span class="sub">${esc(r.workload_id)} · ${esc(r.baseline_id)}/${esc(r.timing_scope||'—')}</span></td><td>${esc(r.evaluation_split)}</td>
    <td class="${r.status==='passed'?'pass':(r.status==='skipped'||r.status==='not_run'?'warn':'fail')}">${esc(r.status)}${r.failure_stage?' ('+esc(r.failure_stage)+')':''}${r.failure_message?'<br><span class="sub">'+esc(r.failure_message).slice(0,160)+'</span>':''}</td>
    <td class="n">${r.max_abs_error!=null?r.max_abs_error.toExponential(2):'—'}</td><td class="n">${fmt(r.baseline_median_ms,4)}${r.baseline_p10_p90_ms?'<br><span class="sub">'+r.baseline_p10_p90_ms.map(x=>x.toFixed(4)).join('–')+'</span>':''}</td>
    <td class="n">${fmt(r.candidate_median_ms,4)}${r.candidate_p10_p90_ms?'<br><span class="sub">'+r.candidate_p10_p90_ms.map(x=>x.toFixed(4)).join('–')+'</span>':''}</td>
    <td class="n">${r.speedup!=null?r.speedup.toFixed(3)+'×':'—'}${r.speedup_ci95?'<br><span class="sub">['+r.speedup_ci95.map(x=>x.toFixed(3)).join(', ')+']</span>':''}</td>
    <td class="n">${r.baseline_peak_bytes!=null?fmtB(r.baseline_peak_bytes)+' → '+fmtB(r.candidate_peak_bytes):'—'}</td><td class="n">${r.memory_reduction_percent!=null?r.memory_reduction_percent.toFixed(1)+'%':'—'}</td>
    <td class="sub">${esc((r.contamination_flags||[]).join(', '))}</td></tr>`).join('') + '</table></section>';
  const withSamples = timed.filter(r => (r.samples.baseline||[]).length);
  if (withSamples.length) h += `<h2>Latency distribution</h2><section class="panel"><select id="distSel">${withSamples.map((r,i)=>`<option value="${i}">${esc(labelOf(r))} · ${esc(r.baseline_id)}/${esc(r.timing_scope)}</option>`).join('')}</select><div id="dist"></div></section>`;
  h += `<h2>Memory metric</h2><section class="panel"><p>${esc(P.memory_definition)}</p><p class="sub">memory reduction = 100 × (1 − candidate_peak / baseline_peak). A record with no memory value was not measured; it is not zero.</p></section>`;
  const exps = D.experiments.filter(e => state.experiment==='all' || e.id===state.experiment);
  h += `<h2>Candidate history</h2><section class="panel scroll"><table><tr><th>Experiment</th><th>Round</th><th>Candidate</th><th>Origin</th><th>State</th><th class="n">Dev geomean</th><th>Failure</th><th class="n">Gen tokens</th></tr>` +
    D.candidates.filter(c => exps.some(e=>e.id===c.experiment_id)).map(c => `<tr><td>${esc(c.experiment_id)}</td><td>${c.round}</td><td><code>${esc(c.id)}</code>${c.parent_id?'<br><span class="sub">parent '+esc(c.parent_id)+'</span>':''}</td><td>${esc(c.origin)}${c.search_mode&&c.search_mode!=='manual'?' / '+esc(c.search_mode):''}</td><td class="${c.dev_score?'pass':(['invalid','compile_failed','incorrect','benchmark_failed','timed_out','worker_error'].includes(c.state)?'fail':'')}">${esc(c.state)}</td><td class="n">${c.dev_score?c.dev_score.toFixed(3)+'×':'—'}</td><td class="sub">${esc(c.failure_stage||'')} ${esc(c.failure_message||'').slice(0,180)}</td><td class="n">${c.tokens||'—'}</td></tr>`).join('') + '</table></section>';
  const gev = D.events.filter(e => (e.event||'').startsWith('generator') && exps.some(x => x.id === e.experiment_id));
  if (gev.length) h += `<h2>Generator lifecycle (stopped and VRAM-checked before every evaluation phase)</h2><section class="panel scroll"><table><tr><th>Experiment</th><th>Event</th><th>At</th><th>Details</th></tr>` +
    gev.map(e => `<tr><td>${esc(e.experiment_id)}</td><td>${esc(e.event)}</td><td>${esc(e.at)}</td><td class="sub">${esc(e.event === 'generator_started'
      ? `pid ${e.pid}, load ${e.load_seconds}s, free VRAM ${e.free_vram_before_mib} → ${e.free_vram_loaded_mib} MiB (server ~${e.server_vram_mib} MiB), host RSS ${e.host_rss_mib} MiB`
      : `shutdown ${e.shutdown_seconds}s, exit ${e.exit_code}, free VRAM after stop ${e.free_vram_after_stop_mib} MiB, released=${e.vram_released}`)}</td></tr>`).join('') + '</table></section>';
  if (P.critics.length) h += `<h2>Critic comparison (generator fixed; only the critic varies)</h2><section class="panel scroll"><table><tr><th>Arm</th><th>Status</th><th class="n">Runs</th><th class="n">Generations (dup)</th><th class="n">Valid-kernel rate</th><th class="n">Invalid / compile / incorrect</th><th class="n">Dev geomean</th><th class="n">Held-out geomean</th><th class="n">Regression rate</th><th class="n">Revisions</th><th class="n">Gen / critic tokens</th></tr>` +
    P.critics.map(r => `<tr><td>${esc(r.arm)}${r.model?'<br><span class="sub">'+esc(r.model)+'</span>':''}<br><span class="sub">generator ${esc(r.generator)}</span></td><td class="${r.status==='completed'?'pass':'warn'}">${esc(r.status)}${r.reason?'<br><span class="sub">'+esc(r.reason)+'</span>':''}</td><td class="n">${r.runs}</td><td class="n">${r.generations!=null?r.generations+' ('+r.duplicates+')':'—'}</td><td class="n">${r.valid_rate!=null?(100*r.valid_rate).toFixed(0)+'%':'—'}</td><td class="n">${r.generated!=null?r.invalid+' / '+r.compile_failed+' / '+r.incorrect:'—'}</td><td class="n">${r.dev_geomean?r.dev_geomean.toFixed(3)+'×':'—'}</td><td class="n">${r.heldout_geomean?r.heldout_geomean.toFixed(3)+'×':'—'}</td><td class="n">${r.regression_rate!=null?(100*r.regression_rate).toFixed(0)+'%':'—'}</td><td class="n">${r.revisions??'—'}</td><td class="n">${r.gen_tokens!=null?r.gen_tokens+' / '+r.critic_tokens:'—'}</td></tr>`).join('') + '</table><p class="sub">Valid-kernel rate = novel candidates that passed every dev workload and were benchmarked ÷ all generations (duplicates of an existing candidate count as generations, not as valid kernels). Cost is not estimated unless actual usage and a documented pricing source are configured.</p></section>';
  h += `<h2>Reproducibility</h2><section class="panel scroll"><table><tr><th>Experiment</th><th>Kind</th><th>Status</th><th>Manifest hashes</th><th>Generator</th><th>Held-out</th></tr>` +
    exps.map(e => { const p=e.progress||{}; const g=p.generator||{}; return `<tr><td>${esc(e.id)}${e.arm?'<br><span class="sub">arm '+esc(e.arm)+' seed '+esc(e.seed)+'</span>':''}</td><td>${esc(e.kind)}</td><td>${esc(e.status)}</td><td class="sub">${Object.entries(p.manifest_hashes||{}).map(([k,v])=>esc(k)+': '+esc(v).slice(0,12)).join('<br>')}</td><td class="sub">${g.source_model?esc(g.source_model)+' '+esc(g.quantization||'')+'<br>'+esc(g.source_file||'')+(g.sha256?'<br>sha256 '+esc(g.sha256).slice(0,16):''):'—'}</td><td class="sub">${p.heldout?'correct_all='+p.heldout.correct_all+', geomean '+fmt(p.heldout.geomean_speedup)+(p.contamination?'<br><span class=fail>'+esc(p.contamination)+'</span>':''):'not run'}</td></tr>`; }).join('') +
    `</table><details><summary>Software / hardware per record</summary><table><tr><th>Record</th><th>GPU</th><th>Driver</th><th>torch / triton</th><th>Evaluator</th><th>Transport</th><th>Candidate hash</th></tr>${rows.slice(0,300).map(r=>`<tr><td class="sub">${esc(r.record_path)}</td><td>${esc(r.gpu)}</td><td>${esc(r.driver)}</td><td>${esc(r.torch)} / ${esc(r.triton)}</td><td><code>${esc(r.evaluator_sha256)}</code></td><td>${esc(r.transport)}</td><td><code>${esc((r.candidate_hash||'').slice(0,16))}</code></td></tr>`).join('')}</table></details>
    <details><summary>Correctness tolerances (fixed before search)</summary><pre>${esc(JSON.stringify(P.tolerances,null,1))}</pre></details></section>`;
  content.innerHTML = (measured.length ? '' : content.innerHTML.split('</div>')[0]+'</div>') + h;
  content.querySelectorAll('select[data-k]').forEach(s => s.onchange = () => { state[s.dataset.k] = s.value; draw(); });
  const ds = $('#distSel'); if (ds) { const show = () => $('#dist').innerHTML = dist(withSamples[+ds.value]); ds.onchange = show; show(); }
}
if (D.records.length || D.candidates.length) draw();
</script>
</body></html>
"""
