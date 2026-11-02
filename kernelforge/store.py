"""SQLite experiment state plus immutable artifacts under runs/<experiment_id>/.

SQLite holds status and relationships; sources, prompts, results and logs are files.
State transitions run inside transactions and are checked against the agent state machine.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .contracts import CandidateSpec, EvaluationRecord, utc_now

# Agent state machine. Failure states keep artifacts and retry counts.
STATES = ("queued", "generated", "validated", "compiled", "correctness_checked", "benchmarked",
          "critiqued", "revised", "selected", "heldout_evaluated")
FAILURE_STATES = ("invalid", "duplicate", "compile_failed", "incorrect", "benchmark_failed",
                  "timed_out", "worker_error")
RETRYABLE = ("timed_out", "worker_error")
TRANSITIONS: dict[str, set[str]] = {
    "queued": {"generated", "invalid"},
    "generated": {"validated", "invalid", "duplicate"},
    "validated": {"compiled", "compile_failed", "timed_out", "worker_error", "incorrect", "invalid"},
    "compiled": {"correctness_checked", "incorrect", "timed_out", "worker_error"},
    "correctness_checked": {"benchmarked", "benchmark_failed", "timed_out", "worker_error"},
    "benchmarked": {"critiqued", "revised", "selected"},
    "critiqued": {"revised", "selected"},
    "revised": {"selected", "critiqued"},
    "selected": {"heldout_evaluated"},
    "timed_out": {"validated"},
    "worker_error": {"validated"},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
  id TEXT PRIMARY KEY, operator TEXT, kind TEXT, status TEXT, arm TEXT, seed INTEGER,
  parent_id TEXT, config_json TEXT, progress_json TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS candidates (
  experiment_id TEXT, id TEXT, source_hash TEXT, operator TEXT, origin TEXT, search_mode TEXT,
  round INTEGER, parent_id TEXT, state TEXT, retry_count INTEGER DEFAULT 0, failure_stage TEXT,
  failure_message TEXT, dev_score REAL, spec_path TEXT, meta_json TEXT, created_at TEXT, updated_at TEXT,
  PRIMARY KEY (experiment_id, id));
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, experiment_id TEXT, candidate_id TEXT, kind TEXT, status TEXT,
  task_path TEXT, result_path TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, candidate_id TEXT, workload_id TEXT,
  split TEXT, baseline_id TEXT, timing_scope TEXT, status TEXT, speedup REAL,
  memory_reduction_percent REAL, record_path TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS critic_trials (
  id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, arm TEXT, seed INTEGER, round INTEGER,
  provider TEXT, model TEXT, status TEXT, candidate_id TEXT, critique_path TEXT, input_tokens INTEGER,
  output_tokens INTEGER, duration_s REAL, reason TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS transitions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT, candidate_id TEXT, from_state TEXT,
  to_state TEXT, note TEXT, at TEXT);
"""


class StateError(RuntimeError):
    pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


class Store:
    def __init__(self, runs_dir: Path):
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runs_dir / "kernelforge.sqlite3"
        self._conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise

    # -- artifacts --------------------------------------------------------
    def exp_dir(self, experiment_id: str) -> Path:
        d = self.runs_dir / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_once(self, path: Path, text: str) -> Path:
        """Immutable artifact write: identical rewrites are no-ops, conflicting ones get a suffix."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_text(encoding="utf-8") == text:
                return path
            stem, suffix, n = path.stem, path.suffix, 1
            while path.exists():
                path = path.with_name(f"{stem}.{n}{suffix}")
                n += 1
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return path

    def write_json(self, path: Path, obj: Any) -> Path:
        if hasattr(obj, "model_dump_json"):
            text = obj.model_dump_json(indent=1)
        else:
            text = json.dumps(obj, indent=1, sort_keys=True, default=str)
        return self.write_once(path, text)

    def rel(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.runs_dir.resolve())).replace("\\", "/")
        except ValueError:
            return str(path)

    # -- experiments ------------------------------------------------------
    def create_experiment(self, experiment_id: str, operator: str, kind: str, config: dict,
                          arm: str | None = None, seed: int | None = None, parent_id: str | None = None) -> str:
        now = utc_now()
        with self.tx() as c:
            c.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (experiment_id, operator, kind, "running", arm, seed, parent_id,
                       json.dumps(config, default=str), json.dumps({}), now, now))
        self.exp_dir(experiment_id)
        return experiment_id

    def experiment(self, experiment_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json") or "{}")
        d["progress"] = json.loads(d.pop("progress_json") or "{}")
        return d

    def experiments(self, parent_id: str | None = None) -> list[dict]:
        if parent_id:
            rows = self._conn.execute("SELECT id FROM experiments WHERE parent_id=? ORDER BY created_at", (parent_id,))
        else:
            rows = self._conn.execute("SELECT id FROM experiments ORDER BY created_at")
        return [self.experiment(r["id"]) for r in rows]

    def update_experiment(self, experiment_id: str, status: str | None = None, progress: dict | None = None) -> None:
        with self.tx() as c:
            if status:
                c.execute("UPDATE experiments SET status=?, updated_at=? WHERE id=?", (status, utc_now(), experiment_id))
            if progress is not None:
                c.execute("UPDATE experiments SET progress_json=?, updated_at=? WHERE id=?",
                          (json.dumps(progress, default=str), utc_now(), experiment_id))

    # -- candidates -------------------------------------------------------
    def add_candidate(self, experiment_id: str, spec: CandidateSpec, round_: int, parent_id: str | None,
                      meta: dict, extra_files: dict[str, str] | None = None) -> tuple[str, bool]:
        """Persist a candidate. Returns (candidate_id, is_new); duplicates are recorded once."""
        cid = spec.short_hash
        existing = self.candidate(experiment_id, cid)
        if existing is not None:
            return cid, False
        same = self.semantic_duplicate(experiment_id, spec)
        if same is not None:
            return same, False
        cdir = self.exp_dir(experiment_id) / "candidates" / cid
        self.write_once(cdir / "kernel.py", spec.kernel_source)
        spec_path = self.write_json(cdir / "spec.json", spec)
        for name, text in (extra_files or {}).items():
            self.write_once(cdir / name, text)
        now = utc_now()
        with self.tx() as c:
            c.execute("INSERT INTO candidates (experiment_id,id,source_hash,operator,origin,search_mode,round,parent_id,"
                      "state,retry_count,spec_path,meta_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (experiment_id, cid, spec.source_hash, spec.operator, spec.origin, spec.search_mode, round_,
                       parent_id, "queued", 0, self.rel(spec_path), json.dumps(meta, default=str), now, now))
        return cid, True

    def semantic_duplicate(self, experiment_id: str, spec: CandidateSpec) -> str | None:
        """Existing candidate whose AST + canonical launch policy equals this spec's, if any."""
        key = spec.semantic_hash
        for row in self._conn.execute("SELECT id, spec_path FROM candidates WHERE experiment_id=? AND spec_path "
                                      "IS NOT NULL", (experiment_id,)).fetchall():
            try:
                other = CandidateSpec.model_validate_json((self.runs_dir / row["spec_path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if other.semantic_hash == key:
                return row["id"]
        return None

    def add_invalid_generation(self, experiment_id: str, generation_id: str, text_hash: str, operator: str,
                               round_: int, parent_id: str | None, problems: list[str], meta: dict) -> str:
        """A generator output that never became a valid candidate (kept for valid-kernel-rate stats)."""
        cid = f"invalid-{generation_id}"
        now = utc_now()
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO candidates (experiment_id,id,source_hash,operator,origin,search_mode,round,"
                      "parent_id,state,retry_count,failure_stage,failure_message,meta_json,created_at,updated_at) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (experiment_id, cid, text_hash, operator, "generated", meta.get("search_mode"), round_,
                       parent_id, "invalid", 0, "validation", "; ".join(problems)[:4000],
                       json.dumps(meta, default=str), now, now))
        return cid

    def clear_evaluations(self, experiment_id: str, candidate_id: str, split: str | None = None) -> None:
        """Drop index rows of an interrupted evaluation before it is re-run (files stay as artifacts)."""
        with self.tx() as c:
            if split:
                c.execute("DELETE FROM evaluations WHERE experiment_id=? AND candidate_id=? AND split=?",
                          (experiment_id, candidate_id, split))
            else:
                c.execute("DELETE FROM evaluations WHERE experiment_id=? AND candidate_id=?",
                          (experiment_id, candidate_id))

    def candidate(self, experiment_id: str, candidate_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM candidates WHERE experiment_id=? AND id=?",
                                 (experiment_id, candidate_id)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["meta"] = json.loads(d.pop("meta_json") or "{}")
        return d

    def candidates(self, experiment_id: str) -> list[dict]:
        rows = self._conn.execute("SELECT id FROM candidates WHERE experiment_id=? ORDER BY created_at, id",
                                  (experiment_id,)).fetchall()
        return [self.candidate(experiment_id, r["id"]) for r in rows]

    def load_spec(self, experiment_id: str, candidate_id: str) -> CandidateSpec:
        cand = self.candidate(experiment_id, candidate_id)
        return CandidateSpec.model_validate_json((self.runs_dir / cand["spec_path"]).read_text(encoding="utf-8"))

    def transition(self, experiment_id: str, candidate_id: str, to_state: str, note: str = "",
                   failure_stage: str | None = None, dev_score: float | None = None) -> None:
        with self.tx() as c:
            row = c.execute("SELECT state, retry_count FROM candidates WHERE experiment_id=? AND id=?",
                            (experiment_id, candidate_id)).fetchone()
            if row is None:
                raise StateError(f"unknown candidate {candidate_id}")
            frm = row["state"]
            if to_state not in TRANSITIONS.get(frm, set()):
                raise StateError(f"illegal transition {frm} -> {to_state} for {candidate_id}")
            retry = row["retry_count"] + (1 if frm in RETRYABLE else 0)
            fields = ["state=?", "retry_count=?", "updated_at=?"]
            values: list[Any] = [to_state, retry, utc_now()]
            if to_state in FAILURE_STATES:
                fields += ["failure_stage=?", "failure_message=?"]
                values += [failure_stage or to_state, note[:4000]]
            if dev_score is not None:
                fields.append("dev_score=?")
                values.append(dev_score)
            c.execute(f"UPDATE candidates SET {', '.join(fields)} WHERE experiment_id=? AND id=?",
                      (*values, experiment_id, candidate_id))
            c.execute("INSERT INTO transitions (experiment_id,candidate_id,from_state,to_state,note,at) VALUES (?,?,?,?,?,?)",
                      (experiment_id, candidate_id, frm, to_state, note[:1000], utc_now()))

    def transitions(self, experiment_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM transitions WHERE experiment_id=? ORDER BY id", (experiment_id,))]

    # -- tasks / evaluations / critics ------------------------------------
    def record_task(self, task_id: str, experiment_id: str, candidate_id: str | None, kind: str, status: str,
                    task_path: Path, result_path: Path) -> None:
        now = utc_now()
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?,?)",
                      (task_id, experiment_id, candidate_id, kind, status, self.rel(task_path),
                       self.rel(result_path), now, now))

    def add_evaluation(self, record: EvaluationRecord, candidate_id: str) -> Path:
        d = self.exp_dir(record.experiment_id) / "evaluations" / candidate_id / record.evaluation_split
        scope = record.timing_scope or "none"
        short_id = "-".join(record.workload_id.split("-")[:3])  # '<op>-<split>-<idx>' keeps Windows paths short
        path = self.write_json(d / f"{short_id}__{record.baseline_id}__{scope}.json", record)
        with self.tx() as c:
            c.execute("INSERT INTO evaluations (experiment_id,candidate_id,workload_id,split,baseline_id,timing_scope,"
                      "status,speedup,memory_reduction_percent,record_path,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (record.experiment_id, candidate_id, record.workload_id, record.evaluation_split,
                       record.baseline_id, record.timing_scope, record.status, record.speedup,
                       record.memory_reduction_percent, self.rel(path), utc_now()))
        return path

    def evaluations(self, experiment_id: str, candidate_id: str | None = None, split: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM evaluations WHERE experiment_id=?", [experiment_id]
        if candidate_id:
            q += " AND candidate_id=?"
            args.append(candidate_id)
        if split:
            q += " AND split=?"
            args.append(split)
        return [dict(r) for r in self._conn.execute(q + " ORDER BY id", args)]

    def load_record(self, record_path: str) -> EvaluationRecord:
        return EvaluationRecord.model_validate_json((self.runs_dir / record_path).read_text(encoding="utf-8"))

    def add_critic_trial(self, experiment_id: str, arm: str, seed: int, round_: int, candidate_id: str | None,
                         critique, critique_path: Path | None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO critic_trials (experiment_id,arm,seed,round,provider,model,status,candidate_id,"
                      "critique_path,input_tokens,output_tokens,duration_s,reason,created_at) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (experiment_id, arm, seed, round_, critique.provider, critique.model, critique.status,
                       candidate_id, self.rel(critique_path) if critique_path else None,
                       critique.usage.input_tokens, critique.usage.output_tokens, critique.duration_s,
                       critique.reason, utc_now()))

    def critic_trials(self, experiment_id: str) -> list[dict]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM critic_trials WHERE experiment_id=? ORDER BY id", (experiment_id,))]
