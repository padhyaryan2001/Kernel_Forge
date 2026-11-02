"""Fixed GPU worker entry point.

    python -m kernelforge.worker run --task task.json --result result.json
    python -m kernelforge.worker run-stdin        # task JSON on stdin, result JSON on stdout (SSH adapter)

Exit codes: 0 completed (inspect result status), 2 invalid task bundle, 3 quarantined (CUDA fault).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from pydantic import ValidationError

from .contracts import TaskBundle


def _execute(task_json: str, scratch: Path) -> tuple[str, int]:
    from .evaluate import execute_task

    try:
        task = TaskBundle.model_validate_json(task_json)
    except ValidationError as exc:
        return json.dumps({"error": "invalid task bundle", "detail": exc.errors()[:5]}, default=str), 2
    result = execute_task(task, scratch)
    return result.model_dump_json(indent=1), 3 if result.status == "quarantined" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelforge.worker")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--task", required=True)
    run.add_argument("--result", required=True)
    sub.add_parser("run-stdin")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        task_path = Path(args.task)
        scratch = Path(os.environ.get("KERNELFORGE_SCRATCH", task_path.parent / "scratch"))
        payload, code = _execute(task_path.read_text(encoding="utf-8"), scratch)
        tmp = Path(args.result).with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(args.result)
        return code
    with tempfile.TemporaryDirectory(prefix="kf-") as tmp_dir:
        payload, code = _execute(sys.stdin.read(), Path(tmp_dir))
    sys.stdout.write(payload)
    return code


if __name__ == "__main__":
    sys.exit(main())
