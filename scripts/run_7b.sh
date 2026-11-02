#!/usr/bin/env bash
# Agent runs with the local Qwen2.5-Coder-7B-Instruct Q4_K_M generator, using the same search settings as the
# 3B runs in run_all.sh (4 rounds x 2 candidates; critic comparison none/self, seeds 0,1, 2 rounds, held-out).
# Requires config.yaml generator.* to point at the 7B GGUF. Usage: bash scripts/run_7b.sh [log]
set -uo pipefail
PY=${PY:-$HOME/venvs/kernelforge/bin/python}
LOG=${1:-runs/run_7b.log}
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG")"
latest() { "$PY" -m kernelforge list | awk -v k="$1" '$1 ~ "^"k {id=$1} END {print id}'; }
step() { echo "=== $(date -u +%FT%TZ) + $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; echo "=== exit=${PIPESTATUS[0]}" | tee -a "$LOG"; }

step "$PY" -m kernelforge doctor
step "$PY" -m kernelforge generator-check

step "$PY" -m kernelforge optimize --operator layernorm
OPT_LN=$(latest optimize-layernorm)
step "$PY" -m kernelforge optimize --operator mrope
OPT_MR=$(latest optimize-mrope)
for exp in "$OPT_LN" "$OPT_MR"; do
  [ -n "$exp" ] && step "$PY" -m kernelforge evaluate-heldout --experiment "$exp"
done

step "$PY" -m kernelforge compare-critics --operator layernorm --seeds 0,1 --rounds 2 --heldout

step "$PY" -m kernelforge report
step "$PY" -m kernelforge list
