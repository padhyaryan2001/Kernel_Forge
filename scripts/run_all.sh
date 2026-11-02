#!/usr/bin/env bash
# The full command sequence executed on the RTX 5060 Ti PC to produce RESULTS.md.
# Steps run one at a time (one GPU job at a time); a failing step is logged and the sequence
# continues so independent evidence is still produced. Usage: bash scripts/run_all.sh [log]
set -uo pipefail
PY=${PY:-$HOME/venvs/kernelforge/bin/python}
RLPY=${RLPY:-$HOME/venvs/kernelforge-rl/bin/python}
LOG=${1:-runs/run_all.log}
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG")"
latest() { "$PY" -m kernelforge list | awk -v k="$1" '$1 ~ "^"k {id=$1} END {print id}'; }
step() { echo "=== $(date -u +%FT%TZ) + $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; echo "=== exit=${PIPESTATUS[0]}" | tee -a "$LOG"; }

step "$PY" -m kernelforge doctor
step "$PY" -m kernelforge smoke
step "$PY" -m kernelforge test-reference --device cuda
step "$PY" -m kernelforge selftest-evaluator
step "$PY" -m pytest -rs -p no:cacheprovider

# Local generator: one request inside the memory budget, unloaded before any benchmark.
step "$PY" -m kernelforge generator-check
step bash scripts/demo.sh

# Execution-guided search with the live Qwen2.5-Coder-3B Q4_K_M coder.
step "$PY" -m kernelforge optimize --operator layernorm
OPT_LN=$(latest optimize-layernorm)
step "$PY" -m kernelforge optimize --operator mrope
OPT_MR=$(latest optimize-mrope)

# Frozen held-out evaluations (3 independent sessions + clean-process memory), once each.
for exp in "$(latest tune-layernorm)" "$(latest tune-mrope)" "$OPT_LN" "$OPT_MR"; do
  [ -n "$exp" ] && step "$PY" -m kernelforge evaluate-heldout --experiment "$exp"
done

# Controlled critic comparison (hosted arms are recorded as unavailable without credentials).
step "$PY" -m kernelforge compare-critics --operator layernorm --seeds 0,1 --rounds 2 --heldout

# Development-matrix baselines including torch.compile.
step "$PY" -m kernelforge baseline --operator layernorm --suite dev --sessions 1 --with-compile
step "$PY" -m kernelforge baseline --operator mrope --suite dev --sessions 1 --with-compile

# Optional RL extension (separate environment): smoke-sized GRPO run, then base-vs-adapter evaluation.
[ -x "$RLPY" ] && step "$RLPY" -m kernelforge train-rl --config rl.yaml --kf-config config.yaml --smoke

step "$PY" -m kernelforge report --output runs/report/index.html
for e in $("$PY" -m kernelforge list | awk '{print $1}'); do "$PY" -m kernelforge export-evidence --experiment "$e" >> "$LOG"; done
echo "=== done $(date -u +%FT%TZ)" | tee -a "$LOG"
