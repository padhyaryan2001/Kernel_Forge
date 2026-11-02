#!/usr/bin/env bash
# Short end-to-end demo: baseline -> generation -> failure/feedback -> correct revision ->
# measurement -> report. The generator is a scripted stand-in (labeled "scripted-demo") so the
# loop is reproducible; every candidate still goes through the real GPU evaluator. Use
# `kernelforge optimize` for the live local Qwen coder.
set -euo pipefail
PY=${PY:-~/venvs/kernelforge/bin/python}
PY=$(eval echo "$PY")

echo "== 1. baseline (manual + untuned Triton vs F.layer_norm, smoke suite)"
$PY -m kernelforge baseline --operator layernorm --suite smoke --sessions 1

echo "== 2-5. generation -> rejected candidate -> feedback -> corrected revision -> measurement"
OUT=$($PY -m kernelforge optimize --operator layernorm --scripted-demo --rounds 2)
echo "$OUT"
EXP=$(echo "$OUT" | grep -o '"experiment_id": "[^"]*"' | head -1 | cut -d'"' -f4)

echo "== candidate history"
$PY - "$EXP" <<'EOF'
import sys
from kernelforge.config import load_config
from kernelforge.store import Store
s = Store(load_config(None).runs_path())
for c in s.candidates(sys.argv[1]):
    print(f"round {c['round']}  {c['id']:14s} {c['origin']:16s} {c['state']:12s} "
          f"dev={c['dev_score'] and round(c['dev_score'], 3)}  {(c['failure_message'] or '')[:110]}")
EOF

echo "== 6. report"
$PY -m kernelforge report --experiment "$EXP"
