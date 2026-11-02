#!/usr/bin/env bash
# Environment setup used on the RTX 5060 Ti PC (WSL2 Ubuntu; also fine on native Linux).
# Idempotent; run from the repository root. Needs no sudo: Python comes from uv, CUDA
# runtime libraries come with the PyTorch wheel and the llama.cpp release tarball.
# Do NOT install a Linux NVIDIA display driver inside WSL: the Windows driver is projected.
set -euo pipefail

TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
LLAMA_TAG=${LLAMA_TAG:-b12024}
GGUF_REV=${GGUF_REV:-f74adce6aa16316c625447af059dbebe4983757c}
GGUF_SHA=724fb256bec1ff062b2f65e4569e871ad2e95ab2a3989723d1769c54294730b7
export PATH="$HOME/.local/bin:$PATH"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv

echo "== evaluator / controller environment"
[ -x ~/venvs/kernelforge/bin/python ] || uv venv --python 3.12 ~/venvs/kernelforge
uv pip install --python ~/venvs/kernelforge/bin/python "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
uv pip install --python ~/venvs/kernelforge/bin/python -e ".[test]"
~/venvs/kernelforge/bin/python scripts/cuda_smoke.py

echo "== llama.cpp ${LLAMA_TAG} (official CUDA 13.4 build; sm_120 capable)"
DEST=~/opt/llama.cpp-${LLAMA_TAG}
mkdir -p "$DEST" ~/models ~/opt/dl
for f in "llama-${LLAMA_TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz" "cudart-llama-${LLAMA_TAG}-bin-ubuntu-cuda-13.4-x64.tar.gz"; do
  [ -f ~/opt/dl/"$f" ] || curl -sSL --fail -o ~/opt/dl/"$f" "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/$f"
  tar -xzf ~/opt/dl/"$f" -C "$DEST"
done

echo "== Qwen2.5-Coder-3B-Instruct GGUF Q4_K_M (only this quantization)"
GGUF=~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf
[ -f "$GGUF" ] || curl -sSL --fail -o "$GGUF" \
  "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/${GGUF_REV}/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
echo "${GGUF_SHA}  ${GGUF}" | sha256sum -c -

[ -f config.yaml ] || sed "s#  python: null .*#  python: $HOME/venvs/kernelforge/bin/python#" config.example.yaml > config.yaml
~/venvs/kernelforge/bin/python -m kernelforge doctor
