# Notices and attribution

## Adapted material

- **Triton LayerNorm tutorial** (`triton-lang/triton`, `python/tutorials/05-layer-norm.py`, MIT License,
  Copyright 2018-2020 Philippe Tillet, Copyright 2020-2022 OpenAI). The basic LayerNorm kernel in
  `kernelforge/kernels/layernorm.py` follows the tutorial's forward structure (one row per program, masked
  loads, FP32 reduction, fused affine). The multi-row 2D tiling, launch policy and tuning are independent.

  > Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
  > associated documentation files (the "Software"), to deal in the Software without restriction, including
  > without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  > copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
  > following conditions: The above copyright notice and this permission notice shall be included in all
  > copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
  > KIND.

- **Qwen2-VL multimodal RoPE convention** (`huggingface/transformers`,
  `src/transformers/models/qwen2_vl/modeling_qwen2_vl.py` at revision
  `f90e93c4f83974f634860e246cdfd2c68666d605`, Apache License 2.0). `references.hf_qwen2vl_mrope_transcription`
  transcribes the upstream axis-selection and `rotate_half` logic solely to verify the convention. The fused
  Triton kernel is independent work.

## Independent work

Kernel changes (multi-row LayerNorm tiles, fused MRoPE with trig reuse across heads), the trusted
evaluator and measurement methodology, the agent integration and its state machine, the controlled
critic comparison, and the reproducibility infrastructure (contracts, evidence records, report).

## Models and tools (not redistributed)

- Qwen2.5-Coder-3B-Instruct-GGUF (Qwen, Apache 2.0), downloaded by `scripts/setup_wsl.sh`.
- llama.cpp release binaries (ggml-org, MIT), downloaded by `scripts/setup_wsl.sh`.
