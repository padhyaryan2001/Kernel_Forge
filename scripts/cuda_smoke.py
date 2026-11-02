"""Standalone Blackwell (sm_120) smoke check. Run with the evaluator interpreter:

    ~/venvs/kernelforge/bin/python scripts/cuda_smoke.py

A successful import or torch.cuda.is_available() is not enough: this executes a CUDA matmul
and a freshly compiled Triton kernel on the device, synchronizes to surface asynchronous
errors, and prints the versions to record. Exit code 0 only if both real executions pass.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    import torch

    report: dict = {"torch": torch.__version__, "torch_cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        print(json.dumps({**report, "ok": False, "diagnosis": "torch.cuda.is_available() is False: check the NVIDIA "
                          "driver (Windows driver for WSL), and that a CUDA wheel (not +cpu) is installed."}, indent=1))
        return 1
    cap = torch.cuda.get_device_capability(0)
    report.update(device=torch.cuda.get_device_name(0), capability=f"{cap[0]}.{cap[1]}",
                  arch_list=torch.cuda.get_arch_list())
    try:
        x = torch.randn(256, 256, device="cuda")
        y = x @ x
        assert torch.isfinite(y).all().item()
        torch.cuda.synchronize()
        report["matmul"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report.update(ok=False, matmul=f"FAILED: {exc}",
                      diagnosis="CUDA kernels do not run. 'no kernel image is available' means the PyTorch wheel "
                                "lacks sm_120: install a CUDA 12.8+ wheel from the official PyTorch index.")
        print(json.dumps(report, indent=1))
        return 1

    import triton
    import triton.language as tl

    report["triton"] = triton.__version__

    @triton.jit
    def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(c_ptr + offs, tl.load(a_ptr + offs, mask=mask) + tl.load(b_ptr + offs, mask=mask), mask=mask)

    try:
        n = 98_765
        a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        c = torch.empty_like(a)
        add_kernel[(triton.cdiv(n, 1024),)](a, b, c, n, BLOCK=1024)
        torch.cuda.synchronize()
        assert torch.equal(c, a + b)
        report["triton_vector_add"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report.update(ok=False, triton_vector_add=f"FAILED: {exc}",
                      diagnosis="Triton cannot compile/execute for this GPU. Use the Triton version pinned by the "
                                "installed PyTorch wheel; do not install a separate, incompatible triton.")
        print(json.dumps(report, indent=1))
        return 1
    report["sm_target_in_arch_list"] = f"sm_{cap[0]}{cap[1]}" in report["arch_list"]
    report["ok"] = True
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
