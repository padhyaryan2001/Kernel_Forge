from __future__ import annotations

import functools

import pytest


@functools.lru_cache(maxsize=1)
def cuda_triton_available() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    try:
        import triton  # noqa: F401
    except ImportError:
        return False, "triton not installed"
    return True, ""


def pytest_collection_modifyitems(config, items):
    ok, reason = cuda_triton_available()
    if ok:
        return
    skip = pytest.mark.skip(reason=f"CUDA test skipped (not passed): {reason}")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tmp_cfg(tmp_path):
    from kernelforge.config import KernelForgeConfig

    return KernelForgeConfig(runs_dir=str(tmp_path / "runs"))


@pytest.fixture
def store(tmp_cfg):
    from kernelforge.store import Store

    s = Store(tmp_cfg.runs_path())
    yield s
    s.close()
