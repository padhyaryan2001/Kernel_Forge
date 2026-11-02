"""Declared workload manifests: smoke, search (agent benchmark subset), dev, and held-out.

Development seeds start at 1000; held-out seeds start at 9000 and are never placed in a
generator prompt. Held-out shapes (e.g. N=1536/5120, S=768/2048/3000) do not occur in dev.
"""
from __future__ import annotations

import itertools

from .contracts import LayerNormCase, Manifest, MRoPECase, Split

DTYPES = ("float32", "float16", "bfloat16")
SECTIONS = {64: (8, 12, 12), 128: (16, 24, 24)}


def _ln(split: str, idx: int, seed_base: int, **kw) -> LayerNormCase:
    kind = kw.get("input_kind", "normal")
    eps = kw.get("eps", 1e-5)
    wid = f"ln-{split}-{idx:03d}-M{kw['M']}-N{kw['N']}-{kw['dtype']}"
    if kind != "normal":
        wid += f"-{kind}"
    if eps != 1e-5:
        wid += f"-eps{eps:g}"
    return LayerNormCase(workload_id=wid, seed=seed_base + idx, **kw)


def _mr(split: str, idx: int, seed_base: int, **kw) -> MRoPECase:
    d = kw["D"]
    kw.setdefault("sections", SECTIONS[d])
    c = "pre" if kw.get("contract") == "precomputed_trig" else "pos"
    wid = (f"mr-{split}-{idx:03d}-B{kw['B']}-Hq{kw['Hq']}-Hkv{kw['Hkv']}-S{kw['S']}-D{d}-{kw['dtype']}"
           f"-{kw.get('position_kind', 'text')}-{c}")
    if kw.get("position_offset"):
        wid += f"-off{kw['position_offset']}"
    return MRoPECase(workload_id=wid, seed=seed_base + idx, **kw)


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------

def layernorm_manifest(split: Split) -> Manifest:
    rows: list[dict] = []
    if split == "smoke":
        rows = [
            dict(M=32, N=768, dtype="float32"),
            dict(M=256, N=1024, dtype="float16"),
            dict(M=1, N=4096, dtype="bfloat16"),
            dict(M=2048, N=8192, dtype="float16"),
            dict(M=64, N=1000, dtype="float32"),
            dict(M=32, N=768, dtype="float32", input_kind="constant_rows"),
        ]
        seed_base = 100
    elif split == "search":
        rows = [
            dict(M=2048, N=768, dtype="float16"),
            dict(M=2048, N=1024, dtype="float16"),
            dict(M=2048, N=4096, dtype="float16"),
            dict(M=256, N=8192, dtype="float16"),
            dict(M=32, N=3072, dtype="float16"),
            dict(M=2048, N=1024, dtype="float32"),
            dict(M=2048, N=4096, dtype="bfloat16"),
        ]
        seed_base = 500
    elif split == "dev":
        for m, n, dt in itertools.product((1, 32, 256, 2048), (128, 768, 1024, 3072, 4096, 8192), DTYPES):
            rows.append(dict(M=m, N=n, dtype=dt))
        rows += [
            dict(M=64, N=1000, dtype="float32"),
            dict(M=64, N=2500, dtype="float16"),
            dict(M=17, N=333, dtype="bfloat16"),
            dict(M=32, N=768, dtype="float32", input_kind="constant_rows"),
            dict(M=32, N=4096, dtype="float16", input_kind="constant_rows"),
            dict(M=32, N=1024, dtype="float32", input_kind="low_variance"),
            dict(M=32, N=3072, dtype="bfloat16", input_kind="low_variance"),
            dict(M=32, N=768, dtype="float16", input_kind="negative"),
            dict(M=32, N=8192, dtype="float32", input_kind="large_offset"),
            dict(M=32, N=1024, dtype="float32", eps=1e-6),
            dict(M=32, N=1024, dtype="float16", eps=1e-3),
        ]
        seed_base = 1000
    elif split == "heldout":
        for m, n, dt in itertools.product((64, 512, 1536), (1536, 5120), DTYPES):
            rows.append(dict(M=m, N=n, dtype=dt))
        rows += [
            dict(M=3, N=640, dtype="float32"),
            dict(M=777, N=1536, dtype="float16", input_kind="negative"),
            dict(M=48, N=5120, dtype="float32", input_kind="low_variance"),
            dict(M=48, N=1536, dtype="bfloat16", input_kind="constant_rows"),
            dict(M=48, N=5120, dtype="float16", eps=1e-4),
        ]
        seed_base = 9000
    else:  # pragma: no cover - guarded by Split
        raise ValueError(split)
    cases = [_ln(split, i, seed_base, **r) for i, r in enumerate(rows)]
    return Manifest(name=f"layernorm-{split}", operator="layernorm", split=split, cases=cases)


# ---------------------------------------------------------------------------
# MRoPE
# ---------------------------------------------------------------------------

def mrope_manifest(split: Split) -> Manifest:
    rows: list[dict] = []
    if split == "smoke":
        rows = [
            dict(B=1, Hq=8, Hkv=2, S=128, D=64, dtype="float32", position_kind="text"),
            dict(B=2, Hq=16, Hkv=4, S=1024, D=128, dtype="float16", position_kind="image"),
            dict(B=1, Hq=16, Hkv=8, S=1024, D=128, dtype="bfloat16", position_kind="video"),
            dict(B=1, Hq=8, Hkv=8, S=128, D=128, dtype="float16", position_kind="text", contract="precomputed_trig"),
            dict(B=1, Hq=8, Hkv=2, S=256, D=128, dtype="float32", position_kind="video", position_offset=200_000),
        ]
        seed_base = 100
    elif split == "search":
        rows = [
            dict(B=1, Hq=16, Hkv=4, S=4096, D=128, dtype="float16", position_kind="image"),
            dict(B=2, Hq=16, Hkv=2, S=1024, D=128, dtype="bfloat16", position_kind="video"),
            dict(B=2, Hq=8, Hkv=8, S=1024, D=64, dtype="float16", position_kind="text"),
            dict(B=1, Hq=16, Hkv=4, S=4096, D=128, dtype="float16", position_kind="image", contract="precomputed_trig"),
            dict(B=1, Hq=8, Hkv=2, S=128, D=128, dtype="float32", position_kind="video"),
        ]
        seed_base = 500
    elif split == "dev":
        kinds = ("text", "image", "video")
        i = 0
        for s, d, dt in itertools.product((128, 1024, 4096), (64, 128), DTYPES):
            b = 1 + (i % 2)
            hq = (8, 16)[(i // 2) % 2]
            hkv = max(1, hq // (2, 4, 1)[i % 3])
            rows.append(dict(B=b, Hq=hq, Hkv=hkv, S=s, D=d, dtype=dt, position_kind=kinds[i % 3]))
            i += 1
        for s, d, dt in itertools.product((128, 1024, 4096), (64, 128), ("float16", "float32")):
            rows.append(dict(B=1, Hq=8, Hkv=2, S=s, D=d, dtype=dt, position_kind=kinds[s % 3],
                             contract="precomputed_trig"))
        rows += [
            dict(B=1, Hq=8, Hkv=2, S=1024, D=128, dtype="float32", position_kind="video", position_offset=500_000),
            dict(B=2, Hq=16, Hkv=4, S=1000, D=128, dtype="float16", position_kind="image", position_offset=100_000),
            dict(B=1, Hq=4, Hkv=1, S=77, D=64, dtype="bfloat16", position_kind="text"),
        ]
        seed_base = 1000
    elif split == "heldout":
        rows = [
            dict(B=1, Hq=16, Hkv=4, S=2048, D=128, dtype="float16", position_kind="video"),
            dict(B=2, Hq=8, Hkv=2, S=3000, D=128, dtype="bfloat16", position_kind="image"),
            dict(B=1, Hq=16, Hkv=16, S=768, D=64, dtype="float32", position_kind="text"),
            dict(B=2, Hq=16, Hkv=8, S=2048, D=64, dtype="float16", position_kind="image"),
            dict(B=1, Hq=12, Hkv=4, S=3000, D=128, dtype="float32", position_kind="video", position_offset=750_000),
            dict(B=1, Hq=16, Hkv=4, S=2048, D=128, dtype="float16", position_kind="video", contract="precomputed_trig"),
            dict(B=2, Hq=8, Hkv=2, S=768, D=128, dtype="bfloat16", position_kind="text", contract="precomputed_trig"),
            dict(B=1, Hq=8, Hkv=1, S=333, D=64, dtype="float16", position_kind="video"),
        ]
        seed_base = 9000
    else:  # pragma: no cover
        raise ValueError(split)
    cases = [_mr(split, i, seed_base, **r) for i, r in enumerate(rows)]
    return Manifest(name=f"mrope-{split}", operator="mrope", split=split, cases=cases)


def manifest(operator: str, split: Split) -> Manifest:
    if operator == "layernorm":
        return layernorm_manifest(split)
    if operator == "mrope":
        return mrope_manifest(split)
    raise ValueError(f"unknown operator {operator!r}")


def shape_key(case) -> tuple:
    """Shape identity used to keep held-out shapes disjoint from development shapes."""
    if case.operator == "layernorm":
        return (case.N,)
    return (case.S,)
