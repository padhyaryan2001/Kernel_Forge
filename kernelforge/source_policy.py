"""AST allowlist for untrusted candidate source.

A first filter only: the hard boundary is the disposable worker process/container. The
policy admits a module made of imports of triton / triton.language / libdevice / math,
numeric constants, and functions decorated with @triton.jit. No host code is allowed.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .contracts import MAX_SOURCE_CHARS, MAX_SOURCE_LINES

ALLOWED_IMPORTS = {"triton", "triton.language", "math"}
ALLOWED_FROM_IMPORTS = {
    "triton": {"jit", "language", "cdiv"},
    "triton.language": {"constexpr"},
    "triton.language.extra": {"libdevice"},
    "triton.language.extra.cuda": {"libdevice"},
}
ALLOWED_GLOBAL_NAMES = {"tl", "triton", "libdevice", "math"}
ALLOWED_BUILTINS = {"range", "min", "max", "abs", "int", "float", "bool", "len", "True", "False", "None"}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr", "globals",
    "locals", "vars", "input", "breakpoint", "help", "os", "sys", "subprocess", "socket", "shutil",
    "pathlib", "importlib", "builtins", "print", "memoryview", "type", "object", "super",
}
FORBIDDEN_ATTRS = {"inline_asm_elementwise", "device_print", "static_print", "device_assert",
                   "extern_elementwise"}
FORBIDDEN_NODES = (
    ast.ClassDef, ast.Lambda, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Global, ast.Nonlocal,
    ast.Await, ast.Yield, ast.YieldFrom, ast.AsyncFunctionDef, ast.Delete, ast.ListComp, ast.SetComp,
    ast.DictComp, ast.GeneratorExp, ast.Starred, ast.JoinedStr, ast.NamedExpr, ast.Match,
)

# Exact parameter lists required for each operator's entry point (the trusted wrapper binds them).
ENTRY_SIGNATURES = {
    "layernorm": ["X", "Y", "W", "B", "stride_x_row", "stride_y_row", "M", "N", "eps",
                  "BLOCK_SIZE", "ROWS_PER_PROGRAM"],
    "mrope": ["Q", "K", "QO", "KO", "POS", "INV_FREQ", "COS", "SIN", "S", "HQ", "HKV", "HALF_D",
              "SEC_T", "SEC_TH", "stride_qb", "stride_qh", "stride_qs", "stride_kb", "stride_kh",
              "stride_ks", "stride_pa", "stride_pb", "stride_cb", "USE_PRECOMPUTED", "BLOCK_S",
              "BLOCK_D", "HEAD_SPLITS"],
}
ENTRY_CONSTEXPRS = {
    "layernorm": {"BLOCK_SIZE", "ROWS_PER_PROGRAM"},
    "mrope": {"USE_PRECOMPUTED", "BLOCK_S", "BLOCK_D", "HEAD_SPLITS"},
}


@dataclass
class PolicyReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    jit_functions: list[str] = field(default_factory=list)


def _is_jit_decorator(dec: ast.expr) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute):
        return isinstance(target.value, ast.Name) and target.value.id == "triton" and target.attr == "jit"
    return isinstance(target, ast.Name) and target.id == "jit"


def _annotation_is_constexpr(node: ast.expr | None) -> bool:
    if node is None:
        return False
    return (isinstance(node, ast.Attribute) and node.attr == "constexpr") or (
        isinstance(node, ast.Name) and node.id == "constexpr")


def check_source(source: str, operator: str, entry_point: str) -> PolicyReport:
    problems: list[str] = []
    if len(source) > MAX_SOURCE_CHARS:
        problems.append(f"source exceeds {MAX_SOURCE_CHARS} characters")
    if source.count("\n") + 1 > MAX_SOURCE_LINES:
        problems.append(f"source exceeds {MAX_SOURCE_LINES} lines")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return PolicyReport(False, problems + [f"syntax error: {exc.msg} (line {exc.lineno})"])

    module_names = set(ALLOWED_GLOBAL_NAMES)
    jit_functions: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    problems.append(f"import of {alias.name!r} is not allowed")
                    continue
                if alias.name == "triton.language" and alias.asname not in (None, "tl"):
                    problems.append("triton.language must be imported as tl")
                module_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod not in ALLOWED_FROM_IMPORTS:
                problems.append(f"from-import of {mod!r} is not allowed")
                continue
            for alias in node.names:
                if alias.name in ALLOWED_FROM_IMPORTS[mod]:
                    module_names.add(alias.asname or alias.name)
                else:
                    problems.append(f"from {mod} import {alias.name} is not allowed")
        elif isinstance(node, ast.FunctionDef):
            if not node.decorator_list or not all(_is_jit_decorator(d) for d in node.decorator_list):
                problems.append(f"function {node.name!r} must be decorated with @triton.jit (no host functions)")
            jit_functions.append(node.name)
            module_names.add(node.name)
        elif isinstance(node, ast.Assign):
            if not (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and _is_constant_expr(node.value)):
                problems.append(f"line {node.lineno}: only NAME = <numeric constant> assignments allowed at module level")
            else:
                module_names.add(node.targets[0].id)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue  # docstring
        else:
            problems.append(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} not allowed at module level")

    for fn in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        problems.extend(_check_function(fn, module_names))

    entry = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point), None)
    if entry is None:
        problems.append(f"entry point {entry_point!r} not found")
    elif operator in ENTRY_SIGNATURES:
        params = [a.arg for a in entry.args.args]
        if params != ENTRY_SIGNATURES[operator]:
            problems.append(f"entry point parameters must be exactly {ENTRY_SIGNATURES[operator]}, got {params}")
        else:
            for a in entry.args.args:
                if a.arg in ENTRY_CONSTEXPRS[operator] and not _annotation_is_constexpr(a.annotation):
                    problems.append(f"parameter {a.arg} must be annotated tl.constexpr")
        if entry.args.vararg or entry.args.kwarg or entry.args.kwonlyargs or entry.args.defaults:
            problems.append("entry point may not use defaults, *args, **kwargs or keyword-only arguments")
    return PolicyReport(not problems, problems, jit_functions)


def _is_constant_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, bool))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_constant_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_constant_expr(node.left) and _is_constant_expr(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "constexpr":
        return len(node.args) == 1 and _is_constant_expr(node.args[0]) and not node.keywords
    return False


def _check_function(fn: ast.FunctionDef, module_names: set[str]) -> list[str]:
    problems: list[str] = []
    local_names = {a.arg for a in fn.args.args}
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, FORBIDDEN_NODES):
            problems.append(f"{fn.name}: line {node.lineno}: {type(node).__name__} is not allowed")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            problems.append(f"{fn.name}: line {node.lineno}: imports inside kernels are not allowed")
        if isinstance(node, ast.FunctionDef):
            problems.append(f"{fn.name}: line {node.lineno}: nested functions are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS:
                problems.append(f"{fn.name}: line {node.lineno}: attribute {node.attr!r} is not allowed")
        if isinstance(node, (ast.Name, ast.arg)):
            ident = node.id if isinstance(node, ast.Name) else node.arg
            if ident in FORBIDDEN_NAMES or ident.startswith("__"):
                problems.append(f"{fn.name}: line {node.lineno}: name {ident!r} is not allowed")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_names.add(ident)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) > 200:
            problems.append(f"{fn.name}: line {node.lineno}: long string literals are not allowed")
    # every loaded bare name must be a local, a parameter, a module-level name or an allowed builtin
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in local_names and node.id not in module_names and node.id not in ALLOWED_BUILTINS:
                problems.append(f"{fn.name}: line {node.lineno}: unknown name {node.id!r}")
    return sorted(set(problems))
