"""Generator and critic clients, the managed llama.cpp server lifecycle, prompt building and
structured-output parsing.

The local generator is any OpenAI-compatible endpoint (llama.cpp's llama-server by default,
an optional remote Qwen3-Coder server as a separately identified configuration). Hosted
critics (OpenAI, Anthropic, Google) are optional and imported lazily; a missing SDK or
credential disables only that critic arm. Secrets are read from the environment and never
persisted.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import ValidationError

from .config import GeneratorConfig, KernelForgeConfig, gpu_free_mib, server_env
from .contracts import (LAUNCH_SPACE, SELECTOR_NAME, CandidateSpec, CritiqueResult, LaunchPolicy, LaunchTier,
                        TokenUsage)
from .kernels import module_for


class GeneratorError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    duration_s: float
    model: str | None
    request: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# ---------------------------------------------------------------------------
# Generator clients
# ---------------------------------------------------------------------------

def _post_json(url: str, body: dict, timeout: float, headers: dict | None = None) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class OpenAICompatibleClient:
    """Minimal /v1/chat/completions client (no API key needed for the loopback llama-server)."""

    kind = "openai_compatible"

    def __init__(self, base_url: str, model: str, timeout: float = 600.0, api_key_env: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key_env = api_key_env

    @property
    def root_url(self) -> str:
        p = urlparse(self.base_url)
        return f"{p.scheme}://{p.netloc}"

    def chat(self, messages: list[dict], max_tokens: int, temperature: float, seed: int | None = None) -> ChatResult:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature, "stream": False}
        if seed is not None:
            body["seed"] = seed
        headers = {}
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        t0 = time.monotonic()
        try:
            data = _post_json(self.base_url + "/chat/completions", body, self.timeout, headers)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise GeneratorError(f"generator request failed: {exc}") from exc
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ChatResult(text=choice["message"].get("content") or "", finish_reason=choice.get("finish_reason"),
                          input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                          duration_s=time.monotonic() - t0, model=data.get("model"),
                          request={k: v for k, v in body.items() if k != "messages"})

    def count_tokens(self, text: str) -> int | None:
        """Exact token count via llama-server /tokenize when available."""
        try:
            data = _post_json(self.root_url + "/tokenize", {"content": text}, 30)
            return len(data.get("tokens", []))
        except Exception:  # noqa: BLE001 - optional capability
            return None


class ScriptedClient:
    """Replays fixed responses. Used by the demo and tests; its candidates are labeled 'scripted'.

    It never produces measurements: every candidate it emits still goes through the trusted
    evaluator, and in offline fixture mode no GPU evaluation is run at all.
    """

    kind = "scripted"

    def __init__(self, responses: list[str], model: str = "scripted-demo"):
        self.responses = list(responses)
        self.model = model
        self.calls = 0

    def chat(self, messages, max_tokens, temperature, seed=None) -> ChatResult:
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(text=text, finish_reason="stop", input_tokens=sum(len(m["content"]) for m in messages) // 4,
                          output_tokens=len(text) // 4, duration_s=0.0, model=self.model)

    def count_tokens(self, text: str) -> int | None:
        return None


# ---------------------------------------------------------------------------
# Managed llama.cpp server
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


class LlamaServer:
    """Owns exactly one llama-server process started by this project.

    stop() requests a graceful shutdown, waits with a bound, confirms process exit and that
    free VRAM recovered. It never touches processes it did not start.
    """

    def __init__(self, gcfg: GeneratorConfig, log_path: Path, gpu_index: int = 0):
        self.g = gcfg
        self.log_path = log_path
        self.gpu_index = gpu_index
        self.proc: subprocess.Popen | None = None
        self.metrics: dict[str, Any] = {}
        url = urlparse(gcfg.base_url)
        self.host, self.port = url.hostname or "127.0.0.1", url.port or 80
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            raise GeneratorError("managed llama-server must bind to loopback")

    def command(self) -> list[str]:
        if not self.g.server_binary or not self.g.model_path:
            raise GeneratorError("generator.server_binary and generator.model_path must be configured "
                                 "for a managed llama.cpp server")
        return [os.path.expanduser(self.g.server_binary), "-m", os.path.expanduser(self.g.model_path),
                "--alias", self.g.model, "--host", self.host, "--port", str(self.port),
                "-c", str(self.g.context_tokens), "-ngl", str(self.g.n_gpu_layers),
                "--parallel", str(self.g.max_concurrent_requests), "--cache-ram", "0", "--no-webui",
                *self.g.server_extra_args]

    def start(self) -> dict[str, Any]:
        if _port_open(self.host, self.port):
            raise GeneratorError(f"{self.host}:{self.port} is already in use by a process this project did not "
                                 "start; refusing to share or stop it")
        env = server_env(self.g)
        free_before = gpu_free_mib(self.gpu_index)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "a", encoding="utf-8")
        t0 = time.monotonic()
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        deadline = t0 + self.g.startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise GeneratorError(f"llama-server exited during startup (code {self.proc.returncode}); "
                                     f"see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://{self.host}:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(0.5)
        else:
            self.stop()
            raise GeneratorError(f"llama-server not healthy after {self.g.startup_timeout_s}s")
        free_loaded = gpu_free_mib(self.gpu_index)
        try:
            import psutil

            rss = psutil.Process(self.proc.pid).memory_info().rss
        except Exception:  # noqa: BLE001
            rss = None
        self.metrics = {"pid": self.proc.pid, "load_seconds": round(time.monotonic() - t0, 2),
                        "free_vram_before_mib": free_before, "free_vram_loaded_mib": free_loaded,
                        "server_vram_mib": (free_before - free_loaded) if free_before and free_loaded else None,
                        "host_rss_mib": round(rss / 2**20) if rss else None, "command": self.command()}
        return self.metrics

    def stop(self) -> dict[str, Any]:
        if self.proc is None:
            return {"stopped": True, "note": "not running"}
        t0 = time.monotonic()
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=self.g.shutdown_timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=30)
        exit_code = self.proc.returncode
        target = self.metrics.get("free_vram_before_mib")
        free_after = gpu_free_mib(self.gpu_index)
        released = None
        if target is not None:
            deadline = time.monotonic() + 30
            while free_after is not None and free_after < target - self.g.release_tolerance_mib and time.monotonic() < deadline:
                time.sleep(1)
                free_after = gpu_free_mib(self.gpu_index)
            released = free_after is not None and free_after >= target - self.g.release_tolerance_mib
        self.proc = None
        out = {"stopped": True, "exit_code": exit_code, "shutdown_seconds": round(time.monotonic() - t0, 2),
               "free_vram_after_stop_mib": free_after, "vram_released": released}
        self.metrics.update(out)
        return out

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GEN_SYSTEM = (
    "You are an expert Triton GPU kernel engineer. Reply in exactly this format:\n"
    "```python\n<complete Python module: only imports and @triton.jit functions>\n```\n"
    "```json\n<launch policy JSON>\n```\n"
    "EXPLANATION: <one or two sentences about the optimization>"
)
GEN_SYSTEM_LAUNCH_ONLY = (
    "You tune launch parameters for a fixed, verified Triton kernel. Reply in exactly this format:\n"
    "```json\n<launch policy JSON>\n```\n"
    "EXPLANATION: <one sentence>"
)
RULES = (
    "Rules: allowed imports are `import triton`, `import triton.language as tl`, "
    "`from triton.language.extra import libdevice`, `import math`. Every function must be decorated with "
    "@triton.jit; no host Python functions, prints, file/OS/network access, classes or lambdas. "
    "Keep the exact entry-point signature. Mask every load/store at M/N (or S/HALF_D) edges, accumulate in "
    "float32, never write to inputs."
)


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1  # conservative for code-heavy prompts


def _launch_doc(operator: str) -> str:
    space = {k: v for k, v in LAUNCH_SPACE[operator].items()}
    if operator == "layernorm":
        space["BLOCK_SIZE"] = "0 (= next_pow2(N)) or a power of two >= 16"
    return (f"Launch policy JSON: {{\"tiers\": [{{\"upto\": <int or null>, \"params\": {{...}}}}]}}. "
            f"Tiers are chosen by {SELECTOR_NAME[operator]} <= upto; the last tier must have upto null. "
            f"Allowed params: {json.dumps(space)}")


def format_feedback(items: list[dict], limit_chars: int = 2400) -> str:
    lines = []
    for it in items:
        head = f"- candidate {it.get('candidate')}: {it.get('status')}"
        if it.get("stage"):
            head += f" at {it['stage']}"
        if it.get("geomean_speedup"):
            head += f"; geomean speedup vs {it.get('baseline', 'baseline')} {it['geomean_speedup']:.3f}x"
        lines.append(head)
        for detail in it.get("details", [])[:3]:
            lines.append("    " + detail.replace("\n", " ")[:400])
    text = "\n".join(lines)
    return text if len(text) <= limit_chars else text[:limit_chars] + "\n    ..."


def critique_text(c: CritiqueResult | None) -> str:
    if c is None or c.status != "completed":
        return ""
    parts = []
    for title, items in (("Bottlenecks", c.bottlenecks), ("Correctness risks", c.correctness_risks),
                         ("Suggested changes", c.changes)):
        if items:
            parts.append(title + ":\n" + "\n".join(f"- {x}" for x in items[:5]))
    return "\n".join(parts)


def build_generation_messages(operator: str, mode: str, parent: CandidateSpec, parent_result: str,
                              feedback: list[dict], critique: CritiqueResult | None, gpu: str,
                              max_prompt_tokens: int, counter=None) -> tuple[list[dict], dict]:
    """Compact prompt within the generator context budget. The operator spec is never trimmed;
    reviewer notes, then feedback, are dropped first when over budget."""
    mod = module_for(operator)
    count = counter or estimate_tokens
    fb_items = list(feedback)
    crit = critique_text(critique)
    for _attempt in range(len(fb_items) + 3):
        sections = [f"## Operator\n{mod.SPEC_DOC}", f"## GPU\n{gpu}", f"## {_launch_doc(operator)}"]
        if mode == "full":
            sections.insert(1, f"## Required entry point\n{mod.SIGNATURE_DOC}\n{RULES}")
            sections.append(f"## Current best correct kernel\n```python\n{parent.kernel_source}```\n"
                            f"launch: {json.dumps(parent.launch.model_dump())}\nmeasured: {parent_result}")
        else:
            sections.append(f"## Fixed kernel (do not change)\n```python\n{parent.kernel_source}```\n"
                            f"current launch: {json.dumps(parent.launch.model_dump())}\nmeasured: {parent_result}")
        if fb_items:
            sections.append("## Feedback from the evaluator (last round)\n" + format_feedback(fb_items))
        if crit:
            sections.append("## Reviewer suggestions\n" + crit)
        task = ("Write a faster kernel that stays exactly correct. Make one or two focused changes."
                if mode == "full" else "Propose a better launch policy for this kernel.")
        sections.append("## Task\n" + task)
        user = "\n\n".join(sections)
        system = GEN_SYSTEM if mode == "full" else GEN_SYSTEM_LAUNCH_ONLY
        tokens = count(system + user) or estimate_tokens(system + user)
        if tokens <= max_prompt_tokens:
            return ([{"role": "system", "content": system}, {"role": "user", "content": user}],
                    {"prompt_tokens_estimate": tokens, "feedback_items": len(fb_items), "critique_included": bool(crit)})
        if crit:
            crit = ""
        elif fb_items:
            fb_items.pop(0)
        else:
            break
    raise GeneratorError(f"prompt exceeds the {max_prompt_tokens}-token budget even without feedback; "
                         "reduce the operator description or raise context_tokens")


_PY_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
_JSON_BLOCK = re.compile(r"```json[ \t]*\n(.*?)```", re.S)


def _parse_policy(raw: str) -> LaunchPolicy:
    data = json.loads(raw)
    if isinstance(data, dict) and "tiers" in data:
        return LaunchPolicy.model_validate(data)
    if isinstance(data, dict) and "params" in data:
        return LaunchPolicy(tiers=[LaunchTier(upto=None, params=data["params"])])
    if isinstance(data, dict):
        return LaunchPolicy(tiers=[LaunchTier(upto=None, params=data)])
    raise ValueError("launch policy must be a JSON object")


def parse_generation(text: str, operator: str, mode: str, parent: CandidateSpec,
                     truncated: bool = False) -> tuple[CandidateSpec | None, list[str], dict]:
    """Structured output -> CandidateSpec. Truncated or incomplete output is rejected, never executed."""
    problems: list[str] = []
    meta: dict[str, Any] = {}
    if truncated:
        return None, ["generator output was truncated (finish_reason=length); not executed"], meta
    mod = module_for(operator)
    if mode == "full":
        blocks = [b for b in _PY_BLOCK.findall(text) if "@triton.jit" in b or "triton.jit" in b]
        if not blocks:
            if text.count("```") % 2 == 1:
                problems.append("unterminated code block (incomplete output)")
            else:
                problems.append("no ```python block containing a @triton.jit kernel")
            return None, problems, meta
        source = blocks[0].rstrip() + "\n"
    else:
        source = parent.kernel_source
    launch = parent.launch
    jblocks = _JSON_BLOCK.findall(text)
    if jblocks:
        try:
            launch = _parse_policy(jblocks[0])
        except (ValueError, ValidationError, TypeError) as exc:
            problems.append(f"invalid launch policy JSON: {str(exc)[:300]}")
            return None, problems, meta
    elif mode == "launch_only":
        return None, ["no ```json launch policy block"], meta
    else:
        meta["launch_inherited"] = True
    m = re.search(r"EXPLANATION:\s*(.+)", text, re.S)
    explanation = (m.group(1).strip() if m else _PY_BLOCK.sub("", text).strip())[:1500]
    try:
        spec = CandidateSpec(operator=operator, kernel_source=source, entry_point=mod.ENTRY_POINT, launch=launch,
                             explanation=explanation, origin="generated", search_mode=mode)
    except ValidationError as exc:
        return None, [f"candidate schema: {exc.errors()[0]['msg']}"], meta
    return spec, problems, meta


# ---------------------------------------------------------------------------
# Critics
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = (
    "You review Triton GPU kernels for performance. You cannot run code; an evaluator measures every "
    "revision. Respond with one JSON object only, no prose: {\"bottlenecks\": [<hypotheses>], "
    "\"correctness_risks\": [<risks>], \"changes\": [<concrete, specific code or launch changes>]}. "
    "At most 5 items per list."
)
CRITIC_ARMS = ("none", "self", "openai", "anthropic", "google")


def critique_payload(operator: str, spec: CandidateSpec, summary: str, feedback: list[dict], gpu: str) -> str:
    """Identical feedback fields for every critic arm."""
    mod = module_for(operator)
    return (f"Operator contract:\n{mod.SPEC_DOC}\n\nGPU: {gpu}\n\nKernel:\n```python\n{spec.kernel_source}```\n"
            f"Launch policy: {json.dumps(spec.launch.model_dump())}\n\nMeasured on development workloads: "
            f"{summary}\n\nEvaluator feedback:\n{format_feedback(feedback)}")


def _parse_critique(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in critic response")
    data = json.loads(m.group(0))
    out = {}
    for key in ("bottlenecks", "correctness_risks", "changes"):
        vals = data.get(key, [])
        out[key] = [str(v)[:500] for v in (vals if isinstance(vals, list) else [vals])][:5]
    return out


def critic_availability(arm: str, cfg: KernelForgeConfig) -> tuple[bool, str | None, str | None]:
    """(available, model, reason-if-not)."""
    c = cfg.critics
    if arm == "none":
        return False, None, "control arm (no critic)"
    if arm == "self":
        return (True, cfg.generator.model, None) if c.self_critique else (False, None, "self_critique disabled")
    model = {"openai": c.openai_model, "anthropic": c.anthropic_model, "google": c.google_model}[arm]
    if not model:
        return False, None, f"critics.{arm}_model not configured"
    sdk = {"openai": "openai", "anthropic": "anthropic", "google": "google.genai"}[arm]
    try:
        __import__(sdk)
    except ImportError:
        return False, model, f"python package for {arm} not installed (pip install '.[critics]')"
    keys = {"openai": ["OPENAI_API_KEY"], "anthropic": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
            "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]}[arm]
    has_profile = arm == "anthropic" and (Path.home() / ".config" / "anthropic").exists()
    if not any(os.environ.get(k) for k in keys) and not has_profile:
        return False, model, f"no credentials ({' or '.join(keys)} not set)"
    return True, model, None


def _openai_critique(model: str, payload: str, max_tokens: int) -> tuple[str, TokenUsage]:
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=model, max_completion_tokens=max_tokens, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": payload}])
    u = resp.usage
    return resp.choices[0].message.content or "", TokenUsage(input_tokens=getattr(u, "prompt_tokens", None),
                                                             output_tokens=getattr(u, "completion_tokens", None))


def _anthropic_critique(model: str, payload: str, max_tokens: int) -> tuple[str, TokenUsage, str | None]:
    import anthropic

    # No refusal fallbacks: the arm must be served by exactly the configured model.
    resp = anthropic.Anthropic().messages.create(
        model=model, max_tokens=max_tokens, system=CRITIC_SYSTEM,
        messages=[{"role": "user", "content": payload}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, TokenUsage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens), resp.stop_reason


def _google_critique(model: str, payload: str, max_tokens: int) -> tuple[str, TokenUsage]:
    from google import genai
    from google.genai import types

    resp = genai.Client().models.generate_content(
        model=model, contents=payload,
        config=types.GenerateContentConfig(system_instruction=CRITIC_SYSTEM, max_output_tokens=max_tokens,
                                           response_mime_type="application/json"))
    um = resp.usage_metadata
    return resp.text or "", TokenUsage(input_tokens=getattr(um, "prompt_token_count", None),
                                       output_tokens=getattr(um, "candidates_token_count", None))


def run_critic(arm: str, cfg: KernelForgeConfig, payload: str, local_client=None) -> CritiqueResult:
    available, model, reason = critic_availability(arm, cfg)
    provider = {"self": "local", "none": "none"}.get(arm, arm)
    if arm == "none":
        return CritiqueResult(arm=arm, provider=provider, model=None, status="not_run", reason=reason)
    if not available:
        return CritiqueResult(arm=arm, provider=provider, model=model, status="unavailable", reason=reason)
    max_tokens = cfg.critics.max_output_tokens
    t0 = time.monotonic()
    stop_reason = None
    try:
        if arm == "self":
            if local_client is None:
                return CritiqueResult(arm=arm, provider=provider, model=model, status="unavailable",
                                      reason="local generator not running")
            r = local_client.chat([{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": payload}],
                                  max_tokens=min(max_tokens, cfg.generator.max_output_tokens), temperature=0.2,
                                  seed=cfg.search.seed)
            text, usage = r.text, TokenUsage(input_tokens=r.input_tokens, output_tokens=r.output_tokens)
            stop_reason = "max_tokens" if r.truncated else None
        elif arm == "openai":
            text, usage = _openai_critique(model, payload, max_tokens)
        elif arm == "anthropic":
            text, usage, stop_reason = _anthropic_critique(model, payload, max_tokens)
        else:
            text, usage = _google_critique(model, payload, max_tokens)
    except Exception as exc:  # noqa: BLE001 - provider failures are recorded per trial
        return CritiqueResult(arm=arm, provider=provider, model=model, status="error",
                              reason=f"{type(exc).__name__}: {str(exc)[:400]}", duration_s=time.monotonic() - t0)
    duration = time.monotonic() - t0
    if stop_reason == "refusal":
        return CritiqueResult(arm=arm, provider=provider, model=model, status="refused", usage=usage,
                              reason="provider returned stop_reason=refusal", duration_s=duration)
    try:
        parsed = _parse_critique(text)
    except (ValueError, json.JSONDecodeError) as exc:
        return CritiqueResult(arm=arm, provider=provider, model=model, status="error", usage=usage,
                              reason=f"unparseable critique ({stop_reason or 'stop'}): {exc}", duration_s=duration)
    return CritiqueResult(arm=arm, provider=provider, model=model, status="completed", usage=usage,
                          duration_s=duration, **parsed)
