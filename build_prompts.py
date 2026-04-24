"""
Load prompt text from ``prompts/`` and build user messages for the KernelBench agent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load(rel: str) -> str:
    return (_PROMPTS_DIR / rel).read_text(encoding="utf-8")


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n // 2] + "\n...\n" + s[-(n // 2) :]


# Match lines the LLM should see to fix JIT / link failures (avoids middle-cut losing "error: ..." lines).
_SIGNAL_LINE = re.compile(
    r"""(?ix)
    (?:^|\s) (?:error|Error) [`':] .+ |
    not\ declared\ in\ this\ scope |
    not\ declared\ |
    \bFAILED: |
    \bninja:\s+build\s+stopped |
    Error\s+building\ extension |
    undefined\ reference\ to |
    main\.cpp:\d+:\d+:\s*error: |
    \.cu\(\d+\):\s*error: |
    \.cpp\(\d+\):\s*error: |
    \b(?:c\+\+|g\+\+)\b.*\berror: |
    \bld: .* error: |
    CalledProcessError: |
    \bTraceback\ \( |
    \bFile\ \".*\",\ line\ \d+,
    """,
    re.VERBOSE,
)


def _truncate_error_for_prompt(s: str, n: int) -> str:
    """
    Prefer *compiler / linker* lines; keep short head+tail. Falls back to :func:`_truncate` if
    heuristics find nothing (e.g. only Python trace with no 'error:' lines).
    """
    s = s.strip()
    if not s or len(s) <= n:
        return s
    lines = s.splitlines()
    key_lines: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        t = ln.strip()
        if not t or t in seen:
            continue
        if _SIGNAL_LINE.search(t) or re.search(
            r"(?i)error:|\bFAILED:\b|not declared|ninja: build|Error building", t
        ):
            seen.add(t)
            key_lines.append(ln)
            if len(key_lines) >= 120:
                key_lines.append("... (more key lines omitted) ...")
                break
    block = "\n".join(key_lines)
    if len(block) < 80 and len(key_lines) < 4:
        return _truncate(s, n)
    if len(block) > n * 2 // 3:
        block = _truncate(block, n * 2 // 3)
    head = s[: min(1600, n // 5)]
    mid = f"\n\n--- key build / compile / link lines (high signal) ---\n{block}\n"
    tail = ""
    if len(s) > 2000:
        tail = f"\n--- (trace end, {min(1200, n // 5)} chars) ---\n" + s[-(min(1200, n // 5)) :]
    out = f"{head}{mid}{tail}".strip()
    if len(out) <= n:
        return out
    if len(key_lines) >= 4 and len(block) <= n - 20:
        return f"{(head + mid + tail).strip()[:n]}\n[truncated]"
    return out[: n // 2] + "\n...\n" + out[-(n // 2) :]


def system_prompt_round0() -> str:
    return _load("round0_system.txt").strip()


def system_prompt_roundk() -> str:
    return _load("roundk_system.txt").strip()


def build_user_prompt_round0(reference_source: str) -> str:
    intro = _load("round0_user_intro.txt").rstrip()
    return (
        f"{intro}\n\n"
        "Reference KernelBench file:\n\n"
        f"```python\n{reference_source.rstrip()}\n```\n"
    )


def build_user_prompt_roundk(
    reference_source: str,
    previous_kernel: str,
    previous_metrics_summary: str,
    *,
    best_previous_round: Optional[Tuple[int, float, str, str]] = None,
    previous_round_index: int = 0,
) -> str:
    intro = _load("roundk_user_intro.txt").rstrip()
    s = (
        f"{intro}\n\n"
        "Original reference:\n\n"
        f"```python\n{reference_source.rstrip()}\n```\n\n"
    )
    if best_previous_round is not None:
        br, bsp, bkern, bsum = best_previous_round
        s += f"\nBest kernel.py in previous rounds (highest benchmark speedup so far: {bsp:.6g} in round {br}):\n"
        if br == previous_round_index:
            s += (
                "This is the same kernel as **Previous round** above; code and metrics are already listed there.\n"
            )
        else:
            s += (
                f"\n```python\n{bkern.rstrip()}\n```\n\n"
                f"Metrics for that best round:\n\n{bsum}\n"
            )
    s += (
        "Previous round kernel.py:\n\n"
        f"```python\n{previous_kernel.rstrip()}\n```\n\n"
        "Previous round metrics / ncu summary (for optimization):\n\n"
        f"{previous_metrics_summary}\n"
    )
    return s


def summarize_metrics_for_prompt(metrics_path: Path, max_chars: int = 12000) -> str:
    if not metrics_path.is_file():
        return "(no previous metrics.json)"
    try:
        data: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"(failed to read metrics: {e})"
    parts: list[str] = [f"status={data.get('status')}", f"runnable={data.get('runnable')}"]
    ncu = data.get("ncu") if isinstance(data.get("ncu"), dict) else {}
    metrics = ncu.get("metrics") if isinstance(ncu.get("metrics"), dict) else {}
    has_ncu_metrics = bool(metrics)

    if has_ncu_metrics:
        parts.append(
            "ncu.metrics (aggregated):\n"
            + _truncate(json.dumps(metrics, ensure_ascii=False, indent=2), max_chars // 2)
        )
    err_keys: tuple[str, ...] = (
        "compile_error",
        "runtime_error",
        "numerical_error",
        "parse_error",
        "ncu_error",
    )
    n_payload = sum(1 for k in err_keys if data.get(k))
    for key in err_keys:
        if not data.get(key):
            continue
        raw = str(data[key])
        if key in ("compile_error", "runtime_error"):
            # Large JIT / ninja logs: keep g++/nvcc "error:" lines, not just head+tail of a blob.
            per = min(14000, max(7000, max_chars - 400) // max(1, n_payload))
            parts.append(f"{key}:\n{_truncate_error_for_prompt(raw, per)}")
        else:
            parts.append(f"{key}:\n{_truncate(raw, max(2500, max_chars // max(4, n_payload + 2)))}")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        # Single tail trim (preserves first ~all high-signal content we just built)
        return text[: max(0, max_chars - 40)] + "\n[metrics summary trimmed]\n"
    return text


def speedup_from_metrics(data: dict[str, Any]) -> Optional[float]:
    """``benchmark_timing.speedup`` as float, or ``None`` if missing."""
    bt = data.get("benchmark_timing")
    if not isinstance(bt, dict):
        return None
    sp = bt.get("speedup")
    if isinstance(sp, (int, float)):
        return float(sp)
    return None


def best_round_tuple_for_prompt(
    work_dir: Path, round_idx: int, speedup: float
) -> Optional[Tuple[int, float, str, str]]:
    """``(round, speedup, kernel_text, metrics_summary)`` for the prompt block, or ``None`` if files missing."""
    rd = work_dir / f"round_{round_idx:03d}"
    kpath, mpath = rd / "kernel.py", rd / "metrics.json"
    if not kpath.is_file() or not mpath.is_file():
        return None
    try:
        ksrc = kpath.read_text(encoding="utf-8")
    except OSError:
        return None
    summ = summarize_metrics_for_prompt(mpath, max_chars=12000)
    return (round_idx, speedup, ksrc, summ)


def find_best_previous_round(
    work_dir: Path, current_round: int
) -> Optional[Tuple[int, float, str, str]]:
    """
    Full scan: ``round_000`` … past ``round_{current_round-1}``, highest
    ``benchmark_timing.speedup`` (tie: later round). Used for one-time backfill only.
    """
    if current_round <= 0:
        return None
    best: Optional[Tuple[int, float, str, str]] = None
    for r in range(0, current_round):
        rd = work_dir / f"round_{r:03d}"
        kpath = rd / "kernel.py"
        mpath = rd / "metrics.json"
        if not kpath.is_file() or not mpath.is_file():
            continue
        try:
            data: dict[str, Any] = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not data.get("runnable"):
            continue
        sp = speedup_from_metrics(data)
        if sp is None:
            continue
        try:
            ksrc = kpath.read_text(encoding="utf-8")
        except OSError:
            continue
        summ = summarize_metrics_for_prompt(mpath, max_chars=12000)
        if best is None:
            best = (r, sp, ksrc, summ)
        elif sp > best[1] or (sp == best[1] and r > best[0]):
            best = (r, sp, ksrc, summ)
    return best
