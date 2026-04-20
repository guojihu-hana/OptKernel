"""
Load prompt text from ``prompts/`` and build user messages for the KernelBench agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load(rel: str) -> str:
    return (_PROMPTS_DIR / rel).read_text(encoding="utf-8")


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n // 2] + "\n...\n" + s[-(n // 2) :]


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
) -> str:
    intro = _load("roundk_user_intro.txt").rstrip()
    return (
        f"{intro}\n\n"
        "Original reference:\n\n"
        f"```python\n{reference_source.rstrip()}\n```\n\n"
        "Previous round kernel.py:\n\n"
        f"```python\n{previous_kernel.rstrip()}\n```\n\n"
        "Previous round metrics / ncu summary (for optimization):\n\n"
        f"{previous_metrics_summary}\n"
    )


def summarize_metrics_for_prompt(metrics_path: Path, max_chars: int = 6000) -> str:
    if not metrics_path.is_file():
        return "(no previous metrics.json)"
    try:
        data: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"(failed to read metrics: {e})"
    parts = [f"status={data.get('status')}", f"runnable={data.get('runnable')}"]
    ncu = data.get("ncu") if isinstance(data.get("ncu"), dict) else {}
    metrics = ncu.get("metrics") if isinstance(ncu.get("metrics"), dict) else {}
    has_ncu_metrics = bool(metrics)

    if has_ncu_metrics:
        parts.append(
            "ncu.metrics (aggregated):\n"
            + _truncate(json.dumps(metrics, ensure_ascii=False, indent=2), max_chars // 3)
        )
    else:
        for key in ("compile_error", "runtime_error", "numerical_error", "parse_error", "ncu_error"):
            if data.get(key):
                parts.append(f"{key}:\n{_truncate(str(data[key]), max_chars // 4)}")
    text = "\n\n".join(parts)
    return _truncate(text, max_chars)
