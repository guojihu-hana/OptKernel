"""
Isolated :func:`query_server` (LLM) worker for the KernelBench agent.

The agent calls :func:`run_llm_subprocess` (fresh ``python`` process) so HTTP/OOM issues
in the OpenAI-compatible client and streaming are less likely to take down the parent.
One JSON object is written to stdout by the child, mirroring
:func:`run_validation.run_forward_validation_subprocess` / :func:`run_ncu.run_ncu_profile_subprocess`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Optional


def _subprocess_io_capture(
    proc: CompletedProcess[str], cmd: list[str], *, max_chars: int = 16_000
) -> dict[str, Any]:
    return {
        "returncode": proc.returncode,
        "command": cmd,
        "stderr": (proc.stderr or "")[:max_chars],
        "stdout": (proc.stdout or "")[:max_chars],
    }


def _parse_llm_worker_json_stdout(raw: str) -> dict[str, Any]:
    """
    The llm child finally prints a single line ``json.dumps({...})`` to stdout. Legacy paths
    may prepend colored ``Finish reason`` / ``Usage:`` lines to **stdout**; read the last
    line that decodes as JSON, or the last ``{...}`` from ``\\{\"ok\"``.
    """
    s = (raw or "").strip()
    if not s:
        raise json.JSONDecodeError("empty stdout", "", 0)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        for line in reversed(s.splitlines()):
            t = line.strip()
            if t.startswith("{"):
                try:
                    return json.loads(t)
                except json.JSONDecodeError:
                    continue
        i = s.rfind('{"ok"')
        if i >= 0:
            return json.loads(s[i:])
    raise json.JSONDecodeError("no valid JSON object in child stdout", s, 0)


def run_llm_subprocess(
    system_path: Path,
    user_path: Path,
    round_idx: int,
    stream_dump_path: Optional[Path],
    *,
    temperature: float,
    max_tokens: int,
    server_type: str,
    server_address: str,
    server_port: int,
    model_name: str,
    is_reasoning_model: bool,
    openai_compatible_api_key: str,
    repetition_penalty: float,
    max_context_length: int,
) -> dict[str, Any]:
    """
    Run a single :func:`query_server` call in a **fresh Python process**.
    Child prints one JSON with ``ok: true`` and ``text`` / ``llm_output_dumped`` on success.
    On failure, ``ok: false`` plus ``runtime_error`` and/or ``subprocess`` diagnostics
    (same style as the validation worker). When ``max_context_length`` is set, the server
    caps output tokens only on **continuation** segments of a single chat (not the first request).
    """
    root = Path(__file__).resolve().parent
    script = root / "run_llm.py"
    cmd: list[str] = [
        sys.executable,
        str(script),
        "llm-call",
        "--system-file",
        str(Path(system_path).resolve()),
        "--user-file",
        str(Path(user_path).resolve()),
        "--round-idx",
        str(int(round_idx)),
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(int(max_tokens)),
        "--server-type",
        str(server_type),
        "--server-address",
        str(server_address),
        "--server-port",
        str(int(server_port)),
        "--model",
        str(model_name),
        "--is-reasoning",
        "1" if is_reasoning_model else "0",
        "--api-key",
        str(openai_compatible_api_key or ""),
        "--repetition-penalty",
        str(float(repetition_penalty)),
        "--max-context-length",
        str(int(max_context_length or 0)),
    ]
    if stream_dump_path is not None:
        cmd += ["--stream-dump-path", str(Path(stream_dump_path).resolve())]
    else:
        cmd += ["--no-stream-dump"]

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(root),
        env=os.environ.copy(),
    )
    out = (proc.stdout or "").strip()
    if not out:
        cap = _subprocess_io_capture(proc, cmd)
        return {
            "ok": False,
            "runtime_error": (
                f"llm worker process exited with code {proc.returncode} and no JSON on stdout. "
                "Check subprocess.stderr in metrics (traceback, OOM, signal)."
            ),
            "subprocess": cap,
        }
    try:
        parsed: dict[str, Any] = _parse_llm_worker_json_stdout(out)
    except json.JSONDecodeError as e:
        cap = _subprocess_io_capture(proc, cmd)
        return {
            "ok": False,
            "runtime_error": (
                f"llm worker: invalid JSON on stdout ({e!s}). "
                "See subprocess.stderr/stdout in metrics for the raw process output."
            ),
            "subprocess": cap,
        }
    if proc.returncode not in (0, None) and "subprocess_exit_warning" not in parsed:
        parsed = dict(parsed)
        parsed["subprocess_exit_warning"] = _subprocess_io_capture(proc, cmd)
    return parsed


def _main_llm_call_worker() -> int:
    """``python run_llm.py llm-call --system-file ...`` — one JSON on stdout."""
    import argparse

    from query_server import query_server

    p = argparse.ArgumentParser(description="Isolated query_server / LLM worker (for agent subprocess).")
    p.add_argument("--system-file", type=Path, required=True)
    p.add_argument("--user-file", type=Path, required=True)
    p.add_argument("--round-idx", type=int, default=-1)
    p.add_argument(
        "--stream-dump-path",
        type=Path,
        default=None,
        help="Path for streaming LLM output; omit with --no-stream-dump",
    )
    p.add_argument("--no-stream-dump", action="store_true", help="Do not pass stream_dump_path")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--server-type", type=str, default="local")
    p.add_argument("--server-address", type=str, default="localhost")
    p.add_argument("--server-port", type=int, default=30000)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--is-reasoning", type=str, default="1", help="0 or 1")
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--max-context-length", type=int, default=0)
    args = p.parse_args()
    if args.no_stream_dump:
        sdp: Optional[str] = None
    else:
        sdp = str(args.stream_dump_path) if args.stream_dump_path is not None else None

    system = Path(args.system_file).read_text(encoding="utf-8")
    user = Path(args.user_file).read_text(encoding="utf-8")
    is_reasoning = (args.is_reasoning or "1").strip() not in {"0", "false", "False", ""}
    try:
        raw, llm_dumped = query_server(
            user,
            system_prompt=system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            server_type=args.server_type,
            server_address=args.server_address,
            server_port=int(args.server_port),
            model_name=args.model,
            is_reasoning_model=bool(is_reasoning),
            call_type="kernel_bench_agent",
            round_idx=int(args.round_idx),
            stream_dump_path=sdp,
            openai_compatible_api_key=args.api_key,
            repetition_penalty=float(args.repetition_penalty),
            max_context_length=int(args.max_context_length or 0),
        )
        if isinstance(raw, dict):
            text = str(raw.get("text", ""))
        elif isinstance(raw, list):
            text = str(raw[0] if raw else "")
        else:
            text = str(raw or "")
        result: dict[str, Any] = {
            "ok": True,
            "text": text,
            "llm_output_dumped": bool(llm_dumped),
        }
    except Exception as e:  # noqa: BLE001
        result = {
            "ok": False,
            "runtime_error": f"query_server raised: {e!r}\n{traceback.format_exc()}",
        }
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "llm-call":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        raise SystemExit(_main_llm_call_worker())
    print("Usage: run_llm.py llm-call --system-file S --user-file U ...", file=sys.stderr)
    raise SystemExit(2)
