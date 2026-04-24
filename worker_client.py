"""
HTTP client for :mod:`worker` (``POST /validation``, ``POST /ncu``).

The worker resolves ``task_file``, ``kernel_file``, and ``work_dir`` on **its** filesystem;
the agent and worker must share the same absolute paths (e.g. a common NFS mount) or
run on the same machine.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_TIMEOUT_S = 86400.0


def _base(url: str) -> str:
    s = (url or "").strip().rstrip("/")
    if not s:
        return ""
    return s if re.match(r"^https?://", s, re.I) else f"http://{s}"


def _post_json(full_url: str, payload: dict[str, Any], *, is_validation: bool) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        full_url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    raw: str
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except (OSError, TypeError) as e2:  # noqa: BLE001
            return _err_payload(
                is_validation,
                f"HTTP {e.code} from worker; could not read body: {e2!r}",
            )
    except (urllib.error.URLError, OSError) as e:
        return _err_payload(is_validation, f"request failed: {e!r}")
    return _parse_worker_response(raw, is_validation)


def _err_payload(is_validation: bool, msg: str) -> dict[str, Any]:
    if is_validation:
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": msg,
        }
    return {
        "returncode": 1,
        "stderr": msg,
    }


def _parse_worker_response(raw: str, is_validation: bool) -> dict[str, Any]:
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as ex:
        return _err_payload(
            is_validation,
            f"invalid JSON from worker: {ex!r}; head={raw[:1500]!r}",
        )
    if d.get("ok") is True and "result" in d:
        res = d["result"]
        return res if isinstance(res, dict) else _err_payload(is_validation, "result is not a dict")
    err = str(d.get("error", "unknown error"))
    tb = str(d.get("traceback", ""))[:8_000]
    if is_validation:
        return {
            "runnable": False,
            "status": "runtime_error",
            "runtime_error": err,
            "queue_timing": d.get("queue_timing"),
            "remote_traceback": tb,
        }
    return {
        "returncode": 1,
        "stderr": err,
        "remote": d,
    }


def run_validation_via_worker(
    base_url: str,
    task_path: Path,
    kernel_path: Path,
    seed: int,
    atol: float,
    rtol: float,
    gen_module_name: str = "kernelbench_generated_uniq",
) -> dict[str, Any]:
    b = _base(base_url)
    p = {
        "task_file": str(task_path.resolve()),
        "kernel_file": str(kernel_path.resolve()),
        "seed": seed,
        "atol": atol,
        "rtol": rtol,
        "gen_module_name": gen_module_name,
    }
    return _post_json(f"{b}/validation", p, is_validation=True)


def run_ncu_via_worker(
    base_url: str,
    kernel_path: Path,
    work_dir: Path,
    ncu_binary: str,
    metrics_comma: str,
    extra_args: list[str],
    *,
    launch_skip: int,
    launch_count: int,
) -> dict[str, Any]:
    b = _base(base_url)
    p: dict[str, Any] = {
        "kernel_file": str(Path(kernel_path).resolve()),
        "work_dir": str(Path(work_dir).resolve()),
        "ncu_binary": ncu_binary,
        "metrics": metrics_comma,
        "extra_args": extra_args,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
    }
    return _post_json(f"{b}/ncu", p, is_validation=False)
