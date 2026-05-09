from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from agent import AgentConfig, KernelBenchAgent, extract_python_module
from run_ncu import PROFILE_K, SKIP_K, effective_ncu_metrics, nccu_bin, run_ncu_profile_subprocess
from run_validation import run_forward_validation_subprocess

# Appended to every speculative prompt so the model returns a single parseable ```python block.
SPEC_PROMPT_OUTPUT_TAIL = (
    "\n\n---\n\n"
    "Final instructions: keep internal reasoning brief, then respond with **only** one complete "
    "```python ... ``` fenced code block containing the full working kernel implementation—no "
    "preamble before the fence and no commentary after it."
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _resolve_output_dir(target_dir: Path) -> Path:
    """Mirror ``.../runs/...`` to ``.../spec_act_runs/...``; else keep original directory."""
    parts = target_dir.parts
    try:
        i = parts.index("runs")
    except ValueError:
        return target_dir
    repo_root = Path(*parts[:i]) if i > 0 else Path("/")
    relative_under_runs = Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path()
    return repo_root / "spec_act_runs" / relative_under_runs


def _encode_ids(text: str, tokenizer_name: str | None = None) -> list[int]:
    """Encode ``text`` to token ids using the same logic as ``_first_n_tokens_text``."""
    if not text:
        return []
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            return list(tok.encode(text, add_special_tokens=False))
        except Exception:
            pass
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        return list(enc.encode(text))
    except Exception:
        return []


def _count_tokens(text: str, tokenizer_name: str | None = None) -> int:
    ids = _encode_ids(text, tokenizer_name)
    if ids:
        return len(ids)
    return len(text.split())


def _discover_round_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        raise NotADirectoryError(str(parent))
    rounds = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("round_")]
    return sorted(rounds, key=lambda p: p.name)


def _head_sizes_through_full(total_tokens: int, step_n: int) -> list[int]:
    """Return ``[step_n, 2*step_n, ..., total_tokens]`` (multiples of ``step_n``, then full if needed)."""
    if total_tokens <= 0 or step_n <= 0:
        return []
    out: list[int] = []
    k = 1
    while True:
        h = k * step_n
        if h >= total_tokens:
            if not out or out[-1] != total_tokens:
                out.append(total_tokens)
            break
        out.append(h)
        k += 1
    return out


def _first_n_tokens_text(text: str, n: int, tokenizer_name: str | None = None) -> str:
    if n <= 0 or not text:
        return ""
    ids = _encode_ids(text, tokenizer_name)
    if not ids:
        return " ".join(text.split()[:n])
    ids = ids[:n]
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            return tok.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(ids)
    except Exception:
        return " ".join(text.split()[:n])


def _split_system_user(prompt_text: str) -> tuple[str, str]:
    ms = "--- system ---"
    mu = "--- user ---"
    i_sys = prompt_text.find(ms)
    i_usr = prompt_text.find(mu)
    if i_sys >= 0 and i_usr > i_sys:
        return (
            prompt_text[i_sys + len(ms) : i_usr].strip(),
            prompt_text[i_usr + len(mu) :].strip(),
        )
    return ("", prompt_text.strip())


def _round_head_suffix(target_dir: Path, output_subdir: str | None) -> str:
    r = _round_index_from_dirname(target_dir.name)
    round_part = f"r{r:03d}" if r is not None else "r000"
    head_part = "h0"
    if output_subdir:
        m = re.fullmatch(r"head(\d+)", output_subdir.strip())
        if m:
            head_part = f"h{m.group(1)}"
    return f"{round_part}_{head_part}"


def _uniquify_load_inline_name(py_src: str, suffix: str) -> str:
    """Append unique suffix to first ``load_inline(name=...)`` occurrence."""
    if not py_src or not suffix:
        return py_src
    pat = re.compile(r"(load_inline\s*\(\s*.*?\bname\s*=\s*)([\"'])([^\"']+)\2", re.DOTALL)

    def _repl(m: re.Match[str]) -> str:
        old = m.group(3)
        if old.endswith(f"_{suffix}"):
            new = old
        else:
            new = f"{old}_{suffix}"
        return f"{m.group(1)}{m.group(2)}{new}{m.group(2)}"

    return pat.sub(_repl, py_src, count=1)


def _to_float(v: object) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _baseline_speedup_from_prompt(prompt_text: str) -> float | None:
    m = re.search(r"highest benchmark speedup so far:\s*([0-9eE+\-.]+)", prompt_text)
    if not m:
        return None
    return _to_float(m.group(1))


def _round_index_from_dirname(name: str) -> int | None:
    m = re.fullmatch(r"round_(\d+)", name.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _best_spec_candidate_from_existing(round_dir: Path) -> tuple[float, str, dict[str, object], Path] | None:
    """Find best speculative result (historical rounds up to current) by validation speedup."""
    out_round_dir = _resolve_output_dir(round_dir)
    out_task_dir = out_round_dir.parent
    cur_idx = _round_index_from_dirname(out_round_dir.name)
    if not out_task_dir.is_dir():
        return None
    best_speed = float("-inf")
    best_kernel: str | None = None
    best_ncu_metrics: dict[str, object] | None = None
    best_dir: Path | None = None
    round_dirs: list[Path] = []
    for rd in sorted(out_task_dir.glob("round_*")):
        if not rd.is_dir():
            continue
        ridx = _round_index_from_dirname(rd.name)
        if ridx is None:
            continue
        if cur_idx is not None and ridx > cur_idx:
            continue
        round_dirs.append(rd)

    for rd in round_dirs:
        for p in sorted(rd.glob("head*")):
            if not p.is_dir():
                continue
            mp = p / "metrics.json"
            kp = p / "kernel.py"
            if not mp.is_file() or not kp.is_file():
                continue
            try:
                m = json.loads(_read_text(mp))
            except Exception:
                continue
            val = m.get("validation") if isinstance(m, dict) else None
            bt = val.get("benchmark_timing") if isinstance(val, dict) else None
            sp = _to_float(bt.get("speedup")) if isinstance(bt, dict) else None
            if sp is None:
                continue
            if sp <= best_speed:
                continue
            try:
                kernel_src = _read_text(kp)
            except Exception:
                continue
            ncu_obj = m.get("ncu") if isinstance(m, dict) else {}
            ncu_metrics = ncu_obj.get("metrics") if isinstance(ncu_obj, dict) else None
            if not isinstance(ncu_metrics, dict):
                ncu_metrics = {}
            best_speed = sp
            best_kernel = kernel_src
            best_ncu_metrics = dict(ncu_metrics)
            best_dir = p
    if best_kernel is None or best_ncu_metrics is None or best_dir is None:
        return None
    return best_speed, best_kernel, best_ncu_metrics, best_dir


def _replace_prompt_kernel_and_ncu_metrics(
    prompt_text: str,
    kernel_src: str,
    ncu_metrics: dict[str, object],
) -> str:
    replaced = prompt_text
    k_pat = re.compile(r"(Previous round kernel\.py:\s*\n\s*```python\n)(.*?)(\n```)", re.DOTALL)
    replaced = k_pat.sub(r"\1" + kernel_src.rstrip() + r"\3", replaced, count=1)

    ncu_json = json.dumps(ncu_metrics, ensure_ascii=False, indent=2, default=str)
    n_pat = re.compile(r"(ncu\.metrics \(aggregated\):\s*\n)\{.*?\}", re.DOTALL)
    replaced = n_pat.sub(r"\1" + ncu_json, replaced, count=1)
    return replaced


def _append_best_spec_block(prompt_text: str, kernel_src: str, ncu_metrics: dict[str, object]) -> str:
    ncu_json = json.dumps(ncu_metrics, ensure_ascii=False, indent=2, default=str)
    block = (
        "\n\n---\n\n"
        "Historical best speculative candidate (append mode):\n\n"
        "Best kernel:\n\n"
        f"```python\n{kernel_src.rstrip()}\n```\n\n"
        "ncu.metrics (aggregated):\n"
        f"{ncu_json}\n"
    )
    return f"{prompt_text.rstrip()}{block}"


def _host_to_addr_port(host: str) -> tuple[str, int]:
    h = host.strip()
    if "://" not in h:
        h = "http://" + h
    u = urlparse(h)
    if not u.hostname:
        raise ValueError(f"Invalid --host: {host!r}")
    port = int(u.port or 80)
    return (u.hostname, port)


def _discover_model(host: str, api_key: str) -> str:
    base = host.strip()
    if "://" not in base:
        base = "http://" + base
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/models")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    rows = data.get("data") or []
    if not rows:
        raise RuntimeError("No models found at /v1/models")
    m = rows[0]
    for k in ("id", "root", "model"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise RuntimeError("Unable to infer model name from /v1/models")


def _write_round_sweep_log(round_dir: Path, lines: list[str]) -> None:
    out_round_dir = _resolve_output_dir(round_dir)
    out_round_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_round_dir / "sweep_status.txt"
    txt = "\n".join(lines).rstrip() + "\n"
    log_path.write_text(txt, encoding="utf-8")


def build_spec_prompt(
    target_dir: Path,
    head_n: int,
    tokenizer_name: str | None = None,
    output_name: str = "spec_prompt.txt",
    *,
    output_subdir: str | None = None,
    best_context_mode: str = "replace",
) -> Path:
    prompt_path = target_dir / "prompt.txt"
    llm_output_path = target_dir / "llm_output.txt"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing file: {prompt_path}")
    if not llm_output_path.is_file():
        raise FileNotFoundError(f"Missing file: {llm_output_path}")
    if head_n < 0:
        raise ValueError("--head must be >= 0")

    out_dir = _resolve_output_dir(target_dir)
    if output_subdir:
        out_dir = out_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / output_name

    prompt_text = _read_text(prompt_path)
    base_speedup = _baseline_speedup_from_prompt(prompt_text)
    mode = (best_context_mode or "replace").strip().lower()
    best_spec = _best_spec_candidate_from_existing(target_dir)
    if mode != "off" and best_spec is not None:
        best_spec_speed, best_kernel_src, best_ncu_metrics, best_dir = best_spec
        baseline_cmp = base_speedup if base_speedup is not None else float("-inf")
        if best_spec_speed > baseline_cmp:
            if mode == "append":
                prompt_text = _append_best_spec_block(prompt_text, best_kernel_src, best_ncu_metrics)
            else:
                prompt_text = _replace_prompt_kernel_and_ncu_metrics(prompt_text, best_kernel_src, best_ncu_metrics)
            note_path = out_dir / "spec_replacement_note.txt"
            note = (
                f"Applied best speculative candidate in {mode} mode.\n"
                f"baseline_speedup={base_speedup}\n"
                f"selected_spec_speedup={best_spec_speed}\n"
                f"selected_spec_dir={best_dir}\n"
                f"selected_kernel={best_dir / 'kernel.py'}\n"
                f"selected_metrics={best_dir / 'metrics.json'}\n"
            )
            note_path.write_text(note, encoding="utf-8")
    llm_text = _read_text(llm_output_path)
    llm_prefix = _first_n_tokens_text(llm_text, head_n, tokenizer_name)
    merged = f"{prompt_text.rstrip()}\n\n{llm_prefix}{SPEC_PROMPT_OUTPUT_TAIL}"
    out.write_text(merged, encoding="utf-8")
    return out


def run_speculative_pipeline(
    args: argparse.Namespace,
    *,
    target_dir: Path | None = None,
    head_n: int | None = None,
    output_subdir: str | None = None,
) -> dict[str, object]:
    td = (target_dir or args.path).resolve()
    hn = int(head_n if head_n is not None else args.head)
    subdir = output_subdir
    if subdir is None:
        subdir = (getattr(args, "output_subdir", "") or "").strip() or None
    spec_prompt_path = build_spec_prompt(
        target_dir=td,
        head_n=hn,
        tokenizer_name=(args.tokenizer or "").strip() or None,
        output_name=args.output_name,
        output_subdir=subdir,
        best_context_mode=str(getattr(args, "best_context_mode", "replace")),
    )
    out_dir = spec_prompt_path.parent
    llm_out_path = out_dir / "llm_output.txt"
    kernel_path = out_dir / "kernel.py"
    metrics_path = out_dir / "metrics.json"

    system, user = _split_system_user(_read_text(spec_prompt_path))
    addr, port = _host_to_addr_port(args.host)
    model_name = (args.model or "").strip() or _discover_model(args.host, args.api_key or "")
    round_idx = _round_index_from_dirname(td.name)
    common_meta: dict[str, object] = {
        "model_name": model_name,
        "round_idx": (round_idx if round_idx is not None else int(args.round_idx)),
        "head": int(hn),
    }
    cfg = AgentConfig(
        task_path=Path(args.task_file).resolve(),
        work_dir=out_dir,
        start_round=0,
        max_rounds=1,
        seed=int(args.seed),
        atol=float(args.atol),
        rtol=float(args.rtol),
        server_type="vllm",
        server_address=addr,
        server_port=port,
        model_name=model_name,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        max_context_length=int(args.max_context_length),
        reasoning_enabled=not bool(args.no_reasoning),
        thinking_budget_tokens=int(args.thinking_budget_tokens or 0),
        reasoning_effort=str(args.reasoning_effort or "medium"),
        run_ncu=not bool(args.no_ncu),
        ncu_binary=args.ncu_binary,
        ncu_metrics=[x.strip() for x in (args.ncu_metrics or "").split(",") if x.strip()],
        ncu_extra_args=args.ncu_extra.split() if (args.ncu_extra or "").strip() else [],
        ncu_launch_skip=int(args.ncu_launch_skip),
        ncu_launch_count=int(args.ncu_launch_count),
        openai_compatible_api_key=args.api_key or "",
        repetition_penalty=float(args.repetition_penalty),
        worker_url=(args.worker_url or "").strip(),
    )
    agent = KernelBenchAgent(cfg)

    ll_t0 = time.perf_counter()
    ll_ts0 = _utc_iso()
    llm = agent.call_llm(system=system, user=user, round_idx=int(args.round_idx), llm_output_path=llm_out_path)
    ll_t1 = time.perf_counter()
    ll_ts1 = _utc_iso()
    llm_eval_timing = {
        "started_at": ll_ts0,
        "finished_at": ll_ts1,
        "seconds": round(ll_t1 - ll_t0, 6),
    }
    if not llm.get("ok"):
        payload: dict[str, object] = {
            **common_meta,
            "status": "llm_subprocess_error",
            "runnable": False,
            "llm": llm,
            "spec_prompt": str(spec_prompt_path),
            "kernel_path": str(kernel_path),
            "eval_timing": {"llm": llm_eval_timing},
        }
        agent.write_metrics(metrics_path, payload)
        return payload

    raw = str(llm.get("text", ""))
    if not bool(llm.get("llm_output_dumped", False)):
        llm_out_path.write_text(raw, encoding="utf-8")

    py_src = extract_python_module(raw)
    if py_src is None:
        payload = {
            **common_meta,
            "status": "parse_error",
            "runnable": False,
            "parse_error": "No ```python ... ``` block found in LLM output.",
            "spec_prompt": str(spec_prompt_path),
            "kernel_path": str(kernel_path),
            "eval_timing": {"llm": llm_eval_timing},
        }
        agent.write_metrics(metrics_path, payload)
        return payload

    suffix = _round_head_suffix(td, subdir)
    py_src = _uniquify_load_inline_name(py_src, suffix)
    kernel_path.write_text(py_src, encoding="utf-8")

    val = run_forward_validation_subprocess(
        Path(args.task_file).resolve(),
        kernel_path,
        seed=int(args.seed),
        atol=float(args.atol),
        rtol=float(args.rtol),
        gen_module_name=str(args.gen_module_name),
        optkernel_worker_url=(args.worker_url or "").strip() or None,
    )
    base: dict[str, object] = {
        **common_meta,
        "status": val.get("status"),
        "runnable": bool(val.get("runnable")),
        "spec_prompt": str(spec_prompt_path),
        "kernel_path": str(kernel_path),
        "llm_output_path": str(llm_out_path),
        "validation": val,
        "eval_timing": {"llm": llm_eval_timing},
    }
    if not bool(val.get("runnable")):
        agent.write_metrics(metrics_path, base)
        return base

    if not cfg.run_ncu:
        base["ncu"] = {"skipped": True, "reason": "run_ncu disabled"}
        base["status"] = "success"
        agent.write_metrics(metrics_path, base)
        return base

    wurl = (cfg.worker_url or "").strip()
    if not (wurl or shutil.which(nccu_bin(cfg.ncu_binary))):
        base["ncu"] = {"skipped": True, "reason": "ncu not found on PATH"}
        base["status"] = "success"
        agent.write_metrics(metrics_path, base)
        return base

    metric_names = effective_ncu_metrics(cfg.ncu_metrics)
    metrics_args: list[str] = ["--metrics", ",".join(metric_names)]
    ncu_info = run_ncu_profile_subprocess(
        kernel_path,
        out_dir,
        cfg.ncu_binary,
        metrics_args,
        cfg.ncu_extra_args,
        launch_skip=cfg.ncu_launch_skip,
        launch_count=cfg.ncu_launch_count,
        optkernel_worker_url=wurl or None,
    )
    base["ncu"] = ncu_info
    if ncu_info.get("returncode") != 0:
        base["status"] = "ncu_error"
    else:
        base["status"] = "success"
    agent.write_metrics(metrics_path, base)
    return base


def _run_speculative_llm_stage(
    args: argparse.Namespace,
    *,
    target_dir: Path,
    head_n: int,
    output_subdir: str,
) -> dict[str, object]:
    """Stage-1: build spec prompt + run LLM + parse/write kernel; no validation/ncu here."""
    td = target_dir.resolve()
    spec_prompt_path = build_spec_prompt(
        target_dir=td,
        head_n=int(head_n),
        tokenizer_name=(args.tokenizer or "").strip() or None,
        output_name=args.output_name,
        output_subdir=output_subdir,
    )
    out_dir = spec_prompt_path.parent
    llm_out_path = out_dir / "llm_output.txt"
    kernel_path = out_dir / "kernel.py"
    metrics_path = out_dir / "metrics.json"

    system, user = _split_system_user(_read_text(spec_prompt_path))
    addr, port = _host_to_addr_port(args.host)
    model_name = (args.model or "").strip() or _discover_model(args.host, args.api_key or "")
    round_idx = _round_index_from_dirname(td.name)
    common_meta: dict[str, object] = {
        "model_name": model_name,
        "round_idx": (round_idx if round_idx is not None else int(args.round_idx)),
        "head": int(head_n),
    }
    cfg = AgentConfig(
        task_path=Path(args.task_file).resolve(),
        work_dir=out_dir,
        start_round=0,
        max_rounds=1,
        seed=int(args.seed),
        atol=float(args.atol),
        rtol=float(args.rtol),
        server_type="vllm",
        server_address=addr,
        server_port=port,
        model_name=model_name,
        temperature=float(args.temperature),
        max_tokens=int(args.max_tokens),
        max_context_length=int(args.max_context_length),
        reasoning_enabled=not bool(args.no_reasoning),
        thinking_budget_tokens=int(args.thinking_budget_tokens or 0),
        reasoning_effort=str(args.reasoning_effort or "medium"),
        run_ncu=not bool(args.no_ncu),
        ncu_binary=args.ncu_binary,
        ncu_metrics=[x.strip() for x in (args.ncu_metrics or "").split(",") if x.strip()],
        ncu_extra_args=args.ncu_extra.split() if (args.ncu_extra or "").strip() else [],
        ncu_launch_skip=int(args.ncu_launch_skip),
        ncu_launch_count=int(args.ncu_launch_count),
        openai_compatible_api_key=args.api_key or "",
        repetition_penalty=float(args.repetition_penalty),
        worker_url=(args.worker_url or "").strip(),
    )
    agent = KernelBenchAgent(cfg)

    ll_t0 = time.perf_counter()
    ll_ts0 = _utc_iso()
    llm = agent.call_llm(system=system, user=user, round_idx=int(args.round_idx), llm_output_path=llm_out_path)
    ll_t1 = time.perf_counter()
    ll_ts1 = _utc_iso()
    llm_eval_timing = {
        "started_at": ll_ts0,
        "finished_at": ll_ts1,
        "seconds": round(ll_t1 - ll_t0, 6),
    }
    if not llm.get("ok"):
        payload: dict[str, object] = {
            **common_meta,
            "status": "llm_subprocess_error",
            "runnable": False,
            "llm": llm,
            "spec_prompt": str(spec_prompt_path),
            "kernel_path": str(kernel_path),
            "eval_timing": {"llm": llm_eval_timing},
        }
        agent.write_metrics(metrics_path, payload)
        return {"ready_for_eval": False, "result": payload}

    raw = str(llm.get("text", ""))
    if not bool(llm.get("llm_output_dumped", False)):
        llm_out_path.write_text(raw, encoding="utf-8")

    py_src = extract_python_module(raw)
    if py_src is None:
        payload = {
            **common_meta,
            "status": "parse_error",
            "runnable": False,
            "parse_error": "No ```python ... ``` block found in LLM output.",
            "spec_prompt": str(spec_prompt_path),
            "kernel_path": str(kernel_path),
            "eval_timing": {"llm": llm_eval_timing},
        }
        agent.write_metrics(metrics_path, payload)
        return {"ready_for_eval": False, "result": payload}

    suffix = _round_head_suffix(td, output_subdir)
    py_src = _uniquify_load_inline_name(py_src, suffix)
    kernel_path.write_text(py_src, encoding="utf-8")
    return {
        "ready_for_eval": True,
        "cfg": cfg,
        "common_meta": common_meta,
        "spec_prompt_path": str(spec_prompt_path),
        "kernel_path": str(kernel_path),
        "llm_output_path": str(llm_out_path),
        "metrics_path": str(metrics_path),
        "eval_timing": {"llm": llm_eval_timing},
    }


def _run_speculative_eval_stage(
    args: argparse.Namespace,
    llm_stage_payload: dict[str, object],
) -> dict[str, object]:
    """Stage-2: validation + ncu from an already materialized kernel.py."""
    cfg = llm_stage_payload["cfg"]  # type: ignore[assignment]
    assert isinstance(cfg, AgentConfig)
    spec_prompt_path = Path(str(llm_stage_payload["spec_prompt_path"]))
    kernel_path = Path(str(llm_stage_payload["kernel_path"]))
    llm_out_path = Path(str(llm_stage_payload["llm_output_path"]))
    metrics_path = Path(str(llm_stage_payload["metrics_path"]))
    out_dir = kernel_path.parent
    agent = KernelBenchAgent(cfg)
    cm = llm_stage_payload.get("common_meta")
    common_meta: dict[str, object] = dict(cm) if isinstance(cm, dict) else {}
    if not common_meta:
        common_meta = {
            "model_name": getattr(cfg, "model_name", ""),
            "round_idx": int(args.round_idx),
            "head": None,
        }
    et = llm_stage_payload.get("eval_timing")
    eval_timing = dict(et) if isinstance(et, dict) else {}

    val = run_forward_validation_subprocess(
        Path(args.task_file).resolve(),
        kernel_path,
        seed=int(args.seed),
        atol=float(args.atol),
        rtol=float(args.rtol),
        gen_module_name=str(args.gen_module_name),
        optkernel_worker_url=(args.worker_url or "").strip() or None,
    )
    base: dict[str, object] = {
        **common_meta,
        "status": val.get("status"),
        "runnable": bool(val.get("runnable")),
        "spec_prompt": str(spec_prompt_path),
        "kernel_path": str(kernel_path),
        "llm_output_path": str(llm_out_path),
        "validation": val,
        "eval_timing": eval_timing,
    }
    if not bool(val.get("runnable")):
        agent.write_metrics(metrics_path, base)
        return base

    if not cfg.run_ncu:
        base["ncu"] = {"skipped": True, "reason": "run_ncu disabled"}
        base["status"] = "success"
        agent.write_metrics(metrics_path, base)
        return base

    wurl = (cfg.worker_url or "").strip()
    if not (wurl or shutil.which(nccu_bin(cfg.ncu_binary))):
        base["ncu"] = {"skipped": True, "reason": "ncu not found on PATH"}
        base["status"] = "success"
        agent.write_metrics(metrics_path, base)
        return base

    metric_names = effective_ncu_metrics(cfg.ncu_metrics)
    metrics_args: list[str] = ["--metrics", ",".join(metric_names)]
    ncu_info = run_ncu_profile_subprocess(
        kernel_path,
        out_dir,
        cfg.ncu_binary,
        metrics_args,
        cfg.ncu_extra_args,
        launch_skip=cfg.ncu_launch_skip,
        launch_count=cfg.ncu_launch_count,
        optkernel_worker_url=wurl or None,
    )
    base["ncu"] = ncu_info
    if ncu_info.get("returncode") != 0:
        base["status"] = "ncu_error"
    else:
        base["status"] = "success"
    agent.write_metrics(metrics_path, base)
    return base


def main() -> int:
    p = argparse.ArgumentParser(
        description="Speculative runner: build spec_prompt, call vLLM via agent.py interfaces, then validation+ncu."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Directory with prompt.txt + llm_output.txt; with --sweep, parent directory containing round_* subdirs.",
    )
    p.add_argument(
        "--head",
        type=int,
        required=True,
        help="Token prefix length: single-run uses first N tokens; --sweep uses N as the step (N, 2N, …, full).",
    )
    p.add_argument("--task-file", type=str, required=True, help="KernelBench reference .py for validation")
    p.add_argument("--host", type=str, required=True, help="vLLM host, e.g. http://127.0.0.1:8000")
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name served by vLLM; if omitted, auto-detect first model from /v1/models",
    )
    p.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--tokenizer", type=str, default="", help="Optional HF tokenizer for --head tokenization")
    p.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Optional subdirectory under spec_act_runs/.../<round>/ for single-run output (default: flat under round).",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Walk path/round_*; for each round take prefixes of length N,2N,…,full (tokens); write to …/head<N_tokens>/",
    )
    p.add_argument(
        "--start-round",
        type=int,
        default=0,
        help="With --sweep: start processing from this round index (inclusive), e.g. 20 -> start from round_020.",
    )
    p.add_argument("--output-name", type=str, default="spec_prompt.txt")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-context-length", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--no-reasoning", action="store_true")
    p.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=int(os.environ.get("KERNEL_THINKING_BUDGET_TOKENS", "0") or 0),
        help="Thinking budget tokens for reasoning models (0 = provider default/unlimited).",
    )
    p.add_argument(
        "--reasoning-effort",
        type=str,
        default=(os.environ.get("KERNEL_REASONING_EFFORT", "medium") or "medium"),
        choices=["low", "medium", "high"],
        help="Reasoning effort for OpenAI reasoning APIs (ignored by vLLM/local).",
    )
    p.add_argument("--round-idx", type=int, default=0, help="Tag only (for llm_output progress)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--gen-module-name", type=str, default="kernelbench_generated_spec")
    p.add_argument(
        "--worker-url",
        type=str,
        default=os.environ.get("OPTKERNEL_WORKER_URL", "") or "",
        help="Optional http://host:port for remote validation/NCU; if unset, run locally (same as OPTKERNEL_WORKER_URL).",
    )
    p.add_argument("--no-ncu", action="store_true")
    p.add_argument("--ncu-binary", type=str, default=os.environ.get("NCU_BINARY", "ncu"))
    p.add_argument("--ncu-metrics", type=str, default="")
    p.add_argument("--ncu-extra", type=str, default="")
    p.add_argument(
        "--best-context-mode",
        type=str,
        default="off",
        choices=["off", "replace", "append"],
        help="How to use best speculative context when speedup exceeds baseline: off / replace / append.",
    )
    p.add_argument("--ncu-launch-skip", type=int, default=SKIP_K)
    p.add_argument("--ncu-launch-count", type=int, default=PROFILE_K)
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output: do not print full JSON result payloads.",
    )
    p.add_argument(
        "--head-parallelism",
        type=int,
        default=0,
        help="With --sweep: max concurrent head jobs per round. 0 means all heads in parallel; 1 means sequential.",
    )
    p.add_argument(
        "--async",
        dest="async_mode",
        action="store_true",
        help="Pipeline mode for --sweep: enqueue all round-head jobs in order; run LLM queue and validation+NCU queue asynchronously.",
    )
    p.add_argument(
        "--llm-queue-parallelism",
        type=int,
        default=20,
        help="With --async: max concurrent LLM requests (default 20).",
    )
    p.add_argument(
        "--eval-queue-parallelism",
        type=int,
        default=1,
        help="With --async: max concurrent validation+NCU workers (default 1).",
    )
    args = p.parse_args()
    if int(args.thinking_budget_tokens or 0) < 0:
        print("--thinking-budget-tokens must be >= 0", file=sys.stderr)
        return 1
    if int(args.head_parallelism or 0) < 0:
        print("--head-parallelism must be >= 0", file=sys.stderr)
        return 1
    if int(args.llm_queue_parallelism or 0) <= 0:
        print("--llm-queue-parallelism must be > 0", file=sys.stderr)
        return 1
    if int(args.eval_queue_parallelism or 0) <= 0:
        print("--eval-queue-parallelism must be > 0", file=sys.stderr)
        return 1
    if int(args.start_round or 0) < 0:
        print("--start-round must be >= 0", file=sys.stderr)
        return 1

    if args.sweep:
        parent = args.path.resolve()
        step_n = int(args.head)
        if step_n <= 0:
            print("With --sweep, --head must be the positive token step N.", file=sys.stderr)
            return 1
        try:
            rounds = _discover_round_dirs(parent)
        except OSError as e:
            print(str(e), file=sys.stderr)
            return 1
        if not rounds:
            print(f"No round_* directories under {parent}", file=sys.stderr)
            return 1
        start_r = int(args.start_round or 0)
        if start_r > 0:
            rounds = [rd for rd in rounds if ((_round_index_from_dirname(rd.name) or -1) >= start_r)]
            if not rounds:
                print(
                    f"No round_* directories with index >= {start_r} under {parent}",
                    file=sys.stderr,
                )
                return 1
            print(f"Resume sweep from round_{start_r:03d}", file=sys.stderr, flush=True)
        if not (args.model or "").strip():
            args.model = _discover_model(args.host, args.api_key or "")
            print(f"Auto-discovered model: {args.model}", file=sys.stderr, flush=True)
        tok = (args.tokenizer or "").strip() or None
        overall_ok = True
        if args.async_mode:
            jobs: list[tuple[Path, int, str]] = []
            for rd in rounds:
                llm_out = rd / "llm_output.txt"
                prompt_p = rd / "prompt.txt"
                if not llm_out.is_file() or not prompt_p.is_file():
                    print(f"Skip {rd}: need prompt.txt and llm_output.txt", file=sys.stderr)
                    continue
                text = _read_text(llm_out)
                total = _count_tokens(text, tok)
                heads = _head_sizes_through_full(total, step_n)
                if not heads:
                    print(f"Skip {rd}: empty llm_output.txt", file=sys.stderr)
                    continue
                print(
                    f"Sweep {rd.relative_to(parent)}: tokens≈{total}, heads={heads}",
                    file=sys.stderr,
                    flush=True,
                )
                for h in heads:
                    jobs.append((rd, h, f"head{h}"))

            if not jobs:
                print("No runnable round-head jobs discovered.", file=sys.stderr)
                return 1

            print(
                f"Async pipeline: llm_workers={int(args.llm_queue_parallelism)}, eval_workers={int(args.eval_queue_parallelism)}, jobs={len(jobs)}",
                file=sys.stderr,
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=int(args.llm_queue_parallelism)) as llm_ex, ThreadPoolExecutor(
                max_workers=int(args.eval_queue_parallelism)
            ) as eval_ex:
                llm_futs = {
                    llm_ex.submit(
                        _run_speculative_llm_stage,
                        args,
                        target_dir=rd,
                        head_n=h,
                        output_subdir=label,
                    ): (rd, h, label)
                    for (rd, h, label) in jobs
                }
                eval_futs: dict[object, tuple[Path, int, str]] = {}
                for lf in as_completed(llm_futs):
                    rd, h, label = llm_futs[lf]
                    try:
                        stage = lf.result()
                    except Exception as e:  # noqa: BLE001
                        stage = {"ready_for_eval": False, "result": {"status": "runner_exception", "error": repr(e)}}
                    if bool(stage.get("ready_for_eval")):
                        ef = eval_ex.submit(_run_speculative_eval_stage, args, stage)
                        eval_futs[ef] = (rd, h, label)
                        print(f"llm done -> enqueued eval {rd.name}/{label}", file=sys.stderr, flush=True)
                    else:
                        result = dict(stage.get("result") or {})
                        print(
                            f"llm done {rd.name}/{label}: status={result.get('status')}",
                            file=sys.stderr,
                            flush=True,
                        )
                        if not args.quiet:
                            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                        if str(result.get("status")) != "success":
                            overall_ok = False

                pending_eval = set(eval_futs.keys())
                while pending_eval:
                    done, pending_eval = wait(pending_eval, return_when=FIRST_COMPLETED)
                    for ef in done:
                        rd, h, label = eval_futs[ef]
                        try:
                            result = ef.result()
                        except Exception as e:  # noqa: BLE001
                            result = {
                                "status": "runner_exception",
                                "runnable": False,
                                "round_dir": str(rd),
                                "head": h,
                                "error": repr(e),
                            }
                        print(f"eval done {rd.name}/{label}: status={result.get('status')}", file=sys.stderr, flush=True)
                        if not args.quiet:
                            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                        if str(result.get("status")) != "success":
                            overall_ok = False
            return 0 if overall_ok else 1

        for rd in rounds:
            llm_out = rd / "llm_output.txt"
            prompt_p = rd / "prompt.txt"
            if not llm_out.is_file() or not prompt_p.is_file():
                print(f"Skip {rd}: need prompt.txt and llm_output.txt", file=sys.stderr)
                continue
            text = _read_text(llm_out)
            total = _count_tokens(text, tok)
            heads = _head_sizes_through_full(total, step_n)
            if not heads:
                print(f"Skip {rd}: empty llm_output.txt", file=sys.stderr)
                continue
            round_log_lines: list[str] = []
            sweep_line = f"Sweep {rd.name}: tokens≈{total}, heads={heads}"
            round_log_lines.append(sweep_line)
            print(
                f"Sweep {rd.relative_to(parent)}: tokens≈{total}, heads={heads}",
                file=sys.stderr,
                flush=True,
            )
            req_parallel = int(args.head_parallelism or 0)
            max_workers = len(heads) if req_parallel == 0 else min(len(heads), max(1, req_parallel))
            if max_workers <= 1:
                seq_line = "  parallel heads: workers=1"
                round_log_lines.append(seq_line)
                for h in heads:
                    label = f"head{h}"
                    print(f"  -> {label}", file=sys.stderr, flush=True)
                    result = run_speculative_pipeline(
                        args,
                        target_dir=rd,
                        head_n=h,
                        output_subdir=label,
                    )
                    if not args.quiet:
                        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                    done_line = f"  done {label}: status={result.get('status')}"
                    round_log_lines.append(done_line)
                    if str(result.get("status")) != "success":
                        overall_ok = False
                _write_round_sweep_log(rd, round_log_lines)
                continue

            par_line = f"  parallel heads: workers={max_workers}"
            round_log_lines.append(par_line)
            print(par_line, file=sys.stderr, flush=True)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        run_speculative_pipeline,
                        args,
                        target_dir=rd,
                        head_n=h,
                        output_subdir=f"head{h}",
                    ): h
                    for h in heads
                }
                for fut in as_completed(futs):
                    h = futs[fut]
                    label = f"head{h}"
                    try:
                        result = fut.result()
                    except Exception as e:  # noqa: BLE001
                        result = {
                            "status": "runner_exception",
                            "runnable": False,
                            "round_dir": str(rd),
                            "head": h,
                            "error": repr(e),
                        }
                    done_line = f"  done {label}: status={result.get('status')}"
                    round_log_lines.append(done_line)
                    print(done_line, file=sys.stderr, flush=True)
                    if not args.quiet:
                        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                    if str(result.get("status")) != "success":
                        overall_ok = False
            _write_round_sweep_log(rd, round_log_lines)
        return 0 if overall_ok else 1

    result = run_speculative_pipeline(args)
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"status={result.get('status')}", file=sys.stderr, flush=True)
    return 0 if str(result.get("status")) == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
