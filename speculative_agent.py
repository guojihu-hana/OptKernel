from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from agent import AgentConfig, KernelBenchAgent, extract_python_module
from run_ncu import PROFILE_K, SKIP_K, effective_ncu_metrics, nccu_bin, run_ncu_profile_subprocess
from run_validation import run_forward_validation_subprocess


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_output_dir(target_dir: Path) -> Path:
    """Mirror ``.../runs/...`` to ``.../runs/spec/...``; else keep original directory."""
    parts = target_dir.parts
    try:
        i = parts.index("runs")
    except ValueError:
        return target_dir
    runs_root = Path(*parts[: i + 1])
    relative_under_runs = Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path()
    return runs_root / "spec" / relative_under_runs


def _first_n_tokens_text(text: str, n: int, tokenizer_name: str | None = None) -> str:
    if n <= 0 or not text:
        return ""
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-not-found]

            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            ids = tok.encode(text, add_special_tokens=False)[:n]
            return tok.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)[:n]
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


def build_spec_prompt(
    target_dir: Path,
    head_n: int,
    tokenizer_name: str | None = None,
    output_name: str = "spec_prompt.txt",
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
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / output_name

    prompt_text = _read_text(prompt_path)
    llm_text = _read_text(llm_output_path)
    llm_prefix = _first_n_tokens_text(llm_text, head_n, tokenizer_name)
    merged = f"{prompt_text.rstrip()}\n\n{llm_prefix}"
    out.write_text(merged, encoding="utf-8")
    return out


def run_speculative_pipeline(args: argparse.Namespace) -> dict[str, object]:
    target_dir = args.path.resolve()
    spec_prompt_path = build_spec_prompt(
        target_dir=target_dir,
        head_n=int(args.head),
        tokenizer_name=(args.tokenizer or "").strip() or None,
        output_name=args.output_name,
    )
    out_dir = spec_prompt_path.parent
    llm_out_path = out_dir / "llm_output.txt"
    kernel_path = out_dir / "kernel.py"
    metrics_path = out_dir / "metrics.json"

    system, user = _split_system_user(_read_text(spec_prompt_path))
    addr, port = _host_to_addr_port(args.host)
    model_name = (args.model or "").strip() or _discover_model(args.host, args.api_key or "")
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

    llm = agent.call_llm(system=system, user=user, round_idx=int(args.round_idx), llm_output_path=llm_out_path)
    if not llm.get("ok"):
        payload: dict[str, object] = {
            "status": "llm_subprocess_error",
            "runnable": False,
            "llm": llm,
            "spec_prompt": str(spec_prompt_path),
            "kernel_path": str(kernel_path),
        }
        agent.write_metrics(metrics_path, payload)
        return payload

    raw = str(llm.get("text", ""))
    if not bool(llm.get("llm_output_dumped", False)):
        llm_out_path.write_text(raw, encoding="utf-8")

    py_src = extract_python_module(raw)
    if py_src is None:
        payload = {
            "status": "parse_error",
            "runnable": False,
            "parse_error": "No ```python ... ``` block found in LLM output.",
            "spec_prompt": str(spec_prompt_path),
            "kernel_path": str(kernel_path),
        }
        agent.write_metrics(metrics_path, payload)
        return payload

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
        "status": val.get("status"),
        "runnable": bool(val.get("runnable")),
        "spec_prompt": str(spec_prompt_path),
        "kernel_path": str(kernel_path),
        "llm_output_path": str(llm_out_path),
        "validation": val,
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
    p.add_argument("path", type=Path, help="Directory containing prompt.txt and llm_output.txt")
    p.add_argument("--head", type=int, required=True, help="Take first N tokens from llm_output.txt")
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
    p.add_argument("--output-name", type=str, default="spec_prompt.txt")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--max-context-length", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--no-reasoning", action="store_true")
    p.add_argument("--round-idx", type=int, default=0, help="Tag only (for llm_output progress)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--gen-module-name", type=str, default="kernelbench_generated_spec")
    p.add_argument("--worker-url", type=str, default=os.environ.get("OPTKERNEL_WORKER_URL", "") or "")
    p.add_argument("--no-ncu", action="store_true")
    p.add_argument("--ncu-binary", type=str, default=os.environ.get("NCU_BINARY", "ncu"))
    p.add_argument("--ncu-metrics", type=str, default="")
    p.add_argument("--ncu-extra", type=str, default="")
    p.add_argument("--ncu-launch-skip", type=int, default=SKIP_K)
    p.add_argument("--ncu-launch-count", type=int, default=PROFILE_K)
    args = p.parse_args()

    result = run_speculative_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if str(result.get("status")) == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
