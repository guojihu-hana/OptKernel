"""
KernelBench CUDA rewrite agent: multi-round LLM -> kernel.py -> run_validation -> ncu.

Per round under work_dir/round_{k:03d}/: kernel.py, prompt.txt, llm_output.txt, metrics.json.
Default work_dir (if --work-dir omitted): ./runs/<YYYYMMDDHHMMSS>/<task_file_stem>/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Set

# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------

from query_server import query_server


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -----------------------------------------------------------------------------
# Prompts (text under prompts/; construction in build_prompts.py)
# -----------------------------------------------------------------------------

from build_prompts import (
    build_user_prompt_round0,
    build_user_prompt_roundk,
    summarize_metrics_for_prompt,
    system_prompt_round0,
    system_prompt_roundk,
)
from run_ncu import (
    PROFILE_K,
    SKIP_K,
    effective_ncu_metrics,
    nccu_bin,
    run_ncu_profile,
)
from run_validation import run_forward_validation

# -----------------------------------------------------------------------------
# Extract ```python``` from LLM output
# -----------------------------------------------------------------------------

_FENCED_PYTHON = re.compile(
    r"```(?:python|py)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_ANY_FENCE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_python_module(text: str) -> Optional[str]:
    """
    Extract ```python``` / ```py``` content. If there are multiple fences, use the
    **last** one (models often put the final full file after earlier snippets).
    """
    if not text or not text.strip():
        return None

    py_blocks = [b.strip() for b in _FENCED_PYTHON.findall(text) if b.strip()]
    if py_blocks:
        return py_blocks[-1]

    generic: list[str] = []
    for gm in _ANY_FENCE.finditer(text):
        block = gm.group(1).strip()
        if "import" in block and "Model" in block:
            generic.append(block)
    if generic:
        return generic[-1]
    return None


# -----------------------------------------------------------------------------
# Agent config & run loop
# -----------------------------------------------------------------------------


@dataclass
class AgentConfig:
    task_path: Path
    work_dir: Path
    max_rounds: int = 5
    stop_on_success: bool = False
    seed: int = 0
    atol: float = 1e-4
    rtol: float = 1e-4
    server_type: str = "local"
    server_address: str = "localhost"
    server_port: int = 30000
    model_name: str = "GLM-5.1-FP8"
    temperature: float = 0.1
    max_tokens: int = 16384
    # Thinking / reasoning API (see query_server is_reasoning_model). Default: on.
    reasoning_enabled: bool = True
    """If set, only these round indices use reasoning; all others do not."""
    reasoning_only_rounds: Optional[Set[int]] = None
    """Rounds where reasoning is forced off (when reasoning_only_rounds is unset)."""
    reasoning_except_rounds: Optional[Set[int]] = None
    run_ncu: bool = True
    ncu_binary: str = field(default_factory=lambda: shutil.which("ncu") or "ncu")
    ncu_metrics: list[str] = field(default_factory=list)
    ncu_extra_args: list[str] = field(default_factory=list)
    ncu_launch_skip: int = SKIP_K
    ncu_launch_count: int = PROFILE_K


class KernelBenchAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._reference_source = config.task_path.read_text(encoding="utf-8")

    def reasoning_for_round(self, round_idx: int) -> bool:
        """Whether to enable thinking/reasoning for this round (query_server flag)."""
        c = self.config
        if not c.reasoning_enabled:
            return False
        if c.reasoning_only_rounds is not None:
            return round_idx in c.reasoning_only_rounds
        if c.reasoning_except_rounds:
            return round_idx not in c.reasoning_except_rounds
        return True

    def round_dir(self, round_idx: int) -> Path:
        return self.config.work_dir / f"round_{round_idx:03d}"

    def write_metrics(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def call_llm(
        self,
        system: str,
        user: str,
        round_idx: int,
        llm_output_path: Optional[Path] = None,
    ) -> tuple[str, bool]:
        """Returns (completion_text, llm_output_already_written_to_disk)."""
        c = self.config
        raw, llm_dumped = query_server(
            user,
            system_prompt=system,
            temperature=c.temperature,
            max_tokens=c.max_tokens,
            server_type=c.server_type,
            server_address=c.server_address,
            server_port=c.server_port,
            model_name=c.model_name,
            is_reasoning_model=self.reasoning_for_round(round_idx),
            call_type="kernel_bench_agent",
            round_idx=round_idx,
            stream_dump_path=str(llm_output_path) if llm_output_path else None,
        )
        if isinstance(raw, dict):
            return str(raw.get("text", "")), llm_dumped
        if isinstance(raw, list):
            return (str(raw[0]) if raw else ""), llm_dumped
        return str(raw or ""), llm_dumped

    def run_round_llm_only(self, round_idx: int) -> dict[str, Any]:
        """
        Task (from init), build prompt, call LLM, extract ```python```, write
        ``prompt.txt``, ``llm_output.txt``, and ``kernel.py`` if extraction succeeds.

        If parsing fails, writes ``metrics.json`` and returns a **final** result dict
        (``status: parse_error``). Otherwise returns a partial dict without
        ``run_validation.run_forward_validation`` / ncu; :meth:`run_round` continues from there.
        """
        c = self.config
        rd = self.round_dir(round_idx)
        rd.mkdir(parents=True, exist_ok=True)

        kernel_path = rd / "kernel.py"
        prompt_path = rd / "prompt.txt"
        llm_out_path = rd / "llm_output.txt"
        metrics_path = rd / "metrics.json"

        if round_idx == 0:
            system = system_prompt_round0()
            user_body = build_user_prompt_round0(self._reference_source)
        else:
            prev_rd = self.round_dir(round_idx - 1)
            pk = prev_rd / "kernel.py"
            prev_kernel = (
                pk.read_text(encoding="utf-8")
                if pk.is_file()
                else "# (previous round did not write kernel.py — e.g. parse_error)\n"
            )
            prev_metrics = prev_rd / "metrics.json"
            summary = summarize_metrics_for_prompt(prev_metrics)
            system = system_prompt_roundk()
            user_body = build_user_prompt_roundk(self._reference_source, prev_kernel, summary)

        if not c.reasoning_enabled:
            _no_reasoning_banner = (
                "NO THINKING\n\nCRITICAL DIRECTLY OUTPUT THE CODE\n\n"
            )
            user = _no_reasoning_banner + user_body
            prompt_path.write_text(
                _no_reasoning_banner
                + f"--- system ---\n{system}\n\n--- user ---\n{user_body}",
                encoding="utf-8",
            )
        else:
            user = user_body
            prompt_path.write_text(
                f"--- system ---\n{system}\n\n--- user ---\n{user_body}",
                encoding="utf-8",
            )

        ll_t0 = time.perf_counter()
        ll_ts0 = _utc_iso()
        llm_raw, llm_written = self.call_llm(system, user, round_idx, llm_out_path)
        ll_t1 = time.perf_counter()
        ll_ts1 = _utc_iso()
        llm_eval_timing = {
            "started_at": ll_ts0,
            "finished_at": ll_ts1,
            "seconds": round(ll_t1 - ll_t0, 6),
        }
        if not llm_written:
            llm_out_path.write_text(llm_raw, encoding="utf-8")

        py_src = extract_python_module(llm_raw)
        base: dict[str, Any] = {
            "round": round_idx,
            "task_path": str(c.task_path.resolve()),
            "work_dir": str(rd.resolve()),
            "eval_timing": {"llm": llm_eval_timing},
        }

        if py_src is None:
            base.update(
                {
                    "runnable": False,
                    "status": "parse_error",
                    "parse_error": "No ```python ... ``` block found in LLM output.",
                }
            )
            self.write_metrics(metrics_path, base)
            return base

        kernel_path.write_text(py_src, encoding="utf-8")
        return base

    def run_round(self, round_idx: int) -> dict[str, Any]:
        c = self.config
        rd = self.round_dir(round_idx)
        kernel_path = rd / "kernel.py"
        metrics_path = rd / "metrics.json"

        base = self.run_round_llm_only(round_idx)
        if base.get("status") == "parse_error":
            return base

        gen_mod_name = f"kernelbench_generated_r{round_idx}"
        eval_timing: dict[str, Any] = dict(base.get("eval_timing") or {})
        v_t0 = time.perf_counter()
        v_ts0 = _utc_iso()
        val = run_forward_validation(
            c.task_path,
            kernel_path,
            seed=c.seed,
            atol=c.atol,
            rtol=c.rtol,
            gen_module_name=gen_mod_name,
        )
        v_t1 = time.perf_counter()
        v_ts1 = _utc_iso()
        eval_timing["validation"] = {
            "started_at": v_ts0,
            "finished_at": v_ts1,
            "seconds": round(v_t1 - v_t0, 6),
        }
        base.update(val)
        base["eval_timing"] = eval_timing

        if not val.get("runnable"):
            self.write_metrics(metrics_path, base)
            return base

        # runnable: optional ncu
        if c.run_ncu and shutil.which(nccu_bin(c.ncu_binary)):
            metric_names = effective_ncu_metrics(c.ncu_metrics)
            metrics_args: list[str] = ["--metrics", ",".join(metric_names)]

            n_t0 = time.perf_counter()
            n_ts0 = _utc_iso()
            ncu_info = run_ncu_profile(
                kernel_path,
                rd,
                c.ncu_binary,
                metrics_args,
                c.ncu_extra_args,
                metric_names,
                launch_skip=c.ncu_launch_skip,
                launch_count=c.ncu_launch_count,
            )
            n_t1 = time.perf_counter()
            n_ts1 = _utc_iso()
            eval_timing["ncu"] = {
                "started_at": n_ts0,
                "finished_at": n_ts1,
                "seconds": round(n_t1 - n_t0, 6),
            }
            base["ncu"] = ncu_info
            if ncu_info.get("returncode") != 0:
                base["status"] = "ncu_error"
                base["ncu_error"] = {
                    "message": "ncu process returned non-zero",
                    "returncode": ncu_info.get("returncode"),
                    "stderr_tail": ncu_info.get("stderr"),
                }
            else:
                base["status"] = "success"
        elif c.run_ncu:
            base["ncu"] = {"skipped": True, "reason": "ncu not found on PATH"}
            eval_timing["ncu"] = {"skipped": True, "reason": "ncu not found on PATH"}
            base["status"] = "success"
        else:
            eval_timing["ncu"] = {"skipped": True, "reason": "run_ncu disabled in config"}
            base["status"] = "success"

        self.write_metrics(metrics_path, base)
        return base

    def run(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        c = self.config
        c.work_dir.mkdir(parents=True, exist_ok=True)

        for r in range(c.max_rounds):
            m = self.run_round(r)
            results.append(m)
            if c.stop_on_success and m.get("status") == "success":
                break

        return results


def parse_args(argv: Optional[list[str]] = None) -> AgentConfig:
    p = argparse.ArgumentParser(description="KernelBench CUDA agent (multi-round)")
    p.add_argument(
        "--task-file",
        type=str,
        required=True,
        help="Path to KernelBench reference .py",
    )
    p.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Output root (rounds go under work_dir/round_{k:03d}/). "
        "If omitted: ./runs/<YYYYMMDDHHMMSS>/<task_file_stem>/ using current time and --task-file basename.",
    )
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--stop-on-success", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--server-type", type=str, default=os.environ.get("KERNEL_AGENT_SERVER", "local"))
    p.add_argument("--server-address", type=str, default="localhost")
    p.add_argument("--server-port", type=int, default=30000)
    p.add_argument("--model", type=str, default=os.environ.get("KERNEL_AGENT_MODEL", "gpt-4o-mini"))
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--no-ncu", action="store_true", help="Skip ncu profiling")
    p.add_argument(
        "--ncu-metrics",
        type=str,
        default="",
        help="Comma-separated ncu metrics (--metrics a,b,c). "
        "If omitted, uses run_ncu.DEFAULT_NCU_METRICS.",
    )
    p.add_argument(
        "--ncu-extra",
        type=str,
        default="",
        help="Extra args for ncu (space-separated)",
    )
    p.add_argument("--ncu-binary", type=str, default=os.environ.get("NCU_BINARY", "ncu"))
    p.add_argument(
        "--ncu-launch-skip",
        type=int,
        default=SKIP_K,
        help=f"ncu --launch-skip (default {SKIP_K}, run_ncu.SKIP_K)",
    )
    p.add_argument(
        "--ncu-launch-count",
        type=int,
        default=PROFILE_K,
        help=f"ncu --launch-count (default {PROFILE_K}, run_ncu.PROFILE_K)",
    )
    p.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable thinking/reasoning mode for all rounds (query_server is_reasoning_model=False).",
    )
    p.add_argument(
        "--reasoning-only-rounds",
        type=str,
        default="",
        help="Comma-separated round indices (0-based) where reasoning is ON; other rounds OFF. "
        "If set, overrides --reasoning-except-rounds.",
    )
    p.add_argument(
        "--reasoning-except-rounds",
        type=str,
        default="",
        help="Comma-separated rounds where reasoning is OFF; all other rounds ON (default thinking on).",
    )
    args = p.parse_args(argv)

    task_path = Path(args.task_file).resolve()
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        work_dir = (Path.cwd() / "runs" / stamp / task_path.stem).resolve()

    def _parse_round_set(s: str) -> Optional[set[int]]:
        s = (s or "").strip()
        if not s:
            return None
        out: set[int] = set()
        for part in s.split(","):
            part = part.strip()
            if part:
                out.add(int(part))
        return out

    metrics_list = [x.strip() for x in args.ncu_metrics.split(",") if x.strip()]
    extra_list = args.ncu_extra.split() if args.ncu_extra.strip() else []

    only = _parse_round_set(args.reasoning_only_rounds)
    exc = _parse_round_set(args.reasoning_except_rounds)

    return AgentConfig(
        task_path=task_path,
        work_dir=work_dir,
        max_rounds=args.max_rounds,
        stop_on_success=args.stop_on_success,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        server_type=args.server_type,
        server_address=args.server_address,
        server_port=args.server_port,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_enabled=not args.no_reasoning,
        reasoning_only_rounds=only,
        reasoning_except_rounds=exc,
        run_ncu=not args.no_ncu,
        ncu_binary=args.ncu_binary,
        ncu_metrics=metrics_list,
        ncu_extra_args=extra_list,
        ncu_launch_skip=args.ncu_launch_skip,
        ncu_launch_count=args.ncu_launch_count,
    )


def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_args(argv)
    if not cfg.task_path.is_file():
        print(f"Task file not found: {cfg.task_path}", file=sys.stderr)
        return 1

    agent = KernelBenchAgent(cfg)
    agent.run()
    print(f"Done. Artifacts under {cfg.work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
