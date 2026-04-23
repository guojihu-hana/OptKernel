"""
KernelBench CUDA rewrite agent: multi-round LLM (``run_llm`` subprocess) -> kernel.py
-> run_validation -> ncu.

Per round under work_dir/round_{k:03d}/: kernel.py, prompt.txt, llm_output.txt, metrics.json.
Default work_dir (if --work-dir omitted): ./runs/<YYYYMMDDHHMMSS>/<task_file_stem>/.
Resume: pass the same --task-file and an existing --work-dir, then --start-round <k> (0-based);
round_{k-1} should exist so round k can build the prior-kernel prompt (see run_generation).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------

from run_llm import run_llm_subprocess

_REPO_ROOT = Path(__file__).resolve().parent


def _llm_subproc_staging_dir() -> Path:
    """Scratch for ``run_llm`` file inputs: ``<repo>/temp/llm_subproc/``; each call uses a subdir, deleted after."""
    return _REPO_ROOT / "temp" / "llm_subproc"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -----------------------------------------------------------------------------
# Prompts (text under prompts/; construction in build_prompts.py)
# -----------------------------------------------------------------------------

from build_prompts import (
    best_round_tuple_for_prompt,
    build_user_prompt_round0,
    build_user_prompt_roundk,
    find_best_previous_round,
    speedup_from_metrics,
    summarize_metrics_for_prompt,
    system_prompt_round0,
    system_prompt_roundk,
)
from run_ncu import (
    PROFILE_K,
    SKIP_K,
    effective_ncu_metrics,
    nccu_bin,
    run_ncu_profile_subprocess,
)
from run_validation import run_forward_validation_subprocess

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
    # First round index (0-based); run() uses for r in range(start_round, max_rounds).
    start_round: int = 0
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
    # 0 = disabled. Else cap max_tokens on LLM continuation rounds only (see llm_local).
    max_context_length: int = 0
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
    # Bearer for OpenAI-compatible servers (local/vllm); required there via CLI --api-key.
    openai_compatible_api_key: str = ""
    # vLLM extra_body.repetition_penalty; default 1.1 for MiniMax, else 1.0 (see parse_args).
    repetition_penalty: float = 1.0


class KernelBenchAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._reference_source = config.task_path.read_text(encoding="utf-8")
        # Best (round_index, speedup) by ``benchmark_timing.speedup``; in-memory only.
        # If resuming, :meth:`run` may seed with one ``find_best_previous_round`` call.
        self._best_by_speedup: Optional[Tuple[int, float]] = None

    def _current_best_for_prompt(self, work_dir: Path, current_round: int) -> Optional[Tuple[int, float, str, str]]:
        t = self._best_by_speedup
        if t is None or t[0] >= current_round:
            return None
        return best_round_tuple_for_prompt(work_dir, t[0], t[1])

    def _update_best_from_metrics(self, round_idx: int, base: dict[str, Any]) -> None:
        if not base.get("runnable"):
            return
        sp = speedup_from_metrics(base)
        if sp is None:
            return
        t = self._best_by_speedup
        o_r, o_sp = (t[0], t[1]) if t is not None else (-1, float("-inf"))
        if not (sp > o_sp or (sp == o_sp and round_idx > o_r)):
            return
        self._best_by_speedup = (round_idx, sp)

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
    ) -> dict[str, Any]:
        """
        Single LLM call via a child process (see :func:`run_llm.run_llm_subprocess`).

        On success: ``{"ok": true, "text", "llm_output_dumped"}``.

        On failure: the child/parent result dict (``ok: false`` plus ``runtime_error`` and
        often ``subprocess`` with command/stderr/stdout) so the agent can store it in
        ``metrics.json``, same style as validation/ncu.
        """
        c = self.config
        if llm_output_path is not None:
            llm_output_path.parent.mkdir(parents=True, exist_ok=True)
        res: dict[str, Any] = {}
        work: Optional[Path] = None
        staging = _llm_subproc_staging_dir()
        staging.mkdir(parents=True, exist_ok=True)
        work = staging / uuid.uuid4().hex
        try:
            work.mkdir(parents=True, exist_ok=True)
            sys_p = work / "system.txt"
            usr_p = work / "user.txt"
            sys_p.write_text(system, encoding="utf-8")
            usr_p.write_text(user, encoding="utf-8")
            res = run_llm_subprocess(
                sys_p,
                usr_p,
                round_idx,
                llm_output_path,
                temperature=c.temperature,
                max_tokens=c.max_tokens,
                server_type=c.server_type,
                server_address=c.server_address,
                server_port=c.server_port,
                model_name=c.model_name,
                is_reasoning_model=self.reasoning_for_round(round_idx),
                openai_compatible_api_key=c.openai_compatible_api_key,
                repetition_penalty=c.repetition_penalty,
                max_context_length=c.max_context_length,
            )
        finally:
            if work is not None and work.is_dir():
                shutil.rmtree(work, ignore_errors=True)
        if res.get("ok"):
            return {
                "ok": True,
                "text": str(res.get("text", "")),
                "llm_output_dumped": bool(res.get("llm_output_dumped", False)),
            }
        return res

    def run_generation(self, round_idx: int) -> dict[str, Any]:
        """
        Task (from init), build prompt, call LLM, extract ```python```, write
        ``prompt.txt``, ``llm_output.txt``, and ``kernel.py`` if extraction succeeds.

        If the LLM subprocess fails, writes ``metrics.json`` with
        ``status: llm_subprocess_error`` and a ``llm`` payload (``runtime_error``, ``subprocess``).
        If parsing fails, writes ``metrics.json`` and returns ``status: parse_error``.
        Otherwise returns a partial dict without validation/ncu; :meth:`run_round` continues from there.
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
            best = self._current_best_for_prompt(c.work_dir, round_idx)
            system = system_prompt_roundk()
            user_body = build_user_prompt_roundk(
                self._reference_source,
                prev_kernel,
                summary,
                best_previous_round=best,
                previous_round_index=round_idx - 1,
            )

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
        llm_res = self.call_llm(system, user, round_idx, llm_out_path)
        ll_t1 = time.perf_counter()
        ll_ts1 = _utc_iso()
        llm_eval_timing = {
            "started_at": ll_ts0,
            "finished_at": ll_ts1,
            "seconds": round(ll_t1 - ll_t0, 6),
        }
        if not llm_res.get("ok"):
            base: dict[str, Any] = {
                "round": round_idx,
                "task_path": str(c.task_path.resolve()),
                "work_dir": str(rd.resolve()),
                "model_name": c.model_name,
                "runnable": False,
                "status": "llm_subprocess_error",
                "llm": llm_res,
                "eval_timing": {"llm": llm_eval_timing},
            }
            self.write_metrics(metrics_path, base)
            return base

        llm_raw = str(llm_res.get("text", ""))
        llm_written = bool(llm_res.get("llm_output_dumped", False))
        if not llm_written:
            llm_out_path.write_text(llm_raw, encoding="utf-8")

        py_src = extract_python_module(llm_raw)
        base = {
            "round": round_idx,
            "task_path": str(c.task_path.resolve()),
            "work_dir": str(rd.resolve()),
            "model_name": c.model_name,
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

        base = self.run_generation(round_idx)
        if base.get("status") in ("parse_error", "llm_subprocess_error"):
            return base

        gen_mod_name = f"kernelbench_generated_r{round_idx}"
        eval_timing: dict[str, Any] = dict(base.get("eval_timing") or {})
        v_t0 = time.perf_counter()
        v_ts0 = _utc_iso()
        # Fresh Python process per round: CUDA / native crashes do not take down the LLM agent.
        val = run_forward_validation_subprocess(
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
            ncu_info = run_ncu_profile_subprocess(
                kernel_path,
                rd,
                c.ncu_binary,
                metrics_args,
                c.ncu_extra_args,
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
        self._update_best_from_metrics(round_idx, base)
        return base

    def run(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        c = self.config
        c.work_dir.mkdir(parents=True, exist_ok=True)
        if c.start_round > 0:
            fb = find_best_previous_round(c.work_dir, c.start_round)
            if fb is not None:
                self._best_by_speedup = (fb[0], fb[1])

        for r in range(c.start_round, c.max_rounds):
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
    p.add_argument(
        "--start-round",
        type=int,
        default=0,
        help="0-based first round to run, for resuming in an existing --work-dir. "
        "Runs r in [start-round, max-rounds). Requires round_{start-1} when start-round > 0 "
        "(e.g. start 4 needs .../round_003/ for the previous-kernel prompt).",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Exclusive end of round indices: runs r in [start-round, max-rounds) (default 5 → rounds 0–4).",
    )
    p.add_argument("--stop-on-success", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--rtol", type=float, default=1e-4)
    p.add_argument("--server-type", type=str, default=os.environ.get("KERNEL_AGENT_SERVER", "local"))
    p.add_argument("--server-address", type=str, default="localhost")
    p.add_argument("--server-port", type=int, default=30000)
    p.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Bearer token for OpenAI-compatible servers. Required when --server-type is local or vllm "
        "(must match vLLM --api-key). Ignored for other server types.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=(
            os.environ.get("KERNEL_AGENT_MODEL")
            or os.environ.get("MODEL_NAME")
            or "gpt-4o-mini"
        ),
    )
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument(
        "--max-context-length",
        type=int,
        default=0,
        help="Model context window (0 = off). On LLM continuation requests only (after length truncation, k>0), "
        "cap output tokens to min(--max-tokens, max_context_length - estimated prompt tokens). "
        "The first completion uses full --max-tokens.",
    )
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="vLLM sampling repetition_penalty (sent in extra_body). "
        "If omitted: 1.1 when --model name contains 'MiniMax' (case-insensitive), else 1.0.",
    )
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

    _st = args.server_type.strip().lower()
    _api_key = (args.api_key or "").strip()
    if _st in {"local", "vllm"} and not _api_key:
        p.error("--api-key is required when --server-type is local or vllm")

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

    def _default_repetition_penalty(model: str) -> float:
        if "minimax" in (model or "").lower():
            return 1.1
        return 1.0

    if args.repetition_penalty is not None:
        _rep = args.repetition_penalty
    else:
        _rep = _default_repetition_penalty(args.model)

    if args.start_round < 0:
        p.error("--start-round must be >= 0")
    if args.start_round >= args.max_rounds:
        p.error("--start-round must be < --max-rounds (run range is [start-round, max-rounds))")
    if int(args.max_context_length or 0) < 0:
        p.error("--max-context-length must be >= 0 (0 = disabled)")

    return AgentConfig(
        task_path=task_path,
        work_dir=work_dir,
        start_round=args.start_round,
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
        max_context_length=int(args.max_context_length or 0),
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
        openai_compatible_api_key=_api_key,
        repetition_penalty=_rep,
    )


def main(argv: Optional[list[str]] = None) -> int:
    cfg = parse_args(argv)
    if not cfg.task_path.is_file():
        print(f"Task file not found: {cfg.task_path}", file=sys.stderr)
        return 1

    agent = KernelBenchAgent(cfg)
    if cfg.start_round > 0:
        print(
            f"Resuming: rounds {cfg.start_round}..{cfg.max_rounds - 1} under {cfg.work_dir}",
            file=sys.stderr,
        )
    agent.run()
    print(f"Done. Artifacts under {cfg.work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
