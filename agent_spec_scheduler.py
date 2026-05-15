"""
Inline speculative scheduling for :class:`agent.KernelBenchAgent` when ``--spec`` is set.

Imported lazily from :meth:`KernelBenchAgent.run_round` to avoid import cycles
(``speculative_agent`` imports ``agent``).
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

from build_prompts import speedup_from_metrics
from interruption import InterruptionCoordinator, InterruptionContext, LoggingInterruptionPolicy
from spec_token_utils import count_tokens, first_n_tokens_text, iter_token_head_targets

if TYPE_CHECKING:
    from agent import KernelBenchAgent

# Match speculative_agent wording (keep in sync for KV / prompt behavior).
SPEC_PROMPT_THINKING_PREFIX = (
    "Read and learn from the following thinking content (maybe partial), then generate **only** the final ```python ... ``` fenced code "
    "block containing the full working kernel implementation.\n\n"
    "--- thinking content ---"
)
SPEC_PROMPT_THINKING_SUFFIX = (
    "\n\n--- thinking content over --- directly output the final ```python ... ``` fenced code without other outputs."
)


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


def _uniquify_load_inline_name(py_src: str, suffix: str) -> str:
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


def _benchmark_ok(metrics: dict[str, Any]) -> bool:
    bt = metrics.get("benchmark_timing") if isinstance(metrics, dict) else None
    benchmark_speedup = bt.get("speedup") if isinstance(bt, dict) else None
    return bool(isinstance(bt, dict) and not bool(bt.get("skipped")) and benchmark_speedup is not None)


def _successful_speedup(metrics: dict[str, Any]) -> Optional[float]:
    if metrics.get("status") != "success":
        return None
    if not _benchmark_ok(metrics):
        return None
    return speedup_from_metrics(metrics)


@runtime_checkable
class SpecBatchSizer(Protocol):
    def width(
        self,
        *,
        round_idx: int,
        trigger_id: str,
        worker_hint: Optional[int],
        queued_generations: int,
        queued_validations: int,
        spec_jobs_scheduled: int,
        spec_max_candidates: int,
    ) -> int:
        ...


class ConstantSpecBatchSizer:
    def __init__(self, n: int) -> None:
        self._n = max(1, int(n))

    def width(
        self,
        *,
        round_idx: int,
        trigger_id: str,
        worker_hint: Optional[int],
        queued_generations: int,
        queued_validations: int,
        spec_jobs_scheduled: int,
        spec_max_candidates: int,
    ) -> int:
        _ = round_idx, trigger_id, worker_hint, queued_generations, queued_validations, spec_jobs_scheduled, spec_max_candidates
        return self._n


@dataclass
class _CandRecord:
    candidate_id: str
    role: str
    work_dir: Path
    metrics: dict[str, Any]


def _new_token_heads(nt: int, fired: set[int], *, step: int, explicit: list[int]) -> list[int]:
    new: list[int] = []
    if explicit:
        for h in sorted({int(x) for x in explicit if int(x) > 0}):
            if nt >= h and h not in fired:
                new.append(h)
                fired.add(h)
        return sorted(new)
    for h in iter_token_head_targets(step):
        if nt < h:
            break
        if h not in fired:
            new.append(h)
            fired.add(h)
    return sorted(new)


def execute_spec_round(agent: KernelBenchAgent, round_idx: int) -> dict[str, Any]:
    c = agent.config
    rd = agent.round_dir(round_idx)
    rd.mkdir(parents=True, exist_ok=True)
    cand_root = rd / "candidates"
    cand_root.mkdir(parents=True, exist_ok=True)

    tokenizer = (c.spec_tokenizer_name or "").strip() or None
    batch_sizer: SpecBatchSizer = ConstantSpecBatchSizer(c.batch_spec)

    prompt_path = rd / "prompt.txt"
    llm_out_path = rd / "llm_output.txt"
    metrics_path = rd / "metrics.json"

    bundle = agent.build_round_prompt_bundle(round_idx)
    system = str(bundle["system"])
    user = str(bundle["user"])
    prompt_file_text = str(bundle["prompt_file_text"])
    prompt_path.write_text(prompt_file_text, encoding="utf-8")

    fired_heads: set[int] = set()
    spec_jobs_scheduled = 0
    spec_lock = threading.Lock()
    futures: list[tuple[str, Future[_CandRecord]]] = []
    interrupt_coord = InterruptionCoordinator(
        c.enable_interruption,
        LoggingInterruptionPolicy() if c.enable_interruption else None,
    )

    gen_executor = ThreadPoolExecutor(max_workers=max(1, int(c.spec_generation_parallelism)))

    def _enqueue_specs_for_heads(new_heads: list[int], snapshot: str) -> None:
        nonlocal spec_jobs_scheduled
        if not new_heads:
            return

        nh = sorted(new_heads)
        coordinator_detail = ",".join(str(h) for h in nh)
        interrupt_coord.emit(
            InterruptionContext(
                round_idx=round_idx,
                trigger_kind="spec_token_heads",
                detail=f"crossed_heads={coordinator_detail}",
                approx_output_tokens=count_tokens(snapshot or "", tokenizer),
                output_path=str(llm_out_path),
            )
        )

        cap_mc = int(c.spec_max_candidates or 0)
        for head_n in nh:
            qgen = sum(1 for _cid, fu in futures if not fu.done())
            slot_jobs: list[tuple[str, Path, str, int]] = []
            with spec_lock:
                remaining_cap = (
                    10**12 if cap_mc <= 0 else max(0, cap_mc - spec_jobs_scheduled)
                )
                if remaining_cap <= 0:
                    break
                raw_w = batch_sizer.width(
                    round_idx=round_idx,
                    trigger_id=f"head_{head_n}",
                    worker_hint=None,
                    queued_generations=qgen,
                    queued_validations=0,
                    spec_jobs_scheduled=spec_jobs_scheduled,
                    spec_max_candidates=cap_mc,
                )
                n_batch = max(0, min(int(raw_w), int(remaining_cap)))
                if n_batch > 0:
                    for slot_ix in range(n_batch):
                        remaining_cap_live = (
                            10**12 if cap_mc <= 0 else max(0, cap_mc - spec_jobs_scheduled)
                        )
                        if remaining_cap_live <= 0:
                            break
                        spec_jobs_scheduled += 1
                        cid = f"spec_head{head_n}_s{slot_ix}"
                        wd = cand_root / cid
                        wd.mkdir(parents=True, exist_ok=True)
                        suf = f"r{round_idx:03d}_h{head_n}_s{slot_ix}"
                        slot_jobs.append((cid, wd, suf, slot_ix))

            for cid, wd, suf, slot_ix in slot_jobs:
                fut = gen_executor.submit(
                    _run_one_spec_generation,
                    agent,
                    round_idx,
                    wd,
                    prompt_file_text,
                    snapshot,
                    head_n,
                    slot_ix,
                    system,
                    user,
                    tokenizer,
                    cid,
                    suf,
                )
                futures.append((cid, fut))

    main_box: list[dict[str, Any]] = []

    def _main_worker() -> None:
        from agent import _log_phase_start, _utc_iso

        ll_t0 = time.perf_counter()
        ll_ts0 = _utc_iso()
        _log_phase_start(round_idx, "llm", ll_ts0, c.task_path.stem)
        llm_res = agent.call_llm(system, user, round_idx, llm_out_path)
        ll_t1 = time.perf_counter()
        ll_ts1 = _utc_iso()
        llm_eval_timing = {
            "started_at": ll_ts0,
            "finished_at": ll_ts1,
            "seconds": round(ll_t1 - ll_t0, 6),
        }
        main_box.append({"llm": llm_res, "llm_eval_timing": llm_eval_timing})

    t_main = threading.Thread(target=_main_worker, name="main-llm", daemon=True)
    t_main.start()

    poll = max(0.05, float(c.spec_poll_interval or 1.0))
    while t_main.is_alive():
        snap = ""
        try:
            if llm_out_path.is_file():
                snap = llm_out_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            snap = ""
        nt = count_tokens(snap, tokenizer)
        if (c.spec_trigger or "token-heads").strip().lower() in {"token-heads", "token_heads"}:
            heads = _new_token_heads(
                nt,
                fired_heads,
                step=max(1, int(c.spec_head_step or 2000)),
                explicit=list(c.spec_heads or []),
            )
            if heads:
                _enqueue_specs_for_heads(heads, snap)
        time.sleep(poll)

    t_main.join()
    # Final poll for heads that complete together with stream end
    snap = ""
    if llm_out_path.is_file():
        snap = llm_out_path.read_text(encoding="utf-8", errors="replace")
    nt = count_tokens(snap, tokenizer)
    heads = _new_token_heads(
        nt,
        fired_heads,
        step=max(1, int(c.spec_head_step or 2000)),
        explicit=list(c.spec_heads or []),
    )
    if heads:
        _enqueue_specs_for_heads(heads, snap)

    gen_executor.shutdown(wait=True)

    gathered: dict[str, _CandRecord] = {}

    llm_combo = main_box[0] if main_box else {"llm": {"ok": False}, "llm_eval_timing": {}}
    llm_res = llm_combo["llm"]
    llm_eval_timing = llm_combo["llm_eval_timing"]

    main_dir = cand_root / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    main_metrics_path = main_dir / "metrics.json"
    main_kernel_path = main_dir / "kernel.py"
    shutil.copy2(prompt_path, main_dir / "prompt.txt")

    main_base = _finalize_main_generation(
        agent,
        rd,
        main_dir,
        round_idx,
        llm_out_path,
        llm_res,
        llm_eval_timing,
        main_metrics_path,
        main_kernel_path,
    )

    gathered["main"] = _CandRecord("main", "main", main_dir, main_base)

    for cid, fut in futures:
        try:
            rec = fut.result()
            gathered[cid] = rec
        except Exception as e:
            wd = cand_root / cid
            wd.mkdir(parents=True, exist_ok=True)
            err = {
                "round": round_idx,
                "task_path": str(c.task_path.resolve()),
                "work_dir": str(wd.resolve()),
                "model_name": c.model_name,
                "candidate_id": cid,
                "runnable": False,
                "status": "spec_future_error",
                "runtime_error": str(e),
                "candidate_role": "spec",
            }
            mp = wd / "metrics.json"
            agent.write_metrics(mp, err)
            gathered[cid] = _CandRecord(cid, "spec", wd, err)

    # Pipeline validate + NCU — parallel breadth
    work_items: list[_CandRecord] = list(gathered.values())
    val_exec = ThreadPoolExecutor(
        max_workers=max(1, int(c.validation_parallelism), int(c.profile_parallelism))
    )

    validated: dict[str, _CandRecord] = {cr.candidate_id: cr for cr in gathered.values()}

    val_futs: list[tuple[str, Future[_CandRecord]]] = []

    def _pipeline_one(cr: _CandRecord) -> _CandRecord:
        kp = cr.work_dir / "kernel.py"
        gen_mod = f"kernelbench_generated_r{round_idx}_{re.sub(r'[^a-zA-Z0-9_]', '_', cr.candidate_id)}"
        extra = {
            "candidate_id": cr.candidate_id,
            "candidate_role": cr.role,
        }
        m = agent._validate_and_ncu_candidate(
            round_idx=round_idx,
            kernel_path=kp,
            cand_work_dir=cr.work_dir,
            metrics_path=cr.work_dir / "metrics.json",
            gen_mod_name=gen_mod,
            seed_eval_timing=dict(cr.metrics.get("eval_timing") or {}),
            extra_top=extra,
        )
        return _CandRecord(cr.candidate_id, cr.role, cr.work_dir, m)

    for cr in work_items:
        if cr.role == "main" or cr.metrics.get("kernel_extracted"):
            val_futs.append((cr.candidate_id, val_exec.submit(_pipeline_one, cr)))

    for cid, vf in val_futs:
        try:
            validated[cid] = vf.result()
        except Exception as e:
            cr = gathered[cid]
            m = dict(cr.metrics)
            m.setdefault("candidate_id", cid)
            m["pipeline_error"] = str(e)
            agent.write_metrics(cr.work_dir / "metrics.json", m)
            validated[cid] = _CandRecord(cid, cr.role, cr.work_dir, m)

    val_exec.shutdown(wait=True)

    winner = pick_winner(candidate_metrics={k: v.metrics for k, v in validated.items()})
    final_metrics = finalize_round_layout(
        agent=agent,
        rd=rd,
        cand_root=cand_root,
        validated=validated,
        winner_id=winner,
        metrics_path=metrics_path,
        round_idx=round_idx,
    )
    agent._update_best_from_metrics(round_idx, final_metrics)
    return final_metrics


def pick_winner(*, candidate_metrics: dict[str, dict[str, Any]]) -> str:
    best_id: Optional[str] = None
    best_sp = float("-inf")
    for cid, m in candidate_metrics.items():
        sp = _successful_speedup(m)
        if sp is not None and sp > best_sp:
            best_sp = sp
            best_id = cid
    if best_id is not None:
        return best_id
    if "main" in candidate_metrics:
        return "main"
    if candidate_metrics:
        return next(iter(candidate_metrics.keys()))
    return "main"


def finalize_round_layout(
    *,
    agent: KernelBenchAgent,
    rd: Path,
    cand_root: Path,
    validated: dict[str, _CandRecord],
    winner_id: str,
    metrics_path: Path,
    round_idx: int,
) -> dict[str, Any]:
    c = agent.config
    win = validated.get(winner_id) or validated.get("main")
    if win is None:
        win = next(iter(validated.values()))

    candidate_summary = []
    for cid, cr in sorted(validated.items()):
        candidate_summary.append(
            {
                "candidate_id": cid,
                "role": cr.role,
                "work_dir": str(cr.work_dir.resolve()),
                "status": cr.metrics.get("status"),
                "speedup": speedup_from_metrics(cr.metrics) if cr.metrics.get("runnable") else None,
            }
        )

    base_out = dict(win.metrics)
    base_out["round"] = round_idx
    base_out["task_path"] = str(c.task_path.resolve())
    base_out["work_dir"] = str(rd.resolve())
    base_out["candidate_summary"] = candidate_summary
    base_out["winner_candidate_id"] = win.candidate_id

    wk = win.work_dir / "kernel.py"
    if wk.is_file():
        shutil.copy2(wk, rd / "kernel.py")
    wlo = win.work_dir / "llm_output.txt"
    if wlo.is_file():
        if win.candidate_id != "main":
            if (rd / "llm_output.txt").is_file():
                shutil.copy2(rd / "llm_output.txt", rd / "llm_output_main.txt")
        shutil.copy2(wlo, rd / "llm_output.txt")

    agent.write_metrics(metrics_path, base_out)
    return base_out


def _finalize_main_generation(
    agent: KernelBenchAgent,
    rd: Path,
    main_dir: Path,
    round_idx: int,
    llm_out_path: Path,
    llm_res: dict[str, Any],
    llm_eval_timing: dict[str, Any],
    main_metrics_path: Path,
    main_kernel_path: Path,
) -> dict[str, Any]:
    from agent import extract_python_module

    c = agent.config
    base: dict[str, Any] = {
        "round": round_idx,
        "task_path": str(c.task_path.resolve()),
        "work_dir": str(main_dir.resolve()),
        "model_name": c.model_name,
        "candidate_id": "main",
        "candidate_role": "main",
        "eval_timing": {"llm": llm_eval_timing},
    }
    if not llm_res.get("ok"):
        base.update(
            {
                "runnable": False,
                "status": "llm_subprocess_error",
                "llm": llm_res,
            }
        )
        agent.write_metrics(main_metrics_path, base)
        return base

    llm_raw = str(llm_res.get("text", ""))
    if not bool(llm_res.get("llm_output_dumped", False)):
        llm_out_path.write_text(llm_raw, encoding="utf-8")
    shutil.copy2(llm_out_path, main_dir / "llm_output.txt")

    py_src = extract_python_module(llm_raw)
    if py_src is None:
        base.update(
            {
                "runnable": False,
                "status": "parse_error",
                "parse_error": "No ```python ... ``` block found in LLM output.",
            }
        )
        agent.write_metrics(main_metrics_path, base)
        return base

    py_src = _uniquify_load_inline_name(py_src, f"r{round_idx:03d}_main")
    main_kernel_path.write_text(py_src, encoding="utf-8")
    base["kernel_extracted"] = True
    agent.write_metrics(main_metrics_path, base)
    return base


def _run_one_spec_generation(
    agent: KernelBenchAgent,
    round_idx: int,
    work_dir: Path,
    prompt_file_text: str,
    snapshot_text: str,
    head_n: int,
    slot_ix: int,
    system: str,
    user: str,
    tokenizer: str | None,
    candidate_id: str,
    suffix: str,
) -> _CandRecord:
    from agent import extract_python_module

    c = agent.config
    llm_prefix = first_n_tokens_text(snapshot_text, head_n, tokenizer)
    merged = (
        f"{prompt_file_text.rstrip()}\n\n{SPEC_PROMPT_THINKING_PREFIX}\n\n"
        f"{llm_prefix}{SPEC_PROMPT_THINKING_SUFFIX}"
    )
    spec_prompt_path = work_dir / "spec_prompt.txt"
    spec_prompt_path.write_text(merged, encoding="utf-8")
    system_s, user_s = _split_system_user(merged)

    ll_t0 = time.perf_counter()
    llm_out = work_dir / "llm_output.txt"
    llm_res = agent.call_llm(system_s, user_s, round_idx, llm_out)
    ll_t1 = time.perf_counter()
    llm_eval_timing = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "seconds": round(ll_t1 - ll_t0, 6),
    }

    base: dict[str, Any] = {
        "round": round_idx,
        "task_path": str(c.task_path.resolve()),
        "work_dir": str(work_dir.resolve()),
        "model_name": c.model_name,
        "candidate_id": candidate_id,
        "candidate_role": "spec",
        "spec_head_tokens": head_n,
        "spec_slot": slot_ix,
        "spec_prompt_path": str(spec_prompt_path),
        "eval_timing": {"llm": llm_eval_timing},
    }

    metrics_path = work_dir / "metrics.json"
    if not llm_res.get("ok"):
        base.update({"runnable": False, "status": "llm_subprocess_error", "llm": llm_res})
        agent.write_metrics(metrics_path, base)
        return _CandRecord(candidate_id, "spec", work_dir, base)

    raw = str(llm_res.get("text", ""))
    if not bool(llm_res.get("llm_output_dumped", False)):
        llm_out.write_text(raw, encoding="utf-8")

    py_src = extract_python_module(raw)
    if py_src is None:
        base.update(
            {
                "runnable": False,
                "status": "parse_error",
                "parse_error": "No ```python ... ``` block found in LLM output.",
            }
        )
        agent.write_metrics(metrics_path, base)
        return _CandRecord(candidate_id, "spec", work_dir, base)

    py_src = _uniquify_load_inline_name(py_src, suffix)
    (work_dir / "kernel.py").write_text(py_src, encoding="utf-8")
    base["kernel_extracted"] = True
    agent.write_metrics(metrics_path, base)
    return _CandRecord(candidate_id, "spec", work_dir, base)
