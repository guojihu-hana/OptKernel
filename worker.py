"""
Validation + ncu profile microservice (FCFS job queue, **idle-GPU** assignment).

* Incoming jobs are placed on a **FIFO** queue. A small **pump** thread
  ``ThreadPoolExecutor.submit``'s to workers (``max_workers = len(gpus)``). Each
  run **first blocks on a free-GPU handle** (``queue.Queue`` of size ``G``) then
  runs the child subprocess. If all ``G`` are busy, further jobs **wait** in the
  executor queue; when a run finishes, it **returns the GPU to the free pool** and
  the next job acquires that GPU.

* ``queue_timing.wait_s`` = from enqueue to **acquisition of an idle GPU** (i.e. start
  of the subprocess is imminent). ``execute_s`` = that subprocess run only.

* ``GET /health`` includes ``gpus_occupation``: each GPU is ``null`` if idle, else
  ``{"task_id", "kind"}`` for the in-flight request.

* ``POST /validation`` / ``POST /ncu`` — see :func:`run_validation.run_forward_validation_subprocess`
  and :func:`run_ncu.run_ncu_profile_subprocess` (or ``"metrics"`` as comma list).

Env: ``WORKER_HOST`` (``0.0.0.0``), ``WORKER_PORT`` (``9876``), ``WORKER_GPUS`` (e.g. ``0,1``).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, cast

_lock = threading.Lock()
_next_id = 0


def _new_task_id() -> int:
    global _next_id
    with _lock:
        _next_id += 1
        return _next_id


class _Job:
    __slots__ = (
        "task_id",
        "kind",
        "t_enqueued",
        "payload",
        "device",
        "out",
        "err",
        "wait_s",
        "exec_s",
        "event",
    )

    def __init__(self, kind: str, payload: dict[str, Any]) -> None:
        self.task_id = _new_task_id()
        self.kind = kind
        self.t_enqueued = time.perf_counter()
        self.payload = payload
        self.device = ""
        self.out: Any = None
        self.err: Optional[BaseException] = None
        self.wait_s = 0.0
        self.exec_s = 0.0
        self.event = threading.Event()


def _pre_import() -> None:
    from run_validation import run_forward_validation_subprocess  # noqa: F401
    from run_ncu import run_ncu_profile_subprocess  # noqa: F401


def _run_job_body(j: _Job, device: str) -> None:
    from run_validation import run_forward_validation_subprocess
    from run_ncu import run_ncu_profile_subprocess, PROFILE_K, SKIP_K, effective_ncu_metrics

    if j.kind == "validation":
        p = j.payload
        j.out = run_forward_validation_subprocess(
            Path(p["task_file"]).resolve(),
            Path(p["kernel_file"]).resolve(),
            int(p.get("seed", 0)),
            float(p.get("atol", 1e-4)),
            float(p.get("rtol", 1e-4)),
            str(p.get("gen_module_name", "kernelbench_generated_uniq")),
            cuda_visible_device=device,
        )
    else:
        p = j.payload
        margs = p.get("metrics_args")
        if margs is not None and isinstance(margs, list):
            margs2 = [str(x) for x in margs]
        elif p.get("metrics") is not None:
            names = effective_ncu_metrics(
                [x.strip() for x in str(p["metrics"]).split(",") if x.strip()]
            )
            margs2 = ["--metrics", ",".join(names)]
        else:
            margs2 = None
        if not margs2 or len(margs2) < 2:
            raise ValueError("ncu: provide metrics_args or \"metrics\" (comma list)")
        extra = p.get("extra_args", [])
        if not isinstance(extra, list):
            extra = []
        j.out = run_ncu_profile_subprocess(
            Path(p["kernel_file"]).resolve(),
            Path(p["work_dir"]).resolve(),
            str(p.get("ncu_binary", "ncu")),
            margs2,
            [str(x) for x in extra],
            launch_skip=int(p.get("launch_skip", SKIP_K)),
            launch_count=int(p.get("launch_count", PROFILE_K)),
            cuda_visible_device=device,
        )


class _ServiceContext:
    """Holds work queue, free-GPU pool, thread pool, pump thread, GPU occupation for /health."""

    def __init__(self, glist: list[str]) -> None:
        self.glist: list[str] = list(glist)
        g = len(self.glist)
        self._free: "queue.Queue[str]" = cast(queue.Queue, queue.Queue(maxsize=0))
        for d in self.glist:
            self._free.put(d)
        self.occupation: dict[str, Optional[tuple[int, str]]] = {d: None for d in self.glist}
        self._occ_lock = threading.Lock()
        self._work: "queue.Queue[Optional[_Job]]" = cast(queue.Queue, queue.Queue())
        self._pool: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=g,
            thread_name_prefix="optkernel_gpu",
        )
        self._pump: Optional[threading.Thread] = None
        self._pump = threading.Thread(target=self._pump_fn, name="optkernel_pump", daemon=True)
        self._pump.start()

    def _pump_fn(self) -> None:
        """Move jobs from FIFO ``_work`` into the pool; a task **blocks in pool** or at ``_free.get()``."""
        w = self._work
        p = self._pool
        while True:
            j = w.get()
            if j is None:
                w.task_done()
                break
            try:
                p.submit(self._on_job, j)
            except (RuntimeError, OSError) as e:
                j.err = e
                j.event.set()
                w.task_done()

    def _on_job(self, j: _Job) -> None:
        t0: float = 0.0
        d: str = self._free.get()  # wait for an idle device before starting subprocess
        try:
            t_slot = time.perf_counter()
            j.wait_s = t_slot - j.t_enqueued
            j.device = d
            with self._occ_lock:
                self.occupation[d] = (j.task_id, j.kind)
            t0 = time.perf_counter()
            try:
                _run_job_body(j, d)
            except BaseException as e:  # noqa: BLE001
                j.err = e
        finally:
            t1 = time.perf_counter()
            if t0 > 0.0:
                j.exec_s = t1 - t0
            with self._occ_lock:
                self.occupation[d] = None
            self._free.put(d)
            j.event.set()
            self._work.task_done()

    def shutdown(self) -> None:
        self._work.put(None)
        if self._pump and self._pump.is_alive():
            self._pump.join(timeout=5.0)
        self._pool.shutdown(wait=True, cancel_futures=False)

    @property
    def work(self) -> "queue.Queue[Optional[_Job]]":
        return self._work


def make_request_handler(
    gpus: list[str],
) -> tuple[type, _ServiceContext]:
    glist: list[str] = list(gpus)
    if not glist:
        glist = ["0"]
    ctx = _ServiceContext(glist)
    wq = ctx.work

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        gpus: list[str] = glist
        svc: _ServiceContext = ctx
        work_q: "queue.Queue[Optional[_Job]]" = wq

        def log_message(self, f: str, *a: Any) -> None:  # noqa: A003
            sys.stderr.write(
                f"[worker] {self.log_date_time_string()} {self.client_address[0]!r} {f % a if a else f}\n"
            )
            sys.stderr.flush()

        def do_GET(self) -> None:  # noqa: N802
            pth = (self.path or "/").split("?")[0].rstrip("/")
            if pth in ("", "/health", "/v1/health"):
                o: dict[str, Any] = {}
                with H.svc._occ_lock:  # noqa: SLF001
                    for d in H.gpus:
                        v = H.svc.occupation.get(d)  # noqa: SLF001
                        o[d] = (
                            {"task_id": v[0], "kind": v[1]} if v is not None else None
                        )
                wq: Any = H.work_q
                incomplete = int(getattr(wq, "unfinished_tasks", 0) or 0)
                b = json.dumps(
                    {
                        "ok": True,
                        "service": "optkernel.validation_ncu",
                        "gpus": H.gpus,
                        "gpus_occupation": o,
                        "work_queue": {
                            "incomplete": incomplete,
                            "qsize": wq.qsize(),
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self._send(b, "application/json; charset=utf-8")
            else:
                self.send_error(404, "Not Found")

        def do_POST(self) -> None:  # noqa: N802
            pth = (self.path or "/").rstrip("/") or "/"
            if pth in ("/validation", "/v1/validation", "/v1/forward-validation"):
                k = "validation"
            elif pth in ("/ncu", "/v1/ncu", "/v1/ncu-profile"):
                k = "ncu"
            else:
                self.send_error(404, "Not Found")
                return
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                raw = self.rfile.read(n) if n else b"{}"
                body = json.loads(raw.decode("utf-8", errors="replace") or "{}")
            except (json.JSONDecodeError, UnicodeError) as e:
                self._send_err(400, f"Invalid JSON: {e!r}")
                return
            if not isinstance(body, dict):
                self._send_err(400, "JSON object required")
                return
            try:
                _check_payload(body, k)
            except ValueError as e:
                self._send_err(422, str(e))
                return
            job = _Job(k, body)
            H.work_q.put(job)
            to = 86400.0
            if not job.event.wait(timeout=to):
                self._send_err(
                    504,
                    "Timeout waiting for worker",
                    extra={"task_id": job.task_id},
                )
                return
            qt = {
                "task_id": job.task_id,
                "wait_s": round(job.wait_s, 6),
                "execute_s": round(job.exec_s, 6),
                "gpu": job.device,
            }
            if job.err is not None:
                b = {
                    "ok": False,
                    "error": str(job.err),
                    "error_type": type(job.err).__name__,
                    "traceback": "".join(
                        traceback.format_exception(
                            type(job.err), job.err, job.err.__traceback__
                        )
                    ),
                    "queue_timing": qt,
                }
                r = 500
                if isinstance(job.err, (ValueError, OSError, FileNotFoundError)):
                    r = 400
                self._send(
                    json.dumps(b, ensure_ascii=False, default=str).encode("utf-8"),
                    "application/json; charset=utf-8",
                    r,
                )
                return
            b = {
                "ok": True,
                "result": job.out,
                "queue_timing": qt,
            }
            self._send(
                json.dumps(b, ensure_ascii=False, default=str).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_err(
            self,
            code: int,
            message: str,
            extra: Optional[dict] = None,
        ) -> None:
            p: dict[str, Any] = {"ok": False, "error": message}
            if extra:
                p.update(extra)
            b = json.dumps(p, ensure_ascii=False).encode("utf-8")
            self._send(b, "application/json; charset=utf-8", code=code)

        def _send(self, data: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return H, ctx


def _check_payload(d: dict[str, Any], kind: str) -> None:
    if kind == "validation":
        for k in ("task_file", "kernel_file"):
            if k not in d:
                raise ValueError(f"validation: missing {k!r}")
        if not Path(d["task_file"]).is_file():
            raise ValueError(f"task_file not a file: {d['task_file']!r}")
        if not Path(d["kernel_file"]).is_file():
            raise ValueError(f"kernel_file not a file: {d['kernel_file']!r}")
    else:
        for k in ("kernel_file", "work_dir"):
            if k not in d:
                raise ValueError(f"ncu: missing {k!r}")
        if not Path(d["kernel_file"]).is_file():
            raise ValueError(f"kernel_file not a file: {d['kernel_file']!r}")
        w = Path(d["work_dir"])
        if not w.is_dir():
            raise ValueError(f"work_dir not a directory: {d['work_dir']!r}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="FCFS jobs + free-GPU pool; ThreadPool size = |gpus|.",
    )
    ap.add_argument(
        "-p",
        "--port",
        type=int,
        default=int(os.environ.get("WORKER_PORT", "9876")),
    )
    ap.add_argument(
        "-b",
        "--bind",
        default=os.environ.get("WORKER_HOST", "0.0.0.0"),
    )
    ap.add_argument(
        "--gpus",
        default=os.environ.get("WORKER_GPUS", "0"),
        help="Comma/space separated CUDA index list, e.g. 0,1 (pool size = number of devices)",
    )
    args = ap.parse_args(argv)
    gpus = [x for x in re.split(r"[\s,]+", (args.gpus or "").strip()) if x]
    if not gpus:
        gpus = ["0"]

    _pre_import()
    H, ctx = make_request_handler(gpus)
    server = ThreadingHTTPServer(
        (args.bind, int(args.port)),
        H,  # type: ignore[misc,arg-type]
    )
    print(
        f"http://{args.bind}:{int(args.port)}  free-GPU pool size={len(gpus)}  "
        f"POST /validation  POST /ncu  gpus={gpus!r}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("shutdown", file=sys.stderr)
    finally:
        ctx.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
