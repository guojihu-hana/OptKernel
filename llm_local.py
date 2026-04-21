"""
* Connects to local server (multi-GPU backend)
* Replaces HF Transformers inference
* Compatible with query_server interface
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
import os
import sys


def llm_streaming_enabled() -> bool:
    """Set KERNEL_LLM_STREAM=0 to disable streaming (buffered completion only)."""
    v = os.environ.get("KERNEL_LLM_STREAM", "1").strip().lower()
    return v not in ("0", "false", "no")


def max_token_continue_enabled() -> bool:
    """Set KERNEL_LLM_MAX_TOKEN_CONTINUE=0 to disable max_tokens truncation auto-continue."""
    v = os.environ.get("KERNEL_LLM_MAX_TOKEN_CONTINUE", "1").strip().lower()
    return v not in ("0", "false", "no")


def max_continuation_rounds() -> int:
    """Max extra completions after a length-truncated response (env KERNEL_LLM_MAX_CONTINUATIONS, default 64)."""
    try:
        return max(0, int(os.environ.get("KERNEL_LLM_MAX_CONTINUATIONS", "64")))
    except ValueError:
        return 64


def _dump_progress_tag(round_idx: Optional[int], continuation_k: int) -> str:
    """Suffix for [llm_output.txt] stderr progress: agent round and which continuation request."""
    r = "?" if round_idx is None else str(round_idx)
    if continuation_k <= 0:
        cont = "init"
    else:
        cont = f"ctn#{continuation_k}"
    return f"round={r} {cont}"


def is_max_tokens_truncation(reason: Optional[str]) -> bool:
    if reason is None:
        return False
    r = str(reason).strip()
    if not r:
        return False
    ru = r.upper()
    rl = r.lower()
    return ru == "MAX_TOKENS" or rl in ("length", "max_tokens")


def consume_chat_completion_stream(
    stream: Any,
    dump_path: Optional[str] = None,
    *,
    append_dump: bool = False,
    round_idx: Optional[int] = None,
    continuation_k: int = 0,
) -> tuple[str, Optional[str], bool]:
    """Iterate OpenAI chat completion stream; print tokens to stdout; optional incremental UTF-8 file write.

    Returns (full_text, finish_reason, file_incrementally_written).
    When dump_path is set: truncate write (``w``) unless append_dump is True, then append (``a``).
    Progress: stderr only, one line refreshed with ``\\r`` + clear-to-EOL (no stdout tokens), so the count is the only thing that moves.
    """
    parts: list[str] = []
    finish_reason: Optional[str] = None
    n_chars = 0
    f = None
    file_dumped = False
    if dump_path:
        if append_dump and Path(dump_path).exists():
            try:
                n_chars = len(Path(dump_path).read_text(encoding="utf-8"))
            except OSError:
                n_chars = 0
        mode = "a" if append_dump else "w"
        f = open(dump_path, mode, encoding="utf-8")
        file_dumped = True
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            ch0 = chunk.choices[0]
            delta = getattr(ch0, "delta", None)
            if delta is not None:
                piece = getattr(delta, "content", None) or ""
                if piece:
                    parts.append(piece)
                    if f is not None:
                        f.write(piece)
                        f.flush()
                        n_chars += len(piece)
                        # stderr only: stdout token stream interleaves with \\r and breaks single-line progress
                        tag = _dump_progress_tag(round_idx, continuation_k)
                        print(
                            f"\r\033[2K[llm_output.txt] {tag} written: {n_chars} chars",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        print(piece, end="", flush=True)
            fr = getattr(ch0, "finish_reason", None)
            if fr:
                finish_reason = str(fr)
        if f is None:
            print(flush=True)
        else:
            print(file=sys.stderr)
    finally:
        if f is not None:
            f.close()
    return "".join(parts), finish_reason, file_dumped


def openai_chat_completion_with_truncation_retry(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    original_user: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int],
    extra_body: Optional[dict[str, Any]],
    use_stream: bool,
    dump_path: Optional[str],
    max_continuations: int,
    round_idx: Optional[int] = None,
) -> tuple[str, Optional[str], bool]:
    """
    Repeat chat.completions until finish_reason is not max_tokens truncation or cap hit.

    Each continuation uses user message ``original_user + accumulated_text`` (single user turn).
    Returns (full_text, last_finish_reason, any_incremental_dump_written).
    """
    eff_max_cont = max(0, max_continuations)
    accumulated = ""
    last_fr: Optional[str] = None
    file_dumped = False

    for k in range(eff_max_cont + 1):
        user_content = original_user + accumulated
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        if use_stream:
            kwargs["stream"] = True
            stream = client.chat.completions.create(**kwargs)
            segment, fr, dumped = consume_chat_completion_stream(
                stream,
                dump_path,
                append_dump=(k > 0 and bool(dump_path)),
                round_idx=round_idx,
                continuation_k=k,
            )
            file_dumped = file_dumped or dumped
        else:
            response = client.chat.completions.create(**kwargs)
            segment = ""
            fr = None
            if response.choices:
                c0 = response.choices[0]
                segment = str(getattr(c0.message, "content", "") or "")
                fr_raw = getattr(c0, "finish_reason", None)
                fr = str(fr_raw) if fr_raw is not None else None
            if dump_path:
                Path(dump_path).write_text(accumulated + segment, encoding="utf-8")
                tag = _dump_progress_tag(round_idx, k)
                print(
                    f"\r[llm_output.txt] {tag} written: {len(accumulated) + len(segment)} chars",
                    file=sys.stderr,
                    flush=True,
                )
                print(file=sys.stderr)
                file_dumped = True

        accumulated += segment
        last_fr = fr

        if not is_max_tokens_truncation(fr):
            break
        if k >= eff_max_cont:
            print(
                "[llm_local] max_tokens truncation: reached KERNEL_LLM_MAX_CONTINUATIONS "
                f"({eff_max_cont}); response may still be incomplete.",
                file=sys.stderr,
            )
            break

    return accumulated, last_fr, file_dumped


# -------------------------------
@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 40
    repetition_penalty: float = 1.05
    seed: Optional[int] = None
    stream: bool = True
    enable_thinking: bool = True
    thinking_budget_tokens: int = 0
    stream_dump_path: Optional[str] = None
    # If set, overrides env KERNEL_LLM_MAX_CONTINUATIONS for this call.
    max_continuations: Optional[int] = None
    # Agent round index for [llm_output.txt] stderr progress only (optional).
    round_idx: Optional[int] = None


# -------------------------------
class LLM:
    """OpenAI-compatible client."""

    def __init__(
        self,
        model: str,
        server_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ):
        self.model = model
        timeout_s = float(os.environ.get("SERVER_TIMEOUT_S", "3600"))
        max_retries = int(os.environ.get("SERVER_MAX_RETRIES", "2"))
        self.client = OpenAI(
            base_url=server_url,
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    def chat(self, system: str, user: str, cfg: GenerationConfig | None = None) -> tuple[str, bool]:
        """Returns (full_text, llm_output_written_to_disk). Second True if stream_dump_path was used."""
        cfg = cfg or GenerationConfig()
        dump = (cfg.stream_dump_path or "").strip() or None

        # For non-chat models like MPT, fallback to generate()
        if "mpt" in self.model.lower() or "deepseek-coder" in self.model.lower():
            prompt = f"{system.strip()}\n{user.strip()}"
            text = self.generate(prompt, cfg)
            if dump:
                Path(dump).write_text(text, encoding="utf-8")
                tag = _dump_progress_tag(cfg.round_idx, 0)
                print(
                    f"\r[llm_output.txt] {tag} written: {len(text)} chars",
                    file=sys.stderr,
                    flush=True,
                )
                print(file=sys.stderr)
            return text, bool(dump)

        extra_body: Optional[dict[str, Any]] = None
        if cfg.enable_thinking:
            think_obj: dict[str, Any] = {"type": "enabled"}
            if int(cfg.thinking_budget_tokens or 0) > 0:
                think_obj["budget_tokens"] = int(cfg.thinking_budget_tokens)
            extra_body = {"thinking": think_obj}

        use_stream = cfg.stream and llm_streaming_enabled()
        max_cont = 0 if not max_token_continue_enabled() else (
            cfg.max_continuations if cfg.max_continuations is not None else max_continuation_rounds()
        )

        text, _fr, dumped = openai_chat_completion_with_truncation_retry(
            self.client,
            model=self.model,
            system_prompt=system,
            original_user=user,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
            extra_body=extra_body,
            use_stream=use_stream,
            dump_path=dump,
            max_continuations=max_cont,
            round_idx=cfg.round_idx,
        )
        return text, dumped

    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig()
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )
        return response.choices[0].text


@lru_cache(maxsize=32)
def get_llm(
    model_id: str,
    server_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
) -> LLM:
    return LLM(model=model_id, server_url=server_url, api_key=api_key)
