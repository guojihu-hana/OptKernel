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


def llm_stream_include_reasoning() -> bool:
    """
    Whether to merge streaming reasoning tokens into the main llm_output text stream.

    Default is OFF for safety: keep ``llm_output.txt`` identical to assistant visible content.
    Enable explicitly with ``KERNEL_LLM_STREAM_INCLUDE_REASONING=1``.
    """
    v = os.environ.get("KERNEL_LLM_STREAM_INCLUDE_REASONING", "1").strip().lower()
    return v in ("1", "true", "yes", "on")


def llm_safe_output_enabled() -> bool:
    """
    Safer llm_output mode: avoid chunk-wise assembly and write finalized text only.

    Set ``KERNEL_LLM_SAFE_OUTPUT=1`` to force non-streaming persistence even when
    streaming is enabled for normal operation.
    """
    v = os.environ.get("KERNEL_LLM_SAFE_OUTPUT", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


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


# Appended to the user message on max_tokens continuation requests (k > 0) only.
_CONTINUATION_DIRECT_CODE_SUFFIX = "\n\nNO THINKING, DIRECTLY OUTPUT THE CODE"

# ~bytes of chat template overhead (roles, padding); conservative fudge on top of body tokens.
_CHAT_FORMAT_OVERHEAD_TOKENS = 8


def estimate_chat_prompt_tokens(system_prompt: str, user_text: str) -> int:
    """
    Best-effort prompt token count for a single system + one user turn (vLLM/OpenAI chat).
    Uses ``tiktoken`` (cl100k_base) if installed, else **chars / 2.5** (mixed text heuristic).
    """
    sys_t = system_prompt or ""
    usr_t = user_text or ""
    try:
        import tiktoken  # type: ignore[import-not-found]

        enc = tiktoken.get_encoding("cl100k_base")
        return int(len(enc.encode(sys_t)) + len(enc.encode(usr_t)) + _CHAT_FORMAT_OVERHEAD_TOKENS)
    except Exception:
        n = len(sys_t) + len(usr_t)
        return max(1, (n * 2 + 4) // 5 + _CHAT_FORMAT_OVERHEAD_TOKENS)


def _completion_max_tokens_capped(
    request_max: int, system_prompt: str, user_content: str, max_context_length: int
) -> int:
    """``min(request_max, max(1, max_context - prompt_tokens))`` when max_context is set."""
    if max_context_length <= 0:
        return request_max
    pt = estimate_chat_prompt_tokens(system_prompt, user_content)
    room = max_context_length - pt
    if room < 1:
        print(
            f"[llm_local] max_context_length={max_context_length} with ~{pt} est. prompt tokens "
            f"leaves {room} completion budget — capping to 1; raise --max-context-length or shorten prompt.",
            file=sys.stderr,
        )
    cap = min(request_max, max(1, room))
    return cap


def coerce_reasoning_to_str(val: Any) -> str:
    """Normalize reasoning / thinking payloads from OpenAI-compatible APIs (str, dict, list)."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        for k in ("text", "reasoning", "content", "value", "thinking", "thought"):
            t = val.get(k)
            if isinstance(t, str) and t:
                return t
            nested = coerce_reasoning_to_str(t)
            if nested:
                return nested
        return ""
    if isinstance(val, list):
        return "".join(coerce_reasoning_to_str(x) for x in val)
    return str(val)


def extract_reasoning_from_assistant_message(msg: Any) -> str:
    """
    Best-effort chain-of-thought from a chat completion *message* or streaming *delta*.
    vLLM reasoning parsers often set ``delta.reasoning`` (see vLLM streaming example); some
    gateways use ``reasoning_content``, nested dicts, or pydantic extras stripped from typed attrs.
    """
    if msg is None:
        return ""
    parts: list[str] = []

    def add_raw(val: Any) -> None:
        s = coerce_reasoning_to_str(val).strip()
        if s:
            parts.append(s)

    for attr in ("reasoning_content", "reasoning", "thinking", "thought"):
        add_raw(getattr(msg, attr, None))
    md = getattr(msg, "model_dump", None)
    if callable(md):
        try:
            d = md()
            if isinstance(d, dict):
                for k in (
                    "reasoning_content",
                    "reasoning",
                    "thinking",
                    "thought",
                    "reasoning_details",
                ):
                    add_raw(d.get(k))
        except Exception:
            pass
    extra = getattr(msg, "__pydantic_extra__", None)
    if isinstance(extra, dict):
        for k in ("reasoning_content", "reasoning", "thinking", "thought"):
            add_raw(extra.get(k))
    return "\n\n".join(parts) if parts else ""


def iter_stream_delta_text_pieces(delta: Any) -> list[tuple[str, str]]:
    """
    Incremental text pieces for one SSE chunk, in server order.
    Returns ``[(kind, text)]`` where ``kind`` is ``"reasoning"`` or ``"content"``.
    """
    if delta is None:
        return []
    content = getattr(delta, "content", None)
    if isinstance(content, list):
        out: list[tuple[str, str]] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str) and txt:
                    typ = str(item.get("type", "")).strip().lower()
                    kind = "reasoning" if ("reason" in typ or "think" in typ) else "content"
                    out.append((kind, txt))
                    continue
                for k in ("reasoning", "thinking", "thought", "reasoning_content"):
                    t2 = item.get(k)
                    if isinstance(t2, str) and t2:
                        out.append(("reasoning", t2))
            elif isinstance(item, str) and item:
                out.append(("content", item))
        if out:
            return out
    out2: list[tuple[str, str]] = []
    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
    if isinstance(reasoning, str) and reasoning:
        out2.append(("reasoning", reasoning))
    if isinstance(content, str) and content:
        out2.append(("content", content))
    return out2


def _stream_piece_to_emit(kind: str, prev: str, piece: str) -> str:
    """
    Merge stream fragments without corrupting Python/code indentation.

    Many gateways send **cumulative prefixes** for ``reasoning`` / ``reasoning_content`` (repeat
    growing text). OpenAI-style ``content`` is usually **token deltas**; applying the same
    prefix-dedup to ``content`` can falsely treat indent spaces as a shared prefix and strip
    characters from the next line (e.g. ``def forward`` losing one leading space).
    """
    if not piece:
        return ""
    if not prev:
        return piece
    if piece == prev:
        return ""
    if kind == "reasoning":
        if piece.startswith(prev) and len(piece) > len(prev):
            return piece[len(prev) :]
        return piece
    # content: append deltas as-is (only drop exact duplicate chunks)
    return piece


def assistant_output_str_from_message(msg: Any) -> str:
    """Full assistant text for logs: optional ``<thinking>`` block + string ``content``."""
    raw = getattr(msg, "content", None)
    if not isinstance(raw, str):
        return str(raw or "") if raw is not None else ""
    reasoning = extract_reasoning_from_assistant_message(msg)
    if not reasoning.strip():
        return raw
    return f"<thinking>\n{reasoning.strip()}\n</thinking>\n\n{raw}"


def _atomic_write_text(path: str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def consume_chat_completion_stream(
    stream: Any,
    dump_path: Optional[str] = None,
    *,
    append_dump: bool = False,
    round_idx: Optional[int] = None,
    continuation_k: int = 0,
) -> tuple[str, Optional[str], bool]:
    """Iterate OpenAI chat completion stream; print tokens to stdout; optional incremental UTF-8 file write.

    Streams assistant ``content`` tokens to output file.
    ``reasoning`` tokens are ignored by default for safety; opt-in via
    ``KERNEL_LLM_STREAM_INCLUDE_REASONING=1``.

    Returns (full_text, finish_reason, file_incrementally_written).
    When dump_path is set: truncate write (``w``) unless append_dump is True, then append (``a``).
    Progress: stderr only, one line refreshed with ``\\r`` + clear-to-EOL (no stdout tokens), so the count is the only thing that moves.
    """
    parts: list[str] = []
    finish_reason: Optional[str] = None
    n_chars = 0
    f = None
    file_dumped = False
    seen_by_kind: dict[str, str] = {"reasoning": "", "content": ""}
    include_reasoning = llm_stream_include_reasoning()
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
                for kind, piece in iter_stream_delta_text_pieces(delta):
                    if not piece:
                        continue
                    if kind == "reasoning" and not include_reasoning:
                        # Keep main llm_output stable (content-only) unless explicitly requested.
                        seen_by_kind[kind] = piece
                        continue
                    add = _stream_piece_to_emit(kind, seen_by_kind.get(kind, ""), piece)
                    if not add:
                        seen_by_kind[kind] = piece
                        continue
                    seen_by_kind[kind] = piece
                    parts.append(add)
                    if f is not None:
                        f.write(add)
                        f.flush()
                        n_chars += len(add)
                        # stderr only: stdout token stream interleaves with \\r and breaks single-line progress
                        tag = _dump_progress_tag(round_idx, continuation_k)
                        print(
                            f"\r\033[2K[llm_output.txt] {tag} written: {n_chars} chars",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                    else:
                        print(add, end="", flush=True)
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
    repetition_penalty: Optional[float] = None,
    max_context_length: int = 0,
) -> tuple[str, Optional[str], bool]:
    """
    Repeat chat.completions until finish_reason is not max_tokens truncation or cap hit.

    The **first** request (``k == 0``) always uses the full ``max_tokens`` (no
    :func:`_completion_max_tokens_capped`).

    **Continuation** requests (``k > 0``), when ``max_context_length > 0``, cap ``max_tokens``
    to the remaining room vs. estimated prompt size so the growing user message does not overflow
    the context window.

    Each continuation uses user message ``original_user + accumulated_text`` (single user turn).
    Returns (full_text, last_finish_reason, any_incremental_dump_written).
    """
    eff_max_cont = max(0, max_continuations)
    accumulated = ""
    last_fr: Optional[str] = None
    file_dumped = False

    safe_output = bool(dump_path) and llm_safe_output_enabled()
    effective_stream = bool(use_stream) and (not safe_output)
    if safe_output:
        print(
            "[llm_local] KERNEL_LLM_SAFE_OUTPUT=1: disable chunk-wise streaming assembly "
            "for llm_output; writing finalized text per request.",
            file=sys.stderr,
            flush=True,
        )

    for k in range(eff_max_cont + 1):
        user_content = original_user + accumulated
        if k > 0:
            user_content = user_content + _CONTINUATION_DIRECT_CODE_SUFFIX
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        if k == 0:
            eff_max = int(max_tokens)
        else:
            eff_max = _completion_max_tokens_capped(
                max_tokens, system_prompt, user_content, max_context_length
            )
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=eff_max,
        )
        if seed is not None:
            kwargs["seed"] = seed
        eb: Optional[dict[str, Any]] = extra_body
        if repetition_penalty is not None:
            eb = {**(eb or {}), "repetition_penalty": repetition_penalty}
        if eb is not None:
            kwargs["extra_body"] = eb

        if effective_stream:
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
                segment = assistant_output_str_from_message(c0.message)
                fr_raw = getattr(c0, "finish_reason", None)
                fr = str(fr_raw) if fr_raw is not None else None
            if dump_path:
                _atomic_write_text(dump_path, accumulated + segment)
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
    # 0 = disabled; else on **continuation** chat requests only, cap max_tokens to context room.
    max_context_length: int = 0


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
        else:
            # For GLM-5/5.1 on vLLM, thinking is enabled by default; explicitly disable it.
            # https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html#openai-client-example
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

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
            repetition_penalty=cfg.repetition_penalty,
            max_context_length=cfg.max_context_length,
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
