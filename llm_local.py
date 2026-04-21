"""
* Connects to local server (multi-GPU backend)
* Replaces HF Transformers inference
* Compatible with query_server interface
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from openai import OpenAI
import os
import sys

# -------------------------------
def llm_streaming_enabled() -> bool:
    """Set KERNEL_LLM_STREAM=0 to disable streaming (buffered completion only)."""
    v = os.environ.get("KERNEL_LLM_STREAM", "1").strip().lower()
    return v not in ("0", "false", "no")


def consume_chat_completion_stream(
    stream: Any,
    dump_path: Optional[str] = None,
) -> tuple[str, Optional[str], bool]:
    """Iterate OpenAI chat completion stream; print tokens to stdout; optional incremental UTF-8 file write.

    Returns (full_text, finish_reason, file_incrementally_written).
    When dump_path is set, opens it once with mode "w" (truncate), then appends each chunk with flush.
    Progress: stderr only, one line refreshed with ``\\r`` + clear-to-EOL (no stdout tokens), so the count is the only thing that moves.
    """
    parts: list[str] = []
    finish_reason: Optional[str] = None
    n_chars = 0
    f = None
    file_dumped = False
    if dump_path:
        f = open(dump_path, "w", encoding="utf-8")
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
                        print(
                            f"\r\033[2K[llm_output.txt] written: {n_chars} chars",
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


# -------------------------------
class LLM:
    """OpenAI-compatible client."""

    def __init__(self, model: str, server_url: str = "http://localhost:8000/v1"):
        self.model = model
        timeout_s = float(os.environ.get("SERVER_TIMEOUT_S", "3600"))
        max_retries = int(os.environ.get("SERVER_MAX_RETRIES", "2"))
        self.client = OpenAI(
            base_url=server_url,
            api_key="EMPTY",
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
                print(
                    f"\r[llm_output.txt] written: {len(text)} chars",
                    file=sys.stderr,
                    flush=True,
                )
                print(file=sys.stderr)
            return text, bool(dump)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )
        if cfg.enable_thinking:
            think_obj = {"type": "enabled"}
            if int(cfg.thinking_budget_tokens or 0) > 0:
                think_obj["budget_tokens"] = int(cfg.thinking_budget_tokens)
            kwargs["extra_body"] = {"thinking": think_obj}

        use_stream = cfg.stream and llm_streaming_enabled()
        if use_stream:
            kwargs["stream"] = True
            stream = self.client.chat.completions.create(**kwargs)
            text, _, dumped = consume_chat_completion_stream(stream, dump)
            return text, dumped

        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        if dump:
            Path(dump).write_text(text, encoding="utf-8")
            print(
                f"\r[llm_output.txt] written: {len(text)} chars",
                file=sys.stderr,
                flush=True,
            )
            print(file=sys.stderr)
            return text, True
        return text, False

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


# -------------------------------
from functools import lru_cache


@lru_cache(maxsize=2)
def get_llm(model_id: str, server_url: str = "http://localhost:8000/v1") -> LLM:
    return LLM(model=model_id, server_url=server_url)
