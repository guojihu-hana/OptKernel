"""
Token estimation helpers shared by speculative / agent.spec paths.

Copied from speculative_agent logic to avoid import cycles (speculative_agent imports agent).
"""

from __future__ import annotations


def encode_ids(text: str, tokenizer_name: str | None = None) -> list[int]:
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


def count_tokens(text: str, tokenizer_name: str | None = None) -> int:
    ids = encode_ids(text, tokenizer_name)
    if ids:
        return len(ids)
    return len(text.split())


def first_n_tokens_text(text: str, n: int, tokenizer_name: str | None = None) -> str:
    if n <= 0 or not text:
        return ""
    ids = encode_ids(text, tokenizer_name)
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


def iter_token_head_targets(step_n: int) -> range:
    """1-based multiples of ``step_n`` used as breakpoints (caller caps by nt)."""
    if step_n <= 0:
        return range(0, 0)
    return range(step_n, 10_000_000_000, step_n)
