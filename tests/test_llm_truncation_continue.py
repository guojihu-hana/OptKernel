"""Tests for max_tokens truncation auto-continue in llm_local."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from llm_local import (
    _CONTINUATION_DIRECT_CODE_SUFFIX,
    consume_chat_completion_stream,
    is_max_tokens_truncation,
    openai_chat_completion_with_truncation_retry,
)


@pytest.mark.parametrize(
    "reason,expected",
    [
        (None, False),
        ("", False),
        ("stop", False),
        ("length", True),
        ("max_tokens", True),
        ("MAX_TOKENS", True),
        ("Length", True),
    ],
)
def test_is_max_tokens_truncation(reason: Optional[str], expected: bool) -> None:
    assert is_max_tokens_truncation(reason) is expected


class _Delta:
    __slots__ = ("content",)

    def __init__(self, content: Optional[str] = None) -> None:
        self.content = content


class _Choice:
    __slots__ = ("delta", "finish_reason")

    def __init__(
        self,
        *,
        delta: Optional[_Delta] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        self.delta = delta
        self.finish_reason = finish_reason


class _Chunk:
    __slots__ = ("choices",)

    def __init__(self, choices: list[_Choice]) -> None:
        self.choices = choices


class _FakeCompletions:
    def __init__(self, streams: list[Any]) -> None:
        self._streams = streams
        self.call_count = 0
        self.last_kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.last_kwargs.append(kwargs)
        i = self.call_count
        self.call_count += 1
        return self._streams[i]


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAIClient:
    """Minimal stand-in for OpenAI client.chat.completions.create."""

    def __init__(self, streams: list[Any]) -> None:
        self._completions = _FakeCompletions(streams)
        self.chat = _FakeChat(self._completions)


def _stream_parts(parts: list[tuple[Optional[str], Optional[str]]]) -> list[_Chunk]:
    """Each tuple: (delta_content, finish_reason on same chunk)."""
    out: list[_Chunk] = []
    for content, fr in parts:
        d = _Delta(content) if content is not None else _Delta(None)
        out.append(_Chunk([_Choice(delta=d, finish_reason=fr)]))
    return out


def test_consume_stream_append_dump(tmp_path: Path) -> None:
    p = str(tmp_path / "out.txt")
    s1 = _stream_parts([("aa", None), (None, "length")])
    s2 = _stream_parts([("bb", None), (None, "stop")])
    t1, fr1, d1 = consume_chat_completion_stream(iter(s1), p, append_dump=False)
    assert t1 == "aa"
    assert fr1 == "length"
    assert d1
    t2, fr2, d2 = consume_chat_completion_stream(iter(s2), p, append_dump=True)
    assert t2 == "bb"
    assert fr2 == "stop"
    assert d2
    assert Path(p).read_text(encoding="utf-8") == "aabb"


def test_openai_truncation_retry_two_stream_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_LLM_MAX_TOKEN_CONTINUE", "1")
    monkeypatch.setenv("KERNEL_LLM_MAX_CONTINUATIONS", "8")

    stream1 = _stream_parts([("part1", None), (None, "length")])
    stream2 = _stream_parts([("part2", None), (None, "stop")])
    client = _FakeOpenAIClient([iter(stream1), iter(stream2)])

    text, fr, dumped = openai_chat_completion_with_truncation_retry(
        client,  # type: ignore[arg-type]
        model="m",
        system_prompt="sys",
        original_user="U0",
        temperature=0.0,
        top_p=1.0,
        max_tokens=10,
        seed=None,
        extra_body=None,
        use_stream=True,
        dump_path=None,
        max_continuations=8,
    )
    assert text == "part1part2"
    assert fr == "stop"
    assert not dumped
    assert client._completions.call_count == 2
    assert client._completions.last_kwargs[0]["messages"][1]["content"] == "U0"
    assert client._completions.last_kwargs[1]["messages"][1]["content"] == (
        "U0part1" + _CONTINUATION_DIRECT_CODE_SUFFIX
    )


def test_openai_truncation_retry_respects_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_LLM_MAX_TOKEN_CONTINUE", "0")
    # Single truncated response: without continue, only one call; still return partial text.
    stream1 = _stream_parts([("x", None), (None, "length")])
    client = _FakeOpenAIClient([iter(stream1)])

    text, fr, _ = openai_chat_completion_with_truncation_retry(
        client,  # type: ignore[arg-type]
        model="m",
        system_prompt="s",
        original_user="U",
        temperature=0.0,
        top_p=1.0,
        max_tokens=5,
        seed=None,
        extra_body=None,
        use_stream=True,
        dump_path=None,
        max_continuations=0,
    )
    assert text == "x"
    assert fr == "length"
    assert client._completions.call_count == 1


def test_openai_truncation_retry_non_stream_two_rounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_LLM_MAX_TOKEN_CONTINUE", "1")

    class Msg:
        def __init__(self, content: str) -> None:
            self.content = content

    class Ch:
        def __init__(self, content: str, fr: Optional[str]) -> None:
            self.message = Msg(content)
            self.finish_reason = fr

    class Resp:
        def __init__(self, content: str, fr: Optional[str]) -> None:
            self.choices = [Ch(content, fr)]

    responses = [Resp("a", "length"), Resp("b", "stop")]

    class NC:
        def __init__(self) -> None:
            self.i = 0
            self.calls = 0

        def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            r = responses[self.i]
            self.i += 1
            return r

    nc = NC()

    class FakeCompletionsNS:
        def create(self, **kwargs: Any) -> Any:
            return nc.create(**kwargs)

    chat_ns = type("ChatNS", (), {"completions": FakeCompletionsNS()})()
    client_ns = type("ClientNS", (), {"chat": chat_ns})()

    text, fr, _ = openai_chat_completion_with_truncation_retry(
        client_ns,  # type: ignore[arg-type]
        model="m",
        system_prompt="s",
        original_user="U",
        temperature=0.0,
        top_p=1.0,
        max_tokens=5,
        seed=None,
        extra_body=None,
        use_stream=False,
        dump_path=None,
        max_continuations=4,
    )
    assert text == "ab"
    assert fr == "stop"
    assert nc.calls == 2
