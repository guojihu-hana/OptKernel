"""
Pluggable interruption / control signals observed alongside spec triggers.

``--enable-interruption`` gates whether policies run; defaults keep behavior unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable


class InterruptionSignal(str, Enum):
    NONE = "none"
    NOTIFY = "notify"
    # Reserved for future: REQUEST_STOP_MAIN, REQUEST_BACKOFF, ...
    REQUEST_STOP_MAIN = "request_stop_main"


@dataclass(frozen=True)
class InterruptionContext:
    round_idx: int
    trigger_kind: str
    detail: str
    approx_output_tokens: int
    output_path: str = ""


@runtime_checkable
class InterruptionPolicy(Protocol):
    def on_context(self, ctx: InterruptionContext) -> InterruptionSignal: ...


class NoOpInterruptionPolicy:
    def on_context(self, ctx: InterruptionContext) -> InterruptionSignal:
        return InterruptionSignal.NONE


class LoggingInterruptionPolicy:
    """v1: record NOTIFY-level signals without changing main generation."""

    def __init__(self, log: Optional[logging.Logger] = None) -> None:
        self._log = log or logging.getLogger("optkernel.interruption")

    def on_context(self, ctx: InterruptionContext) -> InterruptionSignal:
        self._log.info(
            "interruption_notify round=%s kind=%s tokens≈%s detail=%s path=%s",
            ctx.round_idx,
            ctx.trigger_kind,
            ctx.approx_output_tokens,
            ctx.detail,
            ctx.output_path or "-",
        )
        return InterruptionSignal.NOTIFY


class InterruptionCoordinator:
    def __init__(self, enabled: bool, policy: Optional[InterruptionPolicy] = None) -> None:
        self._enabled = bool(enabled)
        self._policy: InterruptionPolicy = policy or NoOpInterruptionPolicy()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def emit(self, ctx: InterruptionContext) -> InterruptionSignal:
        if not self._enabled:
            return InterruptionSignal.NONE
        return self._policy.on_context(ctx)
