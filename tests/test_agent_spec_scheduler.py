from __future__ import annotations

import unittest

from agent_spec_scheduler import ConstantSpecBatchSizer, _new_token_heads, pick_winner
from interruption import InterruptionContext, InterruptionCoordinator, InterruptionSignal


class RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[InterruptionContext] = []

    def on_context(self, ctx: InterruptionContext) -> InterruptionSignal:
        self.calls.append(ctx)
        return InterruptionSignal.NOTIFY


class TestConstantSpecBatchSizer(unittest.TestCase):
    def test_width(self) -> None:
        b = ConstantSpecBatchSizer(5)
        self.assertEqual(
            b.width(
                round_idx=0,
                trigger_id="t",
                worker_hint=None,
                queued_generations=0,
                queued_validations=0,
                spec_jobs_scheduled=0,
                spec_max_candidates=0,
            ),
            5,
        )


class TestPickWinner(unittest.TestCase):
    def test_highest_speedup(self) -> None:
        w = pick_winner(
            candidate_metrics={
                "main": {
                    "status": "success",
                    "runnable": True,
                    "benchmark_timing": {"speedup": 1.0, "skipped": False},
                },
                "spec_head2000_s0": {
                    "status": "success",
                    "runnable": True,
                    "benchmark_timing": {"speedup": 2.5, "skipped": False},
                },
            }
        )
        self.assertEqual(w, "spec_head2000_s0")

    def test_fallback_main(self) -> None:
        w = pick_winner(
            candidate_metrics={
                "main": {"status": "parse_error", "runnable": False},
                "spec_head2000_s0": {"status": "parse_error", "runnable": False},
            }
        )
        self.assertEqual(w, "main")


class TestInterruptionCoordinator(unittest.TestCase):
    def test_disabled_is_noop(self) -> None:
        rec = RecordingPolicy()
        c = InterruptionCoordinator(False, rec)
        sig = c.emit(
            InterruptionContext(
                round_idx=0,
                trigger_kind="x",
                detail="d",
                approx_output_tokens=10,
            )
        )
        self.assertEqual(sig, InterruptionSignal.NONE)
        self.assertEqual(rec.calls, [])

    def test_enabled_invokes_policy(self) -> None:
        rec = RecordingPolicy()
        c = InterruptionCoordinator(True, rec)
        ctx = InterruptionContext(
            round_idx=1,
            trigger_kind="spec_token_heads",
            detail="crossed_heads=2000",
            approx_output_tokens=2000,
            output_path="/tmp/x",
        )
        sig = c.emit(ctx)
        self.assertEqual(sig, InterruptionSignal.NOTIFY)
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.calls[0].detail, "crossed_heads=2000")


class TestNewTokenHeads(unittest.TestCase):
    def test_step_schedule(self) -> None:
        fired: set[int] = set()
        self.assertEqual(_new_token_heads(500, fired, step=1000, explicit=[]), [])
        self.assertEqual(_new_token_heads(1500, fired, step=1000, explicit=[]), [1000])
        self.assertEqual(_new_token_heads(2500, fired, step=1000, explicit=[]), [2000])
        self.assertEqual(fired, {1000, 2000})
        self.assertEqual(_new_token_heads(2500, fired, step=1000, explicit=[]), [])

    def test_explicit_heads(self) -> None:
        fired = set()
        self.assertEqual(_new_token_heads(500, fired, step=999, explicit=[100, 300]), [100, 300])


if __name__ == "__main__":
    unittest.main()
