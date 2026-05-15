from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spec_batch_compute_start_round import main as resume_main


def _capture_resume(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = resume_main(argv)
    return code, buf.getvalue()


class TestResumeStartRound(unittest.TestCase):
    @patch("speculative_agent._count_tokens", return_value=5000)
    def test_earliest_incomplete_in_window_not_highest_round(self, _tok: object) -> None:
        """Regression: hi=52 and window covering 42..52 — incomplete at 049 must resume 049, not 052."""
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / "runs" / "x" / "task_a"
            for lab in ("049", "050"):
                rr = rt / f"round_{lab}"
                rr.mkdir(parents=True)
                (rr / "llm_output.txt").write_text("stub", encoding="utf-8")
                (rr / "prompt.txt").write_text("x", encoding="utf-8")

            sp_root = Path(d) / "spec_act_runs" / "x" / "task_a"
            (sp_root / "round_052").mkdir(parents=True)

            sp49 = sp_root / "round_049"
            sp49.mkdir(parents=True)
            for h in (2000, 4000):
                hd = sp49 / f"head{h}"
                hd.mkdir(parents=True)
                (hd / "metrics.json").write_text("{}", encoding="utf-8")
            (sp49 / "head5000").mkdir(parents=True)

            sp50 = sp_root / "round_050"
            sp50.mkdir(parents=True)
            for h in (2000, 4000, 5000):
                hd = sp50 / f"head{h}"
                hd.mkdir(parents=True)
                (hd / "metrics.json").write_text("{}", encoding="utf-8")

            code, out = _capture_resume([str(rt), "--head-step", "2000", "--parallel-window", "10"])
            self.assertEqual(code, 0)
            self.assertEqual(out, "49")


    @patch("speculative_agent._count_tokens", return_value=5000)
    def test_no_spec_yet_returns_zero(self, _tok: object) -> None:
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / "runs" / "x" / "task_a"
            (rt / "round_000").mkdir(parents=True)
            code, out = _capture_resume([str(rt), "--head-step", "2000", "--parallel-window", "3"])
            self.assertEqual(code, 0)
            self.assertEqual(out, "0")

    @patch("speculative_agent._count_tokens", return_value=5000)
    def test_all_heads_done_advances_round(self, _tok: object) -> None:
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / "runs" / "x" / "task_a"
            r0 = rt / "round_000"
            r0.mkdir(parents=True)
            (r0 / "llm_output.txt").write_text("stub", encoding="utf-8")
            (r0 / "prompt.txt").write_text("x", encoding="utf-8")

            sp = Path(d) / "spec_act_runs" / "x" / "task_a" / "round_000"
            for h in (2000, 4000, 5000):
                hd = sp / f"head{h}"
                hd.mkdir(parents=True)
                (hd / "metrics.json").write_text("{}", encoding="utf-8")

            code, out = _capture_resume([str(rt), "--head-step", "2000", "--parallel-window", "5"])
            self.assertEqual(code, 0)
            self.assertEqual(out, "1")

    @patch("speculative_agent._count_tokens", return_value=5000)
    def test_incomplete_returns_round_without_deleting_partial_head_dir(self, _tok: object) -> None:
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d) / "runs" / "x" / "task_a"
            r0 = rt / "round_000"
            r0.mkdir(parents=True)
            (r0 / "llm_output.txt").write_text("stub", encoding="utf-8")
            (r0 / "prompt.txt").write_text("x", encoding="utf-8")

            sp = Path(d) / "spec_act_runs" / "x" / "task_a" / "round_000"
            for h in (2000, 4000):
                hd = sp / f"head{h}"
                hd.mkdir(parents=True)
                (hd / "metrics.json").write_text("{}", encoding="utf-8")
            tail_dir = sp / "head5000"
            tail_dir.mkdir(parents=True)
            (tail_dir / "partial.txt").write_text("busy", encoding="utf-8")

            code, out = _capture_resume([str(rt), "--head-step", "2000", "--parallel-window", "10"])
            self.assertEqual(code, 0)
            self.assertEqual(out, "0")
            self.assertTrue(tail_dir.is_dir())



if __name__ == "__main__":
    unittest.main()
