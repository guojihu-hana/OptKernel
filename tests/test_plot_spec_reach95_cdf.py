from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from plot_spec_reach95_cdf import _task_reach_ratios


class TestSpecReach95Cdf(unittest.TestCase):
    def test_runs_without_speedup_uses_first_spec_success_head(self) -> None:
        rr = Path("runs/task/round_000")
        sr = Path("spec_act_runs/task/round_000")
        with (
            patch("plot_spec_reach95_cdf._overlap_round_pairs", return_value=[(0, rr, sr)]),
            patch("plot_spec_reach95_cdf._speedup_from_metrics", return_value=None),
            patch("plot_spec_reach95_cdf._round_total_tokens", return_value=100),
            patch("plot_spec_reach95_cdf._first_head_success", return_value=25),
        ):
            ratios, runs_fail_spec_success, not_reached = _task_reach_ratios(
                Path("runs/task"),
                Path("spec_act_runs/task"),
                threshold=0.95,
                start_round=0,
                end_round=None,
                first_success=False,
                success_only=False,
            )

        self.assertEqual(ratios, [0.25])
        self.assertEqual(runs_fail_spec_success, 1)
        self.assertEqual(not_reached, 0)

    def test_first_success_mode_ignores_runs_speedup_threshold(self) -> None:
        rr = Path("runs/task/round_000")
        sr = Path("spec_act_runs/task/round_000")
        with (
            patch("plot_spec_reach95_cdf._overlap_round_pairs", return_value=[(0, rr, sr)]),
            patch("plot_spec_reach95_cdf._speedup_from_metrics", return_value=10.0),
            patch("plot_spec_reach95_cdf._round_total_tokens", return_value=100),
            patch("plot_spec_reach95_cdf._first_head_success", return_value=40),
            patch("plot_spec_reach95_cdf._first_head_reach_target", return_value=None),
        ):
            ratios, runs_fail_spec_success, not_reached = _task_reach_ratios(
                Path("runs/task"),
                Path("spec_act_runs/task"),
                threshold=0.95,
                start_round=0,
                end_round=None,
                first_success=True,
                success_only=False,
            )

        self.assertEqual(ratios, [0.4])
        self.assertEqual(runs_fail_spec_success, 0)
        self.assertEqual(not_reached, 0)

    def test_success_only_skips_round_when_runs_and_spec_both_fail(self) -> None:
        rr = Path("runs/task/round_000")
        sr = Path("spec_act_runs/task/round_000")
        with (
            patch("plot_spec_reach95_cdf._overlap_round_pairs", return_value=[(0, rr, sr)]),
            patch("plot_spec_reach95_cdf._speedup_from_metrics", return_value=None),
            patch("plot_spec_reach95_cdf._round_total_tokens", return_value=100),
            patch("plot_spec_reach95_cdf._first_head_success", return_value=None),
        ):
            ratios, runs_fail_spec_success, not_reached = _task_reach_ratios(
                Path("runs/task"),
                Path("spec_act_runs/task"),
                threshold=0.95,
                start_round=0,
                end_round=None,
                first_success=False,
                success_only=True,
            )

        self.assertEqual(ratios, [])
        self.assertEqual(runs_fail_spec_success, 0)
        self.assertEqual(not_reached, 0)


if __name__ == "__main__":
    unittest.main()
