from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from plot_spec_termination_token_ratios import _task_strategy_ratios


class TestSpecTerminationTokenRatios(unittest.TestCase):
    def test_success_only_skips_round_when_runs_and_spec_both_fail(self) -> None:
        rr = Path("runs/task/round_000")
        sr = Path("spec_act_runs/task/round_000")
        with (
            patch("plot_spec_termination_token_ratios._overlap_round_pairs", return_value=[(0, rr, sr)]),
            patch("plot_spec_termination_token_ratios._round_total_tokens", return_value=100),
            patch("plot_spec_termination_token_ratios._first_head_success", return_value=None),
            patch("plot_spec_termination_token_ratios._speedup_from_metrics", return_value=None),
        ):
            first_mean, reach_mean, count = _task_strategy_ratios(
                Path("runs/task"),
                Path("spec_act_runs/task"),
                threshold=0.95,
                start_round=0,
                end_round=None,
                success_only=True,
            )

        self.assertEqual((first_mean, reach_mean, count), (0.0, 0.0, 0))


if __name__ == "__main__":
    unittest.main()
