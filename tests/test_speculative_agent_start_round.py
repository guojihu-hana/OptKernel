"""Tests for speculative sweep start-round selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from speculative_agent import _resolve_sweep_start_round


class TestResolveSweepStartRound(unittest.TestCase):
    def test_numeric_start_round(self) -> None:
        self.assertEqual(_resolve_sweep_start_round("12", []), 12)

    def test_auto_start_round_resumes_after_largest_output_round(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            input_root = Path(td) / "runs"
            output_root = Path(td) / "spec_act_runs"
            input_root.mkdir()
            output_root.mkdir()
            rounds = []
            for name in ["round_001", "round_099", "round_020"]:
                p = input_root / name
                p.mkdir()
                rounds.append(p)
            output_rounds = []
            for name in ["round_001", "round_082", "round_020"]:
                p = output_root / name
                p.mkdir()
                output_rounds.append(p)

            self.assertEqual(_resolve_sweep_start_round("auto", rounds, output_rounds), 83)

    def test_auto_start_round_without_output_rounds_starts_from_zero(self) -> None:
        self.assertEqual(_resolve_sweep_start_round("auto", [], []), 0)


if __name__ == "__main__":
    unittest.main()
