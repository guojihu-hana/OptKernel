"""Tests for validation result status mapping."""

from __future__ import annotations

import unittest

from run_validation import _validation_status_from_benchmark_timing


class TestValidationStatusFromBenchmarkTiming(unittest.TestCase):
    def test_skipped_benchmark_is_benchmark_error(self) -> None:
        self.assertEqual(
            _validation_status_from_benchmark_timing({"skipped": True, "reason": "boom"}),
            "benchmark_error",
        )

    def test_completed_benchmark_is_success(self) -> None:
        self.assertEqual(
            _validation_status_from_benchmark_timing({"speedup": 1.2}),
            "success",
        )


if __name__ == "__main__":
    unittest.main()
