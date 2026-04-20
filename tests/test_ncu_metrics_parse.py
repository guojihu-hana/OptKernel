"""Tests for ncu Raw CSV metric parsing (no GPU required)."""

from __future__ import annotations

import unittest

from run_ncu import parse_ncu_raw_csv, parse_ncu_wide_csv


class TestParseNcuRawCsv(unittest.TestCase):
    def test_three_columns(self) -> None:
        text = """
==PROF== noise
sm__cycles_active.avg,cycle,1.23 M
smsp__inst_executed_op_shared_st.sum,inst,42
"""
        m = parse_ncu_raw_csv(text, None)
        self.assertIn("sm__cycles_active.avg", m)
        self.assertIn("smsp__inst_executed_op_shared_st.sum", m)
        self.assertEqual(m["smsp__inst_executed_op_shared_st.sum"], 42.0)

    def test_four_columns_kernel_prefix(self) -> None:
        text = """
matmul_kernel,sm__cycles_active.avg,cycle,2.0
vectorized_elementwise_kernel,sm__cycles_active.avg,cycle,4.0
"""
        m = parse_ncu_raw_csv(text, None)
        # Averaged across two launches
        self.assertEqual(m["sm__cycles_active.avg"], 3.0)

    def test_filter_only_names(self) -> None:
        text = "k,a__metric.avg,x,1.0\n"
        m = parse_ncu_raw_csv(text, {"a__metric.avg"})
        self.assertEqual(list(m.keys()), ["a__metric.avg"])

    def test_semicolon_instances(self) -> None:
        text = "m__x.sum,inst,10; 20; 30\n"
        m = parse_ncu_raw_csv(text, None)
        self.assertEqual(m["m__x.sum"], 20.0)

    def test_wide_header_row_metric_name_column(self) -> None:
        text = """Process ID,Kernel Name,Metric Name,Metric Unit,Metric Value
1,matmul_kernel,sm__cycles_active.avg,cycle,100
2,matmul_kernel,sm__cycles_active.avg,cycle,300
"""
        want = {"sm__cycles_active.avg"}
        m = parse_ncu_raw_csv(text, want)
        self.assertEqual(m["sm__cycles_active.avg"], 200.0)

    def test_metric_name_followed_by_another_metric_id_uses_last_column(self) -> None:
        # Second column after the metric name is another metric id, not the value.
        text = (
            "0,matmul,sm__inst_executed.sum,"
            "sm__inst_executed_pipe_tensor.max.pct_of_peak_sustained_active,1.25e6\n"
        )
        want = {"sm__inst_executed.sum"}
        m = parse_ncu_raw_csv(text, want)
        self.assertEqual(m["sm__inst_executed.sum"], 1.25e6)

    def test_wide_table_kernel_per_row(self) -> None:
        text = '''"ID","Kernel Name","sm__cycles_active.avg","dram__bytes.sum.per_second"
"8","matmul_kernel","10.5","100"
"9","other_kernel","20.5","200"
'''
        want = {"sm__cycles_active.avg", "dram__bytes.sum.per_second"}
        m = parse_ncu_wide_csv(text, want)
        self.assertEqual(m["sm__cycles_active.avg"], 15.5)
        self.assertEqual(m["dram__bytes.sum.per_second"], 150.0)


if __name__ == "__main__":
    unittest.main()
