"""
Unit tests for KernelBenchAgent.run_round_llm_only (prompt → LLM → extract → save).

Does not run run_validation.run_forward_validation, ncu, or the real query_server.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# agent.py imports query_server; stub before loading agent to avoid agents.llm_local dependency.
if "query_server" not in sys.modules:
    _stub = types.ModuleType("query_server")

    def _stub_query_server(*_a, **_kw) -> str:
        raise RuntimeError("call_llm is mocked in tests; query_server must not be called")

    _stub.query_server = _stub_query_server
    sys.modules["query_server"] = _stub

from agent import AgentConfig, KernelBenchAgent, extract_python_module


REF_SNIPPET = "KERNELBENCH_REFERENCE_UNIQUE_MARK_12345"

FAKE_KERNEL = f"""```python
import torch
# generated for test — {REF_SNIPPET}
x = 42
class Model(torch.nn.Module):
    def forward(self, t):
        return t
def get_inputs():
    return [torch.zeros(1)]
def get_init_inputs():
    return []
```
"""


class TestRunRoundLlmOnly(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.task_path = root / "task.py"
        self.task_path.write_text(
            f"# reference task\nprint('{REF_SNIPPET}')\n",
            encoding="utf-8",
        )
        self.work_dir = root / "work"
        self.cfg = AgentConfig(
            task_path=self.task_path,
            work_dir=self.work_dir,
            run_ncu=False,
        )

    def test_round0_prompt_llm_extract_and_save(self) -> None:
        def fake_call_llm(self_: KernelBenchAgent, system: str, user: str, round_idx: int, *_a, **_kw):
            self.assertIn(REF_SNIPPET, user)
            self.assertIn("CUDA", system)
            return FAKE_KERNEL, False

        agent = KernelBenchAgent(self.cfg)
        with patch.object(KernelBenchAgent, "call_llm", fake_call_llm):
            out = agent.run_round_llm_only(0)

        self.assertNotIn("status", out)
        self.assertEqual(out["round"], 0)
        rd = self.work_dir / "round_000"
        self.assertTrue((rd / "prompt.txt").is_file())
        self.assertTrue((rd / "llm_output.txt").is_file())
        self.assertTrue((rd / "kernel.py").is_file())
        self.assertFalse((rd / "metrics.json").exists())

        prompt_text = (rd / "prompt.txt").read_text(encoding="utf-8")
        self.assertIn(REF_SNIPPET, prompt_text)
        self.assertIn("--- system ---", prompt_text)
        self.assertIn("--- user ---", prompt_text)

        kernel_text = (rd / "kernel.py").read_text(encoding="utf-8")
        self.assertIn("import torch", kernel_text)
        self.assertNotIn("```", kernel_text)

        llm_raw = (rd / "llm_output.txt").read_text(encoding="utf-8")
        self.assertIn("```python", llm_raw)

    def test_parse_error_writes_metrics_no_kernel(self) -> None:
        def fake_call_llm(
            self_: KernelBenchAgent, system: str, user: str, round_idx: int, *_a, **_kw
        ):
            return "no code fence here, only prose", False

        agent = KernelBenchAgent(self.cfg)
        with patch.object(KernelBenchAgent, "call_llm", fake_call_llm):
            out = agent.run_round_llm_only(0)

        self.assertEqual(out.get("status"), "parse_error")
        self.assertFalse(out.get("runnable"))
        rd = self.work_dir / "round_000"
        self.assertTrue((rd / "metrics.json").is_file())
        self.assertFalse((rd / "kernel.py").exists())
        data = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "parse_error")

    def test_round1_includes_previous_kernel_in_user_prompt(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        prev = self.work_dir / "round_000"
        prev.mkdir(parents=True, exist_ok=True)
        (prev / "kernel.py").write_text("# PREV_KERNEL_LINE\n", encoding="utf-8")
        (prev / "metrics.json").write_text(
            json.dumps({"status": "success", "runnable": True, "round": 0}),
            encoding="utf-8",
        )

        captured: dict[str, str] = {}

        def fake_call_llm(self_: KernelBenchAgent, system: str, user: str, round_idx: int, *_a, **_kw):
            captured["user"] = user
            self.assertEqual(round_idx, 1)
            return FAKE_KERNEL, False

        agent = KernelBenchAgent(self.cfg)
        with patch.object(KernelBenchAgent, "call_llm", fake_call_llm):
            agent.run_round_llm_only(1)

        self.assertIn("PREV_KERNEL_LINE", captured["user"])
        self.assertIn(REF_SNIPPET, captured["user"])
        rd = self.work_dir / "round_001"
        self.assertTrue((rd / "kernel.py").is_file())


class TestExtractPythonModule(unittest.TestCase):
    def test_uses_last_python_fence_when_multiple(self) -> None:
        text = """
Thinking...
```python
from torch.utils.cpp_extension import load_inline
# snippet only
load_inline(name="x", cuda_sources="")
```

Full solution:
```python
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, A, B):
        return A @ B
N = 4
def get_inputs():
    return [torch.zeros(N, N), torch.zeros(N, N)]
def get_init_inputs():
    return []
```
"""
        out = extract_python_module(text)
        assert out is not None
        self.assertIn("class Model", out)
        self.assertIn("def get_inputs", out)
        self.assertIn("def get_init_inputs", out)
        self.assertIn("N = 4", out)
        self.assertNotIn("snippet only", out)


if __name__ == "__main__":
    unittest.main()
