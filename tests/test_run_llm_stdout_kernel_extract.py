"""Regress: mixed LLM child stdout (logs before JSON) + extract ```python``` for kernel.py."""

import json
import unittest

from agent import extract_python_module
from run_llm import _parse_llm_worker_json_stdout

_MINIMAL_FAKE_KERNEL = '''```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.lin = nn.Linear(n, h)

    def forward(self, x):
        return self.lin(x)

def get_inputs():
    return [torch.randn(1, 8)]

def get_init_inputs():
    return [8, 4]
```'''


class TestRunLmStdoutAndKernelExtract(unittest.TestCase):
    def test_mixed_stdout_last_line_is_json(self) -> None:
        # Same pattern as pre-fix: Finish + Usage on stdout, then one JSON line.
        payload = {
            "ok": True,
            "text": _MINIMAL_FAKE_KERNEL,
            "llm_output_dumped": False,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        mixed = (
            "\u001b[92mFinish reason: stop\u001b[0m\n"
            "Usage: In=1, Out=2, Total=3\n"
            f"{line}\n"
        )
        parsed = _parse_llm_worker_json_stdout(mixed)
        self.assertTrue(parsed.get("ok"))
        self.assertIn("class Model", parsed.get("text", ""))

    def test_extract_yields_class_model(self) -> None:
        payload = {
            "ok": True,
            "text": _MINIMAL_FAKE_KERNEL,
            "llm_output_dumped": False,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        mixed = f"garbage\n{line}\n"
        text = str(_parse_llm_worker_json_stdout(mixed).get("text", ""))
        src = extract_python_module(text)
        self.assertIsNotNone(src, "expected fenced python to extract")
        self.assertIn("class Model", src)
        self.assertIn("def forward", src)
        # Writable kernel.py for agent
        self.assertNotIn("```", src, "extracted block should be raw python only")

    def test_rfind_ok_path_multiline_escaped_json_in_one_line(self) -> None:
        # Child prints single-line json; inner text has \\n (still one stdout line)
        p = {
            "ok": True,
            "text": _MINIMAL_FAKE_KERNEL,
            "llm_output_dumped": False,
        }
        one = json.dumps(p, ensure_ascii=False, default=str)
        mixed = "x\n" + "y\n" + one
        parsed = _parse_llm_worker_json_stdout(mixed)
        self.assertTrue(parsed.get("ok"))
        src = extract_python_module(str(parsed.get("text", "")))
        self.assertIsNotNone(src)
        self.assertIn("class Model", src)


if __name__ == "__main__":
    unittest.main()
