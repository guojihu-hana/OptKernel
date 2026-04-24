"""
ReLUSelfAttention reference: RNG order and flat ``[0:100]`` behavior.

``run_forward_validation`` uses ``manual_seed`` → ``get_inputs()`` → ``Model()``,
which is **not** the same as ``Model()`` → ``get_inputs()``. For this task, the
first 100 C-order values often equal ``0.0`` (including 100/100 for ``seed=0``)
because they are ``out[0, 0, 0:100]`` (first batch, first token, first 100
channels) and this architecture can yield exact zeros there after
ReLU-on-attention. That is consistent with long runs of ``ref: 0.0`` in
``mismatch_position_sample``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from run_validation import import_kernelbench_file

_KERNEL_50 = (
    Path(__file__).resolve().parent.parent
    / "KernelBench"
    / "level3"
    / "50_ReLUSelfAttention.py"
)


def _import_fresh(name: str):
    return import_kernelbench_file(_KERNEL_50, name)


@torch.inference_mode()
def test_50_re_lu_self_validation_order_seed0_first_100_all_zero() -> None:
    """Match ``run_forward_validation``: seed → get_inputs() → ``Model()``.

    For default ``seed=0``, the first 100 flattened values are all exactly ``0.0``.
    """
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    m = _import_fresh("kb50_val_order_0")
    x = m.get_inputs()[0]
    model = m.Model(*m.get_init_inputs()).eval()
    out = model(x)
    first100 = out.detach().float().cpu().reshape(-1)[:100]
    assert int((first100 == 0).sum().item()) == 100


@torch.inference_mode()
def test_50_re_lu_self_model_before_inputs_order_first_100_not_all_zero() -> None:
    """Contrasting order: init ``Model`` before building inputs (not used in validation)."""
    torch.manual_seed(0)
    m = _import_fresh("kb50_mod_first")
    model = m.Model(*m.get_init_inputs()).eval()
    out = model(m.get_inputs()[0])
    first100 = out.detach().float().cpu().reshape(-1)[:100]
    assert not bool((first100 == 0).all().item())
