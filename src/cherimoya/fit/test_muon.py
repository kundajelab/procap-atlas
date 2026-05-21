"""Equivalence tests: local muon.Muon vs torch.optim.Muon (PyTorch >= 2.9).

All tests are skipped when torch.optim.Muon is unavailable so the suite still
passes on older PyTorch installs.

Run from the repo root:
    python -m pytest src/cherimoya/fit/test_muon.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
import torch

try:
    from torch.optim import Muon as TorchMuon

    HAS_TORCH_MUON = True
except ImportError:
    HAS_TORCH_MUON = False

from muon import Muon as LocalMuon

needs_torch_muon = pytest.mark.skipif(
    not HAS_TORCH_MUON, reason="torch.optim.Muon requires PyTorch >= 2.9"
)

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clone(p: torch.nn.Parameter) -> torch.nn.Parameter:
    return torch.nn.Parameter(p.detach().clone())


def _pair(shape, **kwargs):
    """Return (upstream_opt, local_opt, p_ref, p_local, shared_grad_fn).

    grad_fn() returns a fresh gradient tensor with the same shape each call.
    """
    p_ref = torch.nn.Parameter(torch.randn(*shape))
    p_local = _clone(p_ref)
    upstream = TorchMuon([p_ref], **kwargs)
    local = LocalMuon([p_local], **kwargs)

    def grad_fn():
        return torch.randn(*shape)

    return upstream, local, p_ref, p_local, grad_fn


def _step(upstream, local, p_ref, p_local, grad):
    p_ref.grad = grad.clone()
    p_local.grad = grad.clone()
    upstream.step()
    local.step()


# ---------------------------------------------------------------------------
# Single-step equivalence
# ---------------------------------------------------------------------------


@needs_torch_muon
def test_single_step_wide():
    """Wide matrix: cols > rows — no transpose in Newton-Schulz."""
    upstream, local, p_ref, p_local, grad_fn = _pair((32, 96), lr=0.01)
    _step(upstream, local, p_ref, p_local, grad_fn())
    torch.testing.assert_close(p_local, p_ref)


@needs_torch_muon
def test_single_step_tall():
    """Tall matrix: rows > cols — triggers the transpose branch in NS."""
    upstream, local, p_ref, p_local, grad_fn = _pair((96, 32), lr=0.01)
    _step(upstream, local, p_ref, p_local, grad_fn())
    torch.testing.assert_close(p_local, p_ref)


@needs_torch_muon
def test_single_step_square():
    """Square matrix."""
    upstream, local, p_ref, p_local, grad_fn = _pair((64, 64), lr=0.01)
    _step(upstream, local, p_ref, p_local, grad_fn())
    torch.testing.assert_close(p_local, p_ref)


# ---------------------------------------------------------------------------
# Multi-step: verifies momentum buffer accumulation
# ---------------------------------------------------------------------------


@needs_torch_muon
def test_multi_step_momentum():
    """5 steps with changing gradients — momentum buffers must stay in sync."""
    upstream, local, p_ref, p_local, grad_fn = _pair((48, 64), lr=0.01)
    for _ in range(5):
        g = grad_fn()
        _step(upstream, local, p_ref, p_local, g)
    torch.testing.assert_close(p_local, p_ref)


# ---------------------------------------------------------------------------
# Hyperparameter variants
# ---------------------------------------------------------------------------


@needs_torch_muon
def test_nesterov_false():
    upstream, local, p_ref, p_local, grad_fn = _pair(
        (32, 64), lr=0.01, nesterov=False
    )
    _step(upstream, local, p_ref, p_local, grad_fn())
    torch.testing.assert_close(p_local, p_ref)


@needs_torch_muon
def test_adjust_lr_match_rms_adamw():
    upstream, local, p_ref, p_local, grad_fn = _pair(
        (32, 64), lr=0.01, adjust_lr_fn="match_rms_adamw"
    )
    _step(upstream, local, p_ref, p_local, grad_fn())
    torch.testing.assert_close(p_local, p_ref)


@needs_torch_muon
def test_weight_decay():
    upstream, local, p_ref, p_local, grad_fn = _pair(
        (32, 64), lr=0.01, weight_decay=0.05
    )
    for _ in range(3):
        _step(upstream, local, p_ref, p_local, grad_fn())
    torch.testing.assert_close(p_local, p_ref)


# ---------------------------------------------------------------------------
# Multiple parameters in one optimizer
# ---------------------------------------------------------------------------


@needs_torch_muon
def test_multi_param_group():
    """Two separate 2D parameters optimized together."""
    shapes = [(32, 64), (96, 32)]
    refs = [torch.nn.Parameter(torch.randn(*s)) for s in shapes]
    locals_ = [_clone(p) for p in refs]

    upstream = TorchMuon(refs, lr=0.01)
    local = LocalMuon(locals_, lr=0.01)

    grads = [torch.randn(*s) for s in shapes]
    for p, g in zip(refs, grads):
        p.grad = g.clone()
    for p, g in zip(locals_, grads):
        p.grad = g.clone()

    upstream.step()
    local.step()

    for p_local, p_ref in zip(locals_, refs):
        torch.testing.assert_close(p_local, p_ref)
