"""Tests for compiled (triton_op) wrappers matching eager implementations.

Verifies that:
1. scan_bidirectional_branched_compiled matches scan_bidirectional_branched
2. Same with scan_grad_scale=True
3. shift_pad_compiled matches shift_pad
4. Compiled scan with normalize_scan_grad_r matches eager
"""

import pytest
import torch

from associative_scan_triton import (
  get_grid,
  scan_bidirectional_branched,
  scan_bidirectional_branched_compiled,
  shift_pad,
  shift_pad_compiled,
)
from test_utils import TEST_SEED, get_device

# Skip entire module if CUDA is not available
pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="CUDA required for compile tests"
)


@pytest.fixture
def device():
  return get_device()


@pytest.fixture(autouse=True)
def set_random_seed():
  torch.manual_seed(TEST_SEED)


def _make_cu_seqlens(batch_size, seq_len, device):
  """Create uniform cu_seqlens for testing."""
  lengths = torch.full(
    (batch_size,), seq_len, device=device, dtype=torch.int32
  )
  cu_seqlens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
  cu_seqlens[1:] = lengths.cumsum(0)
  return cu_seqlens


class TestTritonOpsCorrectness:
  """Verify triton_op wrappers match original implementations."""

  @pytest.mark.parametrize("batch_size", [1, 4])
  @pytest.mark.parametrize("seq_len", [64, 256])
  def test_scan_matches_original(self, batch_size, seq_len, device):
    """scan_bidirectional_branched_compiled must match original."""
    no_channels = 64
    cu_seqlens = _make_cu_seqlens(batch_size, seq_len, device)
    total_len = int(cu_seqlens[-1].item())
    grid = get_grid(len(cu_seqlens), seq_len, 128, no_channels)

    gates_fwd = torch.rand(
      no_channels, total_len, device=device, dtype=torch.float32
    )
    tokens_fwd = torch.randn(
      no_channels, total_len, device=device, dtype=torch.float32
    )
    gates_bwd = torch.rand(
      no_channels, total_len, device=device, dtype=torch.float32
    )
    tokens_bwd = torch.randn(
      no_channels, total_len, device=device, dtype=torch.float32
    )

    args = {"cu_seqlens": cu_seqlens, "chunk_size": 128, "grid": grid}

    ref_fwd, ref_bwd = scan_bidirectional_branched(
      gates_fwd.clone(),
      tokens_fwd.clone(),
      gates_bwd.clone(),
      tokens_bwd.clone(),
      args,
    )
    test_fwd, test_bwd = scan_bidirectional_branched_compiled(
      gates_fwd.clone(),
      tokens_fwd.clone(),
      gates_bwd.clone(),
      tokens_bwd.clone(),
      args,
    )

    torch.testing.assert_close(ref_fwd, test_fwd, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ref_bwd, test_bwd, rtol=1e-5, atol=1e-5)

  @pytest.mark.parametrize("batch_size", [1, 4])
  @pytest.mark.parametrize("seq_len", [64, 256])
  def test_scan_grad_scale_compiled_matches_original(
    self, batch_size, seq_len, device
  ):
    """Compiled scan with scan_grad_scale must match non-compiled gradients."""
    no_channels = 64
    cu_seqlens = _make_cu_seqlens(batch_size, seq_len, device)
    total_len = int(cu_seqlens[-1].item())
    grid = get_grid(len(cu_seqlens), seq_len, 128, no_channels)

    def make_inputs(requires_grad=True):
      torch.manual_seed(TEST_SEED)
      return (
        torch.rand(no_channels, total_len, device=device, requires_grad=requires_grad),
        torch.randn(no_channels, total_len, device=device, requires_grad=requires_grad),
        torch.rand(no_channels, total_len, device=device, requires_grad=requires_grad),
        torch.randn(no_channels, total_len, device=device, requires_grad=requires_grad),
      )

    args = {
      "cu_seqlens": cu_seqlens, "chunk_size": 128, "grid": grid,
      "scan_grad_scale": True,
    }

    # Reference (non-compiled)
    gf1, tf1, gb1, tb1 = make_inputs()
    ref_fwd, ref_bwd = scan_bidirectional_branched(
      gf1, tf1, gb1, tb1, args,
    )
    (ref_fwd.sum() + ref_bwd.sum()).backward()

    # Test (compiled)
    gf2, tf2, gb2, tb2 = make_inputs()
    test_fwd, test_bwd = scan_bidirectional_branched_compiled(
      gf2, tf2, gb2, tb2, args,
    )
    (test_fwd.sum() + test_bwd.sum()).backward()

    # Forward outputs
    rtol, atol = 1e-5, 1e-5
    torch.testing.assert_close(ref_fwd, test_fwd, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_bwd, test_bwd, rtol=rtol, atol=atol)

    # Gradients: d_gates and d_tokens for both directions
    torch.testing.assert_close(gf1.grad, gf2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_gates_fwd mismatch")
    torch.testing.assert_close(tf1.grad, tf2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_tokens_fwd mismatch")
    torch.testing.assert_close(gb1.grad, gb2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_gates_bwd mismatch")
    torch.testing.assert_close(tb1.grad, tb2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_tokens_bwd mismatch")

  @pytest.mark.parametrize("backward", [True, False])
  @pytest.mark.parametrize("pad_value", [0.0, 1.0])
  def test_shift_pad_matches_original(self, backward, pad_value, device):
    """shift_pad_compiled must match shift_pad."""
    C, total_len = 64, 256
    cu_seqlens = _make_cu_seqlens(4, 64, device)
    data = torch.randn(C, total_len, device=device, dtype=torch.float32)

    ref = shift_pad(data, cu_seqlens, pad_value=pad_value, backward=backward)
    test = shift_pad_compiled(
      data, cu_seqlens, pad_value=pad_value, backward=backward
    )

    torch.testing.assert_close(ref, test, rtol=0, atol=0)

  @pytest.mark.parametrize("batch_size", [1, 4])
  @pytest.mark.parametrize("seq_len", [64, 256])
  def test_normalize_scan_grad_r_compiled_matches_original(
    self, batch_size, seq_len, device
  ):
    """Compiled scan with normalize_scan_grad_r must match non-compiled gradients."""
    no_channels = 64
    cu_seqlens = _make_cu_seqlens(batch_size, seq_len, device)
    total_len = int(cu_seqlens[-1].item())
    grid = get_grid(len(cu_seqlens), seq_len, 128, no_channels)

    def make_inputs(requires_grad=True):
      torch.manual_seed(TEST_SEED)
      return (
        torch.rand(no_channels, total_len, device=device, requires_grad=requires_grad),
        torch.randn(no_channels, total_len, device=device, requires_grad=requires_grad),
        torch.rand(no_channels, total_len, device=device, requires_grad=requires_grad),
        torch.randn(no_channels, total_len, device=device, requires_grad=requires_grad),
      )

    args = {
      "cu_seqlens": cu_seqlens, "chunk_size": 128, "grid": grid,
      "normalize_scan": True, "normalize_scan_grad_r": True,
    }

    # Reference (non-compiled)
    gf1, tf1, gb1, tb1 = make_inputs()
    ref_fwd, ref_bwd = scan_bidirectional_branched(
      gf1, tf1, gb1, tb1, args,
    )
    (ref_fwd.sum() + ref_bwd.sum()).backward()

    # Test (compiled)
    gf2, tf2, gb2, tb2 = make_inputs()
    test_fwd, test_bwd = scan_bidirectional_branched_compiled(
      gf2, tf2, gb2, tb2, args,
    )
    (test_fwd.sum() + test_bwd.sum()).backward()

    # Forward outputs
    rtol, atol = 1e-5, 1e-5
    torch.testing.assert_close(ref_fwd, test_fwd, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_bwd, test_bwd, rtol=rtol, atol=atol)

    # Gradients
    torch.testing.assert_close(gf1.grad, gf2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_gates_fwd mismatch")
    torch.testing.assert_close(tf1.grad, tf2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_tokens_fwd mismatch")
    torch.testing.assert_close(gb1.grad, gb2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_gates_bwd mismatch")
    torch.testing.assert_close(tb1.grad, tb2.grad, rtol=1e-4, atol=1e-5,
                               msg="d_tokens_bwd mismatch")
