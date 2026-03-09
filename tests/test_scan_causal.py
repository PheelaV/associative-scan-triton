"""Tests for causal (unidirectional) scan."""

import pytest
import torch

from associative_scan_triton import get_grid, scan_causal
from test_utils import TEST_SEED, get_device

import jax
import jax.numpy as jnp

from jax_utils import grad_scan_multi_channel, scan_multi_channel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestScanCausalNumeric:
  """Numeric correctness tests for scan_causal."""

  @pytest.mark.parametrize("chunk_size", [2, 4, 8])
  def test_causal_scan(self, chunk_size) -> None:
    """Test numeric values against hand-computed expected results."""
    device = get_device()
    torch.manual_seed(TEST_SEED)

    no_channels = 1
    seqlens = torch.tensor([4, 4])
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens.cumsum(dim=0))).to(
      device
    )
    max_seqlen = int(seqlens.max().item())
    grid = get_grid(len(cu_seqlens), max_seqlen, chunk_size, no_channels)

    gates = (
      torch.tensor([[1.0, 1.5, 0.8, 2.0, 1.0, 1.5, 0.8, 2.0]])
      .to(device)
      .requires_grad_(True)
    )
    tokens = (
      torch.tensor([[1.0, -1.0, 0.5, 2.0, 1.0, -1.0, 0.5, 2.0]])
      .to(device)
      .requires_grad_(True)
    )

    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

    # Forward
    result = scan_causal(gates, tokens, args)

    expected_states = torch.tensor(
      [[1, 0.5, 0.9, 3.8, 1, 0.5, 0.9, 3.8]], device=device
    )
    assert torch.allclose(
      result, expected_states, rtol=1e-4, atol=1e-4
    ), f"States mismatch: expected {expected_states}, got {result}"

    # Backward
    result.sum().backward()

    expected_gate_grads = torch.tensor(
      [[0.0, 3.4, 1.5, 0.9, 0.0, 3.4, 1.5, 0.9]], device=device
    )
    expected_token_grads = torch.tensor(
      [[6.1, 3.4, 3.0, 1.0, 6.1, 3.4, 3.0, 1.0]], device=device
    )

    assert torch.allclose(
      gates.grad, expected_gate_grads, rtol=1e-4, atol=1e-4
    ), f"Gate grads mismatch: expected {expected_gate_grads}, got {gates.grad}"
    assert torch.allclose(
      tokens.grad, expected_token_grads, rtol=1e-4, atol=1e-4
    ), f"Token grads mismatch: expected {expected_token_grads}, got {tokens.grad}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestScanCausalJaxReference:
  """Compare scan_causal against JAX associative scan reference."""

  @pytest.mark.parametrize(
    ("chunk_size", "num_channels", "seqlens"),
    [
      (chunk_size, num_channels, seqlens)
      for chunk_size in [2**i for i in range(1, 9)]
      for num_channels in (1, 2, 32)
      for seqlens in [1, 5, 32, 128]
    ],
  )
  def test_scan_causal_vs_jax(
    self, chunk_size, num_channels, seqlens
  ) -> None:
    """scan_causal output and gradients must match JAX reference."""
    device = get_device()
    torch.manual_seed(TEST_SEED)

    seqlens_t = torch.tensor([seqlens])
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens_t)).to(device)
    max_seqlen = seqlens
    grid = get_grid(len(cu_seqlens), max_seqlen, chunk_size, num_channels)

    gates = torch.rand(num_channels, seqlens, device=device).requires_grad_(True)
    tokens = torch.randn(num_channels, seqlens, device=device).requires_grad_(True)

    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

    # Triton forward
    result = scan_causal(gates, tokens, args)
    result.sum().backward()

    # JAX reference
    gates_jax = jnp.array(gates.detach().cpu().numpy())
    tokens_jax = jnp.array(tokens.detach().cpu().numpy())

    _, jax_tokens = scan_multi_channel(gates_jax, tokens_jax, False)
    jax_gates_grad, jax_tokens_grad = grad_scan_multi_channel(
      gates_jax, tokens_jax
    )

    jax_result = torch.tensor(
      jax_tokens.__array__(), device=device, dtype=torch.float32
    )
    jax_g_grad = torch.tensor(
      jax_gates_grad.__array__(), device=device, dtype=torch.float32
    )
    jax_t_grad = torch.tensor(
      jax_tokens_grad.__array__(), device=device, dtype=torch.float32
    )

    rtol, atol = 1e-4, 1e-4
    assert torch.allclose(result, jax_result, rtol=rtol, atol=atol), (
      f"Forward mismatch: max diff {(result - jax_result).abs().max()}"
    )
    assert torch.allclose(gates.grad, jax_g_grad, rtol=rtol, atol=atol), (
      f"Gate grad mismatch: max diff {(gates.grad - jax_g_grad).abs().max()}"
    )
    assert torch.allclose(tokens.grad, jax_t_grad, rtol=rtol, atol=atol), (
      f"Token grad mismatch: max diff {(tokens.grad - jax_t_grad).abs().max()}"
    )
