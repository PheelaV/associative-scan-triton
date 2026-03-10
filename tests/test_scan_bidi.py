import pytest
import torch

from associative_scan_triton import get_grid
from test_utils import (
  TEST_SEED,
  get_device,
)


import jax
import jax.numpy as jnp

from jax_utils import (
  grad_scan_multi_channel_bidi_branched,
  scan_multi_channel_bidi_branched,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestJaxReference:
  @pytest.mark.parametrize(
    ("chunk_size", "num_channels", "seqlens"),
    [
      (chunk_size, num_channels, seqlens)
      for chunk_size in [2**i for i in range(1, 9)]
      for num_channels in (1, 2, 32)
      for seqlens in [1, 5, 32, 128]
    ],
  )
  def test_scan_bidirectional_branched(
    self, chunk_size, num_channels, seqlens
  ):
    # Setup
    from associative_scan_triton import scan_bidirectional_branched

    # chunk_size, num_channels, seqlens = 4, 2, 8
    device = get_device()
    torch.manual_seed(TEST_SEED)

    seqlens = torch.tensor([seqlens])
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens)).to(device)
    max_seqlen = cu_seqlens.diff().max().item()
    # random non-negative data for gates and tokens
    gates_fwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    tokens_fwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    gates_bwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    tokens_bwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    gates_jax_fwd = jnp.array(gates_fwd.detach().cpu().numpy())
    tokens_jax_fwd = jnp.array(tokens_fwd.detach().cpu().numpy())
    gates_jax_bwd = jnp.array(gates_bwd.detach().cpu().numpy())
    tokens_jax_bwd = jnp.array(tokens_bwd.detach().cpu().numpy())

    grid = get_grid(len(cu_seqlens), max_seqlen, chunk_size, num_channels)
    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid":grid}

    # Act
    states_fwd, states_bwd = scan_bidirectional_branched(
      gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args
    )
    result = states_fwd.sum() + states_bwd.sum()
    result.backward()
    gates_grad_fwd = gates_fwd.grad
    tokens_grad_fwd = tokens_fwd.grad
    gates_grad_bwd = gates_bwd.grad
    tokens_grad_bwd = tokens_bwd.grad

    jax_states_result_fwd, jax_states_result_bwd = (
      scan_multi_channel_bidi_branched(
        gates_jax_fwd, tokens_jax_fwd, gates_jax_bwd, tokens_jax_bwd
      )
    )
    grads_jax = grad_scan_multi_channel_bidi_branched(
      gates_jax_fwd, tokens_jax_fwd, gates_jax_bwd, tokens_jax_bwd
    )

    # convert JAX results back to torch tensors for comparison
    jax_states_result_fwd = torch.tensor(
      jax.device_get(jax_states_result_fwd), dtype=torch.float32
    ).to(device)
    jax_states_result_bwd = torch.tensor(
      jax.device_get(jax_states_result_bwd), dtype=torch.float32
    ).to(device)
    jax_gates_grad_fwd = torch.tensor(
      jax.device_get(grads_jax[0]), dtype=torch.float32
    ).to(device)
    jax_tokens_grad_fwd = torch.tensor(
      jax.device_get(grads_jax[1]), dtype=torch.float32
    ).to(device)
    jax_gates_grad_bwd = torch.tensor(
      jax.device_get(grads_jax[2]), dtype=torch.float32
    ).to(device)
    jax_tokens_grad_bwd = torch.tensor(
      jax.device_get(grads_jax[3]), dtype=torch.float32
    ).to(device)

    # Asert
    rtol, atol = 1e-4, 1e-7
    assert torch.allclose(
      states_fwd, jax_states_result_fwd, rtol=rtol, atol=atol
    ), "states fwd"
    assert torch.allclose(
      states_bwd, jax_states_result_bwd, rtol=rtol, atol=atol
    ), "states bwd"
    assert torch.allclose(
      gates_grad_fwd, jax_gates_grad_fwd, rtol=rtol, atol=atol
    ), "gates grad fwd"
    assert torch.allclose(
      tokens_grad_fwd, jax_tokens_grad_fwd, rtol=rtol, atol=atol
    ), "tokens grad fwd"
    assert torch.allclose(
      gates_grad_bwd, jax_gates_grad_bwd, rtol=rtol, atol=atol
    ), "gates grad bwd"
    assert torch.allclose(
      tokens_grad_bwd, jax_tokens_grad_bwd, rtol=rtol, atol=atol
    ), "tokens grad bwd"

  @pytest.mark.parametrize(
    ("chunk_size", "num_channels", "seqlens"),
    [
      (chunk_size, num_channels, seqlens)
      for chunk_size in [2**i for i in range(1, 9)]
      for num_channels in (1, 2, 32)
      for seqlens in [1, 5, 32, 128]
      if chunk_size >= seqlens
    ],
  )
  def test_scan_bidirectional_branched_micro(
    self, chunk_size, num_channels, seqlens
  ):
    """This is testing a microptimization.

    When we are not chunking we do not need to write final gates
    """
    # Setup
    from associative_scan_triton import scan_bidirectional_branched

    # chunk_size, num_channels, seqlens = 4, 2, 8
    device = get_device()
    torch.manual_seed(TEST_SEED)

    seqlens = torch.tensor([seqlens])
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens)).to(device)
    max_seqlen = cu_seqlens.diff().max().item()
    # random non-negative data for gates and tokens
    gates_fwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    tokens_fwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    gates_bwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    tokens_bwd = torch.rand(
      (num_channels, max_seqlen), device=device, requires_grad=True
    )
    gates_jax_fwd = jnp.array(gates_fwd.detach().cpu().numpy())
    tokens_jax_fwd = jnp.array(tokens_fwd.detach().cpu().numpy())
    gates_jax_bwd = jnp.array(gates_bwd.detach().cpu().numpy())
    tokens_jax_bwd = jnp.array(tokens_bwd.detach().cpu().numpy())

    grid = get_grid(len(cu_seqlens), max_seqlen, chunk_size, num_channels)
    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid":grid}

    # Act
    states_fwd, states_bwd = scan_bidirectional_branched(
      gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args, testing=False
    )
    result = states_fwd.sum() + states_bwd.sum()
    result.backward()
    gates_grad_fwd = gates_fwd.grad
    tokens_grad_fwd = tokens_fwd.grad
    gates_grad_bwd = gates_bwd.grad
    tokens_grad_bwd = tokens_bwd.grad

    jax_states_result_fwd, jax_states_result_bwd = (
      scan_multi_channel_bidi_branched(
        gates_jax_fwd, tokens_jax_fwd, gates_jax_bwd, tokens_jax_bwd
      )
    )
    grads_jax = grad_scan_multi_channel_bidi_branched(
      gates_jax_fwd, tokens_jax_fwd, gates_jax_bwd, tokens_jax_bwd
    )

    # convert JAX results back to torch tensors for comparison
    jax_states_result_fwd = torch.tensor(
      jax.device_get(jax_states_result_fwd), dtype=torch.float32
    ).to(device)
    jax_states_result_bwd = torch.tensor(
      jax.device_get(jax_states_result_bwd), dtype=torch.float32
    ).to(device)
    jax_gates_grad_fwd = torch.tensor(
      jax.device_get(grads_jax[0]), dtype=torch.float32
    ).to(device)
    jax_tokens_grad_fwd = torch.tensor(
      jax.device_get(grads_jax[1]), dtype=torch.float32
    ).to(device)
    jax_gates_grad_bwd = torch.tensor(
      jax.device_get(grads_jax[2]), dtype=torch.float32
    ).to(device)
    jax_tokens_grad_bwd = torch.tensor(
      jax.device_get(grads_jax[3]), dtype=torch.float32
    ).to(device)

    # Asert
    rtol, atol = 1e-4, 1e-7
    assert torch.allclose(
      states_fwd, jax_states_result_fwd, rtol=rtol, atol=atol
    ), "states fwd"
    assert torch.allclose(
      states_bwd, jax_states_result_bwd, rtol=rtol, atol=atol
    ), "states bwd"
    assert torch.allclose(
      gates_grad_fwd, jax_gates_grad_fwd, rtol=rtol, atol=atol
    ), "gates grad fwd"
    assert torch.allclose(
      tokens_grad_fwd, jax_tokens_grad_fwd, rtol=rtol, atol=atol
    ), "tokens grad fwd"
    assert torch.allclose(
      gates_grad_bwd, jax_gates_grad_bwd, rtol=rtol, atol=atol
    ), "gates grad bwd"
    assert torch.allclose(
      tokens_grad_bwd, jax_tokens_grad_bwd, rtol=rtol, atol=atol
    ), "tokens grad bwd"
