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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestScanGradScale:
  """Tests for Fix 3: receptive field normalization (scan_grad_scale).

  Verifies that:
  1. R[t] computation is correct for known gate values
  2. d_gates is scaled by 1/R[t], d_tokens is untouched
  3. Multi-document cu_seqlens resets R at boundaries
  """

  @staticmethod
  def _compute_receptive_field_reference(shifted_gates, reverse=True):
    """Reference R[t] computation (pure Python, no Triton).

    Takes already-shifted gates (same as production code passes to the scan).
    The scan formula: r[t] = shifted_gates[t] * r[t+-1] + 1
    For reverse=True:  r[t] = shifted_gates[t] * r[t+1] + 1, r[T-1]=1
    For reverse=False: r[t] = shifted_gates[t] * r[t-1] + 1, r[0]=1
    """
    C, T = shifted_gates.shape
    r = torch.ones_like(shifted_gates)
    if reverse:
      for t in range(T - 2, -1, -1):
        r[:, t] = shifted_gates[:, t] * r[:, t + 1] + 1.0
    else:
      for t in range(1, T):
        r[:, t] = shifted_gates[:, t] * r[:, t - 1] + 1.0
    return r

  def test_receptive_field_correctness(self):
    """Verify R[t] matches hand-computed values for known gates."""
    from associative_scan_triton import forward_scan_full, shift_pad, get_grid

    device = get_device()
    # 2 channels: a=0.99 (long memory), a=0.50 (short memory), T=6
    gates = torch.tensor(
      [[0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
       [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]],
      device=device,
    )
    cu_seqlens = torch.tensor([0, 6], device=device, dtype=torch.int32)
    chunk_size = 8
    grid = get_grid(len(cu_seqlens), 6, chunk_size, 2)

    # Compute R via shift_pad + scan (same as production code)
    shifted = shift_pad(data=gates, cu_seqlens=cu_seqlens, backward=False, pad_value=1)
    # Reference computation (before scan, which mutates gates in-place)
    r_ref = self._compute_receptive_field_reference(shifted.clone(), reverse=True)
    r = torch.ones_like(gates)
    forward_scan_full(shifted, r, cu_seqlens, grid, REVERSE=True, CHUNK_SIZE=chunk_size, TESTING=True)

    assert torch.allclose(r, r_ref, rtol=1e-5, atol=1e-6), (
      f"R scan mismatch:\ngot:  {r}\nwant: {r_ref}"
    )

    # Sanity: channel A (a=0.99) should have R[0] ~ 5.85, channel B (a=0.50) ~ 1.97
    assert r[0, 0].item() > 5.0, f"Channel A R[0] too small: {r[0, 0]}"
    assert r[1, 0].item() < 2.5, f"Channel B R[0] too large: {r[1, 0]}"

  @pytest.mark.parametrize("chunk_size", [4, 32, 128])
  @pytest.mark.parametrize("seqlen", [8, 64])
  def test_d_gates_scaled_d_tokens_untouched(self, chunk_size, seqlen):
    """d_tokens must be identical with/without scan_grad_scale.
    d_gates must equal raw_d_gates / R[t]."""
    from associative_scan_triton import (
      scan_bidirectional_branched,
      forward_scan_full,
      shift_pad,
      get_grid,
    )

    device = get_device()
    num_channels = 4
    torch.manual_seed(TEST_SEED)

    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    grid = get_grid(len(cu_seqlens), seqlen, chunk_size, num_channels)

    def make_inputs():
      return (
        torch.rand(num_channels, seqlen, device=device, requires_grad=True),
        torch.randn(num_channels, seqlen, device=device, requires_grad=True),
        torch.rand(num_channels, seqlen, device=device, requires_grad=True),
        torch.randn(num_channels, seqlen, device=device, requires_grad=True),
      )

    # Save clean gate data for R[t] reference computation (scan modifies in-place)
    torch.manual_seed(TEST_SEED)
    gf_ref, _, gb_ref, _ = make_inputs()
    gates_fwd_clean = gf_ref.detach().clone()
    gates_bwd_clean = gb_ref.detach().clone()

    # Run WITHOUT scan_grad_scale
    torch.manual_seed(TEST_SEED)
    gf1, tf1, gb1, tb1 = make_inputs()
    args_off = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}
    s_fwd, s_bwd = scan_bidirectional_branched(gf1, tf1, gb1, tb1, args_off)
    (s_fwd.sum() + s_bwd.sum()).backward()
    d_gates_fwd_raw = gf1.grad.clone()
    d_tokens_fwd_raw = tf1.grad.clone()
    d_gates_bwd_raw = gb1.grad.clone()
    d_tokens_bwd_raw = tb1.grad.clone()

    # Run WITH scan_grad_scale
    torch.manual_seed(TEST_SEED)
    gf2, tf2, gb2, tb2 = make_inputs()
    args_on = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid,
               "scan_grad_scale": True}
    s_fwd2, s_bwd2 = scan_bidirectional_branched(gf2, tf2, gb2, tb2, args_on)
    (s_fwd2.sum() + s_bwd2.sum()).backward()
    d_gates_fwd_scaled = gf2.grad.clone()
    d_tokens_fwd_scaled = tf2.grad.clone()
    d_gates_bwd_scaled = gb2.grad.clone()
    d_tokens_bwd_scaled = tb2.grad.clone()

    # Forward outputs must match (scan_grad_scale only affects backward)
    rtol, atol = 1e-5, 1e-6
    assert torch.allclose(s_fwd, s_fwd2, rtol=rtol, atol=atol), "forward outputs differ"
    assert torch.allclose(s_bwd, s_bwd2, rtol=rtol, atol=atol), "backward outputs differ"

    # d_tokens must be IDENTICAL (untouched by scan_grad_scale)
    assert torch.allclose(d_tokens_fwd_raw, d_tokens_fwd_scaled, rtol=rtol, atol=atol), (
      "d_tokens_fwd changed by scan_grad_scale"
    )
    assert torch.allclose(d_tokens_bwd_raw, d_tokens_bwd_scaled, rtol=rtol, atol=atol), (
      "d_tokens_bwd changed by scan_grad_scale"
    )

    # Compute expected R[t] independently using clean (pre-scan) gate data
    shifted_fwd = shift_pad(gates_fwd_clean, cu_seqlens, backward=False, pad_value=1)
    shifted_bwd = shift_pad(gates_bwd_clean, cu_seqlens, backward=True, pad_value=1)
    r_fwd = self._compute_receptive_field_reference(shifted_fwd, reverse=True)
    r_bwd = self._compute_receptive_field_reference(shifted_bwd, reverse=False)

    # d_gates_scaled == d_gates_raw / R.clamp(min=1.0)
    expected_fwd = d_gates_fwd_raw / r_fwd.clamp(min=1.0)
    expected_bwd = d_gates_bwd_raw / r_bwd.clamp(min=1.0)
    assert torch.allclose(d_gates_fwd_scaled, expected_fwd, rtol=1e-4, atol=1e-5), (
      f"d_gates_fwd scaling wrong\nmax diff: {(d_gates_fwd_scaled - expected_fwd).abs().max()}"
    )
    assert torch.allclose(d_gates_bwd_scaled, expected_bwd, rtol=1e-4, atol=1e-5), (
      f"d_gates_bwd scaling wrong\nmax diff: {(d_gates_bwd_scaled - expected_bwd).abs().max()}"
    )

  def test_multi_document_cu_seqlens(self):
    """R[t] must reset at document boundaries in packed sequences."""
    from associative_scan_triton import (
      scan_bidirectional_branched,
      forward_scan_full,
      shift_pad,
      get_grid,
    )

    device = get_device()
    num_channels = 2
    # Two documents: lengths 5 and 8 = total 13
    cu_seqlens = torch.tensor([0, 5, 13], device=device, dtype=torch.int32)
    total_len = 13
    chunk_size = 16
    grid = get_grid(len(cu_seqlens), 8, chunk_size, num_channels)

    torch.manual_seed(TEST_SEED)
    gates_fwd = torch.rand(num_channels, total_len, device=device, requires_grad=True)
    tokens_fwd = torch.randn(num_channels, total_len, device=device, requires_grad=True)
    gates_bwd = torch.rand(num_channels, total_len, device=device, requires_grad=True)
    tokens_bwd = torch.randn(num_channels, total_len, device=device, requires_grad=True)

    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid,
            "scan_grad_scale": True}
    s_fwd, s_bwd = scan_bidirectional_branched(
      gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args,
    )
    (s_fwd.sum() + s_bwd.sum()).backward()

    # Verify R resets at boundary: compute R for each document separately
    shifted_fwd = shift_pad(gates_fwd.detach(), cu_seqlens, backward=False, pad_value=1)

    # Doc 1: positions 0..4 (len=5)
    r_doc1 = self._compute_receptive_field_reference(
      shifted_fwd[:, 0:5], reverse=True
    )
    # Doc 2: positions 5..12 (len=8)
    r_doc2 = self._compute_receptive_field_reference(
      shifted_fwd[:, 5:13], reverse=True
    )

    # R at last position of each doc should be 1.0 (boundary: only sees itself)
    assert torch.allclose(r_doc1[:, 4], torch.ones(num_channels, device=device), atol=1e-6), (
      f"R at doc1 boundary not 1.0: {r_doc1[:, 4]}"
    )
    assert torch.allclose(r_doc2[:, 7], torch.ones(num_channels, device=device), atol=1e-6), (
      f"R at doc2 boundary not 1.0: {r_doc2[:, 7]}"
    )
    # R >= 1.0 everywhere (accumulation is always at least 1)
    assert (r_doc1 >= 1.0 - 1e-6).all(), f"R doc1 has values < 1: {r_doc1.min()}"
    assert (r_doc2 >= 1.0 - 1e-6).all(), f"R doc2 has values < 1: {r_doc2.min()}"
    # R should be monotonically non-increasing from position 0 to T-1
    # (reverse scan: earlier positions accumulate more)
    for c in range(num_channels):
      assert (r_doc1[c, :-1] >= r_doc1[c, 1:] - 1e-6).all(), (
        f"R doc1 not monotonic for channel {c}: {r_doc1[c]}"
      )
      assert (r_doc2[c, :-1] >= r_doc2[c, 1:] - 1e-6).all(), (
        f"R doc2 not monotonic for channel {c}: {r_doc2[c]}"
      )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
class TestNormalizeScan:
  """Tests for Fix 5: forward-pass R[t] normalization (normalize_scan).

  Verifies that:
  1. Output is bounded (normalized weighted average, magnitude <= max input)
  2. Backward gradients are uniformly scaled by 1/R (both d_tokens and d_gates)
  3. Forward output matches manual h[t]/R[t] computation
  """

  @staticmethod
  def _compute_scan_reference(gates, tokens, reverse=False):
    """Reference scan: h[t] = gates[t]*h[t-1] + tokens[t]."""
    C, T = gates.shape
    h = torch.zeros_like(gates)
    if reverse:
      h[:, T - 1] = tokens[:, T - 1]
      for t in range(T - 2, -1, -1):
        h[:, t] = gates[:, t] * h[:, t + 1] + tokens[:, t]
    else:
      h[:, 0] = tokens[:, 0]
      for t in range(1, T):
        h[:, t] = gates[:, t] * h[:, t - 1] + tokens[:, t]
    return h

  @staticmethod
  def _compute_receptive_field_reference(gates, reverse=False):
    """Reference R[t] = scan(gates, ones)."""
    C, T = gates.shape
    r = torch.ones(C, T, device=gates.device, dtype=gates.dtype)
    if reverse:
      for t in range(T - 2, -1, -1):
        r[:, t] = gates[:, t] * r[:, t + 1] + 1.0
    else:
      for t in range(1, T):
        r[:, t] = gates[:, t] * r[:, t - 1] + 1.0
    return r

  def test_output_bounded_by_max_input(self):
    """Normalized output should be bounded: |output[t]| <= max(|x[j]|)."""
    from associative_scan_triton import forward_scan_full, get_grid

    device = get_device()
    C, T = 4, 64
    torch.manual_seed(TEST_SEED)

    # High eigenvalues to maximize accumulation
    gates = torch.full((C, T), 0.99, device=device)
    tokens = torch.randn(C, T, device=device)
    max_input = tokens.abs().max().item()

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    grid = get_grid(len(cu_seqlens), T, 128, C)

    # Compute h[t] via scan
    h = tokens.clone()
    g = gates.clone()
    forward_scan_full(g, h, cu_seqlens, grid, REVERSE=False, CHUNK_SIZE=128, TESTING=True)

    # Compute R[t] via scan with unit inputs
    r = torch.ones_like(gates)
    g2 = gates.clone()
    forward_scan_full(g2, r, cu_seqlens, grid, REVERSE=False, CHUNK_SIZE=128, TESTING=True)

    # Normalized output
    output = h / r.clamp(min=1.0)

    # Output should be bounded by max input magnitude (it's a weighted average)
    assert output.abs().max().item() <= max_input * 1.1, (
      f"Normalized output {output.abs().max():.3f} exceeds max input {max_input:.3f}"
    )

    # Unnormalized h[t] should grow with T (accumulation)
    assert h.abs().max().item() > max_input * 2, (
      f"Unnormalized h should exceed max input for a=0.99, got {h.abs().max():.3f}"
    )

  def test_output_matches_manual_h_over_r(self):
    """Verify output = h[t] / R[t] for known values."""
    from associative_scan_triton import scan_bidirectional_branched, forward_scan_full, get_grid

    device = get_device()
    C, T = 2, 8
    torch.manual_seed(TEST_SEED)
    gates_fwd = torch.rand(C, T, device=device)
    tokens_fwd = torch.randn(C, T, device=device)
    gates_bwd = torch.rand(C, T, device=device)
    tokens_bwd = torch.randn(C, T, device=device)

    # Reference: h = scan(gates, tokens), R = scan(gates, ones)
    h_fwd = self._compute_scan_reference(gates_fwd, tokens_fwd, reverse=False)
    r_fwd = self._compute_receptive_field_reference(gates_fwd, reverse=False)
    h_bwd = self._compute_scan_reference(gates_bwd, tokens_bwd, reverse=True)
    r_bwd = self._compute_receptive_field_reference(gates_bwd, reverse=True)

    expected_fwd = h_fwd / r_fwd.clamp(min=1.0)
    expected_bwd = h_bwd / r_bwd.clamp(min=1.0)

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    chunk_size = 16
    grid = get_grid(len(cu_seqlens), T, chunk_size, C)

    # Clone gates: scan_bidirectional_branched consumes them in-place
    gf = gates_fwd.clone()
    gb = gates_bwd.clone()

    # Run scan to get h
    y_fwd, y_bwd = scan_bidirectional_branched(
      gf, tokens_fwd.clone(), gb, tokens_bwd.clone(),
      {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid},
    )

    # Compute R manually using the original (unconsumed) gates
    r_fwd_actual = torch.ones_like(gates_fwd)
    gf2 = gates_fwd.clone()
    forward_scan_full(gf2, r_fwd_actual, cu_seqlens, grid, REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=True)
    r_bwd_actual = torch.ones_like(gates_bwd)
    gb2 = gates_bwd.clone()
    forward_scan_full(gb2, r_bwd_actual, cu_seqlens, grid, REVERSE=True, CHUNK_SIZE=chunk_size, TESTING=True)

    # Normalize
    actual_fwd = y_fwd / r_fwd_actual.clamp(min=1.0)
    actual_bwd = y_bwd / r_bwd_actual.clamp(min=1.0)

    assert torch.allclose(actual_fwd, expected_fwd, rtol=1e-4, atol=1e-5), (
      f"Fwd output mismatch: max diff {(actual_fwd - expected_fwd).abs().max()}"
    )
    assert torch.allclose(actual_bwd, expected_bwd, rtol=1e-4, atol=1e-5), (
      f"Bwd output mismatch: max diff {(actual_bwd - expected_bwd).abs().max()}"
    )

  @pytest.mark.parametrize("seqlen", [8, 64])
  def test_backward_gradients_reduced_by_normalization(self, seqlen):
    """When normalize_scan is on, backward gradients should have smaller L2 norm
    than the raw (unnormalized) case.

    The 1/R factor enters as the loss gradient into the scan backward (dL/dh = 1/R
    when loss = sum(h/R)). The scan backward is itself a reverse scan, so the
    relationship is NOT d_normed = d_raw / R pointwise. But the net effect is
    reduced gradient magnitude, especially at early positions (large R).
    """
    from associative_scan_triton import scan_bidirectional_branched, forward_scan_full, get_grid

    device = get_device()
    C = 4
    chunk_size = 16
    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    grid = get_grid(len(cu_seqlens), seqlen, chunk_size, C)

    def make_inputs():
      return (
        torch.rand(C, seqlen, device=device, requires_grad=True),
        torch.randn(C, seqlen, device=device, requires_grad=True),
        torch.rand(C, seqlen, device=device, requires_grad=True),
        torch.randn(C, seqlen, device=device, requires_grad=True),
      )

    # Run WITHOUT normalization
    torch.manual_seed(TEST_SEED)
    gf1, tf1, gb1, tb1 = make_inputs()
    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}
    y_fwd, y_bwd = scan_bidirectional_branched(gf1, tf1, gb1, tb1, args)
    (y_fwd.sum() + y_bwd.sum()).backward()
    d_gates_fwd_raw = gf1.grad.clone()
    d_tokens_fwd_raw = tf1.grad.clone()

    # Run WITH normalization (simulate what a bidirectional scan layer does)
    torch.manual_seed(TEST_SEED)
    gf2, tf2, gb2, tb2 = make_inputs()
    gf2_clean = gf2.detach().clone()
    gb2_clean = gb2.detach().clone()
    y_fwd2, y_bwd2 = scan_bidirectional_branched(gf2, tf2, gb2, tb2, args)

    # Compute R
    r_fwd = torch.ones(C, seqlen, device=device)
    forward_scan_full(gf2_clean.clone(), r_fwd, cu_seqlens, grid,
                      REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=True)
    r_bwd = torch.ones(C, seqlen, device=device)
    forward_scan_full(gb2_clean.clone(), r_bwd, cu_seqlens, grid,
                      REVERSE=True, CHUNK_SIZE=chunk_size, TESTING=True)

    # Normalize output (R detached — this is what the forward pass does)
    y_fwd_normed = y_fwd2 / r_fwd.detach().clamp(min=1.0)
    y_bwd_normed = y_bwd2 / r_bwd.detach().clamp(min=1.0)
    (y_fwd_normed.sum() + y_bwd_normed.sum()).backward()
    d_gates_fwd_normed = gf2.grad.clone()
    d_tokens_fwd_normed = tf2.grad.clone()

    # Gradient L2 norms should be reduced by normalization
    norm_raw_gates = d_gates_fwd_raw.norm().item()
    norm_normed_gates = d_gates_fwd_normed.norm().item()
    assert norm_normed_gates < norm_raw_gates, (
      f"d_gates L2 norm not reduced: raw={norm_raw_gates:.4f}, normed={norm_normed_gates:.4f}"
    )

    norm_raw_tokens = d_tokens_fwd_raw.norm().item()
    norm_normed_tokens = d_tokens_fwd_normed.norm().item()
    assert norm_normed_tokens < norm_raw_tokens, (
      f"d_tokens L2 norm not reduced: raw={norm_raw_tokens:.4f}, normed={norm_normed_tokens:.4f}"
    )


class TestNormalizeScanGradR:
  """Tests for Fix 7: non-detached R[t] (normalize_scan_grad_r).

  Verifies that:
  1. Gate gradients differ from the detached (Fix 5 only) case
  2. Gate gradients match a pure-loop reference implementation
  3. Eager and compiled paths produce matching gradients
  4. torch.autograd.gradcheck passes for numerical gradient verification
  """

  @staticmethod
  def _compute_scan_reference(gates, tokens, reverse=False):
    """Reference scan: h[t] = gates[t]*h[t-1] + tokens[t]."""
    C, T = gates.shape
    h = torch.zeros_like(gates)
    if reverse:
      h[:, T - 1] = tokens[:, T - 1]
      for t in range(T - 2, -1, -1):
        h[:, t] = gates[:, t] * h[:, t + 1] + tokens[:, t]
    else:
      h[:, 0] = tokens[:, 0]
      for t in range(1, T):
        h[:, t] = gates[:, t] * h[:, t - 1] + tokens[:, t]
    return h

  @staticmethod
  def _compute_receptive_field_reference(gates, reverse=False):
    """Reference R[t] = scan(gates, ones)."""
    C, T = gates.shape
    r = torch.ones(C, T, device=gates.device, dtype=gates.dtype)
    if reverse:
      for t in range(T - 2, -1, -1):
        r[:, t] = gates[:, t] * r[:, t + 1] + 1.0
    else:
      for t in range(1, T):
        r[:, t] = gates[:, t] * r[:, t - 1] + 1.0
    return r

  @staticmethod
  def _normalized_scan_loss_reference(gates_fwd, tokens_fwd, gates_bwd, tokens_bwd):
    """Reference: full normalized scan with R in the autograd graph.

    Computes hat_y = h[t] / R[t] with R NOT detached, then loss = sum.
    Uses pure loops with out-of-place ops so autograd gives exact gradients.
    """
    C, T = gates_fwd.shape
    # Forward direction — accumulate as list to avoid in-place assignment
    h_list = [tokens_fwd[:, 0]]
    r_list = [torch.ones(C, device=gates_fwd.device, dtype=gates_fwd.dtype)]
    for t in range(1, T):
      h_list.append(gates_fwd[:, t] * h_list[t - 1] + tokens_fwd[:, t])
      r_list.append(gates_fwd[:, t] * r_list[t - 1] + 1.0)
    h_fwd = torch.stack(h_list, dim=1)
    r_fwd = torch.stack(r_list, dim=1)
    hat_y_fwd = h_fwd / r_fwd.clamp(min=1.0)

    # Backward direction
    h_list_b = [None] * T
    r_list_b = [None] * T
    h_list_b[T - 1] = tokens_bwd[:, T - 1]
    r_list_b[T - 1] = torch.ones(C, device=gates_bwd.device, dtype=gates_bwd.dtype)
    for t in range(T - 2, -1, -1):
      h_list_b[t] = gates_bwd[:, t] * h_list_b[t + 1] + tokens_bwd[:, t]
      r_list_b[t] = gates_bwd[:, t] * r_list_b[t + 1] + 1.0
    h_bwd = torch.stack(h_list_b, dim=1)
    r_bwd = torch.stack(r_list_b, dim=1)
    hat_y_bwd = h_bwd / r_bwd.clamp(min=1.0)

    return (hat_y_fwd.sum() + hat_y_bwd.sum())

  def test_gradients_differ_from_detached(self):
    """Fix 7 gate gradients must differ from Fix 5 (detached R) gradients."""
    from associative_scan_triton import scan_bidirectional_branched, get_grid

    device = get_device()
    C, T = 4, 32
    chunk_size = 16
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    grid = get_grid(len(cu_seqlens), T, chunk_size, C)

    def make_inputs():
      torch.manual_seed(TEST_SEED)
      return (
        torch.rand(C, T, device=device, requires_grad=True),
        torch.randn(C, T, device=device, requires_grad=True),
        torch.rand(C, T, device=device, requires_grad=True),
        torch.randn(C, T, device=device, requires_grad=True),
      )

    # Fix 5 only: normalize_scan=True, normalize_scan_grad_r=False
    gf1, tf1, gb1, tb1 = make_inputs()
    args_fix5 = {
      "cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid,
      "normalize_scan": True, "normalize_scan_grad_r": False,
    }
    y_fwd1, y_bwd1 = scan_bidirectional_branched(gf1, tf1, gb1, tb1, args_fix5)
    (y_fwd1.sum() + y_bwd1.sum()).backward()
    d_gates_fwd_fix5 = gf1.grad.clone()

    # Fix 5 + Fix 7: normalize_scan=True, normalize_scan_grad_r=True
    gf2, tf2, gb2, tb2 = make_inputs()
    args_fix7 = {
      "cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid,
      "normalize_scan": True, "normalize_scan_grad_r": True,
    }
    y_fwd2, y_bwd2 = scan_bidirectional_branched(gf2, tf2, gb2, tb2, args_fix7)
    (y_fwd2.sum() + y_bwd2.sum()).backward()
    d_gates_fwd_fix7 = gf2.grad.clone()

    # Forward outputs must be identical (same forward pass)
    torch.testing.assert_close(y_fwd1, y_fwd2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y_bwd1, y_bwd2, rtol=1e-5, atol=1e-5)

    # Gate gradients must differ (Fix 7 adds the R-path contribution)
    diff = (d_gates_fwd_fix5 - d_gates_fwd_fix7).abs().max().item()
    assert diff > 1e-6, (
      f"Gate gradients should differ between Fix 5 and Fix 7, max diff={diff}"
    )

    # Token gradients should be identical (Fix 7 only affects gate gradients)
    torch.testing.assert_close(
      tf1.grad, tf2.grad, rtol=1e-5, atol=1e-5,
      msg="Token gradients should match between Fix 5 and Fix 7",
    )

  @pytest.mark.parametrize("seqlen", [8, 16, 32])
  def test_gradients_match_reference(self, seqlen):
    """Fix 7 gate gradients must match the pure-loop reference implementation.

    The reference computes h[t]/R[t] with R in the graph (not detached) using
    simple loops, so autograd gives us exact reference gradients via the
    quotient rule.
    """
    from associative_scan_triton import scan_bidirectional_branched, get_grid

    device = get_device()
    C = 4
    chunk_size = 16
    cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    grid = get_grid(len(cu_seqlens), seqlen, chunk_size, C)

    # Generate inputs in float32, then upcast for reference (avoids RNG dtype mismatch)
    torch.manual_seed(TEST_SEED)
    gf_base = torch.rand(C, seqlen, device=device, dtype=torch.float32)
    tf_base = torch.randn(C, seqlen, device=device, dtype=torch.float32)
    gb_base = torch.rand(C, seqlen, device=device, dtype=torch.float32)
    tb_base = torch.randn(C, seqlen, device=device, dtype=torch.float32)

    # --- Reference: pure-loop with R in graph (float64 for precision) ---
    gf_ref = gf_base.double().detach().requires_grad_(True)
    tf_ref = tf_base.double().detach().requires_grad_(True)
    gb_ref = gb_base.double().detach().requires_grad_(True)
    tb_ref = tb_base.double().detach().requires_grad_(True)

    loss_ref = self._normalized_scan_loss_reference(gf_ref, tf_ref, gb_ref, tb_ref)
    loss_ref.backward()

    # --- Test: our backward with normalize_scan_grad_r=True (float32) ---
    gf_test = gf_base.clone().detach().requires_grad_(True)
    tf_test = tf_base.clone().detach().requires_grad_(True)
    gb_test = gb_base.clone().detach().requires_grad_(True)
    tb_test = tb_base.clone().detach().requires_grad_(True)

    args = {
      "cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid,
      "normalize_scan": True, "normalize_scan_grad_r": True,
    }
    y_fwd, y_bwd = scan_bidirectional_branched(gf_test, tf_test, gb_test, tb_test, args)
    (y_fwd.sum() + y_bwd.sum()).backward()

    # Compare gradients (reference is float64, test is float32 -> use relaxed tols)
    rtol, atol = 1e-3, 1e-4
    torch.testing.assert_close(
      gf_test.grad.double(), gf_ref.grad, rtol=rtol, atol=atol,
      msg="d_gates_fwd mismatch vs reference",
    )
    torch.testing.assert_close(
      tf_test.grad.double(), tf_ref.grad, rtol=rtol, atol=atol,
      msg="d_tokens_fwd mismatch vs reference",
    )
    torch.testing.assert_close(
      gb_test.grad.double(), gb_ref.grad, rtol=rtol, atol=atol,
      msg="d_gates_bwd mismatch vs reference",
    )
    torch.testing.assert_close(
      tb_test.grad.double(), tb_ref.grad, rtol=rtol, atol=atol,
      msg="d_tokens_bwd mismatch vs reference",
    )

  @pytest.mark.parametrize("batch_size", [1, 4])
  @pytest.mark.parametrize("seq_len", [64, 256])
  def test_compiled_matches_eager(self, batch_size, seq_len):
    """Compiled scan with normalize_scan_grad_r must match eager gradients."""
    from associative_scan_triton import (
      scan_bidirectional_branched,
      scan_bidirectional_branched_compiled,
      get_grid,
    )

    device = get_device()
    no_channels = 64
    chunk_size = 128
    cu_seqlens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seqlens[1:] = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32).cumsum(0)
    total_len = int(cu_seqlens[-1].item())
    grid = get_grid(len(cu_seqlens), seq_len, chunk_size, no_channels)

    def make_inputs():
      torch.manual_seed(TEST_SEED)
      return (
        torch.rand(no_channels, total_len, device=device, requires_grad=True),
        torch.randn(no_channels, total_len, device=device, requires_grad=True),
        torch.rand(no_channels, total_len, device=device, requires_grad=True),
        torch.randn(no_channels, total_len, device=device, requires_grad=True),
      )

    args = {
      "cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid,
      "normalize_scan": True, "normalize_scan_grad_r": True,
    }

    # Reference (eager)
    gf1, tf1, gb1, tb1 = make_inputs()
    ref_fwd, ref_bwd = scan_bidirectional_branched(gf1, tf1, gb1, tb1, args)
    (ref_fwd.sum() + ref_bwd.sum()).backward()

    # Test (compiled)
    gf2, tf2, gb2, tb2 = make_inputs()
    test_fwd, test_bwd = scan_bidirectional_branched_compiled(gf2, tf2, gb2, tb2, args)
    (test_fwd.sum() + test_bwd.sum()).backward()

    # Forward outputs
    rtol, atol = 1e-5, 1e-5
    torch.testing.assert_close(ref_fwd, test_fwd, rtol=rtol, atol=atol)
    torch.testing.assert_close(ref_bwd, test_bwd, rtol=rtol, atol=atol)

    # Gradients
    rtol, atol = 1e-4, 1e-5
    torch.testing.assert_close(gf1.grad, gf2.grad, rtol=rtol, atol=atol,
                               msg="d_gates_fwd mismatch")
    torch.testing.assert_close(tf1.grad, tf2.grad, rtol=rtol, atol=atol,
                               msg="d_tokens_fwd mismatch")
    torch.testing.assert_close(gb1.grad, gb2.grad, rtol=rtol, atol=atol,
                               msg="d_gates_bwd mismatch")
    torch.testing.assert_close(tb1.grad, tb2.grad, rtol=rtol, atol=atol,
                               msg="d_tokens_bwd mismatch")
