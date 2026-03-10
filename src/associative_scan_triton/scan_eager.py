"""Eager autograd scan implementations (not torch.compile-safe).

Contains:
  - ScanCausal: unidirectional (causal) scan with autograd
  - ScanBidirectionalBranched: bidirectional scan with separate fwd/bwd gates
"""

import torch

from associative_scan_triton._dispatcher import backward_scan_fused_full, forward_scan_full


# ============================================================
# Causal (unidirectional) scan
# ============================================================


class ScanCausal(torch.autograd.Function):
  """Causal (unidirectional) associative scan with autograd."""

  @staticmethod
  def forward(ctx, gates, tokens, args):
    shape = gates.shape
    assert tokens.shape == shape
    assert gates.is_contiguous()
    assert tokens.is_contiguous()
    cu_seqlens, chunk_size, grid = (
      args["cu_seqlens"], args["chunk_size"], args["grid"],
    )
    need_backward = gates.requires_grad or tokens.requires_grad

    # Non-in-place: gates and tokens are READ-ONLY
    out_tokens = torch.empty_like(tokens)
    forward_scan_full(
      gates, tokens, cu_seqlens, grid,
      REVERSE=False, CHUNK_SIZE=chunk_size,
      TESTING=False, tokens_out=out_tokens,
    )

    if need_backward:
      ctx.grid = grid
      ctx.cu_seqlens = cu_seqlens
      ctx.chunk_size = chunk_size
      # gates is unmodified — save directly, NO CLONE
      ctx.save_for_backward(out_tokens, gates)
    return out_tokens

  @staticmethod
  def backward(ctx, grad_output):
    grid = ctx.grid
    states, gates = ctx.saved_tensors
    cu_seqlens = ctx.cu_seqlens
    chunk_size = ctx.chunk_size
    d_tokens = torch.empty_like(grad_output)
    d_gates = torch.empty_like(gates)
    backward_scan_fused_full(
      grad_output.contiguous(), gates, states, d_tokens, d_gates,
      cu_seqlens, grid, CHUNK_SIZE=chunk_size, CAUSAL=True,
    )
    return d_gates, d_tokens, None


def scan_causal(
  gates: torch.Tensor,
  tokens: torch.Tensor,
  args: dict,
) -> torch.Tensor:
  """Causal (unidirectional) associative scan.

  Args:
    gates: [C, B*L] gate values
    tokens: [C, B*L] token values
    args: dict with keys:
      - "cu_seqlens": [B+1] cumulative sequence lengths
      - "chunk_size": int, scan chunk size
      - "grid": tuple (num_seq, num_chunks, no_channels)

  Returns:
    scanned tokens: [C, B*L]
  """
  return ScanCausal.apply(gates, tokens, args)


# ============================================================
# Bidirectional branched scan
# ============================================================


class ScanBidirectionalBranched(torch.autograd.Function):
  """Bidirectional variant that runs two branches."""

  @staticmethod
  def forward(
    ctx, gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args, testing=False
  ):
    shape = gates_fwd.shape
    assert tokens_fwd.shape == shape
    assert tokens_bwd.shape == shape
    assert gates_fwd.is_contiguous()
    assert gates_bwd.is_contiguous()
    assert tokens_fwd.is_contiguous()
    assert tokens_bwd.is_contiguous()

    cu_seqlens, chunk_size, grid = (
      args["cu_seqlens"],
      args["chunk_size"],
      args["grid"],
    )
    need_backward = (
      gates_fwd.requires_grad or tokens_fwd.requires_grad
      or gates_bwd.requires_grad or tokens_bwd.requires_grad
    )

    # Non-in-place: all inputs are READ-ONLY
    out_tokens_fwd = torch.empty_like(tokens_fwd)
    out_tokens_bwd = torch.empty_like(tokens_bwd)

    forward_scan_full(
      gates_fwd, tokens_fwd, cu_seqlens, grid,
      REVERSE=False, CHUNK_SIZE=chunk_size,
      TESTING=False, tokens_out=out_tokens_fwd,
    )
    forward_scan_full(
      gates_bwd, tokens_bwd, cu_seqlens, grid,
      REVERSE=True, CHUNK_SIZE=chunk_size,
      TESTING=False, tokens_out=out_tokens_bwd,
    )

    if need_backward:
      ctx.grid = grid
      ctx.cu_seqlens = cu_seqlens
      ctx.chunk_size = chunk_size
      ctx.testing = testing
      # gates are unmodified — save directly, NO CLONE
      ctx.save_for_backward(
        out_tokens_fwd, out_tokens_bwd, gates_fwd, gates_bwd
      )
    return out_tokens_fwd, out_tokens_bwd

  @staticmethod
  def backward(ctx, grad_output_fwd, grad_output_bwd):
    grid = ctx.grid
    states_fwd, states_bwd, gates_fwd, gates_bwd = ctx.saved_tensors
    cu_seqlens = ctx.cu_seqlens
    chunk_size = ctx.chunk_size

    d_tokens_fwd = torch.empty_like(grad_output_fwd)
    d_gates_fwd = torch.empty_like(gates_fwd)
    d_tokens_bwd = torch.empty_like(grad_output_bwd)
    d_gates_bwd = torch.empty_like(gates_bwd)

    backward_scan_fused_full(
      grad_output_fwd.contiguous(), gates_fwd, states_fwd,
      d_tokens_fwd, d_gates_fwd,
      cu_seqlens, grid, CHUNK_SIZE=chunk_size, CAUSAL=True,
    )
    backward_scan_fused_full(
      grad_output_bwd.contiguous(), gates_bwd, states_bwd,
      d_tokens_bwd, d_gates_bwd,
      cu_seqlens, grid, CHUNK_SIZE=chunk_size, CAUSAL=False,
    )

    return (
      d_gates_fwd,
      d_tokens_fwd,
      d_gates_bwd,
      d_tokens_bwd,
      None,  # args
      None,  # testing
    )


def scan_bidirectional_branched(
  gates_fwd: torch.Tensor,
  tokens_fwd: torch.Tensor,
  gates_bwd: torch.Tensor,
  tokens_bwd: torch.Tensor,
  args: dict,
  testing: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Bidirectional branched associative scan.

  Runs two scan branches in opposite directions with separate gate/token pairs.

  Args:
    gates_fwd: [C, B*L] forward direction gates
    tokens_fwd: [C, B*L] forward direction tokens
    gates_bwd: [C, B*L] backward direction gates
    tokens_bwd: [C, B*L] backward direction tokens
    args: dict with keys:
      - "cu_seqlens": [B+1] cumulative sequence lengths
      - "chunk_size": int, scan chunk size
      - "grid": tuple (num_seq, num_chunks, no_channels)
    testing: if True, write gate values during scan (needed for gradient checks)

  Returns:
    (scanned_tokens_fwd, scanned_tokens_bwd)
  """
  return ScanBidirectionalBranched.apply(
    gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args, testing
  )
