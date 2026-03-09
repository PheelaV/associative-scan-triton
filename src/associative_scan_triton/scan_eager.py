"""Eager autograd scan implementations (not torch.compile-safe).

Contains:
  - ScanCausal: unidirectional (causal) scan with autograd
  - ScanBidirectionalBranched: bidirectional scan with separate fwd/bwd gates
"""

import torch

from associative_scan_triton._dispatcher import forward_scan_full
from associative_scan_triton._shift_pad import shift_pad


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
    gates_work = gates.detach().clone()
    save_tokens = tokens.detach().clone()
    if need_backward:
      save_gates = gates.detach().clone()
    forward_scan_full(
      gates_work, save_tokens, cu_seqlens, grid,
      REVERSE=False, CHUNK_SIZE=chunk_size,
    )
    if need_backward:
      ctx.grid = grid
      ctx.cu_seqlens = cu_seqlens
      ctx.chunk_size = chunk_size
      ctx.save_for_backward(save_tokens, save_gates)
    return save_tokens

  @staticmethod
  def backward(ctx, grad_output):
    grid = ctx.grid
    states, gates = ctx.saved_tensors
    assert gates.is_contiguous()
    assert states.is_contiguous()
    cu_seqlens = ctx.cu_seqlens
    chunk_size = ctx.chunk_size
    d_tokens = grad_output.clone().detach()
    gates_shift_up = shift_pad(
      data=gates, cu_seqlens=cu_seqlens, backward=False, pad_value=1
    )
    forward_scan_full(
      gates_shift_up, d_tokens, cu_seqlens, grid,
      REVERSE=True, CHUNK_SIZE=chunk_size,
    )
    states_shift_down = shift_pad(
      data=states, cu_seqlens=cu_seqlens, backward=True, pad_value=0
    )
    d_gates = states_shift_down * d_tokens
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
    scan_grad_scale = args.get("scan_grad_scale", False)
    normalize_scan_grad_r = args.get("normalize_scan_grad_r", False)
    need_backward = (
      gates_fwd.requires_grad or tokens_fwd.requires_grad
      or gates_bwd.requires_grad or tokens_bwd.requires_grad
    )

    gates_fwd = gates_fwd.detach()
    save_tokens_fwd = tokens_fwd.detach()
    gates_bwd = gates_bwd.detach()
    save_tokens_bwd = tokens_bwd.detach()

    if need_backward:
      save_gates_fwd = gates_fwd.clone()
      save_gates_bwd = gates_bwd.clone()

    forward_scan_full(
      gates_fwd,
      save_tokens_fwd,
      cu_seqlens,
      grid,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      TESTING=testing,
    )
    forward_scan_full(
      gates_bwd,
      save_tokens_bwd,
      cu_seqlens,
      grid,
      REVERSE=True,
      CHUNK_SIZE=chunk_size,
      TESTING=testing,
    )

    if need_backward:
      ctx.grid = grid
      ctx.cu_seqlens = cu_seqlens
      ctx.chunk_size = chunk_size
      ctx.testing = testing
      ctx.scan_grad_scale = scan_grad_scale
      ctx.normalize_scan_grad_r = normalize_scan_grad_r

      if normalize_scan_grad_r:
        r_fwd = args["_r_fwd"]
        r_bwd = args["_r_bwd"]
        ctx.save_for_backward(
          save_tokens_fwd, save_tokens_bwd, save_gates_fwd, save_gates_bwd,
          r_fwd, r_bwd,
        )
      else:
        ctx.save_for_backward(
          save_tokens_fwd, save_tokens_bwd, save_gates_fwd, save_gates_bwd
        )
    return save_tokens_fwd, save_tokens_bwd

  @staticmethod
  def backward(ctx, grad_output_fwd, grad_output_bwd):
    grid = ctx.grid
    if ctx.normalize_scan_grad_r:
      states_fwd, states_bwd, gates_fwd, gates_bwd, r_fwd, r_bwd = (
        ctx.saved_tensors
      )
    else:
      states_fwd, states_bwd, gates_fwd, gates_bwd = ctx.saved_tensors
    assert gates_fwd.is_contiguous()
    assert gates_bwd.is_contiguous()
    assert states_fwd.is_contiguous()
    assert states_bwd.is_contiguous()
    cu_seqlens = ctx.cu_seqlens
    chunk_size = ctx.chunk_size
    testing = ctx.testing

    d_tokens_fwd = grad_output_fwd.clone()
    d_tokens_bwd = grad_output_bwd.clone()
    gates_shift_up_fwd = shift_pad(
      data=gates_fwd, cu_seqlens=cu_seqlens, backward=False, pad_value=1
    )
    gates_shift_down_bwd = shift_pad(
      data=gates_bwd, cu_seqlens=cu_seqlens, backward=True, pad_value=1
    )
    forward_scan_full(
      gates_shift_up_fwd,
      d_tokens_fwd,
      cu_seqlens,
      grid,
      REVERSE=True,
      CHUNK_SIZE=chunk_size,
      TESTING=testing,
    )
    forward_scan_full(
      gates_shift_down_bwd,
      d_tokens_bwd,
      cu_seqlens,
      grid,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      TESTING=testing,
    )
    states_shift_down_fwd = shift_pad(
      data=states_fwd, cu_seqlens=cu_seqlens, backward=True, pad_value=0
    )
    states_shift_up_bwd = shift_pad(
      data=states_bwd, cu_seqlens=cu_seqlens, backward=False, pad_value=0
    )
    d_gates_fwd = states_shift_down_fwd * d_tokens_fwd
    d_gates_bwd = states_shift_up_bwd * d_tokens_bwd

    if ctx.scan_grad_scale:
      gates_r_fwd = shift_pad(
        data=gates_fwd, cu_seqlens=cu_seqlens, backward=False, pad_value=1
      )
      gates_r_bwd = shift_pad(
        data=gates_bwd, cu_seqlens=cu_seqlens, backward=True, pad_value=1
      )
      r_fwd_fix3 = torch.ones_like(gates_fwd)
      r_bwd_fix3 = torch.ones_like(gates_bwd)
      forward_scan_full(
        gates_r_fwd, r_fwd_fix3, cu_seqlens, grid,
        REVERSE=True, CHUNK_SIZE=chunk_size, TESTING=testing,
      )
      forward_scan_full(
        gates_r_bwd, r_bwd_fix3, cu_seqlens, grid,
        REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=testing,
      )
      d_gates_fwd = d_gates_fwd / r_fwd_fix3.clamp(min=1.0)
      d_gates_bwd = d_gates_bwd / r_bwd_fix3.clamp(min=1.0)

    if ctx.normalize_scan_grad_r:
      r_fwd_c = r_fwd.clamp(min=1.0)
      r_bwd_c = r_bwd.clamp(min=1.0)

      d_r_seed_fwd = -grad_output_fwd * states_fwd / r_fwd_c
      d_r_seed_bwd = -grad_output_bwd * states_bwd / r_bwd_c

      gates_r_shift_fwd = shift_pad(
        data=gates_fwd, cu_seqlens=cu_seqlens, backward=False, pad_value=1
      )
      d_r_fwd = d_r_seed_fwd.clone()
      forward_scan_full(
        gates_r_shift_fwd, d_r_fwd, cu_seqlens, grid,
        REVERSE=True, CHUNK_SIZE=chunk_size, TESTING=testing,
      )

      gates_r_shift_bwd = shift_pad(
        data=gates_bwd, cu_seqlens=cu_seqlens, backward=True, pad_value=1
      )
      d_r_bwd = d_r_seed_bwd.clone()
      forward_scan_full(
        gates_r_shift_bwd, d_r_bwd, cu_seqlens, grid,
        REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=testing,
      )

      r_shift_down_fwd = shift_pad(
        data=r_fwd, cu_seqlens=cu_seqlens, backward=True, pad_value=0
      )
      r_shift_up_bwd = shift_pad(
        data=r_bwd, cu_seqlens=cu_seqlens, backward=False, pad_value=0
      )
      d_gates_fwd = d_gates_fwd + r_shift_down_fwd * d_r_fwd
      d_gates_bwd = d_gates_bwd + r_shift_up_bwd * d_r_bwd

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
      - "normalize_scan": bool (optional), enable output normalization
      - "normalize_scan_grad_r": bool (optional), enable R-gradient
      - "scan_grad_scale": bool (optional), enable gradient scaling
      - "xlstm_normalizer": bool (optional), use xLSTM normalizer
    testing: if True, write gate values during scan (needed for gradient checks)

  Returns:
    (scanned_tokens_fwd, scanned_tokens_bwd)
  """
  normalize_scan = args.get("normalize_scan", False)
  normalize_scan_grad_r = args.get("normalize_scan_grad_r", False)
  xlstm_normalizer = args.get("xlstm_normalizer", False)

  if normalize_scan:
    gates_fwd_clean = gates_fwd.clone()
    gates_bwd_clean = gates_bwd.clone()

  if normalize_scan:
    cu_seqlens = args["cu_seqlens"]
    chunk_size = args["chunk_size"]
    grid = args["grid"]
    if xlstm_normalizer:
      r_fwd = (1.0 - gates_fwd_clean).clone()
      r_bwd = (1.0 - gates_bwd_clean).clone()
    else:
      r_fwd = torch.ones_like(gates_fwd_clean)
      r_bwd = torch.ones_like(gates_bwd_clean)
    forward_scan_full(
      gates_fwd_clean, r_fwd, cu_seqlens, grid,
      REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=testing,
    )
    forward_scan_full(
      gates_bwd_clean, r_bwd, cu_seqlens, grid,
      REVERSE=True, CHUNK_SIZE=chunk_size, TESTING=testing,
    )
    if normalize_scan_grad_r:
      args["_r_fwd"] = r_fwd
      args["_r_bwd"] = r_bwd

  y_fwd, y_bwd = ScanBidirectionalBranched.apply(
    gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args, testing
  )

  if normalize_scan:
    y_fwd = y_fwd / r_fwd.detach().clamp(min=1.0)
    y_bwd = y_bwd / r_bwd.detach().clamp(min=1.0)

  return y_fwd, y_bwd
