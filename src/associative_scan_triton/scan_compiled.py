"""triton_op wrappers for bidirectional scan — compile-compatible replacement.

Wraps the scan kernels with torch.library.triton_op so torch.compile can trace
through them.

Key changes from the eager path:
1. kernel[grid](...) -> wrap_triton(kernel)[grid](...)
2. shift_pad uses the Triton kernel instead of data-dependent indexing

IMPORTANT: All wrap_triton kernel calls are inlined directly into the top-level
triton_ops (scan_bidi_fwd_op / scan_bidi_bwd_op). We do NOT nest triton_ops —
calling a triton_op with mutates_args from inside another triton_op causes
"leaf Variable that requires grad in-place operation" errors during AOT autograd
tracing.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from associative_scan_triton._kernels import (
  forward_scan_chunked,
  forward_scan_onepass_pipelined,
)
from associative_scan_triton._shift_pad import shift_pad_compiled


# ============================================================
# Helper: inline scan (NOT a triton_op — just a plain function)
# ============================================================


def _run_scan_inplace(
  gates,
  tokens,
  cu_seqlens,
  num_chunks,
  no_channels,
  chunk_size,
  reverse,
):
  """Run the 1-kernel pipelined scan in-place on gates/tokens.

  This is a plain function (not a triton_op) that calls wrap_triton directly.
  It is meant to be called from within a triton_op body.
  """
  num_seq = cu_seqlens.shape[0] - 1
  if num_chunks == 1:
    grid = (num_seq, num_chunks, no_channels)
    wrap_triton(forward_scan_chunked)[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=reverse,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=False,
      TESTING=True,
    )
  else:
    grid_onepass = (num_seq, no_channels)
    wrap_triton(forward_scan_onepass_pipelined)[grid_onepass](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=reverse,
      CHUNK_SIZE=chunk_size,
      NUM_CHUNKS=num_chunks,
      TESTING=True,
      NUM_STAGES=3,
    )


# ============================================================
# ScanBidirectionalBranched as triton_op
# ============================================================


@triton_op("associative_scan_triton::scan_bidi_fwd", mutates_args={})
def scan_bidi_fwd_op(
  gates_fwd: torch.Tensor,
  tokens_fwd: torch.Tensor,
  gates_bwd: torch.Tensor,
  tokens_bwd: torch.Tensor,
  cu_seqlens: torch.Tensor,
  num_chunks: int,
  no_channels: int,
  chunk_size: int,
  scan_grad_scale: int = 0,
  normalize_scan_grad_r: int = 0,
  r_fwd_saved: torch.Tensor | None = None,
  r_bwd_saved: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Bidirectional branched scan forward (compile-compatible).

  Returns: (out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd)
  """
  save_gates_fwd = gates_fwd.clone()
  save_gates_bwd = gates_bwd.clone()

  gates_fwd_work = gates_fwd.clone()
  out_tokens_fwd = tokens_fwd.clone()
  gates_bwd_work = gates_bwd.clone()
  out_tokens_bwd = tokens_bwd.clone()

  _run_scan_inplace(
    gates_fwd_work, out_tokens_fwd, cu_seqlens,
    num_chunks, no_channels, chunk_size, reverse=False,
  )
  _run_scan_inplace(
    gates_bwd_work, out_tokens_bwd, cu_seqlens,
    num_chunks, no_channels, chunk_size, reverse=True,
  )

  return out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd


@triton_op("associative_scan_triton::scan_bidi_bwd", mutates_args={})
def scan_bidi_bwd_op(
  grad_fwd: torch.Tensor,
  grad_bwd: torch.Tensor,
  states_fwd: torch.Tensor,
  states_bwd: torch.Tensor,
  gates_fwd: torch.Tensor,
  gates_bwd: torch.Tensor,
  cu_seqlens: torch.Tensor,
  num_chunks: int,
  no_channels: int,
  chunk_size: int,
  scan_grad_scale: int = 0,
  normalize_scan_grad_r: int = 0,
  r_fwd_saved: torch.Tensor | None = None,
  r_bwd_saved: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Bidirectional branched scan backward (compile-compatible).

  Returns: (d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd)
  """
  d_tokens_fwd = grad_fwd.clone()
  d_tokens_bwd = grad_bwd.clone()

  gates_shift_up_fwd = shift_pad_compiled(
    gates_fwd, cu_seqlens, pad_value=1.0, backward=False,
  )
  gates_shift_down_bwd = shift_pad_compiled(
    gates_bwd, cu_seqlens, pad_value=1.0, backward=True,
  )

  _run_scan_inplace(
    gates_shift_up_fwd, d_tokens_fwd, cu_seqlens,
    num_chunks, no_channels, chunk_size, reverse=True,
  )
  _run_scan_inplace(
    gates_shift_down_bwd, d_tokens_bwd, cu_seqlens,
    num_chunks, no_channels, chunk_size, reverse=False,
  )

  states_shift_down_fwd = shift_pad_compiled(
    states_fwd, cu_seqlens, pad_value=0.0, backward=True,
  )
  states_shift_up_bwd = shift_pad_compiled(
    states_bwd, cu_seqlens, pad_value=0.0, backward=False,
  )

  d_gates_fwd = states_shift_down_fwd * d_tokens_fwd
  d_gates_bwd = states_shift_up_bwd * d_tokens_bwd

  if scan_grad_scale:
    gates_r_fwd = shift_pad_compiled(
      gates_fwd, cu_seqlens, pad_value=1.0, backward=False,
    )
    gates_r_bwd = shift_pad_compiled(
      gates_bwd, cu_seqlens, pad_value=1.0, backward=True,
    )
    r_fwd = torch.ones_like(gates_fwd)
    r_bwd = torch.ones_like(gates_bwd)
    _run_scan_inplace(
      gates_r_fwd, r_fwd, cu_seqlens, num_chunks,
      no_channels, chunk_size, reverse=True,
    )
    _run_scan_inplace(
      gates_r_bwd, r_bwd, cu_seqlens, num_chunks,
      no_channels, chunk_size, reverse=False,
    )
    d_gates_fwd = d_gates_fwd / r_fwd.clamp(min=1.0)
    d_gates_bwd = d_gates_bwd / r_bwd.clamp(min=1.0)

  if normalize_scan_grad_r and r_fwd_saved is not None:
    r_fwd_c = r_fwd_saved.clamp(min=1.0)
    r_bwd_c = r_bwd_saved.clamp(min=1.0)

    d_r_seed_fwd = -grad_fwd * states_fwd / r_fwd_c
    d_r_seed_bwd = -grad_bwd * states_bwd / r_bwd_c

    gates_r_shift_fwd = shift_pad_compiled(
      gates_fwd, cu_seqlens, pad_value=1.0, backward=False,
    )
    d_r_fwd = d_r_seed_fwd.clone()
    _run_scan_inplace(
      gates_r_shift_fwd, d_r_fwd, cu_seqlens, num_chunks,
      no_channels, chunk_size, reverse=True,
    )

    gates_r_shift_bwd = shift_pad_compiled(
      gates_bwd, cu_seqlens, pad_value=1.0, backward=True,
    )
    d_r_bwd = d_r_seed_bwd.clone()
    _run_scan_inplace(
      gates_r_shift_bwd, d_r_bwd, cu_seqlens, num_chunks,
      no_channels, chunk_size, reverse=False,
    )

    r_shift_down_fwd = shift_pad_compiled(
      r_fwd_saved, cu_seqlens, pad_value=0.0, backward=True,
    )
    r_shift_up_bwd = shift_pad_compiled(
      r_bwd_saved, cu_seqlens, pad_value=0.0, backward=False,
    )
    d_gates_fwd = d_gates_fwd + r_shift_down_fwd * d_r_fwd
    d_gates_bwd = d_gates_bwd + r_shift_up_bwd * d_r_bwd

  return d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd


# ============================================================
# Autograd registration
# ============================================================


def _scan_bidi_backward(
  ctx, grad_fwd, grad_bwd, _grad_gates_fwd, _grad_gates_bwd
):
  if ctx.normalize_scan_grad_r:
    states_fwd, states_bwd, gates_fwd, gates_bwd, r_fwd, r_bwd = (
      ctx.saved_tensors
    )
  else:
    states_fwd, states_bwd, gates_fwd, gates_bwd = ctx.saved_tensors
    r_fwd = r_bwd = None
  d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd = scan_bidi_bwd_op(
    grad_fwd, grad_bwd,
    states_fwd, states_bwd,
    gates_fwd, gates_bwd,
    ctx.cu_seqlens,
    ctx.num_chunks, ctx.no_channels, ctx.chunk_size,
    ctx.scan_grad_scale, ctx.normalize_scan_grad_r,
    r_fwd, r_bwd,
  )
  return (
    d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd,
    None, None, None, None, None, None, None, None,  # non-tensor args
  )


def _scan_bidi_setup_context(ctx, inputs, output):
  (
    gates_fwd, tokens_fwd, gates_bwd, tokens_bwd,
    cu_seqlens, num_chunks, no_channels, chunk_size,
    scan_grad_scale, normalize_scan_grad_r,
    r_fwd_saved, r_bwd_saved,
  ) = inputs
  out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd = output
  if normalize_scan_grad_r and r_fwd_saved is not None:
    ctx.save_for_backward(
      out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd,
      r_fwd_saved, r_bwd_saved,
    )
  else:
    ctx.save_for_backward(
      out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd
    )
  ctx.cu_seqlens = cu_seqlens
  ctx.num_chunks = num_chunks
  ctx.no_channels = no_channels
  ctx.chunk_size = chunk_size
  ctx.scan_grad_scale = scan_grad_scale
  ctx.normalize_scan_grad_r = normalize_scan_grad_r


scan_bidi_fwd_op.register_autograd(
  _scan_bidi_backward,
  setup_context=_scan_bidi_setup_context,
)


# ============================================================
# Receptive field computation (compile-compatible)
# ============================================================


@triton_op("associative_scan_triton::compute_receptive_field", mutates_args={})
def compute_receptive_field_op(
  gates_fwd: torch.Tensor,
  gates_bwd: torch.Tensor,
  cu_seqlens: torch.Tensor,
  num_chunks: int,
  no_channels: int,
  chunk_size: int,
  xlstm_normalizer: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compute receptive field R[t] for both scan directions (compile-compatible).

  No autograd needed (R is always detached before use).
  """
  if xlstm_normalizer:
    r_fwd = (1.0 - gates_fwd).clone()
  else:
    r_fwd = torch.ones_like(gates_fwd)
  gates_fwd_work = gates_fwd.clone()
  _run_scan_inplace(
    gates_fwd_work, r_fwd, cu_seqlens,
    num_chunks, no_channels, chunk_size, reverse=False,
  )

  if xlstm_normalizer:
    r_bwd = (1.0 - gates_bwd).clone()
  else:
    r_bwd = torch.ones_like(gates_bwd)
  gates_bwd_work = gates_bwd.clone()
  _run_scan_inplace(
    gates_bwd_work, r_bwd, cu_seqlens,
    num_chunks, no_channels, chunk_size, reverse=True,
  )

  return r_fwd, r_bwd


# ============================================================
# Public API (drop-in for scan_bidirectional_branched)
# ============================================================


def scan_bidirectional_branched_compiled(
  gates_fwd: torch.Tensor,
  tokens_fwd: torch.Tensor,
  gates_bwd: torch.Tensor,
  tokens_bwd: torch.Tensor,
  args: dict,
  testing: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Drop-in replacement for scan_bidirectional_branched() using triton_op."""
  cu_seqlens = args["cu_seqlens"]
  chunk_size = args["chunk_size"]
  grid = args["grid"]
  scan_grad_scale = int(args.get("scan_grad_scale", False))
  normalize_scan = int(args.get("normalize_scan", False))
  normalize_scan_grad_r = int(args.get("normalize_scan_grad_r", False))
  xlstm_normalizer = int(args.get("xlstm_normalizer", False))
  _num_seq, num_chunks, no_channels = grid

  r_fwd = r_bwd = None
  if normalize_scan:
    r_fwd, r_bwd = compute_receptive_field_op(
      gates_fwd, gates_bwd,
      cu_seqlens, num_chunks, no_channels, chunk_size,
      xlstm_normalizer,
    )

  out_tokens_fwd, out_tokens_bwd, _, _ = scan_bidi_fwd_op(
    gates_fwd, tokens_fwd, gates_bwd, tokens_bwd,
    cu_seqlens, num_chunks, no_channels, chunk_size,
    scan_grad_scale,
    normalize_scan_grad_r if normalize_scan else 0,
    r_fwd.detach() if r_fwd is not None else None,
    r_bwd.detach() if r_bwd is not None else None,
  )

  if normalize_scan:
    out_tokens_fwd = out_tokens_fwd / r_fwd.detach().clamp(min=1.0)
    out_tokens_bwd = out_tokens_bwd / r_bwd.detach().clamp(min=1.0)

  return out_tokens_fwd, out_tokens_bwd
