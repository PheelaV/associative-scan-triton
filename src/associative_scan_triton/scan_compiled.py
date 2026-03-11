"""triton_op wrappers for scan — compile-compatible replacements.

Wraps the scan kernels with torch.library.triton_op so torch.compile can trace
through them.

Key changes from the eager path:
1. kernel[grid](...) -> wrap_triton(kernel)[grid](...)
2. Non-in-place forward: gates/tokens read-only, output to separate buffer
3. grad_output.contiguous() in backward (handles broadcast grads from .sum())

IMPORTANT: All wrap_triton kernel calls are inlined directly into the top-level
triton_ops. We do NOT nest triton_ops — calling a triton_op with mutates_args
from inside another triton_op causes "leaf Variable that requires grad in-place
operation" errors during AOT autograd tracing.
"""

import torch
from torch.library import triton_op, wrap_triton

from associative_scan_triton._grid import get_num_stages
from associative_scan_triton._kernels import (
  backward_scan_fused,
  backward_scan_fused_single_chunk,
  forward_scan_chunked,
  forward_scan_onepass_pipelined,
)

# ============================================================
# Helper: inline scan (NOT a triton_op — just a plain function)
# ============================================================


def _run_scan(
  gates,
  tokens,
  cu_seqlens,
  num_chunks,
  no_channels,
  chunk_size,
  reverse,
  tokens_out=None,
) -> None:
  """Run the 1-kernel pipelined scan.

  If tokens_out is provided, writes to it (non-in-place).
  If tokens_out is None, writes back to tokens (in-place).

  This is a plain function (not a triton_op) that calls wrap_triton directly.
  It is meant to be called from within a triton_op body.
  """
  num_seq = cu_seqlens.shape[0] - 1
  inplace = tokens_out is None
  if inplace:
    tokens_out = tokens  # dummy, not used when INPLACE=True

  if num_chunks == 1:
    grid = (num_seq, num_chunks, no_channels)
    wrap_triton(forward_scan_chunked)[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      tokens_out_ptr=tokens_out,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=reverse,
      CHUNK_SIZE=chunk_size,
      TESTING=inplace,  # only write gates when in-place (testing mode)
      INPLACE=inplace,
    )
  else:
    grid_onepass = (num_seq, no_channels)
    wrap_triton(forward_scan_onepass_pipelined)[grid_onepass](
      gates_ptr=gates,
      tokens_ptr=tokens,
      tokens_out_ptr=tokens_out,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=reverse,
      CHUNK_SIZE=chunk_size,
      NUM_CHUNKS=num_chunks,
      TESTING=inplace,  # only write gates when in-place (testing mode)
      NUM_STAGES=get_num_stages(num_chunks, kernel="fwd"),
      INPLACE=inplace,
    )


# ============================================================
# ScanCausal as triton_op
# ============================================================


@triton_op("associative_scan_triton::scan_causal_fwd", mutates_args={})
def scan_causal_fwd_op(
  gates: torch.Tensor,
  tokens: torch.Tensor,
  cu_seqlens: torch.Tensor,
  num_chunks: int,
  no_channels: int,
  chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Causal scan forward (compile-compatible).

  Non-in-place: gates and tokens are read-only.
  Returns: (out_tokens, save_gates)
  Note: triton_op requires outputs not alias inputs,
  so gates.clone() is needed here.
  """
  out_tokens = torch.empty_like(tokens)

  _run_scan(
    gates, tokens, cu_seqlens,
    num_chunks, no_channels, chunk_size,
    reverse=False, tokens_out=out_tokens,
  )

  return out_tokens, gates.clone()


def _run_backward(
  grad, gates, states, d_tokens, d_gates,
  cu_seqlens, num_chunks, no_channels, chunk_size, causal,
) -> None:
  """Run the fused backward scan, dispatching single-chunk vs multi-chunk.

  Plain function (not a triton_op) called from within triton_op bodies.
  """
  num_seq = cu_seqlens.shape[0] - 1
  if num_chunks == 1:
    grid = (num_seq, 1, no_channels)
    wrap_triton(backward_scan_fused_single_chunk)[grid](
      grad_ptr=grad, gates_ptr=gates, states_ptr=states,
      d_tokens_ptr=d_tokens, d_gates_ptr=d_gates,
      cu_seqlens_ptr=cu_seqlens,
      CHUNK_SIZE=chunk_size, CAUSAL=causal,
    )
  else:
    grid = (num_seq, no_channels)
    wrap_triton(backward_scan_fused)[grid](
      grad_ptr=grad, gates_ptr=gates, states_ptr=states,
      d_tokens_ptr=d_tokens, d_gates_ptr=d_gates,
      cu_seqlens_ptr=cu_seqlens,
      CHUNK_SIZE=chunk_size, NUM_CHUNKS=num_chunks,
      NUM_STAGES=get_num_stages(num_chunks, kernel="bwd"), CAUSAL=causal,
    )


@triton_op("associative_scan_triton::scan_causal_bwd", mutates_args={})
def scan_causal_bwd_op(
  grad: torch.Tensor,
  gates: torch.Tensor,
  states: torch.Tensor,
  cu_seqlens: torch.Tensor,
  num_chunks: int,
  no_channels: int,
  chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Causal scan backward (compile-compatible).

  Returns: (d_gates, d_tokens)
  """
  grad = grad.contiguous()
  d_tokens = torch.empty_like(grad)
  d_gates = torch.empty_like(gates)

  _run_backward(
    grad, gates, states, d_tokens, d_gates,
    cu_seqlens, num_chunks, no_channels, chunk_size, causal=True,
  )

  return d_gates, d_tokens


def _scan_causal_backward(
  ctx, grad, _grad_gates,
) -> tuple[
  torch.Tensor, torch.Tensor, None, None, None, None,
]:
  states, gates = ctx.saved_tensors
  d_gates, d_tokens = scan_causal_bwd_op(
    grad, gates, states,
    ctx.cu_seqlens,
    ctx.num_chunks, ctx.no_channels, ctx.chunk_size,
  )
  return (
    d_gates, d_tokens,
    None, None, None, None,  # non-tensor args
  )


def _scan_causal_setup_context(ctx, inputs, output) -> None:
  _gates, _tokens, cu_seqlens, num_chunks, no_channels, chunk_size = inputs
  out_tokens, save_gates = output
  ctx.save_for_backward(out_tokens, save_gates)
  ctx.cu_seqlens = cu_seqlens
  ctx.num_chunks = num_chunks
  ctx.no_channels = no_channels
  ctx.chunk_size = chunk_size


scan_causal_fwd_op.register_autograd(
  _scan_causal_backward,
  setup_context=_scan_causal_setup_context,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Bidirectional branched scan forward (compile-compatible).

  Non-in-place: all inputs are read-only.
  Returns: (out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd)
  Note: triton_op requires outputs not alias inputs,
  so gates.clone() is needed here.
  """
  out_tokens_fwd = torch.empty_like(tokens_fwd)
  out_tokens_bwd = torch.empty_like(tokens_bwd)

  _run_scan(
    gates_fwd, tokens_fwd, cu_seqlens,
    num_chunks, no_channels, chunk_size,
    reverse=False, tokens_out=out_tokens_fwd,
  )
  _run_scan(
    gates_bwd, tokens_bwd, cu_seqlens,
    num_chunks, no_channels, chunk_size,
    reverse=True, tokens_out=out_tokens_bwd,
  )

  return out_tokens_fwd, out_tokens_bwd, gates_fwd.clone(), gates_bwd.clone()


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Bidirectional branched scan backward (compile-compatible).

  Uses fused backward kernel: shift_pad + reverse_scan + d_gates in one kernel.

  Returns: (d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd)
  """
  grad_fwd = grad_fwd.contiguous()
  grad_bwd = grad_bwd.contiguous()

  d_tokens_fwd = torch.empty_like(grad_fwd)
  d_gates_fwd = torch.empty_like(gates_fwd)
  d_tokens_bwd = torch.empty_like(grad_bwd)
  d_gates_bwd = torch.empty_like(gates_bwd)

  _run_backward(
    grad_fwd, gates_fwd, states_fwd, d_tokens_fwd, d_gates_fwd,
    cu_seqlens, num_chunks, no_channels, chunk_size, causal=True,
  )
  _run_backward(
    grad_bwd, gates_bwd, states_bwd, d_tokens_bwd, d_gates_bwd,
    cu_seqlens, num_chunks, no_channels, chunk_size, causal=False,
  )

  return d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd


# ============================================================
# Autograd registration
# ============================================================


def _scan_bidi_backward(
  ctx, grad_fwd, grad_bwd, _grad_gates_fwd, _grad_gates_bwd,
) -> tuple[
  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
  None, None, None, None,
]:
  states_fwd, states_bwd, gates_fwd, gates_bwd = ctx.saved_tensors
  d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd = scan_bidi_bwd_op(
    grad_fwd, grad_bwd,
    states_fwd, states_bwd,
    gates_fwd, gates_bwd,
    ctx.cu_seqlens,
    ctx.num_chunks, ctx.no_channels, ctx.chunk_size,
  )
  return (
    d_gates_fwd, d_tokens_fwd, d_gates_bwd, d_tokens_bwd,
    None, None, None, None,  # non-tensor args
  )


def _scan_bidi_setup_context(ctx, inputs, output) -> None:
  (
    _gates_fwd, _tokens_fwd, _gates_bwd, _tokens_bwd,
    cu_seqlens, num_chunks, no_channels, chunk_size,
  ) = inputs
  out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd = output
  ctx.save_for_backward(
    out_tokens_fwd, out_tokens_bwd, save_gates_fwd, save_gates_bwd
  )
  ctx.cu_seqlens = cu_seqlens
  ctx.num_chunks = num_chunks
  ctx.no_channels = no_channels
  ctx.chunk_size = chunk_size


scan_bidi_fwd_op.register_autograd(
  _scan_bidi_backward,
  setup_context=_scan_bidi_setup_context,
)


# ============================================================
# Public API
# ============================================================


def scan_causal_compiled(
  gates: torch.Tensor,
  tokens: torch.Tensor,
  args: dict,
) -> torch.Tensor:
  """Drop-in replacement for scan_causal() using triton_op."""
  cu_seqlens = args["cu_seqlens"]
  chunk_size = args["chunk_size"]
  grid = args["grid"]
  _num_seq, num_chunks, no_channels = grid

  out_tokens, _ = scan_causal_fwd_op(
    gates, tokens, cu_seqlens,
    num_chunks, no_channels, chunk_size,
  )

  return out_tokens


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
  _num_seq, num_chunks, no_channels = grid

  out_tokens_fwd, out_tokens_bwd, _, _ = scan_bidi_fwd_op(
    gates_fwd, tokens_fwd, gates_bwd, tokens_bwd,
    cu_seqlens, num_chunks, no_channels, chunk_size,
  )

  return out_tokens_fwd, out_tokens_bwd
