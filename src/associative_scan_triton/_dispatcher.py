import triton.language as tl

from associative_scan_triton._grid import get_num_stages
from associative_scan_triton._kernels import (
  backward_scan_fused,
  backward_scan_fused_single_chunk,
  forward_scan_chunked,
  forward_scan_onepass_pipelined,
)


def forward_scan_full(
  gates,
  tokens,
  cu_seqlens,
  grid: tuple,
  REVERSE: tl.constexpr,
  CHUNK_SIZE: tl.constexpr,
  TESTING: tl.constexpr = True,
  NUM_STAGES: tl.constexpr = None,
  tokens_out=None,
) -> None:
  """1-kernel onepass scan dispatcher with software pipelining.

  Args:
    tokens_out: If provided, write scanned tokens here (non-in-place).
                If None, write back to tokens (in-place, original behavior).
    NUM_STAGES: If None, auto-select based on num_chunks.
  """
  assert (
    len(cu_seqlens) - 1 == grid[0]
  ), "cu_seqlens do not match grid configuration"

  num_chunks = grid[1]
  inplace = tokens_out is None
  if inplace:
    tokens_out = tokens  # dummy, not used when INPLACE=True

  if num_chunks == 1:
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      tokens_out_ptr=tokens_out,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=REVERSE,
      CHUNK_SIZE=CHUNK_SIZE,
      TESTING=TESTING,
      INPLACE=inplace,
    )
    return

  if NUM_STAGES is None:
    NUM_STAGES = get_num_stages(num_chunks, kernel="fwd")

  num_seq = grid[0]
  num_channels = grid[2]

  grid_onepass = (num_seq, num_channels)
  forward_scan_onepass_pipelined[grid_onepass](
    gates_ptr=gates,
    tokens_ptr=tokens,
    tokens_out_ptr=tokens_out,
    cu_seqlens_ptr=cu_seqlens,
    REVERSE=REVERSE,
    CHUNK_SIZE=CHUNK_SIZE,
    NUM_CHUNKS=num_chunks,
    TESTING=TESTING,
    NUM_STAGES=NUM_STAGES,
    INPLACE=inplace,
  )


def backward_scan_fused_full(
  grad,
  gates,
  states,
  d_tokens_out,
  d_gates_out,
  cu_seqlens,
  grid: tuple,
  CHUNK_SIZE: tl.constexpr,
  CAUSAL: tl.constexpr,
  NUM_STAGES: tl.constexpr = None,
) -> None:
  """Fused backward scan dispatcher."""
  num_seq, num_chunks, no_channels = grid

  if num_chunks == 1:
    grid_launch = (num_seq, 1, no_channels)
    backward_scan_fused_single_chunk[grid_launch](
      grad_ptr=grad,
      gates_ptr=gates,
      states_ptr=states,
      d_tokens_ptr=d_tokens_out,
      d_gates_ptr=d_gates_out,
      cu_seqlens_ptr=cu_seqlens,
      CHUNK_SIZE=CHUNK_SIZE,
      CAUSAL=CAUSAL,
    )
    return

  if NUM_STAGES is None:
    NUM_STAGES = get_num_stages(num_chunks, kernel="bwd")

  grid_launch = (num_seq, no_channels)
  backward_scan_fused[grid_launch](
    grad_ptr=grad,
    gates_ptr=gates,
    states_ptr=states,
    d_tokens_ptr=d_tokens_out,
    d_gates_ptr=d_gates_out,
    cu_seqlens_ptr=cu_seqlens,
    CHUNK_SIZE=CHUNK_SIZE,
    NUM_CHUNKS=num_chunks,
    NUM_STAGES=NUM_STAGES,
    CAUSAL=CAUSAL,
  )
