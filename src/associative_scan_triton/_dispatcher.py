import triton.language as tl

from associative_scan_triton._kernels import (
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
  NUM_STAGES: tl.constexpr = 3,
) -> None:
  """1-kernel onepass scan dispatcher with software pipelining."""
  assert (
    len(cu_seqlens) - 1 == grid[0]
  ), "cu_seqlens do not match grid configuration"

  num_chunks = grid[1]
  if num_chunks == 1:
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=REVERSE,
      CHUNK_SIZE=CHUNK_SIZE,
      FIRST_CALL=False,
      TESTING=TESTING,
    )
    return

  num_seq = grid[0]
  num_channels = grid[2]

  grid_onepass = (num_seq, num_channels)
  forward_scan_onepass_pipelined[grid_onepass](
    gates_ptr=gates,
    tokens_ptr=tokens,
    cu_seqlens_ptr=cu_seqlens,
    REVERSE=REVERSE,
    CHUNK_SIZE=CHUNK_SIZE,
    NUM_CHUNKS=num_chunks,
    TESTING=TESTING,
    NUM_STAGES=NUM_STAGES,
  )
