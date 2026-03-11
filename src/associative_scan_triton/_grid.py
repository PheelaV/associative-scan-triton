

# source: https://github.com/triton-lang/triton/blob/6ddb79b47f3fc07038f09eb96bb0011e6725c8ee/python/triton/__init__.py#L63
def next_power_of_2(n: int) -> int:
  """Return the smallest power of 2 greater than or equal to n."""
  n -= 1
  n |= n >> 1
  n |= n >> 2
  n |= n >> 4
  n |= n >> 8
  n |= n >> 16
  n |= n >> 32
  n += 1
  return n


def get_grid(
  no_seq: int, max_seqlen: int, chunk_size: int, no_channels: int
) -> tuple[int, int, int]:
  """Gets the Triton launch grid.

  Args:
    no_seq: number of sequences + 1 (i.e. len(cu_seqlens))
    max_seqlen: maximum sequence length in the batch
    chunk_size: scan chunk size (must be even)
    no_channels: number of channels (embedding dimension)

  Returns:
    grid: (num_seq, num_chunks, no_channels)
  """
  assert chunk_size % 2 == 0, "chunk size must be divisible by 2"
  num_chunks = (max_seqlen + chunk_size - 1) // chunk_size
  num_chunks = next_power_of_2(num_chunks)
  return (no_seq - 1, num_chunks, no_channels)


def get_num_stages(num_chunks: int, kernel: str = "fwd") -> int:
  """Return optimal NUM_STAGES for software pipelining.

  Tuned on H100 with B=8, C=1536, chunk_size=512.

  Args:
    num_chunks: number of chunks in the scan
    kernel: "fwd" for forward, "bwd" for backward

  Returns:
    Optimal NUM_STAGES value (1-3)
  """
  if num_chunks <= 1:
    return 1  # single-chunk kernels don't use stages
  if kernel == "fwd":
    return 1 if num_chunks <= 4 else 3
  else:  # bwd
    return 2 if num_chunks <= 4 else 3


def get_static_grid(
  max_seqlen: int, chunk_size: int, no_channels: int
) -> tuple[int, int]:
  """Gets the batch-independent parts of the Triton launch grid.

  The first element (num_seq) varies per batch and causes torch.compile
  recompilation when passed as a Python int. This returns only the static
  parts; num_seq should be derived inside the compiled block from
  cu_seqlens.shape[0] - 1.

  Returns:
    (num_chunks, no_channels)
  """
  return get_grid(2, max_seqlen, chunk_size, no_channels)[1:]
