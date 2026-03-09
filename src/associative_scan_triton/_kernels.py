import triton
import triton.language as tl


# source: annotated-mamba/hard
@triton.jit
def op(fl: float, xl: float, fr: float, xr: float) -> tuple[float, float]:
  """First order linear recurrence operation.

  ## parameters:
    - f is a state accummulator
    - x is the result of the current final or intermediate time step
  """
  f = fr * fl
  x = fr * xl + xr
  return f, x


@triton.jit
def forward_scan_chunked(
  gates_ptr,
  tokens_ptr,
  cu_seqlens_ptr,
  REVERSE: tl.constexpr,
  CHUNK_SIZE: tl.constexpr,
  FIRST_CALL: tl.constexpr,
  TESTING: tl.constexpr,
) -> None:
  """
  Computes a packed and chunked first order recurrence.

  ## parameters
    - gates_ptr:tensor= gates for in-place update
    - tokens_ptr:tensor= tokens for in-place update
    - cu_seqlens_ptr:tensor= cummulative sequelce lengths for packing and varlen
    - REVERSE:bool= reverse the order of operation (e.g. for gradients)
    - CHUNK_SIZE:int= size of the chunks, equivalent for all sequences as it is
        dealt with using chunking
    - FIRST_CALL:bool= switch between first and second call on the first call
        - we only update the last chunk states (ports) as to preserve data for
        the second call where we update all except the ports as they are final
        already
        - unless the number of chunks is 1 in which case we simply update all
    - TESTING:bool used for unit testing, will always write corresponging gates
  """
  no_sequences = tl.num_programs(0)
  num_chunks = tl.num_programs(1)

  seq_id = tl.program_id(axis=0)
  chunk_id = tl.program_id(axis=1)
  channel_id = tl.program_id(axis=2)

  seq_start = tl.load(cu_seqlens_ptr + seq_id)
  seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
  # length of all the sequences to get us to the next channel
  stride_bl = tl.load(cu_seqlens_ptr + no_sequences)

  input_range = tl.arange(0, CHUNK_SIZE)
  chunk_offset = chunk_id * CHUNK_SIZE
  if chunk_offset >= seq_end:
    # return early
    return
  channel_offset = stride_bl * channel_id

  seq_range = (
    seq_start
    + ((CHUNK_SIZE - 1 - input_range) if REVERSE else input_range)
    + chunk_offset
  )

  # Create a mask for valid loads and saves overall
  mask = seq_end > seq_range

  # Pointers for this sequence, chunk and channel
  gates_ptr_chunk_channel = gates_ptr + seq_range + channel_offset
  tokens_ptr_chunk_channel = tokens_ptr + seq_range + channel_offset

  # Load gates and tokens
  gates = tl.load(gates_ptr_chunk_channel, mask=mask, other=1)
  tokens = tl.load(tokens_ptr_chunk_channel, mask=mask, other=0)

  # Perform scan
  gates, tokens = tl.associative_scan((gates, tokens), axis=0, combine_fn=op)

  # Create a mask for updating the port on the first call, and avoid
  # updating it on on the second call as it is already final.
  seq_length = seq_end - seq_start
  if num_chunks > 1:
    port_index = (
      chunk_offset
      if REVERSE
      else tl.minimum(seq_length - 1, (chunk_id + 1) * CHUNK_SIZE - 1)
    )
    port_mask = seq_range == (seq_start + port_index)
    # Adjust mask based on whether it's the first or second call
    mask = mask & (port_mask if FIRST_CALL else ~port_mask)
  if num_chunks != 1 or TESTING:
    # If we are not aggregating we do not need to updated gates
    tl.store(gates_ptr_chunk_channel, gates, mask=mask)
  tl.store(tokens_ptr_chunk_channel, tokens, mask=mask)


@triton.jit
def forward_scan_onepass_pipelined(
  gates_ptr,
  tokens_ptr,
  cu_seqlens_ptr,
  REVERSE: tl.constexpr,
  CHUNK_SIZE: tl.constexpr,
  NUM_CHUNKS: tl.constexpr,
  TESTING: tl.constexpr,
  NUM_STAGES: tl.constexpr,
) -> None:
  """Single-kernel onepass scan with software pipelining.

  Same as forward_scan_onepass but uses tl.range(num_stages=NUM_STAGES) to
  overlap loads of chunk N+1 with compute/store of chunk N.
  """
  seq_id = tl.program_id(axis=0)
  channel_id = tl.program_id(axis=1)
  no_sequences = tl.num_programs(0)

  seq_start = tl.load(cu_seqlens_ptr + seq_id)
  seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
  stride_bl = tl.load(cu_seqlens_ptr + no_sequences)
  channel_offset = stride_bl * channel_id

  # Running prefix (identity element of op)
  prefix_gate: tl.float32 = 1.0
  prefix_token: tl.float32 = 0.0

  input_range = tl.arange(0, CHUNK_SIZE)

  for loop_idx in tl.range(NUM_CHUNKS, num_stages=NUM_STAGES):
    # Process chunks in scan order
    if REVERSE:
      chunk_idx = NUM_CHUNKS - 1 - loop_idx
    else:
      chunk_idx = loop_idx

    chunk_offset = chunk_idx * CHUNK_SIZE

    # Element indices (reversed within chunk if REVERSE)
    seq_range = (
      seq_start
      + ((CHUNK_SIZE - 1 - input_range) if REVERSE else input_range)
      + chunk_offset
    )
    mask = seq_end > seq_range

    # Load (2 reads — pipelining prefetches these from next iteration)
    gates_ptrs = gates_ptr + seq_range + channel_offset
    tokens_ptrs = tokens_ptr + seq_range + channel_offset
    gates = tl.load(gates_ptrs, mask=mask, other=1.0)
    tokens = tl.load(tokens_ptrs, mask=mask, other=0.0)

    # Seed element 0 with accumulated prefix (skip first chunk)
    if loop_idx > 0:
      first_mask = input_range == 0
      seeded_gates, seeded_tokens = op(
        prefix_gate, prefix_token, gates, tokens
      )
      gates = tl.where(first_mask, seeded_gates, gates)
      tokens = tl.where(first_mask, seeded_tokens, tokens)

    # Scan
    gates, tokens = tl.associative_scan(
      (gates, tokens), axis=0, combine_fn=op
    )

    # Extract last element as new prefix
    last_mask = (input_range == CHUNK_SIZE - 1).to(gates.dtype)
    prefix_gate = tl.sum(gates * last_mask)
    prefix_token = tl.sum(tokens * last_mask)

    # Write (skip gate writes unless TESTING)
    if TESTING:
      tl.store(gates_ptrs, gates, mask=mask)
    tl.store(tokens_ptrs, tokens, mask=mask)
