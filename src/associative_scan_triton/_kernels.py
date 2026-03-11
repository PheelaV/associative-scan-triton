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
  tokens_out_ptr,
  cu_seqlens_ptr,
  REVERSE: tl.constexpr,
  CHUNK_SIZE: tl.constexpr,
  TESTING: tl.constexpr,
  INPLACE: tl.constexpr,
) -> None:
  """Single-chunk forward scan.

  Only used when the entire sequence fits in one chunk.
  For multi-chunk, see forward_scan_onepass_pipelined.

  Args:
    - gates_ptr: gate values
    - tokens_ptr: token values
    - cu_seqlens_ptr: cumulative sequence lengths (varlen)
    - REVERSE: reverse scan order
    - CHUNK_SIZE: chunk size (must be >= max sequence length)
    - TESTING: if True, write gate values (for gradient checks)
    - INPLACE: if True, write tokens back to tokens_ptr
  """
  no_sequences = tl.num_programs(0)

  seq_id = tl.program_id(axis=0)
  channel_id = tl.program_id(axis=2)

  seq_start = tl.load(cu_seqlens_ptr + seq_id)
  seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
  stride_bl = tl.load(cu_seqlens_ptr + no_sequences)
  channel_offset = stride_bl * channel_id

  input_range = tl.arange(0, CHUNK_SIZE)
  seq_range = (
    seq_start
    + ((CHUNK_SIZE - 1 - input_range) if REVERSE else input_range)
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

  if TESTING:
    # If we are not aggregating we do not need to updated gates
    tl.store(gates_ptr_chunk_channel, gates, mask=mask)
  if INPLACE:
    tl.store(tokens_ptr_chunk_channel, tokens, mask=mask)
  else:
    tokens_out_ptrs = tokens_out_ptr + seq_range + channel_offset
    tl.store(tokens_out_ptrs, tokens, mask=mask)


@triton.jit
def forward_scan_onepass_pipelined(
  gates_ptr,
  tokens_ptr,
  tokens_out_ptr,
  cu_seqlens_ptr,
  REVERSE: tl.constexpr,
  CHUNK_SIZE: tl.constexpr,
  NUM_CHUNKS: tl.constexpr,
  TESTING: tl.constexpr,
  NUM_STAGES: tl.constexpr,
  INPLACE: tl.constexpr,
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

    # Load (2 reads — pipelining prefetches next iteration)
    # Cast to fp32 for scan accumulation (matches prefix dtype,
    # prevents type mismatch in if/else for fp16/bf16 inputs)
    gates_ptrs = gates_ptr + seq_range + channel_offset
    tokens_ptrs = tokens_ptr + seq_range + channel_offset
    gates = tl.load(gates_ptrs, mask=mask, other=1.0).to(tl.float32)
    tokens = tl.load(tokens_ptrs, mask=mask, other=0.0).to(tl.float32)

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
    if INPLACE:
      tl.store(tokens_ptrs, tokens, mask=mask)
    else:
      tokens_out_ptrs = tokens_out_ptr + seq_range + channel_offset
      tl.store(tokens_out_ptrs, tokens, mask=mask)


@triton.jit
def backward_scan_fused(
  grad_ptr,
  gates_ptr,
  states_ptr,
  d_tokens_ptr,
  d_gates_ptr,
  cu_seqlens_ptr,
  CHUNK_SIZE: tl.constexpr,
  NUM_CHUNKS: tl.constexpr,
  NUM_STAGES: tl.constexpr,
  CAUSAL: tl.constexpr,
) -> None:
  """Fused backward: shift_pad + reverse_scan + d_gates in one kernel.

  For CAUSAL=True (causal scan backward):
    - Scan runs in REVERSE
    - Gates shifted LEFT (read pos+1, pad=1.0 at seq_end)
    - States shifted RIGHT (read pos-1, pad=0.0 at seq_start)

  For CAUSAL=False (anti-causal scan backward):
    - Scan runs FORWARD
    - Gates shifted RIGHT (read pos-1, pad=1.0 at seq_start)
    - States shifted LEFT (read pos+1, pad=0.0 at seq_end)

  Grid: (num_seq, no_channels)
  """
  seq_id = tl.program_id(axis=0)
  channel_id = tl.program_id(axis=1)
  no_sequences = tl.num_programs(0)

  seq_start = tl.load(cu_seqlens_ptr + seq_id)
  seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
  stride_bl = tl.load(cu_seqlens_ptr + no_sequences)
  channel_offset = stride_bl * channel_id

  prefix_gate: tl.float32 = 1.0
  prefix_token: tl.float32 = 0.0

  input_range = tl.arange(0, CHUNK_SIZE)

  for loop_idx in tl.range(NUM_CHUNKS, num_stages=NUM_STAGES):
    if CAUSAL:
      chunk_idx = NUM_CHUNKS - 1 - loop_idx
    else:
      chunk_idx = loop_idx

    chunk_offset = chunk_idx * CHUNK_SIZE

    if CAUSAL:
      seq_range = seq_start + (CHUNK_SIZE - 1 - input_range) + chunk_offset
    else:
      seq_range = seq_start + input_range + chunk_offset
    mask = seq_end > seq_range

    # Load grad (no shift) — upcast to fp32 for accumulation
    grad_vals = tl.load(
      grad_ptr + seq_range + channel_offset,
      mask=mask, other=0.0,
    ).to(tl.float32)

    # Load shifted gates (inline shift_pad) — upcast to fp32
    if CAUSAL:
      shifted_gate_pos = seq_range + 1
      gate_boundary = shifted_gate_pos >= seq_end
    else:
      shifted_gate_pos = seq_range - 1
      gate_boundary = shifted_gate_pos < seq_start

    shifted_gate_pos_safe = tl.minimum(
      tl.maximum(shifted_gate_pos, seq_start), seq_end - 1,
    )
    shifted_gates = tl.load(
      gates_ptr + shifted_gate_pos_safe + channel_offset,
      mask=mask & ~gate_boundary, other=1.0,
    ).to(tl.float32)
    shifted_gates = tl.where(gate_boundary & mask, 1.0, shifted_gates)

    # Seed with prefix and scan
    if loop_idx > 0:
      first_mask = input_range == 0
      seeded_gates, seeded_tokens = op(
        prefix_gate, prefix_token, shifted_gates, grad_vals
      )
      shifted_gates = tl.where(first_mask, seeded_gates, shifted_gates)
      grad_vals = tl.where(first_mask, seeded_tokens, grad_vals)

    shifted_gates, d_tokens = tl.associative_scan(
      (shifted_gates, grad_vals), axis=0, combine_fn=op
    )

    last_mask = (input_range == CHUNK_SIZE - 1).to(shifted_gates.dtype)
    prefix_gate = tl.sum(shifted_gates * last_mask)
    prefix_token = tl.sum(d_tokens * last_mask)

    # Load shifted states, compute d_gates (shift_pad + multiply)
    if CAUSAL:
      shifted_state_pos = seq_range - 1
      state_boundary = shifted_state_pos < seq_start
    else:
      shifted_state_pos = seq_range + 1
      state_boundary = shifted_state_pos >= seq_end

    shifted_state_pos_safe = tl.minimum(
      tl.maximum(shifted_state_pos, seq_start), seq_end - 1,
    )
    shifted_states = tl.load(
      states_ptr + shifted_state_pos_safe + channel_offset,
      mask=mask & ~state_boundary, other=0.0,
    )
    shifted_states = tl.where(state_boundary & mask, 0.0, shifted_states)

    d_gates = shifted_states * d_tokens

    # Store both outputs
    tl.store(d_tokens_ptr + seq_range + channel_offset, d_tokens, mask=mask)
    tl.store(d_gates_ptr + seq_range + channel_offset, d_gates, mask=mask)


@triton.jit
def backward_scan_fused_single_chunk(
  grad_ptr,
  gates_ptr,
  states_ptr,
  d_tokens_ptr,
  d_gates_ptr,
  cu_seqlens_ptr,
  CHUNK_SIZE: tl.constexpr,
  CAUSAL: tl.constexpr,
) -> None:
  """Single-chunk fused backward — no loop, no prefix, no pipelining.

  Mirrors forward_scan_chunked: 3D grid (num_seq, 1, no_channels).
  Used when the entire sequence fits in one chunk.
  """
  no_sequences = tl.num_programs(0)
  seq_id = tl.program_id(axis=0)
  channel_id = tl.program_id(axis=2)

  seq_start = tl.load(cu_seqlens_ptr + seq_id)
  seq_end = tl.load(cu_seqlens_ptr + seq_id + 1)
  stride_bl = tl.load(cu_seqlens_ptr + no_sequences)
  channel_offset = stride_bl * channel_id

  input_range = tl.arange(0, CHUNK_SIZE)

  if CAUSAL:
    seq_range = seq_start + (CHUNK_SIZE - 1 - input_range)
  else:
    seq_range = seq_start + input_range
  mask = seq_end > seq_range

  # Load grad (no shift)
  grad_vals = tl.load(
    grad_ptr + seq_range + channel_offset,
    mask=mask, other=0.0,
  )

  # Load shifted gates (inline shift_pad)
  if CAUSAL:
    shifted_gate_pos = seq_range + 1
    gate_boundary = shifted_gate_pos >= seq_end
  else:
    shifted_gate_pos = seq_range - 1
    gate_boundary = shifted_gate_pos < seq_start

  shifted_gate_pos_safe = tl.minimum(
    tl.maximum(shifted_gate_pos, seq_start), seq_end - 1,
  )
  shifted_gates = tl.load(
    gates_ptr + shifted_gate_pos_safe + channel_offset,
    mask=mask & ~gate_boundary, other=1.0,
  )
  shifted_gates = tl.where(gate_boundary & mask, 1.0, shifted_gates)

  # Scan (no prefix seeding — single chunk)
  shifted_gates, d_tokens = tl.associative_scan(
    (shifted_gates, grad_vals), axis=0, combine_fn=op
  )

  # Load shifted states and compute d_gates
  if CAUSAL:
    shifted_state_pos = seq_range - 1
    state_boundary = shifted_state_pos < seq_start
  else:
    shifted_state_pos = seq_range + 1
    state_boundary = shifted_state_pos >= seq_end

  shifted_state_pos_safe = tl.minimum(
    tl.maximum(shifted_state_pos, seq_start), seq_end - 1,
  )
  shifted_states = tl.load(
    states_ptr + shifted_state_pos_safe + channel_offset,
    mask=mask & ~state_boundary, other=0.0,
  )
  shifted_states = tl.where(
    state_boundary & mask, 0.0, shifted_states,
  )

  d_gates = shifted_states * d_tokens

  # Store both outputs
  tl.store(d_tokens_ptr + seq_range + channel_offset, d_tokens, mask=mask)
  tl.store(d_gates_ptr + seq_range + channel_offset, d_gates, mask=mask)
