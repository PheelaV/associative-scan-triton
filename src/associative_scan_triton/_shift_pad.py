"""Shift-pad implementations: eager (torch.roll) and compiled (Triton kernel).

The eager version uses data-dependent indexing (mask[cu_seqlens[:-1]] = True)
which causes graph breaks under torch.compile. The compiled version uses a
precomputed boundary mask for O(1) per-element boundary detection.
"""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

# ============================================================
# Eager shift_pad (PyTorch, not compile-safe)
# ============================================================


def shift_pad(
  data: torch.Tensor,
  cu_seqlens: torch.Tensor,
  pad_value: float = 0.0,
  backward: bool = True,
) -> torch.Tensor:
  """Shifts sequences and pads using torch.roll and masking.

  Parameters:
  - data: [C, BL] where C is the number of channels and BL is the total length
  - cu_seqlens: cumulative sequence lengths
  - pad_value: value to use for padding
  - backward: if True, shift right (backward); if False, shift left (forward)

  Returns:
  - shifted_data: [C, BL] with sequences shifted and padded
  """
  _C, BL = data.shape
  device = data.device
  assert not (
    cu_seqlens[1:] == cu_seqlens[:-1]
  ).any(), "can't shift_pad zero-length seqments"

  shifted_data = torch.roll(data, shifts=1 if backward else -1, dims=-1)

  # Create a mask for padding
  mask = torch.zeros(BL, dtype=torch.bool, device=device)
  if backward:
    # Mask the first element of each sequence
    mask[cu_seqlens[:-1]] = True
  else:
    # Mask the last element of each sequence
    mask[cu_seqlens[1:] - 1] = True

  # Apply the mask
  shifted_data[:, mask] = pad_value

  return shifted_data


# ============================================================
# Compiled shift_pad (Triton kernel, torch.compile-safe)
# ============================================================


@triton.jit
def _shift_pad_kernel(
  data_ptr,
  out_ptr,
  boundary_ptr,
  stride_c: int,
  stride_l: int,
  total_len: int,
  PAD_VALUE: tl.constexpr,
  BACKWARD: tl.constexpr,  # True = shift right, False = shift left
  BLOCK_L: tl.constexpr,
):
  """Shift and pad sequences using a precomputed boundary mask.

  Grid: (cdiv(total_len, BLOCK_L), C)
  """
  pos_block_id = tl.program_id(0)
  channel_id = tl.program_id(1)

  pos_start = pos_block_id * BLOCK_L
  pos_offsets = tl.arange(0, BLOCK_L)
  pos_abs = pos_start + pos_offsets
  pos_mask = pos_abs < total_len

  # O(1) boundary check: load precomputed boolean
  is_boundary = tl.load(
    boundary_ptr + pos_abs, mask=pos_mask, other=0,
  ).to(tl.int1)

  # Compute source position
  if BACKWARD:
    src_pos = pos_abs - 1  # shift right: read from pos-1
  else:
    src_pos = pos_abs + 1  # shift left: read from pos+1

  # Clamp source to valid range for the load
  # (boundary positions will be overwritten)
  src_pos_safe = tl.minimum(tl.maximum(src_pos, 0), total_len - 1)

  # Load source data
  base_offset = channel_id * stride_c
  src_vals = tl.load(
    data_ptr + base_offset + src_pos_safe * stride_l,
    mask=pos_mask,
    other=0.0,
  )

  # Apply padding at boundaries
  out_vals = tl.where(is_boundary, PAD_VALUE, src_vals)

  # Store
  tl.store(
    out_ptr + base_offset + pos_abs * stride_l,
    out_vals,
    mask=pos_mask,
  )


@triton_op("associative_scan_triton::shift_pad", mutates_args={})
def shift_pad_compiled(
  data: torch.Tensor,
  cu_seqlens: torch.Tensor,
  pad_value: float,
  backward: bool,
) -> torch.Tensor:
  """Compile-compatible shift_pad using Triton kernel.

  Args:
      data: [C, BL] tensor
      cu_seqlens: [num_seq + 1] cumulative sequence lengths
      pad_value: value for padding (1.0 for gates, 0.0 for states)
      backward: True = shift right (pad at seq starts),
        False = shift left (pad at seq ends)

  Returns:
      shifted_data: [C, BL] shifted and padded tensor
  """
  C, BL = data.shape
  out = torch.empty_like(data)

  # Precompute boundary mask from cu_seqlens — O(BS) scatter, negligible cost
  boundary = torch.zeros(BL, dtype=torch.bool, device=data.device)
  if backward:
    # Pad at sequence starts: cu_seqlens[0..num_seq-1]
    boundary[cu_seqlens[:-1]] = True
  else:
    # Pad at sequence ends: cu_seqlens[1..num_seq] - 1
    boundary[cu_seqlens[1:] - 1] = True

  BLOCK_L = 256
  grid = (triton.cdiv(BL, BLOCK_L), C)
  wrap_triton(_shift_pad_kernel)[grid](
    data,
    out,
    boundary,
    data.stride(0),
    data.stride(1),
    BL,
    PAD_VALUE=pad_value,
    BACKWARD=backward,
    BLOCK_L=BLOCK_L,
  )
  return out
