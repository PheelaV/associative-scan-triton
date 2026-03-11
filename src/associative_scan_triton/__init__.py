"""Chunked associative scan in Triton with varlen/packing, bidirectional, and torch.compile support."""

__version__ = "1.0.0"

# Core kernels and dispatcher
from associative_scan_triton._kernels import op, forward_scan_chunked, forward_scan_onepass_pipelined, backward_scan_fused, backward_scan_fused_single_chunk
from associative_scan_triton._dispatcher import forward_scan_full, backward_scan_fused_full
from associative_scan_triton._shift_pad import shift_pad, shift_pad_compiled
from associative_scan_triton._grid import next_power_of_2, get_grid, get_static_grid, get_num_stages

# High-level API
from associative_scan_triton.scan_eager import scan_causal, scan_bidirectional_branched
from associative_scan_triton.scan_compiled import scan_causal_compiled, scan_bidirectional_branched_compiled

__all__ = [
  # Kernels
  "op",
  "forward_scan_chunked",
  "forward_scan_onepass_pipelined",
  "forward_scan_full",
  "backward_scan_fused",
  "backward_scan_fused_single_chunk",
  "backward_scan_fused_full",
  # Shift-pad
  "shift_pad",
  "shift_pad_compiled",
  # Grid utilities
  "next_power_of_2",
  "get_grid",
  "get_static_grid",
  # High-level API
  "scan_causal",
  "scan_causal_compiled",
  "scan_bidirectional_branched",
  "scan_bidirectional_branched_compiled",
]
