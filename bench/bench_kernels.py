"""Kernel-vs-kernel benchmark: direct kernel calls, no autograd.

Two comparison levels:
1. **API-level**: our forward_scan_full() vs warpscan's scan_forward()
   - Both include their respective Python dispatch overhead
   - This is what users actually call
2. **Kernel-level**: our Triton kernel[grid]() vs warpscan's warpscan_forward()
   - Our side: direct kernel launch, all args pre-computed
   - Warpscan side: warpscan_forward() is already a direct C++ kernel call
   - Closest to pure GPU compute comparison

Methodology:
- triton.testing.do_bench() with CUDA events (GPU-side timing)
- L2 cache cleared between reps (cold cache)
- warmup=5000ms, rep=200ms time budgets
- All buffers pre-allocated outside the timed region
- torch.inference_mode() throughout

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run --group bench python bench/bench_kernels.py
"""

import torch
import triton

from associative_scan_triton._dispatcher import forward_scan_full, backward_scan_fused_full
from associative_scan_triton._kernels import (
  forward_scan_chunked,
  forward_scan_onepass_pipelined,
  backward_scan_fused,
  backward_scan_fused_single_chunk,
)
from associative_scan_triton._grid import get_grid, get_num_stages, next_power_of_2

try:
  from accelerated_scan.warp import warpscan_forward, warpscan_backward
  HAS_WARPSCAN = True
except ImportError:
  HAS_WARPSCAN = False
  print("accelerated-scan not installed, skipping warpscan kernels")

B = 8
C = 1536
MAX_CHUNK_SIZE = 512
DEVICE = "cuda"
_ALL_SEQLENS = [2**i for i in range(7, 17)]  # 128..65536, all pow2 for warpscan


def _max_seqlen_for(num_tensors, dtype_bytes=4, margin_gb=1.5):
  """Return the largest power-of-2 seqlen that fits in GPU VRAM."""
  import math
  total = torch.cuda.get_device_properties(0).total_memory
  usable = total - margin_gb * 2**30
  max_elems = usable / (num_tensors * C * B * dtype_bytes)
  return 2 ** int(math.log2(max(max_elems, 128)))


# Forward: 3 tensors (gates, tokens, out)
# Backward: 6 tensors (gates, tokens, out, grad, d_tokens, d_gates)
SEQLENS_FWD = [s for s in _ALL_SEQLENS if s <= _max_seqlen_for(3)]
SEQLENS_BWD = [s for s in _ALL_SEQLENS if s <= _max_seqlen_for(6)]
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
print(f"  fwd seqlens: {SEQLENS_FWD[0]}..{SEQLENS_FWD[-1]} ({len(SEQLENS_FWD)} points)")
print(f"  bwd seqlens: {SEQLENS_BWD[0]}..{SEQLENS_BWD[-1]} ({len(SEQLENS_BWD)} points)")


# ============================================================
# Data factories
# ============================================================

def _make_our_data(seqlen):
  """Create data in our (C, B*T) layout + pre-compute all kernel args."""
  total = seqlen * B
  gates = torch.rand(C, total, device=DEVICE).contiguous()
  tokens = torch.rand(C, total, device=DEVICE).contiguous()
  cu_seqlens = torch.arange(0, B + 1, device=DEVICE, dtype=torch.int32) * seqlen
  chunk_size = min(next_power_of_2(seqlen), MAX_CHUNK_SIZE)
  grid = get_grid(len(cu_seqlens), seqlen, chunk_size, C)
  return gates, tokens, cu_seqlens, chunk_size, grid


def _make_warp_data(seqlen):
  """Create data in warpscan's (B, C, T) layout."""
  gates = torch.rand(B, C, seqlen, device=DEVICE).contiguous()
  tokens = torch.rand(B, C, seqlen, device=DEVICE).contiguous()
  return gates, tokens


def _make_direct_fwd_fn(gates, tokens, out, cu_seqlens, chunk_size, grid):
  """Create a closure that calls our Triton forward kernel directly (no Python dispatch)."""
  num_seq, num_chunks, no_channels = grid
  if num_chunks == 1:
    def fn():
      forward_scan_chunked[grid](
        gates_ptr=gates, tokens_ptr=tokens, tokens_out_ptr=out,
        cu_seqlens_ptr=cu_seqlens,
        REVERSE=False, CHUNK_SIZE=chunk_size,
        TESTING=False, INPLACE=False,
      )
  else:
    grid_onepass = (num_seq, no_channels)
    num_stages = get_num_stages(num_chunks, kernel="fwd")
    def fn():
      forward_scan_onepass_pipelined[grid_onepass](
        gates_ptr=gates, tokens_ptr=tokens, tokens_out_ptr=out,
        cu_seqlens_ptr=cu_seqlens,
        REVERSE=False, CHUNK_SIZE=chunk_size, NUM_CHUNKS=num_chunks,
        TESTING=False, NUM_STAGES=num_stages, INPLACE=False,
      )
  return fn


def _make_direct_bwd_fn(grad, gates, states, d_tokens, d_gates, cu_seqlens, chunk_size, grid):
  """Create a closure that calls our Triton backward kernel directly (no Python dispatch)."""
  num_seq, num_chunks, no_channels = grid
  if num_chunks == 1:
    grid_launch = (num_seq, 1, no_channels)
    def fn():
      backward_scan_fused_single_chunk[grid_launch](
        grad_ptr=grad, gates_ptr=gates, states_ptr=states,
        d_tokens_ptr=d_tokens, d_gates_ptr=d_gates,
        cu_seqlens_ptr=cu_seqlens,
        CHUNK_SIZE=chunk_size, CAUSAL=True,
      )
  else:
    grid_launch = (num_seq, no_channels)
    num_stages = get_num_stages(num_chunks, kernel="bwd")
    def fn():
      backward_scan_fused[grid_launch](
        grad_ptr=grad, gates_ptr=gates, states_ptr=states,
        d_tokens_ptr=d_tokens, d_gates_ptr=d_gates,
        cu_seqlens_ptr=cu_seqlens,
        CHUNK_SIZE=chunk_size, NUM_CHUNKS=num_chunks,
        NUM_STAGES=num_stages, CAUSAL=True,
      )
  return fn


# ============================================================
# Forward kernel benchmark (direct kernel calls)
# ============================================================

providers_fwd = ["triton_kernel"]
if HAS_WARPSCAN:
  providers_fwd.append("warpscan_kernel")


@triton.testing.perf_report(
  [
    triton.testing.Benchmark(
      x_names=["seqlen"],
      x_vals=SEQLENS_FWD,
      xlabel="sequence length",
      ylabel="ms",
      x_log=True,
      y_log=True,
      line_arg="provider",
      line_vals=providers_fwd,
      line_names=[
        "assoc_scan (Triton kernel)",
        *(["warpscan (CUDA kernel)"] if HAS_WARPSCAN else []),
      ],
      plot_name=f"Kernel fwd: ({B}, {C}, seqlen)",
      args={},
    ),
  ]
)
def bench_kernel_fwd(provider, seqlen, device=DEVICE):
  if provider == "warpscan_kernel":
    gates, tokens = _make_warp_data(seqlen)
    out = torch.empty_like(tokens)
    warpscan_forward(gates, tokens, out, False)

    def subject():
      warpscan_forward(gates, tokens, out, False)

  else:  # triton_kernel
    gates, tokens, cu_seqlens, chunk_size, grid = _make_our_data(seqlen)
    out = torch.empty_like(tokens)
    subject = _make_direct_fwd_fn(gates, tokens, out, cu_seqlens, chunk_size, grid)
    subject()  # warmup / compile

  with torch.inference_mode():
    ms = triton.testing.do_bench(subject, warmup=5000, rep=200)
  print(f"  fwd  {provider:>15s}  seqlen={seqlen:>6d}  {ms:.4f} ms")
  return ms


# ============================================================
# Backward kernel benchmark (direct kernel calls)
# ============================================================

providers_bwd = ["triton_kernel"]
if HAS_WARPSCAN:
  providers_bwd.append("warpscan_kernel")


@triton.testing.perf_report(
  [
    triton.testing.Benchmark(
      x_names=["seqlen"],
      x_vals=SEQLENS_BWD,
      xlabel="sequence length",
      ylabel="ms",
      x_log=True,
      y_log=True,
      line_arg="provider",
      line_vals=providers_bwd,
      line_names=[
        "assoc_scan (Triton kernel)",
        *(["warpscan (CUDA kernel)"] if HAS_WARPSCAN else []),
      ],
      plot_name=f"Kernel bwd: ({B}, {C}, seqlen)",
      args={},
    ),
  ]
)
def bench_kernel_bwd(provider, seqlen, device=DEVICE):
  if provider == "warpscan_kernel":
    gates, tokens = _make_warp_data(seqlen)
    out = torch.empty_like(tokens)
    warpscan_forward(gates, tokens, out, False)
    grad = torch.rand_like(out)
    d_gates = torch.empty_like(gates)
    d_tokens = torch.empty_like(tokens)
    warpscan_backward(gates, out, grad, d_gates, d_tokens)

    def subject():
      warpscan_backward(gates, out, grad, d_gates, d_tokens)

  else:  # triton_kernel
    gates, tokens, cu_seqlens, chunk_size, grid = _make_our_data(seqlen)
    out = torch.empty_like(tokens)
    forward_scan_full(gates, tokens, cu_seqlens, grid,
                      REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=False,
                      tokens_out=out)
    grad = torch.rand_like(out)
    d_tokens = torch.empty_like(grad)
    d_gates = torch.empty_like(gates)
    subject = _make_direct_bwd_fn(grad, gates, out, d_tokens, d_gates,
                                   cu_seqlens, chunk_size, grid)
    subject()  # warmup / compile

  with torch.inference_mode():
    ms = triton.testing.do_bench(subject, warmup=5000, rep=200)
  print(f"  bwd  {provider:>15s}  seqlen={seqlen:>6d}  {ms:.4f} ms")
  return ms


# ============================================================
# Combined fwd+bwd kernel benchmark (direct kernel calls)
# ============================================================

providers_both = ["triton_kernel"]
if HAS_WARPSCAN:
  providers_both.append("warpscan_kernel")


@triton.testing.perf_report(
  [
    triton.testing.Benchmark(
      x_names=["seqlen"],
      x_vals=SEQLENS_BWD,
      xlabel="sequence length",
      ylabel="ms",
      x_log=True,
      y_log=True,
      line_arg="provider",
      line_vals=providers_both,
      line_names=[
        "assoc_scan (Triton kernel)",
        *(["warpscan (CUDA kernel)"] if HAS_WARPSCAN else []),
      ],
      plot_name=f"Kernel fwd+bwd: ({B}, {C}, seqlen)",
      args={},
    ),
  ]
)
def bench_kernel_fwdbwd(provider, seqlen, device=DEVICE):
  if provider == "warpscan_kernel":
    gates, tokens = _make_warp_data(seqlen)
    out = torch.empty_like(tokens)
    grad = torch.rand(B, C, seqlen, device=DEVICE).contiguous()
    d_gates = torch.empty_like(gates)
    d_tokens = torch.empty_like(tokens)
    warpscan_forward(gates, tokens, out, False)
    warpscan_backward(gates, out, grad, d_gates, d_tokens)

    def subject():
      warpscan_forward(gates, tokens, out, False)
      warpscan_backward(gates, out, grad, d_gates, d_tokens)

  else:  # triton_kernel
    gates, tokens, cu_seqlens, chunk_size, grid = _make_our_data(seqlen)
    out = torch.empty_like(tokens)
    grad = torch.rand_like(tokens)
    d_tokens = torch.empty_like(grad)
    d_gates = torch.empty_like(gates)
    fwd_fn = _make_direct_fwd_fn(gates, tokens, out, cu_seqlens, chunk_size, grid)
    bwd_fn = _make_direct_bwd_fn(grad, gates, out, d_tokens, d_gates,
                                  cu_seqlens, chunk_size, grid)
    fwd_fn()  # warmup
    bwd_fn()  # warmup

    def subject():
      fwd_fn()
      bwd_fn()

  with torch.inference_mode():
    ms = triton.testing.do_bench(subject, warmup=5000, rep=200)
  print(f"  f+b  {provider:>15s}  seqlen={seqlen:>6d}  {ms:.4f} ms")
  return ms


# ============================================================
# Plotting with speedup ratio
# ============================================================

def plot_with_speedup(csv_path, save_path):
  """Re-plot a benchmark CSV with secondary y-axis showing speedup ratio."""
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt

  df = pd.read_csv(csv_path)
  cols = df.columns.tolist()
  x_col = cols[0]
  seqlens = df[x_col].values

  triton_col = next((c for c in cols[1:] if "triton" in c.lower() or "assoc" in c.lower()), None)
  warp_col = next((c for c in cols[1:] if "warpscan" in c.lower() or "warp" in c.lower() or "cuda" in c.lower()), None)

  fig, ax1 = plt.subplots(figsize=(10, 6))

  if triton_col:
    ax1.loglog(seqlens, df[triton_col].values, 'o-', label='assoc_scan (Triton)',
               color='#2196F3', linewidth=2, markersize=6)
  if warp_col:
    warp = df[warp_col].values
    mask = ~np.isnan(warp)
    ax1.loglog(seqlens[mask], warp[mask], 's-', label='warpscan (CUDA)',
               color='#FF5722', linewidth=2, markersize=6)

  ax1.set_xlabel('sequence length')
  ax1.set_ylabel('ms')
  ax1.legend(loc='upper left')
  ax1.grid(True, alpha=0.3)

  if warp_col and triton_col:
    ax2 = ax1.twinx()
    warp = df[warp_col].values
    triton_vals = df[triton_col].values
    mask = ~np.isnan(warp) & ~np.isnan(triton_vals) & (triton_vals > 0)
    speedup = warp[mask] / triton_vals[mask]
    ax2.plot(seqlens[mask], speedup, 'd--', label='speedup (warpscan / Triton)',
             color='#4CAF50', linewidth=1.5, alpha=0.7, markersize=5)
    ax2.set_ylabel('speedup ratio (>1 = Triton faster)', color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.legend(loc='lower right')

  title = csv_path.stem if hasattr(csv_path, 'stem') else str(csv_path).rsplit('/', 1)[-1].replace('.csv', '')
  ax1.set_title(title)
  fig.tight_layout()

  out_png = save_path / (title + ".png")
  fig.savefig(out_png, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f"  Saved: {out_png}")


if __name__ == "__main__":
  from pathlib import Path

  save_path = Path("bench/results")
  save_path.mkdir(exist_ok=True, parents=True)

  print("=" * 70)
  print("METHODOLOGY: triton.testing.do_bench() with CUDA events")
  print("  - GPU-side timing (start_event.record -> fn() -> end_event.record)")
  print("  - L2 cache cleared between reps")
  print("  - warmup=5000ms, rep=200ms time budgets")
  print("  - All buffers pre-allocated, all kernel args pre-computed")
  print(f"  - Config: B={B}, C={C}, chunk_size<={MAX_CHUNK_SIZE}")
  print("  - Our side: direct kernel[grid]() call (no Python dispatcher)")
  print("  - Warpscan: warpscan_forward/backward (C++ binding, no autograd)")
  print("=" * 70)

  print("\n--- Forward-only kernel benchmark ---")
  bench_kernel_fwd.run(save_path=save_path, print_data=True)

  print("\n--- Backward-only kernel benchmark ---")
  bench_kernel_bwd.run(save_path=save_path, print_data=True)

  print("\n--- Fwd+Bwd kernel benchmark ---")
  bench_kernel_fwdbwd.run(save_path=save_path, print_data=True)

  # Re-plot with speedup ratio
  for csv_file in save_path.glob("Kernel*.csv"):
    plot_with_speedup(csv_file, save_path)

  print("\nDone! Results saved to bench/results/")
