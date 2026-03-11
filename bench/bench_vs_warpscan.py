"""Benchmark: 1-kernel onepass scan vs accelerated-scan (CUDA warp).

Config: (B=8, C=1536, seqlen), inference mode, log-log axes.

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run --group bench python bench/bench_vs_warpscan.py
"""

import os

import torch
import triton

from associative_scan_triton import forward_scan_full, get_grid, next_power_of_2, scan_causal, scan_causal_compiled

# Pre-compile the compiled variant so benchmark measures steady-state
# Each seqlen produces different (chunk_size, num_chunks) → different specialization.
# Raise cache limit to cover all benchmark configs (11 seqlens).
torch._dynamo.config.cache_size_limit = 32
_scan_causal_torch_compiled = torch.compile(scan_causal_compiled)

try:
  from accelerated_scan.warp import scan as warp_scan
  HAS_WARPSCAN = True
except ImportError:
  HAS_WARPSCAN = False
  print("accelerated-scan not installed, skipping warp_scan baseline")
  print("Install with: uv add --group bench accelerated-scan")

B = 8
C = 1536
max_chunk_size = 512
device = "cuda"


def _max_seqlen_for(num_tensors, dtype_bytes=4, margin_gb=1.5):
  """Return the largest power-of-2 seqlen that fits in GPU VRAM."""
  import math
  total = torch.cuda.get_device_properties(0).total_memory
  usable = total - margin_gb * 2**30
  max_elems = usable / (num_tensors * C * B * dtype_bytes)
  return 2 ** int(math.log2(max(max_elems, 128)))


_ALL_SEQLENS = [2**i for i in range(7, 17)]
# API-level needs more margin than kernel-level due to autograd + allocator fragmentation
# Forward: ~4 tensors (gates, tokens, output, + residual from prior provider)
# Fwd+bwd with autograd: ~8 tensors (+ clones, grads, autograd graph)
SEQLENS_FWD = [s for s in _ALL_SEQLENS if s <= _max_seqlen_for(4)]
SEQLENS_BWD = [s for s in _ALL_SEQLENS if s <= _max_seqlen_for(8)]
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
print(f"  fwd seqlens: {SEQLENS_FWD[0]}..{SEQLENS_FWD[-1]} ({len(SEQLENS_FWD)} points)")
print(f"  bwd seqlens: {SEQLENS_BWD[0]}..{SEQLENS_BWD[-1]} ({len(SEQLENS_BWD)} points)")


def _make_data(seqlen):
  """Create packed mock data: C channels, B equal-length sequences."""
  total = seqlen * B
  gates = torch.rand(C, total, device=device).contiguous()
  tokens = torch.rand(C, total, device=device).contiguous()
  cu_seqlens = torch.arange(
    0, B + 1, device=device, dtype=torch.int32,
  ) * seqlen
  return gates, tokens, cu_seqlens


# ============================================================
# Global warmup: JIT-compile all Triton kernels before timing
# ============================================================

def _warmup_all():
  """Pre-compile Triton kernels for all seqlens to avoid JIT in benchmarks."""
  seqlens = SEQLENS_BWD  # warmup does fwd+bwd, use the restricted list
  for sl in seqlens:
    gates, tokens, cu_seqlens = _make_data(sl)
    cs = min(next_power_of_2(sl), max_chunk_size)
    grid = get_grid(len(cu_seqlens), sl, cs, C)
    args = {"cu_seqlens": cu_seqlens, "chunk_size": cs, "grid": grid}
    # Eager forward + backward (triggers Triton JIT)
    g = gates.clone().requires_grad_(True)
    t = tokens.clone().requires_grad_(True)
    out = scan_causal(g, t, args)
    out.sum().backward()
    # Compiled forward + backward
    g = gates.clone().requires_grad_(True)
    t = tokens.clone().requires_grad_(True)
    out = _scan_causal_torch_compiled(g, t, args)
    out.sum().backward()
  torch.cuda.synchronize()
  print("Global warmup done (all kernels JIT-compiled)")

_warmup_all()


# ============================================================
# Forward-only benchmark
# ============================================================

providers_fwd = ["triton_eager"]
providers_fwd.append("triton_compiled")
if HAS_WARPSCAN:
  providers_fwd.append("accelerated_scan")


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
        "assoc_scan (eager)",
        "assoc_scan (compiled)",
        *( ["accelerated_scan (CUDA warp)"] if HAS_WARPSCAN else []),
      ],
      plot_name=f"Scan fwd: ({B}, {C}, seqlen), inference mode",
      args={},
    ),
  ]
)
def bench_fwd(provider, seqlen, device=device):
  if provider == "accelerated_scan":
    if seqlen > 65536:
      return float("nan")  # warpscan caps at 65K
    gates = 0.999 + 0.001 * torch.rand(B, C, seqlen, device=device)
    tokens = torch.rand(B, C, seqlen, device=device)

    def subject():
      return warp_scan(gates, tokens)

  else:
    gates, tokens, cu_seqlens = _make_data(seqlen)
    chunk_size = min(next_power_of_2(seqlen), max_chunk_size)
    grid = get_grid(len(cu_seqlens), seqlen, chunk_size, C)
    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

    if provider == "triton_compiled":
      # Warmup torch.compile (first call triggers compilation)
      _scan_causal_torch_compiled(gates, tokens, args)

      def subject():
        return _scan_causal_torch_compiled(gates, tokens, args)

    else:
      def subject():
        return scan_causal(gates, tokens, args)

  with torch.inference_mode():
    ms = triton.testing.do_bench(subject, warmup=5000, rep=100)
  print(f"{provider:>30s}  {seqlen=:>6d}  {ms=:.4f}")
  return ms


# ============================================================
# Forward + backward benchmark
# ============================================================

providers_fwdbwd = ["triton_eager"]
providers_fwdbwd.append("triton_compiled")
if HAS_WARPSCAN:
  providers_fwdbwd.append("accelerated_scan")


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
      line_vals=providers_fwdbwd,
      line_names=[
        "assoc_scan (eager)",
        "assoc_scan (compiled)",
        *( ["accelerated_scan (CUDA warp)"] if HAS_WARPSCAN else []),
      ],
      plot_name=f"Scan fwd+bwd: ({B}, {C}, seqlen)",
      args={},
    ),
  ]
)
def bench_fwdbwd(provider, seqlen, device=device):
  if provider == "accelerated_scan":
    if seqlen > 65536:
      return float("nan")  # warpscan caps at 65K
    gates_src = 0.999 + 0.001 * torch.rand(B, C, seqlen, device=device)
    tokens_src = torch.rand(B, C, seqlen, device=device)

    def subject():
      g = gates_src.clone().requires_grad_(True)
      t = tokens_src.clone().requires_grad_(True)
      out = warp_scan(g, t)
      out.sum().backward()
      return out

  else:
    gates_src, tokens_src, cu_seqlens = _make_data(seqlen)
    chunk_size = min(next_power_of_2(seqlen), max_chunk_size)
    grid = get_grid(len(cu_seqlens), seqlen, chunk_size, C)
    args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}
    scan_fn = _scan_causal_torch_compiled if provider == "triton_compiled" else scan_causal

    if provider == "triton_compiled":
      # Warmup torch.compile (triggers compilation for this seqlen)
      g = gates_src.clone().requires_grad_(True)
      t = tokens_src.clone().requires_grad_(True)
      out = scan_fn(g, t, args)
      out.sum().backward()

    def subject():
      g = gates_src.clone().requires_grad_(True)
      t = tokens_src.clone().requires_grad_(True)
      out = scan_fn(g, t, args)
      out.sum().backward()
      return out

  ms = triton.testing.do_bench(subject, warmup=500, rep=100)
  print(f"{provider:>30s}  {seqlen=:>6d}  fwd+bwd {ms=:.4f}")
  return ms


# ============================================================
# Custom plotting with speedup ratio on y2 axis
# ============================================================

def plot_with_speedup(csv_path, save_path):
  """Re-plot a benchmark CSV with a secondary y-axis showing speedup ratio."""
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from matplotlib.ticker import NullLocator

  df = pd.read_csv(csv_path)
  cols = df.columns.tolist()
  x_col = cols[0]
  seqlens = df[x_col].values

  # Identify columns by name patterns
  eager_col = next((c for c in cols[1:] if "eager" in c.lower()), None)
  compiled_col = next((c for c in cols[1:] if "compiled" in c.lower()), None)
  warp_col = next((c for c in cols[1:] if "accelerated" in c.lower() or "warp" in c.lower()), None)

  fig, ax1 = plt.subplots(figsize=(10, 6))

  # Primary axis: timing (log-log)
  if eager_col:
    ax1.loglog(seqlens, df[eager_col].values, 'o-', label='assoc_scan (eager)', color='#2196F3', linewidth=2)
  if compiled_col:
    ax1.loglog(seqlens, df[compiled_col].values, '^-', label='assoc_scan (compiled)', color='#9C27B0', linewidth=2)
  if warp_col:
    warp = df[warp_col].values
    mask = ~np.isnan(warp)
    ax1.loglog(seqlens[mask], warp[mask], 's-', label='accelerated_scan (CUDA warp)', color='#FF5722', linewidth=2)

  ax1.set_xlabel('sequence length')
  ax1.set_ylabel('ms')
  ax1.legend(loc='upper left')
  ax1.grid(True, alpha=0.3)
  # Power-of-2 x-axis ticks (data points are powers of 2)
  ax1.set_xticks(seqlens[~np.isnan(seqlens)])
  ax1.set_xticklabels(
    [str(int(s)) for s in seqlens if not np.isnan(s)], fontsize=7,
  )
  ax1.xaxis.set_minor_locator(NullLocator())

  # Secondary axis: speedup ratio (warp / eager)
  # Compare against eager since warpscan doesn't support torch.compile
  if warp_col and eager_col:
    ax2 = ax1.twinx()
    warp = df[warp_col].values
    mask = ~np.isnan(warp)
    eager = df[eager_col].values
    speedup = warp[mask] / eager[mask]
    ax2.plot(seqlens[mask], speedup, 'd--',
             label='speedup (warp / eager)',
             color='#4CAF50', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('speedup ratio', color='#4CAF50')
    ax2.tick_params(axis='y', labelcolor='#4CAF50')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.legend(loc='lower right')

  title = csv_path.stem if hasattr(csv_path, 'stem') else str(csv_path).rsplit('/', 1)[-1].replace('.csv', '')
  ax1.set_title(title)

  fig.tight_layout()
  out_png = save_path / (title + ".png")
  fig.savefig(out_png, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f"Saved plot with speedup axis: {out_png}")


if __name__ == "__main__":
  from pathlib import Path

  save_path = Path("bench/results")
  save_path.mkdir(exist_ok=True, parents=True)

  print("=" * 60)
  print("Forward-only benchmark")
  print("=" * 60)
  bench_fwd.run(save_path=save_path, print_data=True)

  print()
  print("=" * 60)
  print("Forward + backward benchmark")
  print("=" * 60)
  bench_fwdbwd.run(save_path=save_path, print_data=True)

  # Re-plot with speedup ratio on y2 axis
  for csv_file in save_path.glob("*.csv"):
    plot_with_speedup(csv_file, save_path)

  # NOTE: This is an API-level benchmark (includes Python wrapper
  # overhead). For kernel-vs-kernel comparison, use bench_kernels.py
  # which produces the README graph (docs/h100_kernel_benchmark.png).
