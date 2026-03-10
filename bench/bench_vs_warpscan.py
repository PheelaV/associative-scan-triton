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


def _make_data(seqlen):
  """Create packed mock data: C channels, B equal-length sequences."""
  total = seqlen * B
  gates = torch.rand(C, total, device=device).contiguous()
  tokens = torch.rand(C, total, device=device).contiguous()
  cu_seqlens = torch.arange(0, B + 1, device=device, dtype=torch.int32) * seqlen
  return gates, tokens, cu_seqlens


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
      x_vals=[2**i for i in range(7, 18)],
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

    if provider == "triton_compiled":
      args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}
      # Warmup torch.compile (first call triggers compilation)
      _scan_causal_torch_compiled(gates, tokens, args)

      def subject():
        return _scan_causal_torch_compiled(gates, tokens, args)

    else:
      out_tokens = torch.empty_like(tokens)

      def subject():
        return forward_scan_full(
          gates, tokens, cu_seqlens, grid,
          REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=False,
          tokens_out=out_tokens,
        )

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
      x_vals=[2**i for i in range(7, 18)],
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

  # Secondary axis: speedup ratio (warp / our best)
  if warp_col and (eager_col or compiled_col):
    ax2 = ax1.twinx()
    warp = df[warp_col].values
    mask = ~np.isnan(warp)
    # Use compiled if available, else eager
    best_col = compiled_col or eager_col
    best = df[best_col].values
    speedup = warp[mask] / best[mask]
    ax2.plot(seqlens[mask], speedup, 'd--', label=f'speedup (warp / {"compiled" if compiled_col else "eager"})',
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

  # Copy the forward benchmark plot to docs/ for README
  import shutil
  docs_path = Path("docs")
  docs_path.mkdir(exist_ok=True)
  fwd_plot = save_path / f"Scan fwd: ({B}, {C}, seqlen), inference mode.png"
  if fwd_plot.exists():
    shutil.copy(fwd_plot, docs_path / "h100_scan_benchmark.png")
    print(f"Copied forward benchmark plot to {docs_path / 'h100_scan_benchmark.png'}")
