"""Benchmark: 1-kernel onepass scan vs accelerated-scan (CUDA warp).

Config: (B=8, C=1536, seqlen), inference mode, log-log axes.

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run --group bench python bench/bench_vs_warpscan.py
"""

import os

import torch
import triton

from associative_scan_triton import forward_scan_full, get_grid, next_power_of_2

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


providers = ["associative_scan_triton"]
if HAS_WARPSCAN:
  providers.append("accelerated_scan")


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
      line_vals=providers,
      line_names=[
        "associative_scan_triton (Triton, 1-kernel onepass)",
        *( ["accelerated_scan (CUDA warp)"] if HAS_WARPSCAN else []),
      ],
      plot_name=f"Scan kernel benchmark: ({B}, {C}, seqlen), inference mode",
      args={},
    ),
  ]
)
def bench(provider, seqlen, device=device):
  if provider == "accelerated_scan":
    gates = 0.999 + 0.001 * torch.rand(B, C, seqlen, device=device)
    tokens = torch.rand(B, C, seqlen, device=device)

    def subject():
      return warp_scan(gates, tokens)

  else:
    gates, tokens, cu_seqlens = _make_data(seqlen)
    chunk_size = min(next_power_of_2(seqlen), max_chunk_size)
    grid = get_grid(len(cu_seqlens), seqlen, chunk_size, C)

    def subject():
      return forward_scan_full(
        gates, tokens, cu_seqlens, grid,
        REVERSE=False, CHUNK_SIZE=chunk_size, TESTING=False,
      )

  with torch.inference_mode():
    ms = triton.testing.do_bench(subject, warmup=5000, rep=100)
  print(f"{provider:>30s}  {seqlen=:>6d}  {ms=:.4f}")
  return ms


if __name__ == "__main__":
  from pathlib import Path

  save_path = Path("bench/results")
  save_path.mkdir(exist_ok=True, parents=True)
  bench.run(save_path=save_path, print_data=True)
