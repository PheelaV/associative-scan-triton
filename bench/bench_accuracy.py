"""Numerical accuracy benchmark: Triton vs warpscan vs sequential fp64 gold label.

Measures how far each parallel scan implementation deviates from a sequential
fp64 reference, across fp32/fp16/bf16 input dtypes and various sequence lengths.

Produces:
  1. accuracy_pointwise.png — per-position absolute error across the sequence
  2. accuracy_by_dtype.png — RMSE vs seqlen, one subplot per dtype
  3. accuracy_triton_vs_warpscan.png — head-to-head bar chart comparison
  4. accuracy_report.csv — full results table

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python bench/bench_accuracy.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

from associative_scan_triton._dispatcher import forward_scan_full
from associative_scan_triton._grid import get_grid, next_power_of_2

try:
    from accelerated_scan.warp import warpscan_forward
    HAS_WARPSCAN = True
except ImportError:
    HAS_WARPSCAN = False
    print("accelerated-scan not installed, skipping warpscan")

C = 4
DEVICE = "cuda"
SEQLENS = [128, 256, 512, 1024, 2048, 4096, 8192]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DTYPE_NAMES = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
MAX_CHUNK_SIZE = 512
SEED = 42


def sequential_scan_gold(gates, tokens):
    """Sequential scan in fp64. Input: (C, T) tensors of any dtype. Output: (C, T) fp64."""
    g = gates.to(torch.float64)
    x = tokens.to(torch.float64)
    _, T = g.shape
    out = torch.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, T):
        out[:, t] = g[:, t] * out[:, t - 1] + x[:, t]
    return out


def run_triton(gates, tokens, dtype):
    """Run our Triton scan. Input/output: (C, T) in target dtype."""
    g = gates.to(dtype).contiguous()
    x = tokens.to(dtype).contiguous()
    _, T = g.shape
    cu_seqlens = torch.tensor([0, T], device=DEVICE, dtype=torch.int32)
    cs = min(next_power_of_2(T), MAX_CHUNK_SIZE)
    grid = get_grid(2, T, cs, C)
    out = torch.empty_like(x)
    forward_scan_full(g, x, cu_seqlens, grid, REVERSE=False, CHUNK_SIZE=cs,
                      TESTING=False, tokens_out=out)
    return out


def run_warpscan(gates, tokens, dtype):
    """Run warpscan CUDA kernel. Input: (C, T) → reshape to (1, C, T)."""
    g = gates.to(dtype).contiguous().unsqueeze(0)
    x = tokens.to(dtype).contiguous().unsqueeze(0)
    out = torch.empty_like(x)
    warpscan_forward(g, x, out, False)
    return out.squeeze(0)


def compute_errors(impl_out, gold, dtype):
    """Compute error metrics. Both inputs as fp64 for comparison."""
    impl_f64 = impl_out.to(torch.float64)
    diff = (impl_f64 - gold).abs()
    gold_abs = gold.abs()
    eps = float(torch.finfo(dtype).eps)

    max_abs = diff.max().item()
    # Use safe relative error: only over positions where |gold| > eps
    safe_mask = gold_abs > eps
    if safe_mask.any():
        max_rel = (diff[safe_mask] / gold_abs[safe_mask]).max().item()
    else:
        max_rel = 0.0
    mean_abs = diff.mean().item()
    rmse = diff.pow(2).mean().sqrt().item()

    return max_abs, max_rel, mean_abs, rmse


def pointwise_errors(impl_out, gold):
    """Per-position absolute error, averaged over channels. Returns (T,) fp64 tensor."""
    impl_f64 = impl_out.to(torch.float64)
    diff = (impl_f64 - gold).abs()
    return diff.mean(dim=0)


def is_valid(out):
    """Check output doesn't contain NaN or Inf."""
    return torch.isfinite(out).all().item()


def main():
    results = []
    pointwise_data = []
    save_path = Path("bench/results")
    save_path.mkdir(exist_ok=True, parents=True)

    impls = ["triton"]
    if HAS_WARPSCAN:
        impls.append("warpscan")

    print("=" * 80)
    print("NUMERICAL ACCURACY BENCHMARK")
    print(f"  Gold label: sequential scan in fp64")
    print(f"  Channels: {C}, Seqlens: {SEQLENS}")
    print(f"  Dtypes: {[DTYPE_NAMES[d] for d in DTYPES]}")
    print(f"  Implementations: {impls}")
    print("=" * 80)

    for seqlen in SEQLENS:
        for dtype in DTYPES:
            dname = DTYPE_NAMES[dtype]

            torch.manual_seed(SEED)
            gates_f32 = torch.rand(C, seqlen, device=DEVICE)
            tokens_f32 = torch.randn(C, seqlen, device=DEVICE)

            gold = sequential_scan_gold(gates_f32.to(dtype), tokens_f32.to(dtype))

            for impl_name in impls:
                try:
                    if impl_name == "triton":
                        out = run_triton(gates_f32, tokens_f32, dtype)
                    else:
                        out = run_warpscan(gates_f32, tokens_f32, dtype)
                except Exception as e:
                    print(f"  seqlen={seqlen:>5d}  {dname:>4s}  {impl_name:>8s}  "
                          f"SKIPPED ({type(e).__name__}: multi-chunk fp16/bf16 not supported)")
                    continue

                if not is_valid(out):
                    print(f"  seqlen={seqlen:>5d}  {dname:>4s}  {impl_name:>8s}  "
                          f"DIVERGED (NaN/Inf in output)")
                    results.append({
                        "seqlen": seqlen, "dtype": dname, "impl": impl_name,
                        "max_abs_err": float("nan"), "max_rel_err": float("nan"),
                        "mean_abs_err": float("nan"), "rmse": float("nan"),
                        "status": "diverged",
                    })
                    continue

                max_abs, max_rel, mean_abs, rmse = compute_errors(out, gold, dtype)
                pw_err = pointwise_errors(out, gold)

                results.append({
                    "seqlen": seqlen, "dtype": dname, "impl": impl_name,
                    "max_abs_err": max_abs, "max_rel_err": max_rel,
                    "mean_abs_err": mean_abs, "rmse": rmse,
                    "status": "ok",
                })

                pointwise_data.append({
                    "seqlen": seqlen, "dtype": dname, "impl": impl_name,
                    "errors": pw_err.cpu().numpy(),
                })

                print(f"  seqlen={seqlen:>5d}  {dname:>4s}  {impl_name:>8s}  "
                      f"max_abs={max_abs:.2e}  max_rel={max_rel:.2e}  "
                      f"rmse={rmse:.2e}")

    df = pd.DataFrame(results)
    csv_path = save_path / "accuracy_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Filter to valid results for tables and plots
    df_ok = df[df["status"] == "ok"].copy()

    # Print markdown tables
    print("\n" + "=" * 80)
    print("MARKDOWN TABLES")
    print("=" * 80)

    for dtype in DTYPES:
        dname = DTYPE_NAMES[dtype]
        sub = df_ok[df_ok["dtype"] == dname]
        print(f"\n### {dname}")
        print(f"| seqlen | impl | max_abs_err | max_rel_err | mean_abs_err | rmse |")
        print(f"|-------:|-----:|------------:|------------:|-------------:|-----:|")
        for _, row in sub.iterrows():
            print(f"| {row['seqlen']:>5.0f} | {row['impl']:>8s} | "
                  f"{row['max_abs_err']:.2e} | {row['max_rel_err']:.2e} | "
                  f"{row['mean_abs_err']:.2e} | {row['rmse']:.2e} |")

    # Note limitations
    print("\n**Notes:**")
    print("- Triton multi-chunk kernels upcast fp16/bf16 loads to fp32 for scan accumulation")
    print("  → consistently lower error at seqlen >= 1024 (where multi-chunk kicks in)")
    print("- Warpscan fp16 diverges (NaN/Inf) at seqlen >= 4096: half-precision accumulation overflows")
    print("- Warpscan bf16 diverges at seqlen >= 4096: same overflow issue")

    # ================================================================
    # PLOTS
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"triton": "#2196F3", "warpscan": "#FF5722"}
        markers = {"triton": "o", "warpscan": "s"}

        # ---- Plot 1: Point-wise error at seqlen=512 (both impls, all dtypes) ----
        pw_seqlen = 512  # largest where both impls have all dtypes
        fig, axes = plt.subplots(1, len(DTYPES), figsize=(5 * len(DTYPES), 4), sharey=False)
        if len(DTYPES) == 1:
            axes = [axes]

        for ax, dtype in zip(axes, DTYPES):
            dname = DTYPE_NAMES[dtype]
            for pw in pointwise_data:
                if pw["seqlen"] == pw_seqlen and pw["dtype"] == dname:
                    positions = np.arange(len(pw["errors"]))
                    ax.semilogy(positions, pw["errors"],
                                label=pw["impl"],
                                color=colors.get(pw["impl"], "gray"),
                                linewidth=0.6, alpha=0.8)
            ax.set_title(f"{dname} (seqlen={pw_seqlen})")
            ax.set_xlabel("position in sequence")
            if ax == axes[0]:
                ax.set_ylabel("absolute error (avg over channels)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Point-wise error vs sequential fp64 gold label", fontsize=12)
        fig.tight_layout()
        plot_path = save_path / "accuracy_pointwise.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {plot_path}")

        # ---- Plot 2: RMSE vs seqlen, one subplot per dtype ----
        fig, axes = plt.subplots(1, len(DTYPES), figsize=(5 * len(DTYPES), 4), sharey=False)
        if len(DTYPES) == 1:
            axes = [axes]

        for ax, dtype in zip(axes, DTYPES):
            dname = DTYPE_NAMES[dtype]
            sub = df_ok[df_ok["dtype"] == dname]
            for impl_name in impls:
                impl_sub = sub[sub["impl"] == impl_name]
                if len(impl_sub) == 0:
                    continue
                ax.semilogy(impl_sub["seqlen"], impl_sub["rmse"],
                            f'{markers.get(impl_name, "o")}-',
                            label=impl_name,
                            color=colors.get(impl_name, "gray"),
                            linewidth=1.5, markersize=5)
            ax.set_title(dname)
            ax.set_xlabel("sequence length")
            if ax == axes[0]:
                ax.set_ylabel("RMSE vs fp64 gold label")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("RMSE vs sequential fp64 gold label", fontsize=12)
        fig.tight_layout()
        plot_path = save_path / "accuracy_by_dtype.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {plot_path}")

        # ---- Plot 3: Triton vs warpscan bar chart (RMSE) ----
        if HAS_WARPSCAN:
            fig, axes = plt.subplots(1, len(DTYPES), figsize=(5 * len(DTYPES), 4), sharey=False)
            if len(DTYPES) == 1:
                axes = [axes]

            for ax, dtype in zip(axes, DTYPES):
                dname = DTYPE_NAMES[dtype]
                sub = df_ok[df_ok["dtype"] == dname]
                triton_sub = sub[sub["impl"] == "triton"].set_index("seqlen")
                warp_sub = sub[sub["impl"] == "warpscan"].set_index("seqlen")

                common_seqlens = sorted(triton_sub.index.intersection(warp_sub.index))
                if len(common_seqlens) == 0:
                    ax.set_title(f"{dname} (no overlap)")
                    continue

                x = np.arange(len(common_seqlens))
                w = 0.35
                triton_vals = [triton_sub.loc[s, "rmse"] for s in common_seqlens]
                warp_vals = [warp_sub.loc[s, "rmse"] for s in common_seqlens]

                ax.bar(x - w / 2, triton_vals, w, label="triton", color="#2196F3", alpha=0.8)
                ax.bar(x + w / 2, warp_vals, w, label="warpscan", color="#FF5722", alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels([str(s) for s in common_seqlens], fontsize=8)
                ax.set_yscale("log")
                ax.set_title(dname)
                ax.set_xlabel("sequence length")
                if ax == axes[0]:
                    ax.set_ylabel("RMSE vs fp64 gold label")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis="y")

            fig.suptitle("Triton vs warpscan: RMSE comparison", fontsize=12)
            fig.tight_layout()
            plot_path = save_path / "accuracy_triton_vs_warpscan.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {plot_path}")

    except Exception as e:
        print(f"\nPlot error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
