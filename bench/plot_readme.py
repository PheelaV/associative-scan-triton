"""Generate combined fwd/bwd/fwd+bwd benchmark plot for README."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

results_dir = Path("bench/results")

fwd = pd.read_csv(results_dir / "Kernel fwd: (8, 1536, seqlen).csv")
bwd = pd.read_csv(results_dir / "Kernel bwd: (8, 1536, seqlen).csv")
both = pd.read_csv(results_dir / "Kernel fwd+bwd: (8, 1536, seqlen).csv")

# Each CSV may cover different seqlen ranges (VRAM-dependent)
fwd_seqlens = fwd.iloc[:, 0].values
bwd_seqlens = bwd.iloc[:, 0].values
both_seqlens = both.iloc[:, 0].values

fwd_triton = fwd.iloc[:, 1].values
fwd_warp = fwd.iloc[:, 2].values
bwd_triton = bwd.iloc[:, 1].values
bwd_warp = bwd.iloc[:, 2].values
both_triton = both.iloc[:, 1].values
both_warp = both.iloc[:, 2].values

# Use the common seqlens for the ratio subplot
common_seqlens = np.intersect1d(np.intersect1d(fwd_seqlens, bwd_seqlens), both_seqlens)
all_seqlens = np.union1d(np.union1d(fwd_seqlens, bwd_seqlens), both_seqlens)

# GPU name for title
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"

fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1.5]})

# Top: timing curves (each at its own seqlen range)
ax = axes[0]
ax.loglog(fwd_seqlens, fwd_triton, 'o-', label='fwd Triton', color='#2196F3', linewidth=1.8, markersize=5)
ax.loglog(fwd_seqlens, fwd_warp, 's--', label='fwd warpscan', color='#2196F3', linewidth=1.2, markersize=4, alpha=0.6)
ax.loglog(bwd_seqlens, bwd_triton, 'o-', label='bwd Triton', color='#FF9800', linewidth=1.8, markersize=5)
ax.loglog(bwd_seqlens, bwd_warp, 's--', label='bwd warpscan', color='#FF9800', linewidth=1.2, markersize=4, alpha=0.6)
ax.loglog(both_seqlens, both_triton, 'o-', label='fwd+bwd Triton', color='#4CAF50', linewidth=2.2, markersize=6)
ax.loglog(both_seqlens, both_warp, 's--', label='fwd+bwd warpscan', color='#4CAF50', linewidth=1.5, markersize=5, alpha=0.6)
ax.set_ylabel('time (ms)')
ax.set_title(f'Kernel benchmark: assoc_scan (Triton) vs warpscan (CUDA)  —  {gpu_name}, B=8, C=1536')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xticks(all_seqlens)
ax.set_xticklabels([str(int(s)) for s in all_seqlens], fontsize=7)
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_xlim(all_seqlens[0] * 0.7, all_seqlens[-1] * 1.4)

# Bottom: speedup ratio at common seqlens
ax2 = axes[1]
fwd_common_mask = np.isin(fwd_seqlens, common_seqlens)
bwd_common_mask = np.isin(bwd_seqlens, common_seqlens)
both_common_mask = np.isin(both_seqlens, common_seqlens)

fwd_ratio = fwd_warp[fwd_common_mask] / fwd_triton[fwd_common_mask]
bwd_ratio = bwd_warp[bwd_common_mask] / bwd_triton[bwd_common_mask]
both_ratio = both_warp[both_common_mask] / both_triton[both_common_mask]

ax2.semilogx(common_seqlens, fwd_ratio, 'o-', label='fwd', color='#2196F3', linewidth=1.5, markersize=4)
ax2.semilogx(common_seqlens, bwd_ratio, 'o-', label='bwd', color='#FF9800', linewidth=1.5, markersize=4)
ax2.semilogx(common_seqlens, both_ratio, 'o-', label='fwd+bwd', color='#4CAF50', linewidth=2, markersize=5)
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
ax2.fill_between(common_seqlens, 1.0, both_ratio, where=(both_ratio >= 1.0), alpha=0.15, color='#4CAF50')
ax2.fill_between(common_seqlens, 1.0, both_ratio, where=(both_ratio < 1.0), alpha=0.15, color='#F44336')
ax2.set_xlabel('sequence length')
ax2.set_ylabel('speedup ratio')
ax2.set_ylim(0.7, 1.5)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(common_seqlens)
ax2.set_xticklabels([str(int(s)) for s in common_seqlens], fontsize=7)
ax2.xaxis.set_minor_locator(plt.NullLocator())
ax2.set_xlim(common_seqlens[0] * 0.7, common_seqlens[-1] * 1.4)
ax2.text(200, 1.05, 'Triton faster \u2191', fontsize=8, color='#4CAF50', alpha=0.7)
ax2.text(200, 0.92, 'warpscan faster \u2193', fontsize=8, color='#F44336', alpha=0.7)

fig.tight_layout()
# Use GPU-specific filename
gpu_tag = gpu_name.lower().replace(" ", "_").replace("nvidia_geforce_rtx_", "")
out_path = Path(f"docs/{gpu_tag}_kernel_benchmark.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")

# Print markdown tables
print("\n### Fwd+Bwd combined (training pass)")
print("| seqlen | ours (ms) | warpscan (ms) | speedup |")
print("|-------:|----------:|--------------:|--------:|")
for i in range(len(both_seqlens)):
    sl = int(both_seqlens[i])
    t = both_triton[i]
    w = both_warp[i]
    r = w / t
    marker = "**" if r >= 1.0 else ""
    print(f"| {sl} | {t:.3f} | {w:.3f} | {marker}{r:.2f}x{marker} |")

print("\n### Forward-only")
print("| seqlen | ours (ms) | warpscan (ms) | speedup |")
print("|-------:|----------:|--------------:|--------:|")
for i in range(len(fwd_seqlens)):
    sl = int(fwd_seqlens[i])
    t = fwd_triton[i]
    w = fwd_warp[i]
    r = w / t
    marker = "**" if r >= 1.0 else ""
    print(f"| {sl} | {t:.3f} | {w:.3f} | {marker}{r:.2f}x{marker} |")

print("\n### Backward-only")
print("| seqlen | ours (ms) | warpscan (ms) | speedup |")
print("|-------:|----------:|--------------:|--------:|")
for i in range(len(bwd_seqlens)):
    sl = int(bwd_seqlens[i])
    t = bwd_triton[i]
    w = bwd_warp[i]
    r = w / t
    marker = "**" if r >= 1.0 else ""
    print(f"| {sl} | {t:.3f} | {w:.3f} | {marker}{r:.2f}x{marker} |")
