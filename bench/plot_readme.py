"""Generate combined fwd/bwd/fwd+bwd benchmark plot for README."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("bench/results")

fwd = pd.read_csv(results_dir / "Kernel fwd: (8, 1536, seqlen).csv")
bwd = pd.read_csv(results_dir / "Kernel bwd: (8, 1536, seqlen).csv")
both = pd.read_csv(results_dir / "Kernel fwd+bwd: (8, 1536, seqlen).csv")

seqlens = fwd.iloc[:, 0].values
fwd_triton = fwd.iloc[:, 1].values
fwd_warp = fwd.iloc[:, 2].values
bwd_triton = bwd.iloc[:, 1].values
bwd_warp = bwd.iloc[:, 2].values
both_triton = both.iloc[:, 1].values
both_warp = both.iloc[:, 2].values

fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1.5]})

# Top: timing curves
ax = axes[0]
ax.loglog(seqlens, fwd_triton, 'o-', label='fwd Triton', color='#2196F3', linewidth=1.8, markersize=5)
ax.loglog(seqlens, fwd_warp, 's--', label='fwd warpscan', color='#2196F3', linewidth=1.2, markersize=4, alpha=0.6)
ax.loglog(seqlens, bwd_triton, 'o-', label='bwd Triton', color='#FF9800', linewidth=1.8, markersize=5)
ax.loglog(seqlens, bwd_warp, 's--', label='bwd warpscan', color='#FF9800', linewidth=1.2, markersize=4, alpha=0.6)
ax.loglog(seqlens, both_triton, 'o-', label='fwd+bwd Triton', color='#4CAF50', linewidth=2.2, markersize=6)
ax.loglog(seqlens, both_warp, 's--', label='fwd+bwd warpscan', color='#4CAF50', linewidth=1.5, markersize=5, alpha=0.6)
ax.set_ylabel('time (ms)')
ax.set_title('Kernel benchmark: assoc_scan (Triton) vs warpscan (CUDA)  —  H100 80GB, B=8, C=1536')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(seqlens[0] * 0.8, seqlens[-1] * 1.2)

# Bottom: speedup ratio (warpscan_time / triton_time, >1 = Triton faster)
ax2 = axes[1]
fwd_ratio = fwd_warp / fwd_triton
bwd_ratio = bwd_warp / bwd_triton
both_ratio = both_warp / both_triton

ax2.semilogx(seqlens, fwd_ratio, 'o-', label='fwd', color='#2196F3', linewidth=1.5, markersize=4)
ax2.semilogx(seqlens, bwd_ratio, 'o-', label='bwd', color='#FF9800', linewidth=1.5, markersize=4)
ax2.semilogx(seqlens, both_ratio, 'o-', label='fwd+bwd', color='#4CAF50', linewidth=2, markersize=5)
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
ax2.fill_between(seqlens, 1.0, both_ratio, where=(both_ratio >= 1.0), alpha=0.15, color='#4CAF50')
ax2.fill_between(seqlens, 1.0, both_ratio, where=(both_ratio < 1.0), alpha=0.15, color='#F44336')
ax2.set_xlabel('sequence length')
ax2.set_ylabel('speedup ratio')
ax2.set_ylim(0.7, 1.5)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(seqlens[0] * 0.8, seqlens[-1] * 1.2)
ax2.text(200, 1.05, 'Triton faster ↑', fontsize=8, color='#4CAF50', alpha=0.7)
ax2.text(200, 0.92, 'warpscan faster ↓', fontsize=8, color='#F44336', alpha=0.7)

fig.tight_layout()
out_path = Path("docs/h100_kernel_benchmark.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")

# Print markdown table
print("\n### Fwd+Bwd combined (training pass)")
print("| seqlen | ours (ms) | warpscan (ms) | speedup |")
print("|-------:|----------:|--------------:|--------:|")
for i in range(len(seqlens)):
    sl = int(seqlens[i])
    t = both_triton[i]
    w = both_warp[i]
    r = w / t
    marker = "**" if r >= 1.0 else ""
    print(f"| {sl} | {t:.4f} | {w:.4f} | {marker}{r:.2f}x{marker} |")

print("\n### Forward-only")
print("| seqlen | ours (ms) | warpscan (ms) | speedup |")
print("|-------:|----------:|--------------:|--------:|")
for i in range(len(seqlens)):
    sl = int(seqlens[i])
    t = fwd_triton[i]
    w = fwd_warp[i]
    r = w / t
    marker = "**" if r >= 1.0 else ""
    print(f"| {sl} | {t:.4f} | {w:.4f} | {marker}{r:.2f}x{marker} |")

print("\n### Backward-only")
print("| seqlen | ours (ms) | warpscan (ms) | speedup |")
print("|-------:|----------:|--------------:|--------:|")
for i in range(len(seqlens)):
    sl = int(seqlens[i])
    t = bwd_triton[i]
    w = bwd_warp[i]
    r = w / t
    marker = "**" if r >= 1.0 else ""
    print(f"| {sl} | {t:.4f} | {w:.4f} | {marker}{r:.2f}x{marker} |")
