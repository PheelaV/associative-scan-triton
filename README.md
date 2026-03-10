# associative_scan_triton

Chunked [associative scan](https://en.wikipedia.org/wiki/Prefix_sum#Parallel_algorithms) for the first-order linear recurrence `h[t] = g[t] * h[t-1] + x[t]`, implemented in [Triton](https://github.com/triton-lang/triton). Supports variable-length sequences (packing via `cu_seqlens`), bidirectional operation, and `torch.compile`.

The recurrence is the core primitive behind gated linear RNNs (Griffin, Mamba, RWKV, xLSTM, etc.), but this implementation is architecture-agnostic — it only computes the scan.

## quick start

```python
import torch
from associative_scan_triton import scan_causal, get_grid

# 4 channels, single sequence of length 128
C, T = 4, 128
device = "cuda"

gates = torch.rand(C, T, device=device, requires_grad=True)
tokens = torch.randn(C, T, device=device, requires_grad=True)
cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

grid = get_grid(len(cu_seqlens), T, chunk_size=128, no_channels=C)
args = {"cu_seqlens": cu_seqlens, "chunk_size": 128, "grid": grid}

# forward scan: h[t] = gates[t] * h[t-1] + tokens[t]
output = scan_causal(gates, tokens, args)
output.sum().backward()  # gradients flow through
```

For bidirectional scans (two branches, opposite directions):

```python
from associative_scan_triton import scan_bidirectional_branched

y_fwd, y_bwd = scan_bidirectional_branched(
    gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args
)
```

For `torch.compile`-compatible code paths, use the `_compiled` variants:

```python
from associative_scan_triton import scan_bidirectional_branched_compiled

y_fwd, y_bwd = scan_bidirectional_branched_compiled(
    gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args
)
```

## install

```bash
uv add associative-scan-triton --git https://github.com/PheelaV/associative-scan-triton.git
```

Or for development:

```bash
git clone git@github.com:PheelaV/associative-scan-triton.git
cd associative-scan-triton
uv sync
```

Requires PyTorch >= 2.10 and Triton >= 3.6 (ships with recent PyTorch).

## what's in the box

| File | What it does |
|------|-------------|
| `_kernels.py` | Triton JIT kernels: `op`, `forward_scan_chunked`, `forward_scan_onepass_pipelined` |
| `_dispatcher.py` | `forward_scan_full` — routes single-chunk vs multi-chunk (pipelined) |
| `_shift_pad.py` | `shift_pad` (eager, torch.roll) + `shift_pad_compiled` (Triton kernel) |
| `_grid.py` | `get_grid`, `get_static_grid`, `next_power_of_2` |
| `scan_eager.py` | `scan_causal`, `scan_bidirectional_branched` — autograd-compatible |
| `scan_compiled.py` | `scan_bidirectional_branched_compiled` — `torch.compile`-compatible via `@triton_op` |

## performance

Kernel-vs-kernel benchmark on H100 80GB, `(B=8, C=1536, seqlen)`. Direct kernel calls, no autograd overhead on either side. Compared against [accelerated-scan](https://github.com/proger/accelerated-scan) (hand-tuned CUDA warp-level implementation). Timed with `triton.testing.do_bench()` (CUDA events, L2 cache cleared between reps).

![Kernel benchmark on H100 80GB](docs/h100_kernel_benchmark.png)

### Fwd+Bwd combined (training pass)

| seqlen | ours (ms) | warpscan (ms) | speedup |
|-------:|----------:|--------------:|--------:|
| 128 | 0.027 | 0.028 | **1.03x** |
| 256 | 0.043 | 0.053 | **1.22x** |
| 512 | 0.077 | 0.103 | **1.34x** |
| 1024 | 0.161 | 0.155 | 0.96x |
| 2048 | 0.306 | 0.313 | **1.02x** |
| 4096 | 0.602 | 0.542 | 0.90x |
| 8192 | 1.186 | 1.070 | 0.90x |
| 16384 | 2.322 | 2.132 | 0.92x |
| 32768 | 4.602 | 4.264 | 0.93x |
| 65536 | 9.212 | 8.573 | 0.93x |

<details>
<summary>Forward-only and backward-only breakdown</summary>

#### Forward-only

| seqlen | ours (ms) | warpscan (ms) | speedup |
|-------:|----------:|--------------:|--------:|
| 128 | 0.014 | 0.015 | **1.00x** |
| 256 | 0.020 | 0.024 | **1.20x** |
| 512 | 0.033 | 0.044 | **1.33x** |
| 1024 | 0.068 | 0.061 | 0.90x |
| 2048 | 0.126 | 0.120 | 0.95x |
| 4096 | 0.240 | 0.206 | 0.86x |
| 8192 | 0.455 | 0.401 | 0.88x |
| 16384 | 0.926 | 0.794 | 0.86x |
| 32768 | 1.811 | 1.586 | 0.88x |
| 65536 | 3.606 | 3.185 | 0.88x |

#### Backward-only

Our backward fuses the reverse scan + gate gradient computation into a single kernel. Warpscan uses a separate backward kernel.

| seqlen | ours (ms) | warpscan (ms) | speedup |
|-------:|----------:|--------------:|--------:|
| 128 | 0.020 | 0.020 | **1.02x** |
| 256 | 0.029 | 0.035 | **1.20x** |
| 512 | 0.049 | 0.065 | **1.34x** |
| 1024 | 0.099 | 0.099 | **1.00x** |
| 2048 | 0.186 | 0.199 | **1.07x** |
| 4096 | 0.366 | 0.341 | 0.93x |
| 8192 | 0.722 | 0.674 | 0.93x |
| 16384 | 1.429 | 1.344 | 0.94x |
| 32768 | 2.821 | 2.687 | 0.95x |
| 65536 | 5.623 | 5.394 | 0.96x |

</details>

**Summary**: We match or beat warpscan on fwd+bwd at seqlen <= 2048 (the common training regime). At larger sequence lengths, warpscan's hand-tuned CUDA kernel wins by ~7-10% — an inherent [Triton vs CUDA tradeoff](https://github.com/triton-lang/triton/discussions/4472) in scan operations.

### Why use this over warpscan?

- **Variable-length sequences**: native `cu_seqlens` packing — no padding to power-of-2, no wasted compute
- **torch.compile**: works with `@triton_op` wrappers, warpscan is a C++ extension that breaks the compiler
- **Bidirectional scan**: built-in, not bolted on
- **No sequence length limit**: warpscan caps at 65536, we handle arbitrary lengths

To run the benchmark yourself:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group bench python bench/bench_kernels.py
```

## how it works

The scan is split into fixed-size chunks. For a single chunk, `tl.associative_scan` runs entirely in SRAM. For multiple chunks, a single-kernel pipelined pass propagates the running prefix across chunks with software pipelining (`tl.range(num_stages=N)`) to overlap loads with compute.

Variable-length sequences are handled via `cu_seqlens` (cumulative sequence lengths, same convention as flash-attn). Each sequence is scanned independently — no cross-document leakage.

The backward pass computes `d_tokens` via a reverse scan on shifted gates, and `d_gates = shifted_states * d_tokens`.

## tests

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m pytest tests/ -v
```

Tests covering kernel numerics, forward/backward correctness against JAX reference (`jax.lax.associative_scan`), shift-pad, and compiled-vs-eager parity.

## license

MIT
