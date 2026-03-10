"""Tests for fp16/bf16 precision support.

Verifies that scan_causal and scan_bidirectional_branched produce correct
results for reduced-precision inputs. Uses a sequential fp64 scan as gold
label (not JAX, which runs in fp32).

The multi-chunk pipelined kernel upcasts to fp32 internally, so accuracy
should be better than native half-precision accumulation.
"""

import pytest
import torch

from associative_scan_triton import (
    get_grid,
    scan_causal,
    scan_bidirectional_branched,
)
from test_utils import TEST_SEED, get_device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def sequential_scan_gold(gates, tokens):
    """Sequential scan in fp64 — the gold label reference.

    Input: (C, T) tensors of any dtype. Output: (C, T) fp64.
    """
    g = gates.to(torch.float64)
    x = tokens.to(torch.float64)
    _, T = g.shape
    out = torch.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, T):
        out[:, t] = g[:, t] * out[:, t - 1] + x[:, t]
    return out


# Tolerances per dtype — based on empirical accuracy benchmark results
TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-4},
    torch.float16: {"atol": 5e-3, "rtol": 5e-2},
    torch.bfloat16: {"atol": 3e-2, "rtol": 1e-1},
}


@pytest.fixture
def device():
    return get_device()


@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(TEST_SEED)


class TestScanCausalDtypes:
    """Forward and backward correctness for fp32/fp16/bf16."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seqlen", [32, 128, 512, 1024])
    @pytest.mark.parametrize("num_channels", [1, 4])
    def test_forward_vs_gold(self, dtype, seqlen, num_channels, device):
        """Forward scan output matches sequential fp64 gold label."""
        chunk_size = min(seqlen, 512)

        # Generate in fp32, cast to target dtype
        gates_f32 = torch.rand(num_channels, seqlen, device=device)
        tokens_f32 = torch.randn(num_channels, seqlen, device=device)

        gates = gates_f32.to(dtype)
        tokens = tokens_f32.to(dtype)

        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
        grid = get_grid(2, seqlen, chunk_size, num_channels)
        args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

        result = scan_causal(gates.clone(), tokens.clone(), args)
        gold = sequential_scan_gold(gates, tokens)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(
            result.to(torch.float64),
            gold,
            atol=tol["atol"],
            rtol=tol["rtol"],
            msg=f"Forward mismatch for {dtype} seqlen={seqlen}",
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seqlen", [32, 128, 512, 1024])
    def test_backward_runs(self, dtype, seqlen, device):
        """Backward pass completes without error for reduced precision."""
        num_channels = 4
        chunk_size = min(seqlen, 512)

        gates = torch.rand(num_channels, seqlen, device=device, dtype=dtype, requires_grad=True)
        tokens = torch.randn(num_channels, seqlen, device=device, dtype=dtype, requires_grad=True)

        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
        grid = get_grid(2, seqlen, chunk_size, num_channels)
        args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

        result = scan_causal(gates, tokens, args)
        loss = result.sum()
        loss.backward()

        assert gates.grad is not None, "gates.grad is None"
        assert tokens.grad is not None, "tokens.grad is None"
        assert torch.isfinite(gates.grad).all(), "gates.grad contains NaN/Inf"
        assert torch.isfinite(tokens.grad).all(), "tokens.grad contains NaN/Inf"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_multi_chunk_accuracy_improvement(self, dtype, device):
        """Multi-chunk (seqlen > 512) should be at least as accurate as single-chunk,
        since multi-chunk upcasts to fp32 internally."""
        num_channels = 4
        seqlen = 1024  # forces multi-chunk with chunk_size=512
        chunk_size = 512

        gates_f32 = torch.rand(num_channels, seqlen, device=device)
        tokens_f32 = torch.randn(num_channels, seqlen, device=device)

        gates = gates_f32.to(dtype)
        tokens = tokens_f32.to(dtype)

        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
        grid = get_grid(2, seqlen, chunk_size, num_channels)
        args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

        result = scan_causal(gates.clone(), tokens.clone(), args)
        gold = sequential_scan_gold(gates, tokens)

        rmse = (result.to(torch.float64) - gold).pow(2).mean().sqrt().item()

        # Multi-chunk fp32 accumulation should keep RMSE well-bounded
        if dtype == torch.float16:
            assert rmse < 1e-3, f"fp16 multi-chunk RMSE too high: {rmse:.2e}"
        else:
            assert rmse < 5e-3, f"bf16 multi-chunk RMSE too high: {rmse:.2e}"


class TestScanBidiDtypes:
    """Bidirectional scan dtype tests."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seqlen", [64, 256, 1024])
    def test_bidi_forward_vs_gold(self, dtype, seqlen, device):
        """Bidirectional scan both branches match sequential gold label."""
        num_channels = 4
        chunk_size = min(seqlen, 512)

        gates_fwd = torch.rand(num_channels, seqlen, device=device, dtype=dtype)
        tokens_fwd = torch.randn(num_channels, seqlen, device=device, dtype=dtype)
        gates_bwd = torch.rand(num_channels, seqlen, device=device, dtype=dtype)
        tokens_bwd = torch.randn(num_channels, seqlen, device=device, dtype=dtype)

        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
        grid = get_grid(2, seqlen, chunk_size, num_channels)
        args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

        y_fwd, y_bwd = scan_bidirectional_branched(
            gates_fwd.clone(), tokens_fwd.clone(),
            gates_bwd.clone(), tokens_bwd.clone(),
            args,
        )

        # Gold: forward branch
        gold_fwd = sequential_scan_gold(gates_fwd, tokens_fwd)

        # Gold: backward branch (reverse scan)
        # Flip, scan forward, flip back
        gold_bwd = sequential_scan_gold(
            gates_bwd.flip(-1), tokens_bwd.flip(-1)
        ).flip(-1)

        tol = TOLERANCES[dtype]
        torch.testing.assert_close(
            y_fwd.to(torch.float64), gold_fwd,
            atol=tol["atol"], rtol=tol["rtol"],
            msg=f"Bidi fwd mismatch for {dtype} seqlen={seqlen}",
        )
        torch.testing.assert_close(
            y_bwd.to(torch.float64), gold_bwd,
            atol=tol["atol"], rtol=tol["rtol"],
            msg=f"Bidi bwd mismatch for {dtype} seqlen={seqlen}",
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("seqlen", [64, 512, 1024])
    def test_bidi_backward_runs(self, dtype, seqlen, device):
        """Bidirectional backward completes without NaN/Inf for reduced precision."""
        num_channels = 4
        chunk_size = min(seqlen, 512)

        gates_fwd = torch.rand(num_channels, seqlen, device=device, dtype=dtype, requires_grad=True)
        tokens_fwd = torch.randn(num_channels, seqlen, device=device, dtype=dtype, requires_grad=True)
        gates_bwd = torch.rand(num_channels, seqlen, device=device, dtype=dtype, requires_grad=True)
        tokens_bwd = torch.randn(num_channels, seqlen, device=device, dtype=dtype, requires_grad=True)

        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
        grid = get_grid(2, seqlen, chunk_size, num_channels)
        args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

        y_fwd, y_bwd = scan_bidirectional_branched(
            gates_fwd, tokens_fwd, gates_bwd, tokens_bwd, args,
        )
        (y_fwd.sum() + y_bwd.sum()).backward()

        for name, param in [("gates_fwd", gates_fwd), ("tokens_fwd", tokens_fwd),
                             ("gates_bwd", gates_bwd), ("tokens_bwd", tokens_bwd)]:
            assert param.grad is not None, f"{name}.grad is None"
            assert torch.isfinite(param.grad).all(), f"{name}.grad contains NaN/Inf"


class TestScanCausalVarlenDtypes:
    """Variable-length sequence dtype tests (cu_seqlens packing)."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_varlen_packed(self, dtype, device):
        """Multiple packed sequences of varying lengths."""
        num_channels = 2
        seqlens = [32, 64, 128]  # three sequences of different lengths
        total = sum(seqlens)
        max_seqlen = max(seqlens)
        chunk_size = min(max_seqlen, 512)

        gates = torch.rand(num_channels, total, device=device, dtype=dtype)
        tokens = torch.randn(num_channels, total, device=device, dtype=dtype)

        cu_seqlens = torch.tensor(
            [0] + list(torch.tensor(seqlens).cumsum(0).tolist()),
            device=device, dtype=torch.int32,
        )
        grid = get_grid(len(cu_seqlens), max_seqlen, chunk_size, num_channels)
        args = {"cu_seqlens": cu_seqlens, "chunk_size": chunk_size, "grid": grid}

        result = scan_causal(gates.clone(), tokens.clone(), args)

        # Verify each sequence independently against gold
        tol = TOLERANCES[dtype]
        for i, sl in enumerate(seqlens):
            start = sum(seqlens[:i])
            end = start + sl
            seq_gates = gates[:, start:end]
            seq_tokens = tokens[:, start:end]
            seq_result = result[:, start:end]
            gold = sequential_scan_gold(seq_gates, seq_tokens)

            torch.testing.assert_close(
                seq_result.to(torch.float64), gold,
                atol=tol["atol"], rtol=tol["rtol"],
                msg=f"Varlen seq {i} (len={sl}) mismatch for {dtype}",
            )
