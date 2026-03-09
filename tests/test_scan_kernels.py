import numpy as np
import pytest
import torch

from associative_scan_triton import get_grid
from test_utils import (
  TEST_SEED,
  get_device,
  get_mock_data,
  get_mock_data_simple,
)
from conftest import first_order_op


def _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels):
  """Helper that mimics the old get_grid(cu_seqlens, chunk_size, no_channels) API."""
  max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
  no_seq = len(cu_seqlens)
  return get_grid(no_seq, max_seqlen, chunk_size, no_channels)


class TestScanKernels:
  @pytest.mark.parametrize(
    ("a", "b", "c"),
    [
      ((0.5, 0.3), (0.6, 0.9), (0.7, 0.8)),
      ((0.6, 0.9), (0.5, 0.3), (0.7, 0.8)),
      ((0.6, 0.9), (0.7, 0.8), (0.5, 0.3)),
    ],
  )
  def test_op_associativity(self, a, b, c) -> None:
    """Just to be sure."""
    # Act
    # Compute (A * B) * C
    ab = first_order_op(a[0], a[1], b[0], b[1])
    abc1 = first_order_op(ab[0], ab[1], c[0], c[1])

    # Compute A * (B * C)
    bc = first_order_op(b[0], b[1], c[0], c[1])
    abc2 = first_order_op(a[0], a[1], bc[0], bc[1])

    # Assert
    assert np.allclose(abc1, abc2)

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_call(self) -> None:
    """It should not crash."""
    # Setup
    from associative_scan_triton import forward_scan_chunked

    torch.random.manual_seed(TEST_SEED)

    device = get_device()

    no_channels = 2
    L = 8
    chunk_size = 8  # has got to be a power of 2
    seqlens = torch.tensor([3, 5, 3, 8, 2])
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens.cumsum(dim=0))).to(
      device
    )
    grid = _get_grid_from_cu_seqlens(
      cu_seqlens=cu_seqlens, chunk_size=chunk_size, no_channels=no_channels
    )

    gates = torch.rand(L, device=device).contiguous()
    tokens = (
      torch.rand(L, device=device)
      + torch.arange(L, device=device).contiguous()
    )
    cu_seqlens = cu_seqlens.to(device=device)

    # Act
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=True,
      TESTING=True,
    )
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=False,
      TESTING=True,
    )

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_call_chunks(self) -> None:
    """It should not crash on chunking input, still no hard assertions."""
    # Setup
    from associative_scan_triton import forward_scan_chunked

    device = get_device()

    no_channels = 2
    chunk_size = 4
    seqlens = np.array([3, 5, 8])
    params = get_mock_data(
      no_channels=no_channels,
      max_len=8,
      multiplier=1,
      device=device,
      seqlens=seqlens,
    )
    gates, tokens, cu_seqlens = params

    grid = _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels)

    # Act
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=True,
      REVERSE=False,
      TESTING=True,
    )

    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=False,
      REVERSE=False,
      TESTING=True,
    )

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  # fmt: off
  @pytest.mark.parametrize(
    "chunk_size",
    [
      4,
      8,
      16,
    ],
  )
  # fmt: on
  def test_fwd_chunk_numeric_single_chunks(self, chunk_size) -> None:
    """Assert values for a single channel and single chunk."""
    # Setup
    from associative_scan_triton import forward_scan_chunked

    device = get_device()

    no_channels = 1
    seqlens = torch.tensor([4])
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens.cumsum(dim=0))).to(
      device
    )
    grid = _get_grid_from_cu_seqlens(
      cu_seqlens=cu_seqlens, chunk_size=chunk_size, no_channels=no_channels
    )

    gates = torch.tensor([1.0, 1.5, 0.8, 2.0]).squeeze(0).to(device)
    tokens = torch.tensor([1.0, -1.0, 0.5, 2.0]).squeeze(0).to(device)

    expected_result_gates_first_call = (
      torch.tensor([1.0, 1.5, 1.2, 2.4]).squeeze(0).to(device)
    )
    expected_result_tokens_first_call = (
      torch.tensor([1.0, 0.5, 0.9, 3.8]).squeeze(0).to(device)
    )

    expected_result_gates_second_call = (
      torch.tensor([1, 1.5, 1.2, 2.4]).squeeze(0).to(device)
    )
    expected_result_tokens_second_call = (
      torch.tensor([1, 0.5, 0.9, 3.8]).squeeze(0).to(device)
    )
    # Act
    first_gates = gates.clone()
    first_tokens = tokens.clone()

    forward_scan_chunked[grid](
      gates_ptr=first_gates,
      tokens_ptr=first_tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=True,
      TESTING=True,
    )
    # this is redundant as it should not be called twice when using a single
    # chunk, but logic supports it so it is tested
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=False,
      TESTING=True,
    )

    # Assert
    assert torch.allclose(expected_result_gates_first_call, first_gates)
    assert torch.allclose(expected_result_tokens_first_call, first_tokens)

    assert torch.allclose(expected_result_gates_second_call, gates)
    assert torch.allclose(expected_result_tokens_second_call, tokens)

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  # fmt: off
  @pytest.mark.parametrize(
    "chunk_size",
    [
      4,
      8,
      16,
    ],
  )
  # fmt: on
  def test_fwd_chunk_numeric_single_chunks_reverse(self, chunk_size) -> None:
    """Assert values for a single channel and single chunk."""
    # Setup
    from associative_scan_triton import forward_scan_chunked

    device = get_device()

    no_channels = 1
    seqlens = torch.tensor([4])
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens.cumsum(dim=0))).to(
      device
    )
    grid = _get_grid_from_cu_seqlens(
      cu_seqlens=cu_seqlens, chunk_size=chunk_size, no_channels=no_channels
    )

    gates = torch.tensor([1.0, 1.5, 0.8, 2.0]).squeeze(0).to(device)
    tokens = torch.tensor([1.0, -1.0, 0.5, 2.0]).squeeze(0).to(device)

    expected_result_gates = (
      torch.tensor([2.4, 2.4, 1.6, 2.0]).squeeze(0).to(device)
    )
    expected_result_tokens = (
      torch.tensor([3.15, 2.15, 2.1, 2.0]).squeeze(0).to(device)
    )
    # Act
    first_gates = gates.clone()
    first_tokens = tokens.clone()

    forward_scan_chunked[grid](
      gates_ptr=first_gates,
      tokens_ptr=first_tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=True,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=True,
      TESTING=True,
    )
    # this is redundant as it should not be called twice when using a single
    # chunk, but logic supports it so it is tested
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=True,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=False,
      TESTING=True,
    )

    # Assert
    assert torch.allclose(expected_result_gates, first_gates)
    assert torch.allclose(expected_result_tokens, first_tokens)

    assert torch.allclose(expected_result_gates, gates)
    assert torch.allclose(expected_result_tokens, tokens)

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numeric_even_chunks(self) -> None:
    """Assert values for chunking on even chunks and a single channel."""
    # Setup
    from associative_scan_triton import forward_scan_chunked

    device = get_device()
    repeat = 2
    seqlen = 8

    no_channels = 1
    chunk_size = 4
    num_chunks = 2
    seqlens = torch.tensor([seqlen])
    assert (chunk_size * num_chunks >= seqlens).all().item()
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens.cumsum(dim=0))).to(
      device
    )
    gates = (
      torch.tensor([1.0, 1.5, 0.8, 2.0]).repeat(repeat).squeeze(0).to(device)
    )
    tokens = (
      torch.tensor([1.0, -1.0, 0.5, 2.0]).repeat(repeat).squeeze(0).to(device)
    )
    gates_first = gates.clone()
    tokens_first = tokens.clone()
    grid = _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels)
    # there are multiple chunks and this is the second call that does not
    # update ports
    expected_result_gates_first = (
      torch.tensor([1, 1.5, 0.8, 2.4]).repeat(repeat).squeeze(0).to(device)
    )
    expected_result_tokens_first = (
      torch.tensor([1, -1.0, 0.5, 3.8]).repeat(repeat).squeeze(0).to(device)
    )
    expected_result_gates_second = (
      torch.tensor([1, 1.5, 1.2, 2.0]).repeat(repeat).squeeze(0).to(device)
    )
    expected_result_tokens_second = (
      torch.tensor([1, 0.5, 0.9, 2.0]).repeat(repeat).squeeze(0).to(device)
    )

    # Act
    forward_scan_chunked[grid](
      gates_ptr=gates_first,
      tokens_ptr=tokens_first,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=True,
      TESTING=True,
    )
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=False,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=False,
      TESTING=True,
    )

    # Assert
    assert torch.allclose(expected_result_gates_first, gates_first)
    assert torch.allclose(expected_result_tokens_first, tokens_first)
    assert torch.allclose(expected_result_gates_second, gates)
    assert torch.allclose(expected_result_tokens_second, tokens)

  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numeric_even_chunks_reverse(self) -> None:
    """Assert values for chunking on even chunks and a single channel."""
    # Setup
    from associative_scan_triton import forward_scan_chunked

    device = get_device()
    repeat = 2
    seqlen = 8

    no_channels = 1
    chunk_size = 4
    num_chunks = 2
    seqlens = torch.tensor([seqlen])
    assert (chunk_size * num_chunks >= seqlens).all().item()
    cu_seqlens = torch.cat((torch.tensor([0]), seqlens.cumsum(dim=0))).to(
      device
    )
    gates = (
      torch.tensor([1.0, 1.5, 0.8, 2.0]).repeat(repeat).squeeze(0).to(device)
    )
    tokens = (
      torch.tensor([1.0, -1.0, 0.5, 2.0]).repeat(repeat).squeeze(0).to(device)
    )
    grid = _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels)
    # only the ports should be updated
    expected_result_gates = (
      torch.tensor([2.4, 1.5, 0.8, 2.0]).repeat(repeat).squeeze(0).to(device)
    )
    expected_result_tokens = (
      torch.tensor([3.15, -1.0, 0.5, 2.0]).repeat(repeat).squeeze(0).to(device)
    )

    # Act
    forward_scan_chunked[grid](
      gates_ptr=gates,
      tokens_ptr=tokens,
      cu_seqlens_ptr=cu_seqlens,
      REVERSE=True,
      CHUNK_SIZE=chunk_size,
      FIRST_CALL=True,
      TESTING=True,
    )

    # Assert
    assert torch.allclose(expected_result_gates, gates)
    assert torch.allclose(expected_result_tokens, tokens)

  @pytest.mark.parametrize(
    ("seqlen", "chunk_size", "no_channels"),
    [
      (4, 2, 2),
      (8, 4, 1),  # this is just to try to throw it of
      (8, 4, 2),
      (8, 4, 3),
      (8, 2, 1),
      (8, 2, 2),
      (8, 2, 3),  # by also trying with odd channels
      (16, 8, 2),
      (16, 4, 2),  # this was the default before
      (16, 2, 2),
    ],
  )
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numerical_one_vs_multiple_chunks(
    self, seqlen, chunk_size, no_channels
  ) -> None:
    """This is a numerical test, but it uses itself as an assertion.

    When number of chunks is one it *should yield correct* results, when we
    calculate one non-chunked result we should get the same with the chunked
    version.

    This is really just testing the chunking when nicely matched with seqlen,
    so it is foing on the aggregating call.
    """
    # Setup
    from associative_scan_triton import forward_scan_full

    torch.random.manual_seed(TEST_SEED)

    device = get_device()
    reverse = False
    seq_multiplier = 10
    chunked_chunk_size = chunk_size
    params = get_mock_data_simple(
      no_channels=no_channels,
      seqlen=seqlen,
      seq_multiplier=seq_multiplier,
      device=device,
    )
    gates, tokens, cu_seqlens = params
    chunked_grid = _get_grid_from_cu_seqlens(
      cu_seqlens=cu_seqlens,
      chunk_size=chunked_chunk_size,
      no_channels=no_channels,
    )
    chunked_tokens = tokens.clone()
    chunked_gates = gates.clone()
    single_chunk_size = seqlen
    single_grid = _get_grid_from_cu_seqlens(
      cu_seqlens=cu_seqlens,
      chunk_size=single_chunk_size,
      no_channels=no_channels,
    )

    # Act
    forward_scan_full(
      gates=chunked_gates,
      tokens=chunked_tokens,
      cu_seqlens=cu_seqlens,
      CHUNK_SIZE=chunked_chunk_size,
      grid=chunked_grid,
      REVERSE=reverse,
    )
    forward_scan_full(
      gates=gates,
      tokens=tokens,
      cu_seqlens=cu_seqlens,
      CHUNK_SIZE=single_chunk_size,
      grid=single_grid,
      REVERSE=reverse,
    )

    # Assert
    torch.allclose(chunked_gates, gates)
    torch.allclose(chunked_tokens, tokens)

  @pytest.mark.parametrize(
    ("seqlen", "chunk_size", "no_channels"),
    [
      (4, 2, 2),
      (8, 4, 1),  # this is just to try to throw it of
      (8, 4, 2),
      (8, 4, 3),
      (8, 2, 1),
      (8, 2, 2),
      (8, 2, 3),  # by also trying with odd channels
      (16, 8, 2),
      (16, 4, 2),  # this was the default before
      (16, 2, 2),
    ],
  )
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numerical_one_vs_multiple_chunks_reversed(
    self, seqlen, chunk_size, no_channels
  ) -> None:
    """This is a numerical test, but it uses itself as an assertion.

    When number of chunks is one it *should yield correct* results, when we
    calculate one non-chunked result we should get the same with the chunked
    version.

    This is really just testing the chunking when nicely matched with seqlen,
    so it is focusing on the aggregating call. I.e. it will be same right or
    same wrong in some cases.
    """
    # Setup
    from associative_scan_triton import forward_scan_full

    torch.random.manual_seed(TEST_SEED)

    device = get_device()
    reverse = True
    seq_multiplier = 10
    chunked_chunk_size = chunk_size
    params = get_mock_data_simple(
      no_channels=no_channels,
      seqlen=seqlen,
      seq_multiplier=seq_multiplier,
      device=device,
    )

    gates, tokens, cu_seqlens = params
    chunked_grid = _get_grid_from_cu_seqlens(
      cu_seqlens=cu_seqlens,
      chunk_size=chunked_chunk_size,
      no_channels=no_channels,
    )

    single_chunk_size = seqlen
    chunked_tokens = tokens.clone()
    chunked_gates = gates.clone()
    single_grid = _get_grid_from_cu_seqlens(
      cu_seqlens=cu_seqlens,
      chunk_size=single_chunk_size,
      no_channels=no_channels,
    )

    # Act
    forward_scan_full(
      gates=chunked_gates,
      tokens=chunked_tokens,
      cu_seqlens=cu_seqlens,
      CHUNK_SIZE=chunked_chunk_size,
      grid=chunked_grid,
      REVERSE=reverse,
    )
    forward_scan_full(
      gates=gates,
      tokens=tokens,
      cu_seqlens=cu_seqlens,
      CHUNK_SIZE=single_chunk_size,
      grid=single_grid,
      REVERSE=reverse,
    )

    # Assert
    torch.allclose(chunked_gates, gates)
    torch.allclose(chunked_tokens, tokens)

  # fmt: off
  @pytest.mark.parametrize("chunk_size", [
      2,
      4,
      8,
    ])
  # fmt: on
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numerical_simple_expected_copied_sequences(self, chunk_size) -> None:  # noqa: E501
    """Assert correct values are returned for multiplied sequences."""
    # Setup
    from associative_scan_triton import forward_scan_full

    device = get_device()
    repeat = 2
    no_channels = 1
    seqlen = 4
    seqlens = torch.tensor([seqlen]).repeat(repeat)
    cu_seqlens = torch.cat((
      torch.tensor([0]), seqlens.cumsum(dim=0)
    )).to(device)
    gates = torch.tensor(
      [1.0, 1.5, 0.8, 2.0]
    ).squeeze(0).repeat(repeat).to(device)
    tokens = torch.tensor(
      [1.0, -1.0, 0.5, 2.0]
    ).squeeze(0).repeat(repeat).to(device)
    grid = _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels)

    expected_result_gates = torch.tensor(
      [1, 1.5, 1.2, 2.4]
    ).squeeze(0).repeat(repeat).to(device)
    expected_result_tokens = torch.tensor(
      [1, 0.5, 0.9, 3.8]
    ).squeeze(0).repeat(repeat).to(device)
    # Act
    forward_scan_full(
      gates=gates,
      tokens=tokens,
      cu_seqlens=cu_seqlens,
      CHUNK_SIZE=chunk_size,
      grid=grid,
      REVERSE=False,
      TESTING=True,
    )
    # Assert
    assert torch.allclose(expected_result_gates, gates)
    assert torch.allclose(expected_result_tokens, tokens)

  # fmt: off
  @pytest.mark.parametrize("chunk_size", [
      2,
      4,
      8,
    ])
  # fmt: on
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numerical_simple_multiple_varlen_sequences(self, chunk_size) -> None:  # noqa: E501
    """Assert correct values are returned for multiple varlen sequences."""
    # Setup
    from associative_scan_triton import forward_scan_full

    device = get_device()

    no_channels = 1
    seqlens = torch.tensor([4, 5])
    cu_seqlens = torch.cat((
      torch.tensor([0]), seqlens.cumsum(dim=0)
    )).to(device)
    # fmt: off
    gates = torch.tensor([[
      1.0, 1.5, 0.8, 2.0, # seq1 channel 1
      0.3, 0.4, 0.9, 0.5, 0.45 # seq2 channel 1
    ]]).to(device)
    tokens = torch.tensor([[
      1.0, -1.0, 0.5, 2.0, # seq1 channel 1
      1.0, -0.7, 0.8, 0.6, 0.33 # seq2 channel 1
    ]]).to(device)
    grid = _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels)

    expected_result_gates = torch.tensor([[
      1, 1.5, 1.2, 2.4, # seq1 channel 1
      0.3, 0.12, 0.108, 0.054, 0.0243 # seq2 channel 1
      ]]).to(device)
    expected_result_tokens = torch.tensor([[
      1.0, 0.5, 0.9, 3.8, # seq1 channel 1
      1.0, -0.3,  0.53,  0.865,  0.7192 # seq2 channel 1
    ]]
    ).to(device)
    # fmt: on
    # Act
    forward_scan_full(
      gates=gates,
      tokens=tokens,
      cu_seqlens=cu_seqlens,
      CHUNK_SIZE=chunk_size,
      grid=grid,
      REVERSE=False,
      TESTING=True,
    )
    # Assert
    rtol, atol = 1e-4, 1e-7
    assert torch.allclose(expected_result_tokens, tokens, rtol=rtol, atol=atol)
    assert torch.allclose(expected_result_gates, gates, rtol=rtol, atol=atol)

  # fmt: off
  @pytest.mark.parametrize("chunk_size", [
      2,
      4,
      8,
    ])
  # fmt: on
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numerical_simple_multiple_varlen_sequences_and_channels(
    self, chunk_size,
    ) -> None:
    """Correct values are returned for multiple varlen sequences, channels."""
    # Setup
    from associative_scan_triton import forward_scan_full

    device = get_device()

    no_channels = 2
    seqlens = torch.tensor([5, 4])
    cu_seqlens = torch.cat((
      torch.tensor([0]),
      seqlens.cumsum(dim=0)
    )).to(device)
    # dtype= torch.float64
    dtype= torch.float32
    # fmt: off
    gates = torch.tensor([
      [
        0.3, 0.4, 0.9, 0.5, 0.45, # seq2 channel 1
        1.0, 1.5, 0.8, 2.0, # seq1 channel 1
      ],
      [
        0.3, 0.55, 0.96, 0.35, 0.7, # seq2 channel 2
        0.35, 0.85, 0.45, 1.3, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    tokens = torch.tensor([
      [
        1.0, -0.7, 0.8, 0.6, 0.33, # seq2 channel 1
        1.0, -1.0, 0.5, 2.0, # seq1 channel 1
      ],
      [
        0.6, -0.75, 0.4, 0.65, 0.8, # seq2 channel 2
        1.0, -0.55, 0.65, 0.4, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    grid = _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels)
    expected_result_gates = torch.tensor([
      [
        0.3, 0.12, 0.108, 0.054, 0.0243, # seq2 channel 1
        1.0, 1.5, 1.2, 2.4, # seq1 channel 1
      ],
      [
        0.3, 0.165, 0.1584, 0.05544, 0.0388, # seq2 channel 2
        0.35, 0.2975, 0.1339, 0.174, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    expected_result_tokens = torch.tensor([
      [
        1.0, -0.3,  0.53,  0.865,  0.71925, # seq2 channel 1
        1.0, 0.5, 0.9, 3.8, # seq1 channel 1
      ],
      [
        0.60, -0.42, -0.0032,  0.6489,  1.25422, # seq2 channel 2
        1.0, 0.3, 0.7850, 1.4205, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    # fmt: on
    # Act
    forward_scan_full(
      gates=gates,
      tokens=tokens,
      cu_seqlens=cu_seqlens,
      grid=grid,
      CHUNK_SIZE=chunk_size,
      REVERSE=False,
    )
    # Assert
    rtol, atol = 1e-3, 1e-5
    assert torch.allclose(expected_result_gates, gates, rtol=rtol, atol=atol)
    assert torch.allclose(expected_result_tokens, tokens, rtol=rtol, atol=atol)

  # fmt: off
  @pytest.mark.parametrize("chunk_size", [
      2,
      4,
      8,
    ])
  # fmt: on
  @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
  def test_fwd_chunk_numerical_simple_multiple_varlen_sequences_and_channels_reverse(  # noqa: E501
    self, chunk_size,
    ) -> None:
    """Correct values returned for multi varlen sequences, channels."""
    # Setup
    from associative_scan_triton import forward_scan_full

    device = get_device()
    reverse = True
    no_channels = 2
    seqlens = torch.tensor([5, 4])
    cu_seqlens = torch.cat((
      torch.tensor([0]), seqlens.cumsum(dim=0)
    )).to(device)
    # dtype= torch.float64
    dtype= torch.float32
    # fmt: off
    gates = torch.tensor([
      [
        0.3, 0.4, 0.9, 0.5, 0.45, # seq2 channel 1
        1.0, 1.5, 0.8, 2.0, # seq1 channel 1
      ],
      [
        0.3, 0.55, 0.96, 0.35, 0.7, # seq2 channel 2
        0.35, 0.85, 0.45, 1.3, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    tokens = torch.tensor([
      [
        1.0, -0.7, 0.8, 0.6, 0.33, # seq2 channel 1
        1.0, -1.0, 0.5, 2.0, # seq1 channel 1
      ],
      [
        0.6, -0.75, 0.4, 0.65, 0.8, # seq2 channel 2
        1.0, -0.55, 0.65, 0.4, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    grid = _get_grid_from_cu_seqlens(cu_seqlens, chunk_size, no_channels)
    expected_result_gates = torch.tensor([
      [
        0.024300, 0.081000, 0.202500, 0.225000, 0.450000, # seq2 channel 1
        2.400000, 2.400000, 1.600000, 2.000000, # seq1 channel 1
      ],
      [
        0.038808, 0.129360, 0.235200, 0.245000, 0.700000, # seq2 channel 2
        0.174038, 0.497250, 0.585000, 1.300000, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    expected_result_tokens = torch.tensor([
      [
        0.968620, -0.104600,  1.488500,  0.765000,  0.330000, # seq2 channel 1
        3.150000, 2.150000, 2.100000, 2.000000, # seq1 channel 1
      ],
      [
        0.588312, -0.038960,  1.292800,  0.930000,  0.800000, # seq2 channel 2
        1.054425, 0.155500, 0.830000, 0.400000, # seq1 channel 2
      ]
    ]).to(device, dtype=dtype)
    # fmt: on
    # Act
    forward_scan_full(
      gates=gates,
      tokens=tokens,
      cu_seqlens=cu_seqlens,
      grid=grid,
      REVERSE=reverse,
      CHUNK_SIZE=chunk_size,
    )

    # Assert
    rtol, atol = 1e-3, 1e-5
    assert torch.allclose(expected_result_gates, gates, rtol=rtol, atol=atol)
    assert torch.allclose(expected_result_tokens, tokens, rtol=rtol, atol=atol)


class TestManualCalcs:
  # fmt: off
  @staticmethod
  def compute_h3_2(
    gI0, gI1, tI0, tI1, g, u, debug: bool = False  # noqa: N803
  ) -> tuple[float, float]:
    """Two chunks of two values each."""
    assert gI0 == 1, "Initial gates must be one."
    assert gI1 == 1, "Initial gates must be one."
    assert tI0 == 0, "Initial values must be zero"
    assert tI1 == 0, "Initial values must be zero"
    # Initialize the first chunk's initial state

    gates_current = [gI0]
    tokens_current = [tI0]
    # Process the first chunk
    for i in range(len(g) // 2):
      gate_cur, token_cur = first_order_op(
        gates_current[-1], tokens_current[-1], g[i], u[i]
      )
      gates_current.append(gate_cur)
      tokens_current.append(token_cur)
      if debug:
        print(f"{(gate_cur, token_cur, g[i], u[i])=}")

    # Initialize for second chunk
    gates_current.append(gates_current.pop(0))
    tokens_current.append(tokens_current.pop(0))

    # Process the second chunk
    for i in range(len(g) // 2, len(g)):
      gate_cur, token_cur = first_order_op(
        gates_current[-1], tokens_current[-1], g[i], u[i]
      )
      gates_current.append(gate_cur)
      tokens_current.append(token_cur)
      if debug:
        print(f"{(gate_cur, token_cur, g[i], u[i])=}")
    gates_current.pop(2)
    tokens_current.pop(2)

    # Final result using the first-order operator on the two chunks
    fin_gate, fin_token = first_order_op(
      gates_current[1], tokens_current[1], gates_current[3], tokens_current[3]
    )

    return fin_gate, fin_token
  def test_chunking_by_hand(self):
    # Given a solution and an algebraic proof, used ChatGPT and asked it to
    # help me write the tedious part, hence the monsterous and thorough testing
    # below.

    # We firstly derrive two tests of the same problem approached from two
    # paths then we generalize for multiple chunks and inputs, then we use the
    # generalization to test the implementation (chunking, assert values),
    # finally proving the implementation.
    """Using a premise, confirm that results of chunks match with a hand calculation."""  # noqa: E501

    def compute_h3(
      gI0 ,g0, g1, gI1, g2, g3, tI0, t0, t1, tI1, t2, t3,  # noqa: N803
    ) -> tuple[float, float]:
      assert gI0 == 1, "Initial gates must be one."
      assert gI1 == 1, "Initial gates must be one."
      assert tI0 == 0, "Initial values must be zero"
      assert tI1 == 0, "Initial values must be zero"

      state = gI0 * g0 * g1 * gI1 * g2 * g3
      h3 = (
        gI1 * g2 * g3 * g1 * g0 * tI0
        + gI1 * g2 * g3 * g1 * t0
        + gI1 * g2 * g3 * t1
        + g3 * g2 * tI1
        + g3 * t2
        + t3
      )

      return state, h3

    xI0 = 1
    x0 = 1
    x1 = 1.5
    xI1 = 1
    x2 = 0.8
    x3 = 2.0
    uI0 = 0
    u0 = 1.0
    u1 = -1.0
    uI1 = 0
    u2 = 0.5
    u3 = 2.0
    expected_state_h3 = compute_h3(
      xI0, x0, x1, xI1, x2, x3, uI0, u0, u1, uI1, u2, u3
    )
    expected_state_h3 = torch.tensor(expected_state_h3)

    xI0 = 1
    xI1 = 1
    uI0 = 0
    uI1 = 0
    x = torch.tensor([1.0, 1.5, 0.8, 2.0])
    u = torch.tensor([1.0, -1.0, 0.5, 2.0])

    state_h3 = self.compute_h3_2(xI0, xI1, uI0, uI1, x, u)
    state_h3 = torch.tensor(state_h3)

    hand_result_h0_to_h3 = torch.tensor([1.0000, 0.5000, 0.9000, 3.8000])
    hand_result_gate = torch.tensor([2.4])
    assert torch.allclose(
      hand_result_h0_to_h3[-1], expected_state_h3[-1]
    ), "Manual values do not match premise"
    assert torch.allclose(
      hand_result_h0_to_h3[-1], state_h3[-1]
    ), "Manual values do not match premise"
    assert torch.allclose(
      hand_result_gate, state_h3[0]
    ), "Premise does not match chunks"  # noqa: E501
    assert torch.allclose(
      expected_state_h3, state_h3
    ), "Premise does not match chunks"  # noqa: E501
    # fmt: on
  """Originally inspired by annotated-mamba/hard.

  https://srush.github.io/annotated-mamba/hard.html

  Which of code is correct, but the equations needed a small correction
  for the hand results to match.

  In the mamba variant in the blog had a form of
    dot_h_k = a_{k}*h_{k+1} + c_{k}*dot_y_{k}
  corrected to
    dot_h_k = a_{k+1}*h_{k+1} + c_{k}*dot_y_{k}
  leading to the for us important
    dot_a_k = dot_h_{k}*h_{k-1}
  which was originally correct.

  This propagates to first-order recurrence derivatives as
    dot_h_k = g_{k+1}*h_{k+1} + t_{k}
  and gives
    dot_g_{k} = dot_h_{k}*h_{k-1}
  in the same final form.
  """

  @staticmethod
  def dg_dx_manual(gates, tokens) -> tuple[list[float], list[float]]:
    """Hand calculation showing all intermediates."""
    # Fwd
    h_1 = gates[0] * 0 + tokens[0]  # initial h_0 = 0
    h_2 = gates[1] * h_1 + tokens[1]
    h_3 = gates[2] * h_2 + tokens[2]
    # Calculate y_k=h_k
    y_1 = h_1
    y_2 = h_2
    y_3 = h_3
    # loss
    L = y_1 + y_2 + y_3  # noqa: F841
    # dL/dy_k
    dy_1 = 1
    dy_2 = 1
    dy_3 = 1
    # Backward pass to calculate dot_h_k
    dot_h_3 = dy_3  # Base case dot_h_4 = 0
    dot_h_2 = dy_2 + dot_h_3 * gates[2]
    dot_h_1 = dy_1 + dot_h_2 * gates[1]
    # dL/dg_k
    dg_3 = dot_h_3 * h_2
    dg_2 = dot_h_2 * h_1
    dg_1 = dot_h_1 * 0
    return [dg_1, dg_2, dg_3], [dot_h_1, dot_h_2, dot_h_3]

  @staticmethod
  def dg_dx_ref(gates, tokens) -> tuple[torch.Tensor, torch.Tensor]:
    # Hidden states
    def forward(gates, tokens) -> tuple[float, torch.Tensor]:
      y = []
      h = 0
      for i in range(len(tokens)):
        h = gates[i] * h + tokens[i]
        y.append(h)
      return h, torch.stack(y)

    def loss(g, x) -> float:
      return forward(g, x)[1].sum()

    # Backprop
    g_func = torch.func.grad(loss, tuple(range(2)))

    dg, dx = g_func(gates, tokens)

    return dg, dx

  @pytest.mark.parametrize(
    ("gates", "tokens"),
    [
      ([0.5, 0.6, 0.7], [1.0, 2.0, 3.0]),
      ([0.3, 0.45, 0.6], [1.0, 1.4, 0.3]),
    ],
  )
  def test_manual_vs_ref(self, gates, tokens) -> None:
    # Act
    dg_manual, dx_manual = self.dg_dx_manual(gates, tokens)
    dg_ref, dgx_ref = self.dg_dx_ref(torch.tensor(gates), torch.tensor(tokens))
    # Assert
    assert torch.allclose(torch.tensor(dg_manual), dg_ref)
    assert torch.allclose(torch.tensor(dx_manual), dgx_ref)
