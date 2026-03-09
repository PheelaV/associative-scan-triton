"""Tests for shift_pad utility."""

import pytest
import torch

from associative_scan_triton import shift_pad


@pytest.fixture(params=[True, False])
def backward(request: pytest.FixtureRequest):  # noqa: ANN201
  return request.param


@pytest.fixture(params=[-1])
def pad_value(request: pytest.FixtureRequest):  # noqa: ANN201
  return request.param


class TestShiftPad:
  """Tests a simple one-shift and pad functionality."""

  @pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
  )
  def test_shift_pad_single_sequence(
    self, backward: bool, pad_value: int, device: str
  ) -> None:
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=device)
    cu_seqlens = torch.tensor([0, 5], device=device)

    result = shift_pad(data, cu_seqlens, pad_value, backward)

    if backward:
      expected = torch.tensor([[pad_value, 1.0, 2.0, 3.0, 4.0]], device=device)
    else:
      expected = torch.tensor([[2.0, 3.0, 4.0, 5.0, pad_value]], device=device)

    assert torch.allclose(
      result, expected
    ), f"Expected {expected}, but got {result}"

  @pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
  )
  def test_shift_pad_multiple_sequences(
    self, backward: bool, pad_value: int, device: str
  ) -> None:
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], device=device)
    cu_seqlens = torch.tensor([0, 3, 7], device=device)

    result = shift_pad(data, cu_seqlens, pad_value, backward)

    if backward:
      expected = torch.tensor(
        [[pad_value, 1.0, 2.0, pad_value, 4.0, 5.0, 6.0]], device=device
      )
    else:
      expected = torch.tensor(
        [[2.0, 3.0, pad_value, 5.0, 6.0, 7.0, pad_value]], device=device
      )

    assert torch.allclose(
      result, expected
    ), f"Expected {expected}, but got {result}"

  @pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
  )
  def test_shift_pad_multi_dimensional(
    self, backward: bool, pad_value: int, device: str
  ) -> None:
    data = torch.tensor(
      [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0],
      ],
      device=device,
    )
    cu_seqlens = torch.tensor([0, 2, 4, 7], device=device)

    result = shift_pad(data, cu_seqlens, pad_value, backward)

    if backward:
      expected = torch.tensor(
        [
          [pad_value, 1.0, pad_value, 3.0, pad_value, 5.0, 6.0],
          [pad_value, 7.0, pad_value, 9.0, pad_value, 11.0, 12.0],
        ],
        device=device,
      )
    else:
      expected = torch.tensor(
        [
          [2.0, pad_value, 4.0, pad_value, 6.0, 0.0, pad_value],
          [8.0, pad_value, 10.0, pad_value, 12.0, 0.0, pad_value],
        ],
        device=device,
      )

    assert torch.allclose(
      result, expected
    ), f"Expected {expected}, but got {result}"

  @pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
  )
  def test_shift_pad_single_element_sequences(
    self, backward: bool, pad_value: int, device: str
  ) -> None:
    data = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)
    cu_seqlens = torch.tensor([0, 1, 2, 3, 4], device=device)

    result = shift_pad(data, cu_seqlens, pad_value, backward)

    expected = torch.full_like(data, pad_value)
    assert torch.allclose(
      result, expected
    ), f"Expected all pad values, but got {result}"

  @pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
  )
  def test_shift_pad_empty_sequences(
    self, backward: bool, pad_value: int, device: str
  ) -> None:
    data = torch.tensor([[1.0, 2.0, 3.0]], device=device)
    cu_seqlens = torch.tensor([0, 0, 1, 1, 3], device=device)

    with pytest.raises(
      AssertionError, match="can't shift_pad zero-length seqments"
    ):
      shift_pad(data, cu_seqlens, pad_value, backward)
