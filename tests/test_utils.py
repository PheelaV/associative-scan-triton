"""Test utilities for scan tests."""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

TEST_SEED = 1155


def get_device(accelerator: bool = True) -> torch.device:
  if torch.cuda.is_available() and accelerator:
    return torch.device("cuda")
  return torch.device("cpu")


def get_mock_data_simple(
  no_channels: int,
  seqlen: int,
  seq_multiplier: int = 1,
  device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Gets a simple block of data of [C, L*B] shape.

  Returns:
    (gates, tokens, cu_seqlens)
  """
  device = torch.device("cpu") if device is None else device
  gates = (
    torch.rand(no_channels, seqlen * seq_multiplier).to(device).contiguous()
  )
  tokens = (
    torch.rand(no_channels, seqlen * seq_multiplier).to(device).contiguous()
  )
  cu_seqlens = torch.tensor([seqlen]).repeat(seq_multiplier).cumsum(0)
  cu_seqlens = (
    torch.cat([torch.tensor([0]), cu_seqlens], dim=0).to(device).contiguous()
  )

  return gates, tokens, cu_seqlens


def get_mock_data(
  no_channels: int,
  max_len: int,
  multiplier: float,
  device: torch.device,
  series=None,
  seqlens: list[int] | None = None,
  debug: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  if series is None:
    series = lambda max_len: np.arange(1, max_len + 1)
  if seqlens is None:
    seqlens = series(max_len) * multiplier
    seqlens = np.append(seqlens, [1])

  if debug:
    torch.set_printoptions(precision=2)
    print(f"max seq len:{max(seqlens)}")
    print(f"{seqlens=}")
  cu_seqlens = torch.tensor(np.insert(seqlens.cumsum(0), 0, 0), device=device)
  return get_mock_data_cuseqlens(
    cu_seqlens, no_channels, seqlens, device, debug
  )


def get_mock_data_cuseqlens(
  cu_seqlens, no_channels, seqlens, device, debug=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  max_seqlen = max(cu_seqlens[1:] - cu_seqlens[:-1]).item()
  if debug:
    print(f"{cu_seqlens=}")
    print(f"{max_seqlen=}")

  documents = [
    torch.arange(no_channels).repeat(l, 1) + 1 + 0.1 * (i + 1)
    for i, l in enumerate(seqlens)
  ]
  transposed_docs = [doc.mT for doc in documents]
  final_input = torch.cat(transposed_docs, dim=1)
  if debug:
    torch.set_printoptions(precision=4)
  for i in range(len(cu_seqlens) - 1):
    start = cu_seqlens[i]
    end = cu_seqlens[i + 1]
    slen = end - start
    final_input[:, start:end] += (torch.arange(slen) + 1) * 0.01

  final_input = final_input.contiguous()
  final_input_flatenned = final_input.flatten()

  gates = (
    (final_input_flatenned / (2 * max(final_input_flatenned)))
    .clone()
    .to(device)
    .contiguous()
  )
  tokens = final_input_flatenned.clone().to(device).contiguous()
  cu_seqlens = cu_seqlens.to(device=device).contiguous()
  return gates, tokens, cu_seqlens
