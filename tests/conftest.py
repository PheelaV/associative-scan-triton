"""Pytest configuration and fixtures."""

import gc
import sys
from types import ModuleType
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
import torch


def _clear_cuda():
    """Helper to clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@pytest.fixture(autouse=True, scope="class")
def clear_cuda_cache_class():
    """Clear CUDA cache before and after each test class."""
    _clear_cuda()
    yield
    _clear_cuda()


@pytest.fixture(autouse=True, scope="function")
def clear_cuda_cache_function(request):
    """Clear CUDA cache between test functions (not parametrizations)."""
    node = request.node
    parent = node.parent

    if not hasattr(parent, "_last_test_func"):
        parent._last_test_func = None

    current_func = node.originalname if hasattr(node, "originalname") else node.name

    if parent._last_test_func != current_func:
        _clear_cuda()
        parent._last_test_func = current_func

    yield


@pytest.fixture()
def mock_triton(mocker) -> Generator[Any, Any, None]:
  """Create a mock module hierarchy for "importing" Triton on non-CUDA iron."""
  triton = ModuleType("triton")
  triton.language = ModuleType("triton.language")
  triton.jit = MagicMock()
  triton.language.constexpr = MagicMock()
  triton.language.tensor = MagicMock()

  mocker.patch.dict(
    sys.modules,
    {
      "triton": triton,
      "triton.language": triton.language,
    },
  )

  yield None


# Duplicated as the jit decorator breaks if Triton not present
def first_order_op(fl, xl, fr, xr) -> tuple[float, float]:
  f = fr * fl
  x = fr * xl + xr
  return f, x
