import jax.numpy as jnp
from jax import grad, lax, vmap


def op(
  left: tuple[jnp.ndarray, jnp.ndarray], right: tuple[jnp.ndarray, jnp.ndarray]
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Element-wise operation used in associative scan."""
  fl, xl = left
  fr, xr = right
  return fl * fr, fr * xl + xr


def shift_pad(
  array: jnp.ndarray, backward: bool, pad_value: int = 0
) -> jnp.ndarray:
  """Shift and pad the array based on the direction."""
  if backward:
    return jnp.concatenate((jnp.array([pad_value]), array[:-1]))
  else:
    return jnp.concatenate((array[1:], jnp.array([pad_value])))


def scan_causal(
  gates: jnp.ndarray, tokens: jnp.ndarray, reverse: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Perform a forward associative scan on gates and tokens."""
  return lax.associative_scan(op, (gates, tokens), reverse)


__scan_causal_vmap = vmap(scan_causal, in_axes=(0, 0, None))


def scan_multi_channel(
  gates: jnp.ndarray, tokens: jnp.ndarray, reverse: bool = False
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Vectorized scan_fwd across multiple channels."""
  result_gates, result_tokens = __scan_causal_vmap(gates, tokens, reverse)
  return result_gates, result_tokens


def scan_multi_channel_bidi_forked(
  gates: jnp.ndarray, tokens: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Perform bidirectional scan across multiple channels."""
  _, result_tokens_fwd = __scan_causal_vmap(gates, tokens, False)
  _, result_tokens_bwd = __scan_causal_vmap(gates, tokens, True)

  return result_tokens_fwd, result_tokens_bwd


def scan_multi_channel_bidi_branched(
  gates_fwd: jnp.ndarray,
  tokens_fwd: jnp.ndarray,
  gates_bwd: jnp.ndarray,
  tokens_bwd: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Perform bidirectional scan across multiple channels."""
  _, result_tokens_fwd = __scan_causal_vmap(gates_fwd, tokens_fwd, False)
  _, result_tokens_bwd = __scan_causal_vmap(gates_bwd, tokens_bwd, True)

  return result_tokens_fwd, result_tokens_bwd


def scan_loss_causal(
  gates: jnp.ndarray, tokens: jnp.ndarray, reverse: bool = False
) -> jnp.ndarray:
  """Calculate the loss for a forward scan."""
  _, result_tokens = lax.associative_scan(op, (gates, tokens), reverse)
  return result_tokens.sum()


__scan_loss_causal_vmap = vmap(scan_loss_causal, in_axes=(0, 0, None))


def scan_loss_multi_channel(
  gates: jnp.ndarray, tokens: jnp.ndarray
) -> jnp.ndarray:
  """Calculate the loss for a multi-channel scan."""
  return __scan_loss_causal_vmap(gates, tokens, False).sum()


def scan_loss_multi_channel_bidi_forked(
  gates: jnp.ndarray, tokens: jnp.ndarray
) -> jnp.ndarray:
  """Calculate the loss for a bidirectional multi-channel scan."""
  loss_fwd = __scan_loss_causal_vmap(gates, tokens, False).sum()
  loss_bwd = __scan_loss_causal_vmap(gates, tokens, True).sum()
  return loss_fwd + loss_bwd


def scan_loss_multi_channel_bidi_branched(
  gates_fwd: jnp.ndarray,
  tokens_fwd: jnp.ndarray,
  gates_bwd: jnp.ndarray,
  tokens_bwd: jnp.ndarray,
) -> jnp.ndarray:
  """Calculate the loss for a bidirectional multi-channel scan."""
  loss_fwd = __scan_loss_causal_vmap(gates_fwd, tokens_fwd, False).sum()
  loss_bwd = __scan_loss_causal_vmap(gates_bwd, tokens_bwd, True).sum()
  return loss_fwd + loss_bwd


grad_scan_multi_channel = grad(scan_loss_multi_channel, argnums=[0, 1])
grad_scan_multi_channel_bidi_forked = grad(
  scan_loss_multi_channel_bidi_forked, argnums=[0, 1]
)
grad_scan_multi_channel_bidi_branched = grad(
  scan_loss_multi_channel_bidi_branched, argnums=[0, 1, 2, 3]
)
