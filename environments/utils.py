from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(1,2,3))
def vector_norm(x, axis=-1, keepdims: bool=False, eps: float=1e-4):
    return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + eps**2) - eps


@partial(jax.jit, static_argnums=(1))
def smoothrelu(x, smoothing: float=0.0):
    if smoothing == 0.0:
        return jnp.maximum(x, 0.0)
    return smoothing * jax.nn.softplus(x / smoothing)
