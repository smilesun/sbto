from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp


def _kernel_axes(ndim: int) -> Tuple[int, ...]:
    if ndim <= 2:
        return ()
    return tuple(range(2, ndim))


@jax.jit
def gaussian_kernel(x: jnp.ndarray, y: jnp.ndarray, kappa: float) -> jnp.ndarray:
    axes = _kernel_axes(x.ndim)
    dists = jnp.sum((x - y) ** 2, axis=axes)
    return jnp.exp(-0.5 * dists / (kappa ** 2))


@jax.jit
def gaussian_kernel_neg_log(x: jnp.ndarray, y: jnp.ndarray, kappa: float) -> jnp.ndarray:
    axes = _kernel_axes(x.ndim)
    dists = jnp.sum((x - y) ** 2, axis=axes)
    return 0.5 * dists / (kappa ** 2)


@dataclass
class GaussianKernel:
    kappa: float = 1.0

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return gaussian_kernel(x, y, self.kappa)

    def neg_log(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return gaussian_kernel_neg_log(x, y, self.kappa)
