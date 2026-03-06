from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from jax import debug, jit, lax
from jax.scipy.special import logsumexp


@jit
def compute_polar_consensus(
    costs: jnp.ndarray,  # N
    x: jnp.ndarray,  # N * H * Dof
    neg_log_eval: jnp.ndarray,  # N*N
    temperature: float,
    polar_kernel_reg_loss_weight: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute one consensus trajectory per particle for polarized CBO.

    Uses a column-wise softmax over the regularized loss
    ``-polar_kernel_reg_loss_weight * neg_log_eval - costs / temperature``.
    Column ``j`` gives weights for particle ``j``'s consensus, and the output
    keeps the same tail dimensions as ``x``.

    Args:
        costs: Array of shape ``(N,)`` with one objective value per particle.
        x: Particle states/trajectories of shape ``(N, H, DoF)`` (or ``(N, ...)``).
            Axis 0 indexes particles; remaining axes are averaged into consensus.
        neg_log_eval: Pairwise negative-log kernel matrix of shape ``(N, N)``.
            Entry ``(i, j)`` is the regularization term between particles ``i`` and ``j``.
        temperature: Positive softmax temperature for cost weighting. Smaller values
            make consensus focus more on lower-cost particles.
        polar_kernel_reg_loss_weight: Weight of the pairwise regularization term.
            ``0`` removes kernel regularization; larger values increase its influence.

    Returns:
        Consensus array with shape ``(N, H, DoF)`` (or ``(N, ...)`` matching ``x`` tail).
        Row ``j`` is particle ``j``'s consensus state.
    """

    loss_regularized = - polar_kernel_reg_loss_weight * neg_log_eval \
        - 1.0 / temperature * costs[:, None]
    neg_log_mean = jnp.mean(jnp.abs(neg_log_eval))
    cost_mean = jnp.mean(jnp.abs(costs))
    ratio = neg_log_mean / (cost_mean + 1e-12)
    debug.print(
        "neg_log_eval|mean| / costs|mean| ratio: {}", ratio
    )

    # neg_log_eval.shape = N * N  is symmetric
    # costs[:, None].shape = N * 1
    # broadcast of costs from cost (N,) to cost[:, None] with another array of
    # shape (N,N):
    # [[J_1, J_1, ..., J_1];   # for particle 1
    #  [J_2, J_2, ..., J_2];   # for particle 2
    #  [J_3, J_3, ..., J_3];   # for particle 3
    #  ...
    # ]
    # loss_regularized.shape = N * N
    # row i: neighborhood relationship of each particle to particle i,
    # regularized loss of particle i
    # after regularization, each row is still w.r.t. one particle, where
    # interaction with each other particle is added
    # [
    #   [J_1+r(1,1), J_1+r(1,2), ..., J_1+r(1,N)];  # for particle 1
    #   [J_2+r(2,1), J_2+r(2,2), ..., J_2+r(2,N)];  # for particle 2
    #   [J_3+r(3,1), J_3+r(3,2), ..., J_3+r(3,N)];  # for particle 3
    #   ...
    # ]

    # for vanilla CBO, there is only one consensus point,to calculate this
    # consensus point, there is N cost

    # for polarized CBO, each particle has its own consensus point and
    # there is N regularized cost for each particle,
    # row i: J_i+r_{i,j} with j being dummy index
    # below for logsumexp, axis = 0  instead of -1 here

    weights = jnp.exp(loss_regularized -
                      logsumexp(loss_regularized, axis=0, keepdims=True))

    # weights.shape = N*N
    # [
    #   [w11, w12, ..., w1n]
    #   [w21, w22, ..., w2n]
    #   [w31, w32, ..., w3n]
    #   ...
    # ]
    # each **column** corresponds to one particle

    # logsumexp result in N*1
    # there should be N*N weights
    # jax.debug.print("col_sums: {}", jnp.sum(weights, axis=0))

    # Sanity check: when polar_kernel_reg_loss_weight == 0,
    # each column should match.

    def _check_columns(_):
        max_diff = jnp.max(jnp.abs(weights - weights[:, :1]))
        def _raise(_):
            raise ValueError(
                "polar_kernel_reg_loss_weight=0 but weights columns differ"
            )

        def _bad(_):
            debug.print(
                "polar kernel reg weight is 0 but weights columns differ (max diff: {})",
                max_diff,
            )
            debug.callback(_raise, None)
            return None

        return lax.cond(
            max_diff > 1e-6,
            _bad,
            lambda _: None,
            operand=None,
        )

    _ = lax.cond(
        polar_kernel_reg_loss_weight == 0,
        _check_columns,
        lambda _: None,
        operand=None,
    )

    extra_dims = x.ndim - 1
    if extra_dims > 0:
        # broadcast such that H*DoF are treated equally
        weights = jnp.reshape(weights, weights.shape + (1,) * extra_dims)
        # (1,) * 2 result in (1,1), weights.shape will have two extra dimensino

    # Weights are normalized per column (axis=0 above), so each column j gives
    # the weights for **target consensus of particle j**.

    # Broadcast x along a new column axis so weights[i, j] multiplies x[i].
    # x: N*H*DoF, x[:, None, ...] is N*1*H*DoF
    # weights.shape: N*N*1*1 after reshape above
    # x[:, None, ...] * weights -> N*N*H*DoF, sum over axis=0 (i) -> N*H*DoF
    consensus = jnp.sum(x[:, None, ...] * weights, axis=0)
    # consensus.shape (N, H, DoF)

    def _check_consensus_rows(_):
        max_diff = jnp.max(jnp.abs(consensus - consensus[:1]))

        def _raise(_):
            """
            used in function _bad(_)
            """
            raise ValueError(
                "polar_kernel_reg_loss_weight=0 but consensus rows differ"
            )

        def _bad(_):
            """
            used in condition
            """
            debug.print(
                "polar kernel reg weight is 0 but consensus rows differ (max diff: {})",
                max_diff,
            )
            debug.callback(_raise, None)
            return None

        return lax.cond(
            max_diff > 1e-6,
            _bad,
            lambda _: None,
            operand=None,
        )

    _ = lax.cond(
        polar_kernel_reg_loss_weight == 0,
        _check_consensus_rows,
        lambda _: None,
        operand=None,
    )
    return consensus
