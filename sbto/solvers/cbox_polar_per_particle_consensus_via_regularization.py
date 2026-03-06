from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
from jax import debug, jit, lax
from jax.scipy.special import logsumexp

@jit
def compute_per_particle_target_consensus(
    costs: jnp.ndarray,  # N
    u: jnp.ndarray,  # N * H * Dof
    neighborhood_kernel_neg_los_eval: jnp.ndarray,  # N*N
    temperature: float,
    scalar_reg_loss_weight_neighborhood_kernel: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    """

    Compute one target consensus per particle for polarized CBO.

    Uses a column-wise softmax over the regularized loss

    ``-scalar_reg_loss_weight_neighborhood_kernel *
    neighborhood_kernel_neg_los_eval - costs / temperature``.

    Column ``j`` gives weights for particle ``j``'s consensus, and the output
    keeps the same tail dimensions as ``u``.

    Args:
        costs: Array of shape ``(N,)`` with one objective value per particle.

        u: Particle states/trajectories of shape ``(N, H, DoF)``
        (or ``(N, ...)``).
            Axis 0 indexes particles;
            remaining axes are averaged into consensus.

        neighborhood_kernel_neg_los_eval: Pairwise negative-log kernel matrix
            of shape ``(N, N)``.
            Entry ``(i, j)`` is the neighborhood regularization term
            between particles ``i`` and ``j``.

        temperature: Positive softmax temperature for cost weighting.
            Smaller values make consensus focus more on lower-cost particles.

        scalar_reg_loss_weight_neighborhood_kernel: Weight of the pairwise
            regularization term. ``0`` removes neighbor kernel regularization;
            larger values increase its influence.

    Returns:
        Consensus array with shape ``(N, H, DoF)``
            (or ``(N, ...)`` matching ``u`` tail).
            Row ``j`` is particle ``j``'s consensus state.
    """

    loss_regularized = - scalar_reg_loss_weight_neighborhood_kernel * \
        neighborhood_kernel_neg_los_eval \
        - 1.0 / temperature * costs[:, None]  # broadcast new dimension

    # calculate the ratio between neighborhood kernel as loss regularization vs
    # original cost/loss.
    neg_log_mean = jnp.mean(jnp.abs(neighborhood_kernel_neg_los_eval))
    cost_mean = jnp.mean(jnp.abs(costs))
    ratio = neg_log_mean / (cost_mean + 1e-12)
    debug.print(
        "neighborhood_kernel_neg_los_eval|mean| / costs|mean| ratio: {}", ratio
    )

    # neighborhood_kernel_neg_los_eval.shape = N * N  is symmetric
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

    # Sanity check: when scalar_reg_loss_weight_neighborhood_kernel == 0,
    # each column should match.

    def _check_columns(_):
        max_diff = jnp.max(jnp.abs(weights - weights[:, :1]))
        def _raise(_):
            raise ValueError(
                "scalar_reg_loss_weight_neighborhood_kernel=0 but \
                weights columns differ"
            )

        def _bad(_):
            debug.print(
                "polar kernel reg weight is 0 but weights columns \
                differ (max diff: {})",
                max_diff,
            )
            debug.callback(_raise, None)
            return None

        # If column differences are above tolerance,
        # run `_bad` (print + raise);
        # otherwise do nothing.
        return lax.cond(
            max_diff > 1e-6,
            _bad,
            lambda _: None,
            operand=None,
        )

    # Only run the column-consistency check when regularization weight is zero.
    # i.e. set this scalar to 0 to debug if code is correct
    # With non-zero regularization, different columns are expected.
    _ = lax.cond(
        scalar_reg_loss_weight_neighborhood_kernel == 0,
        _check_columns,
        lambda _: None,
        operand=None,
    )

    extra_dims = u.ndim - 1
    if extra_dims > 0:
        # broadcast such that H*DoF are treated equally
        weights = jnp.reshape(weights, weights.shape + (1,) * extra_dims)
        # (1,) * 2 result in (1,1), weights.shape will have two extra dimension

    # Weights are normalized per column (axis=0 above), so each column j gives
    # the weights for **target consensus of particle j**.

    # Broadcast u along a new column axis so weights[i, j] multiplies u[i].
    # u: N*H*DoF, u[:, None, ...] is N*1*H*DoF
    # weights.shape: N*N*1*1 after reshape above
    # u[:, None, ...] * weights -> N*N*H*DoF, sum over axis=0 (i) -> N*H*DoF
    consensus = jnp.sum(u[:, None, ...] * weights, axis=0)
    # consensus.shape (N, H, DoF)

    # Only enforce identical consensus rows when regularization weight is zero.
    # i.e. to debug/infer if this implementation is consistent/correct
    # For non-zero regularization, per-particle consensus is allowed to differ.
    _ = lax.cond(
        scalar_reg_loss_weight_neighborhood_kernel == 0,
        _check_consensus_rows,
        lambda _: None,
        operand=consensus,
    )
    return consensus


def _check_consensus_rows(consensus: jnp.ndarray) -> None:
    """Raise when consensus rows are not identical under zero regularization"""
    max_diff = jnp.max(jnp.abs(consensus - consensus[:1]))

    def _raise(_):
        raise ValueError(
            "scalar_reg_loss_weight_neighborhood_kernel=0 but \
            consensus rows differ"
        )

    def _bad(_):
        debug.print(
            "polar kernel reg weight is 0 but consensus rows differ \
            (max diff: {})",
            max_diff,
        )
        debug.callback(_raise, None)
        return None

    # If consensus rows differ beyond tolerance, run `_bad` (print + raise);
    # otherwise do nothing.
    return lax.cond(
        max_diff > 1e-6,
        _bad,
        lambda _: None,
        operand=None,
    )
