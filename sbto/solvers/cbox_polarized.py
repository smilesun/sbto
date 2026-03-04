from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from numba import njit

from cbo_x import CBOx, CustomCBOParams
from cal_consensus_polar_regularization import persist_consensus_stats
from hydrax.alg_base import Trajectory

Array = npt.NDArray[np.float64]


@njit(cache=True)
def _gaussian_kernel_neg_log(knots: Array, kappa: float) -> Array:
    n_particles, n_dim = knots.shape
    out = np.empty((n_particles, n_particles), dtype=np.float64)
    for i in range(n_particles):
        for j in range(n_particles):
            sq_dist = 0.0
            for d in range(n_dim):
                diff = knots[i, d] - knots[j, d]
                sq_dist += diff * diff
            out[i, j] = 0.5 * kappa * sq_dist
    return out


@njit(cache=True)
def _compute_polar_consensus(
    costs: Array,
    knots: Array,
    neg_log_eval: Array,
    temperature: float,
    polar_kernel_reg_loss_weight: float,
) -> Array:
    eps = 1.0e-12
    n_particles, n_dim = knots.shape
    inv_temp = 1.0 / max(temperature, eps)
    per_particle = np.empty((n_particles, n_dim), dtype=np.float64)

    for i in range(n_particles):
        logits = np.empty(n_particles, dtype=np.float64)
        max_logit = -np.inf
        for j in range(n_particles):
            logits[j] = -inv_temp * (
                costs[j] + polar_kernel_reg_loss_weight * neg_log_eval[i, j]
            )
            if logits[j] > max_logit:
                max_logit = logits[j]

        denom = 0.0
        weights = np.empty(n_particles, dtype=np.float64)
        for j in range(n_particles):
            weights[j] = np.exp(logits[j] - max_logit)
            denom += weights[j]
        denom = max(denom, eps)

        for d in range(n_dim):
            acc = 0.0
            for j in range(n_particles):
                acc += (weights[j] / denom) * knots[j, d]
            per_particle[i, d] = acc

    return per_particle


class CBOxPolarized(CBOx):
    """CBOx variant with a manual softmax in consensus calculation."""

    def cal_consensus(
        self,
        rollouts: Trajectory,
        params: CustomCBOParams,
        *,
        iteration: int | None = None,
    ) -> tuple[Array, Array, CustomCBOParams]:
        """Update the mean with an exponentially weighted average."""
        arr_consensus, costs, params = super().cal_consensus(rollouts, params)

        knots = np.asarray(rollouts.knots, dtype=np.float64)
        costs = np.asarray(costs, dtype=np.float64)

        consensus_every = getattr(self, "per_particle_consensus_every", 1)
        do_update = (params.iteration % consensus_every) == 0

        if do_update:
            kappa = getattr(self, "per_particle_consensus_kappa", 1.0)
            polar_kernel_reg_loss_weight = getattr(
                self, "polar_kernel_reg_loss_weight", 0.0
            )

            neg_log_eval = _gaussian_kernel_neg_log(knots, float(kappa))

            if getattr(self, "polar_auto_weight", False):
                neg_log_mean = np.mean(np.abs(neg_log_eval))
                cost_mean = np.mean(np.abs(costs))
                polar_kernel_reg_loss_weight = cost_mean / (neg_log_mean + 1e-12)

            per_particle_consensus = _compute_polar_consensus(
                costs,
                knots,
                neg_log_eval,
                float(params.temperature),
                float(polar_kernel_reg_loss_weight),
            )

            stats_dir = None
            particles_dir = getattr(getattr(self, "io", None), "_particles_dir", None)
            if particles_dir is not None:
                stats_dir = Path(particles_dir).parent / "stats"
            persist_every = getattr(self, "persist_particles_latest", None)
            should_persist = (
                stats_dir is not None
                and iteration is not None
                and persist_every not in (None, 0)
                and iteration % int(persist_every) == 0
            )
            if should_persist:
                persist_consensus_stats(per_particle_consensus, stats_dir, iteration)
        else:
            per_particle_consensus = np.asarray(
                params.per_particle_consensus,
                dtype=np.float64,
            )

        params = params.replace(per_particle_consensus=per_particle_consensus)
        return arr_consensus, costs, params

    def update_particles(self, params: CustomCBOParams, costs: Array) -> CustomCBOParams:
        params_ppc = params.replace(mean=params.per_particle_consensus)
        params_updated = super().update_particles(params_ppc, costs)
        return params_updated.replace(mean=params.mean)
