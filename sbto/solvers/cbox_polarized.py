from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from cbo_x import CBOx, CustomCBOParams
from cal_consensus_polar_regularization import compute_polar_consensus, persist_consensus_stats
from kernel import gaussian_kernel_neg_log
from hydrax.alg_base import Trajectory


class CBOxPolarized(CBOx):
    """CBOx variant with a manual softmax in consensus calculation."""
    def cal_consensus(
        self, rollouts: Trajectory, params: CustomCBOParams, *, iteration: int | None = None
    ) -> jax.Array:
        """Update the mean with an exponentially weighted average."""
        arr_consensus, costs, params = super().cal_consensus(rollouts, params)
        consensus_every = getattr(self, "per_particle_consensus_every", 1)
        do_update = (params.iteration % consensus_every) == 0

        def _compute_ppc(_):
            kappa = getattr(self, "per_particle_consensus_kappa", 1.0)
            polar_kernel_reg_loss_weight = getattr(
                self, "polar_kernel_reg_loss_weight", 0.0
            )
            neg_log_eval = gaussian_kernel_neg_log(
                rollouts.knots[:, None, :],
                rollouts.knots[None, :, :],
                kappa=kappa,
            )
            if getattr(self, "polar_auto_weight", False):
                neg_log_mean = jnp.mean(jnp.abs(neg_log_eval))
                cost_mean = jnp.mean(jnp.abs(costs))
                polar_kernel_reg_loss_weight = cost_mean / (neg_log_mean + 1e-12)
            per_particle = compute_polar_consensus(
                costs,
                rollouts.knots,
                neg_log_eval,
                params.temperature,
                polar_kernel_reg_loss_weight=polar_kernel_reg_loss_weight,
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
                jax.debug.callback(
                    lambda c, it: persist_consensus_stats(c, stats_dir, it),
                    per_particle,
                    iteration,
                )
            return per_particle

        per_particle_consensus = jax.lax.cond(
            do_update,
            _compute_ppc,  # function to execute if do_update
            lambda _: params.per_particle_consensus,  # function with no args
            operand=None,
        )
        params = params.replace(per_particle_consensus=per_particle_consensus)
        return arr_consensus, costs, params

    def update_particles(self, params: CustomCBOParams,
                         costs) -> CustomCBOParams:
        params_ppc = params.replace(mean=params.per_particle_consensus)
        params_updated = super().update_particles(params_ppc, costs)
        return params_updated.replace(mean=params.mean)
