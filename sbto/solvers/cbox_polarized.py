from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import jax.numpy as jnp

from cbo_x import CBOx, CustomCBOParams
from cal_consensus_polar_regularization import (
    compute_polar_consensus,
    persist_consensus_stats,
)
from kernel import gaussian_kernel_neg_log
from hydrax.alg_base import Trajectory

Array = npt.NDArray[np.float64]


class CBOxPolarized(CBOx):
    """CBOx variant with a manual softmax in consensus calculation."""

    @staticmethod
    def _resolve_shape_config(x: np.ndarray, n_knots: int | None, n_dof: int | None) -> tuple[int, int]:
        if x.ndim == 3:
            return x.shape[1], x.shape[2]
        if x.ndim != 2:
            raise ValueError(f"Expected rollouts.knots with ndim 2 or 3, got {x.ndim}.")
        if n_knots is None or n_dof is None:
            raise ValueError(
                "For flat N x D knots, both n_knots and n_dof must be configured."
            )
        if x.shape[1] != n_knots * n_dof:
            raise ValueError(
                f"D mismatch: got D={x.shape[1]}, expected n_knots*n_dof={n_knots*n_dof}."
            )
        return int(n_knots), int(n_dof)

    @staticmethod
    def _flat_to_structured(x: np.ndarray, n_knots: int, n_dof: int, layout: str) -> np.ndarray:
        if x.ndim == 3:
            return x
        if layout == "knots_dof":
            return x.reshape(x.shape[0], n_knots, n_dof)
        if layout == "dof_knots":
            return x.reshape(x.shape[0], n_dof, n_knots).transpose(0, 2, 1)
        raise ValueError(f"Unsupported layout '{layout}'. Use 'knots_dof' or 'dof_knots'.")

    @staticmethod
    def _structured_to_flat(x: np.ndarray, layout: str) -> np.ndarray:
        if x.ndim == 2:
            return x
        if layout == "knots_dof":
            return x.reshape(x.shape[0], -1)
        if layout == "dof_knots":
            return x.transpose(0, 2, 1).reshape(x.shape[0], -1)
        raise ValueError(f"Unsupported layout '{layout}'. Use 'knots_dof' or 'dof_knots'.")

    def cal_consensus(
        self,
        rollouts: Trajectory,
        params: CustomCBOParams,
        *,
        iteration: int | None = None,
    ) -> tuple[Array, Array, CustomCBOParams]:
        """Update the mean with an exponentially weighted average."""
        arr_consensus, costs, params = super().cal_consensus(rollouts, params)

        knots_raw = np.asarray(rollouts.knots)
        costs = np.asarray(costs)
        layout = getattr(self, "polar_layout", "knots_dof")
        n_knots_cfg = getattr(self, "polar_num_knots", getattr(self, "num_knots", None))
        n_dof_cfg = getattr(self, "polar_num_dof", getattr(self, "num_dof", None))
        n_knots, n_dof = self._resolve_shape_config(knots_raw, n_knots_cfg, n_dof_cfg)
        knots = self._flat_to_structured(knots_raw, n_knots, n_dof, layout)

        consensus_every = getattr(self, "per_particle_consensus_every", 1)
        do_update = (params.iteration % consensus_every) == 0

        if do_update:
            kappa = getattr(self, "per_particle_consensus_kappa", 1.0)
            polar_kernel_reg_loss_weight = getattr(
                self, "polar_kernel_reg_loss_weight", 0.0
            )

            neg_log_eval = gaussian_kernel_neg_log(
                jnp.asarray(knots[:, None, :, :]),
                jnp.asarray(knots[None, :, :, :]),
                kappa=float(kappa),
            )

            if getattr(self, "polar_auto_weight", False):
                neg_log_mean = np.mean(np.abs(np.asarray(neg_log_eval)))
                cost_mean = np.mean(np.abs(costs))
                polar_kernel_reg_loss_weight = cost_mean / (neg_log_mean + 1e-12)

            per_particle_consensus_structured = compute_polar_consensus(
                jnp.asarray(costs),
                jnp.asarray(knots),
                neg_log_eval,
                float(params.temperature),
                float(polar_kernel_reg_loss_weight),
            )
            per_particle_consensus = self._structured_to_flat(
                np.asarray(per_particle_consensus_structured),
                layout,
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
                persist_consensus_stats(
                    np.asarray(per_particle_consensus_structured),
                    stats_dir,
                    iteration,
                )
        else:
            per_particle_consensus = np.asarray(
                params.per_particle_consensus,
            )

        params = params.replace(per_particle_consensus=per_particle_consensus)
        return arr_consensus, costs, params

    def update_particles(self, params: CustomCBOParams, costs: Array) -> CustomCBOParams:
        params_ppc = params.replace(mean=params.per_particle_consensus)
        params_updated = super().update_particles(params_ppc, costs)
        return params_updated.replace(mean=params.mean)
