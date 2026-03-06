
import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver
from sbto.solvers.cbox_polar_per_particle_consensus_via_regularization \
    import compute_per_particle_target_consensus
from sbto.solvers.cbox_kernel import gaussian_kernel_neg_log

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

@dataclass
class ConfigCBO(ConfigSolver):
    """
    beta: Inverse temperature
    noise_model: standard | coordinate
    delta: diffusion term
    dt: step size
    kappa: neighborhood kernel
    """
    beta: float = 1.
    noise_model: str = "anistropic"
    delta: float = 1.e-2
    dt: float = 1.e-2
    kappa: float = 1.
    scalar_reg_loss_weight_neighborhood_kernel: float 1.


class CBO(SamplingBasedSolver):
    """
    CBO solver.
    """
    def __init__(self, D, cfg: ConfigCBO):
        super().__init__(D, cfg)
        self.Id = np.eye(D)

        self.first_it = True
        self._zeros = np.zeros(D)
        self._Id = np.eye(D) * self.cfg.dt
        self._x = np.zeros((cfg.N_samples, D))
        self._consensus = np.zeros((1, D))
        self._per_particle_consensus = np.zeros((cfg.N_samples, D))
        self._delta = self.cfg.delta
        self._dt = self.cfg.dt

    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        argmin = costs.argmin()
        cmin = costs[argmin]
        exponents = -(costs - cmin) * self.cfg.beta
        w = np.exp(exponents)
        s = w.sum()
        w /= s
        self._consensus[:self.n_dim] = w @ samples[:, :self.n_dim]

        neighborhood_kernel_neg_log_eval = gaussian_kernel_neg_log(
            jnp.asarray(samples[:, None, :, :]),
            jnp.asarray(samples[None, :, :, :]),
            kappa=float(self.cfg.kappa),
        )

        self._per_particle_consensus = compute_per_particle_target_consensus(
            costs, samples, neighborhood_kernel_neg_log_eval,
            self.cfg.beta,
            self.cfg.scalar_reg_loss_weight_neighborhood_kernel)

        return argmin, cmin

    def get_samples(self) -> Array:
        """
        Get samples from distribution parametrized
        by the current state.
        """
        if self.first_it:
            noise = self.sampler.sample(
                mean=self._zeros,
                cov=self._Id,
            )
            self._consensus[:] = self.state.mean
            self._x[:] = self.state.mean + self.cfg.delta * noise
            return self._x

        noise = np.sqrt(self._delta) * self.sampler.sample(
            mean=self._zeros[:self.n_dim],
            cov=self._Id[:self.n_dim, :self.n_dim],
        )

        drift = self._x[:, :self.n_dim] - \
            self._per_particle_consensus[:, :self.n_dim]

        drift_norm_nx1 = np.linalg.norm(drift, axis=-1, keepdims=True)
        self.logs["s"] = drift_norm_nx1

        if self.cfg.noise_model == "isotropic":
            noise = isotropic_noise = jnp.multiply(
              drift_norm_nx1, noise) # drift_norm [Nx1] * noise [N x D]
 
        elif self.cfg.noise_model == "anistropic":
            noise = jnp.multiply(drift, noise) # drift [N x D] vs noise[N x D]

        else:
            raise ValueError(f"Invalid noise config ({self.cfg.noise_model}).")

        self._x[:, :self.n_dim] -= self._dt * drift - noise

        return self._x

    def update_distrib_param(self, state: SolverState, samples: Array) -> None:
        state.mean, state.cov = self.sampler.estimate_params(samples)

    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update the solver state from elite samples.
        """
        arg_min, min_cost = self.update_mean(samples, costs)
        best = samples[arg_min]
        self.update_min_cost_best(self.state, min_cost, best)
        self.update_distrib_param(self.state, samples)

        self.first_it = False
