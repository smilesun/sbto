
import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass
import jax.numpy as jnp

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
    beta: float = 1.e6
    noise_model: str = "anistropic"
    delta: float = 1.e-2
    dt: float = 1.e-2
    lambda_: float = 1.0
    kappa: float = 1.
    scalar_reg_loss_weight_neighborhood_kernel: float = 1.
    flag_auto_weight: bool = False
    min_it_per_knot: int = 100
    _target_: str = "sbto.solvers.cbox_polar.CBO"


class CBO(SamplingBasedSolver):
    """
    CBO solver.
    """
    def __init__(self, D, cfg: ConfigCBO):
        super().__init__(D, cfg)
        self.Id = np.eye(D)
        self.logs = {}

        self.first_it = True
        self._zeros = np.zeros(D)
        self._Id = np.eye(D) * self.cfg.dt
        self._x = np.zeros((cfg.N_samples, D))
        self._consensus = np.zeros((1, D))
        self._per_particle_consensus = np.zeros((cfg.N_samples, D))
        self._delta = self.cfg.delta
        self._dt = self.cfg.dt
        # Warm-start uses increment_value() to decide whether to keep iterating
        # at the current knot count. In CBO, the empirical particle covariance
        # becomes tiny very quickly, so we track a separate search-scale
        # covariance that decays more gradually.
        self._increment_cov = np.eye(D) * self.cfg.sigma0**2
        # Small smoothing factor so the stopping signal shrinks steadily rather
        # than collapsing to the current empirical covariance in one update.
        self._increment_cov_alpha = 0.1
        # Minimum number of solver updates to keep for each knot stage before
        # allowing the warm-start stopping criterion to terminate the stage.
        self._min_it_per_knot = self.cfg.min_it_per_knot
        self._it_current_knot = 0

    def opt_first_dim(self, n_dim: int = -1):
        super().opt_first_dim(n_dim)
        self._it_current_knot = 0

    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        argmin = costs.argmin()
        cmin = costs[argmin]
        exponents = -(costs - cmin) * self.cfg.beta
        w = np.exp(exponents)
        s = w.sum()
        w /= s
        self._consensus[:, :self.n_dim] = w @ samples[:, :self.n_dim]

        neighborhood_kernel_neg_log_eval = gaussian_kernel_neg_log(
            jnp.asarray(samples[:, None, :]),
            jnp.asarray(samples[None, :, :]),
            kappa=float(self.cfg.kappa),
        )

        self._per_particle_consensus = compute_per_particle_target_consensus(
            costs, samples, neighborhood_kernel_neg_log_eval,
            1.0 / self.cfg.beta,
            self.cfg.scalar_reg_loss_weight_neighborhood_kernel,
            flag_auto_weight=self.cfg.flag_auto_weight)

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

        # use n_dim to optimize only the first few dimensions

        noise = np.sqrt(self._dt) * self._delta * self.sampler.sample(
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

        self._x[:, :self.n_dim] -= self.cfg.lambda_ * self._dt * drift - noise

        return self._x

    def update_distrib_param(self, state: SolverState, samples: Array) -> None:
        mean, cov = self.sampler.estimate_params(samples)
        state.mean, state.cov = mean, cov

        s = slice(0, self.n_dim)
        # Only update the currently optimized block so knot-increment logic
        # follows the active dimensions selected by opt_first_dim().
        self._increment_cov[s, s] += self._increment_cov_alpha * (
            cov[s, s] - self._increment_cov[s, s]
        )

    def increment_value(self) -> float:
        if self._it_current_knot < self._min_it_per_knot:
            return np.inf
        # Report the warm-start search scale instead of raw particle spread.
        return np.max(np.diag(self._increment_cov[:self.n_dim, :self.n_dim]))

    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update the solver state from elite samples.
        """
        arg_min, min_cost = self.update_mean(samples, costs)
        best = samples[arg_min]
        self.update_min_cost_best(self.state, min_cost, best, best_id=int(arg_min))
        self.update_distrib_param(self.state, samples)
        self._it_current_knot += 1

        self.first_it = False
