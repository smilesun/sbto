import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass
import jax.numpy as jnp

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver
from sbto.solvers.initial_sampling import load_mean_cov_from_solver_state

Array = npt.NDArray[np.float64]


@dataclass
class ConfigCBO(ConfigSolver):
    """
    beta: Inverse temperature.
    noise_model: isotropic | anistropic.
    delta: Diffusion term.
    dt: Step size.
    lambda_: Drift coefficient toward the consensus.
    """
    beta: float = 1.e6
    noise_model: str = "anistropic"
    delta: float = 1.e-2
    dt: float = 1.e-2
    lambda_: float = 1.0
    min_it_per_knot: int = 100
    load_initial_sampling_state: bool = True
    use_loaded_mean_only: bool = True
    ini_dist_path: str = ""
    _target_: str = "sbto.solvers.cbox.CBO"


class CBO(SamplingBasedSolver):
    """
    Consensus-based optimization with one global consensus point.
    """
    def __init__(self, D, cfg: ConfigCBO):
        super().__init__(D, cfg)

        self.first_it = True
        self._zeros = np.zeros(D)
        self._Id = np.eye(D)
        self._x = np.zeros((cfg.N_samples, D))
        self._consensus = np.zeros((1, D))
        self._delta = self.cfg.delta
        self._dt = self.cfg.dt
        self._min_it_per_knot = self.cfg.min_it_per_knot
        self._it_current_knot = 0
        self._initial_sampling_state_loaded = False

    def opt_first_dim(self, n_dim: int = -1):
        super().opt_first_dim(n_dim)
        self._it_current_knot = 0

    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        argmin = costs.argmin()
        cmin = costs[argmin]
        # Shift costs by the minimum before exponentiation. This does not
        # change the normalized weights, but keeps the best sample at exp(0)=1
        # and all others at exp(negative), which is more numerically stable
        # than exponentiating large-magnitude raw costs directly.
        exponents = -(costs - cmin) * self.cfg.beta
        w = np.exp(exponents)
        w /= w.sum()
        self._consensus[:, :self.n_dim] = w @ samples[:, :self.n_dim]
        return int(argmin), float(cmin)

    def _maybe_load_initial_sampling_state(self) -> None:
        if self._initial_sampling_state_loaded or not self.cfg.load_initial_sampling_state:
            return
        if not self.cfg.ini_dist_path:
            raise ValueError(
                "ini_dist_path must be set when "
                "load_initial_sampling_state=True."
            )

        mean, cov = load_mean_cov_from_solver_state(self.cfg.ini_dist_path)
        self.state.mean = mean
        self.state.cov = np.eye(self.D) if self.cfg.use_loaded_mean_only else cov
        self._initial_sampling_state_loaded = True

    def get_samples(self) -> Array:
        """
        Get samples from distribution parametrized by the current state.
        """
        if self.first_it:
            self._maybe_load_initial_sampling_state()
            self._x[:] = self.sampler.sample(
                mean=self.state.mean,
                cov=self.state.cov,
            )
            self._consensus[:] = self.state.mean
            return self._x

        noise = np.sqrt(self._dt) * self._delta * self.sampler.sample(
            mean=self._zeros[:self.n_dim],
            cov=self._Id[:self.n_dim, :self.n_dim],
        )

        drift = self._x[:, :self.n_dim] - self._consensus[:, :self.n_dim]

        if self.cfg.noise_model == "isotropic":
            drift_norm = np.linalg.norm(drift, axis=-1, keepdims=True)
            noise = jnp.multiply(drift_norm, noise)
        elif self.cfg.noise_model in ("anistropic", "anisotropic"):
            noise = jnp.multiply(drift, noise)
        else:
            raise ValueError(f"Invalid noise config ({self.cfg.noise_model}).")

        self._x[:, :self.n_dim] -= self.cfg.lambda_ * self._dt * drift - noise

        return self._x

    def update_distrib_param(self, state: SolverState, samples: Array) -> None:
        state.mean, state.cov = self.sampler.estimate_params(samples)

    def increment_value(self) -> float:
        if self._it_current_knot < self._min_it_per_knot:
            return np.inf
        return super().increment_value()

    def update(
        self,
        samples: Array,
        costs: Array,
    ) -> None:
        """
        Update the solver state from the current particle cloud.
        """
        arg_min, min_cost = self.update_mean(samples, costs)
        best = samples[arg_min]
        self.update_min_cost_best(self.state, min_cost, best, best_id=arg_min)
        self.update_distrib_param(self.state, samples)
        self._it_current_knot += 1

        self.first_it = False
