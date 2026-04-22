
import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass
import jax.numpy as jnp

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver
from sbto.solvers.initial_sampling import load_mean_cov_from_solver_state
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
    kappa: neighborhood kernel bandwidth (used when adaptive_kappa=False)
    adaptive_kappa: set kappa each iteration from the actual inter-particle
        distance distribution; fixes the common failure mode where kappa<<
        typical particle distance causes the kernel to vanish identically
    kappa_subsample: number of particles used to estimate adaptive kappa
    """
    beta: float = 1.e6
    noise_model: str = "anistropic"
    delta: float = 1.e-2
    dt: float = 1.e-2
    lambda_: float = 1.0
    kappa: float = 1.
    adaptive_kappa: bool = True
    kappa_subsample: int = 128
    scalar_reg_loss_weight_neighborhood_kernel: float = 1.
    flag_auto_weight: bool = False
    min_it_per_knot: int = 100
    load_initial_sampling_state: bool = True
    use_loaded_mean_only: bool = True
    ini_dist_path: str = ""
    mixed_init_frac: float = 0.5
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
        self._initial_sampling_state_loaded = False

    def opt_first_dim(self, n_dim: int = -1):
        super().opt_first_dim(n_dim)
        self._it_current_knot = 0

    def _compute_adaptive_kappa(self, samples: Array) -> float:
        """Median inter-particle distance estimated from a cheap subsample."""
        N = samples.shape[0]
        m = min(self.cfg.kappa_subsample, N)
        rng = np.random.default_rng(self._it_current_knot)
        idx = rng.choice(N, m, replace=False)
        sub = samples[idx, :self.n_dim]
        diff = sub[:, None, :] - sub[None, :, :]   # (m, m, n_dim)
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))  # (m, m)
        upper = dists[np.triu_indices(m, k=1)]
        kappa = float(np.median(upper)) if upper.size > 0 else float(self.cfg.kappa)
        # Guard against degenerate collapsed particles.
        return max(kappa, 1e-6)

    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        argmin = costs.argmin()
        cmin = costs[argmin]
        exponents = -(costs - cmin) * self.cfg.beta
        w = np.exp(exponents)
        w /= w.sum()
        self._consensus[:, :self.n_dim] = w @ samples[:, :self.n_dim]

        kappa = self._compute_adaptive_kappa(samples) if self.cfg.adaptive_kappa \
            else float(self.cfg.kappa)

        # Restrict to active dimensions to avoid O(N² · D) intermediate array.
        s_active = jnp.asarray(samples[:, :self.n_dim])
        neighborhood_kernel_neg_log_eval = gaussian_kernel_neg_log(
            s_active[:, None, :],
            s_active[None, :, :],
            kappa=kappa,
        )

        per_particle = compute_per_particle_target_consensus(
            costs, s_active, neighborhood_kernel_neg_log_eval,
            1.0 / self.cfg.beta,
            self.cfg.scalar_reg_loss_weight_neighborhood_kernel,
            flag_auto_weight=self.cfg.flag_auto_weight,
        )
        self._per_particle_consensus[:, :self.n_dim] = np.asarray(per_particle)

        return argmin, cmin

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

        On the first iteration, when load_initial_sampling_state=True, uses a
        mixed initialisation: mixed_init_frac particles come from the loaded
        distribution while the remainder are drawn from a broad sigma0-Gaussian
        centred at zero.  This diversity in the initial swarm helps PCBO find
        multiple trajectory modes.
        """
        if self.first_it:
            self._maybe_load_initial_sampling_state()

            frac = self.cfg.mixed_init_frac if self.cfg.load_initial_sampling_state else 1.0
            n_loaded = int(round(frac * self.cfg.N_samples))
            n_random = self.cfg.N_samples - n_loaded

            noise = self.sampler.sample(mean=self._zeros, cov=self.state.cov)
            self._x[:n_loaded] = self.state.mean + self.cfg.delta * noise[:n_loaded]

            if n_random > 0:
                broad_cov = np.eye(self.D) * self.cfg.sigma0 ** 2
                noise_broad = self.sampler.sample(mean=self._zeros, cov=broad_cov)
                self._x[n_loaded:] = noise_broad[:n_random]

            return self._x

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
