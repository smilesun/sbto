"""
Stein Variational CMA-ES (SV-CMA-ES) solver wrapper for SBTO.

Wraps the evosax SV_CMA_ES algorithm (Braun et al., 2024, arXiv:2410.10390)
as a drop-in SBTO solver.  SV-CMA-ES runs `num_populations` independent
CMA-ES instances that repel each other via a Stein variational kernel gradient,
encouraging diverse solution clusters — conceptually similar to PCBO but using
the SVGD repulsion mechanism instead of the polarization potential.

Install evosax before use:
    uv add evosax
    # or: pip install evosax

API compatibility note
-----------------------
evosax works with JAX float32; SBTO uses numpy float64.
Conversion is done at the boundary (get_samples / update).

Cost convention
---------------
evosax *maximises* fitness; SBTO *minimises* cost.
The wrapper negates costs: fitness = −cost.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver

Array = npt.NDArray[np.float64]


@dataclass
class ConfigSVCMAES(ConfigSolver):
    """
    Parameters
    ----------
    num_populations : int
        Number of independent CMA-ES populations (sub-swarms).
        N_samples must be divisible by num_populations.
    kernel_std : float
        RBF kernel bandwidth for SVGD repulsion between population means.
        Larger values = weaker repulsion (means can be closer together).
    alpha : float
        Weighting factor for the SVGD repulsive gradient.
        Larger values = stronger repulsion.
    """
    num_populations: int = 4
    kernel_std: float = 1.0
    alpha: float = 1.0
    _target_: str = "sbto.solvers.sv_cma_es_solver.SVCMAESSolver"


class SVCMAESSolver(SamplingBasedSolver):
    """
    Stein Variational CMA-ES solver for SBTO.

    Each call to get_samples() returns num_populations × population_size samples
    where population_size = N_samples // num_populations.
    """

    def __init__(self, D: int, cfg: ConfigSVCMAES):
        super().__init__(D, cfg)

        try:
            import jax
            import jax.numpy as jnp
            from evosax.algorithms.distribution_based.sv.sv_cma_es import SV_CMA_ES
        except ImportError as e:
            raise ImportError(
                "evosax is required for SVCMAESSolver. "
                "Install it with: uv add evosax"
            ) from e

        self._jax = jax
        self._jnp = jnp

        num_pops = cfg.num_populations
        if cfg.N_samples % num_pops != 0:
            warnings.warn(
                f"N_samples={cfg.N_samples} is not divisible by "
                f"num_populations={num_pops}. "
                f"Rounding down to {cfg.N_samples // num_pops * num_pops} samples."
            )
        pop_size = cfg.N_samples // num_pops
        self._pop_size = pop_size
        self._num_pops = num_pops
        self._N = pop_size * num_pops

        solution = jnp.zeros(D, dtype=jnp.float32)
        self._strategy = SV_CMA_ES(
            population_size=pop_size,
            num_populations=num_pops,
            solution=solution,
        )

        # Override default kernel/alpha params; use sigma0 as initial CMA-ES step size
        base_params = self._strategy._default_params
        self._es_params = base_params.replace(
            std_init=float(cfg.sigma0),
            kernel_std=float(cfg.kernel_std),
            alpha=float(cfg.alpha),
        )

        key = jax.random.PRNGKey(cfg.seed)
        self._key = key

        # Initialize all populations at zero mean.
        # If a warm-start mean is set on solver.state before the first get_samples()
        # call, _apply_warm_start() will reinitialize from that mean.
        means = jnp.zeros((num_pops, D), dtype=jnp.float32)
        self._es_state = self._strategy.init(key, means, self._es_params)

        # Cached last population (numpy float64) returned by get_samples()
        self._last_pop: Optional[Array] = None
        self._first_it = True
        self._warm_start_applied = False

    def _split_key(self):
        self._key, subkey = self._jax.random.split(self._key)
        return subkey

    def _apply_warm_start(self) -> None:
        """Reinitialize evosax means from solver.state.mean (set by warm-start)."""
        jnp = self._jnp
        warm_mean = self.state.mean.astype(np.float32)
        # All populations start at the same warm-start mean; SVGD repulsion
        # will spread them out during optimisation
        means = jnp.array(
            np.tile(warm_mean, (self._num_pops, 1)), dtype=jnp.float32
        )
        self._es_state = self._strategy.init(
            self._split_key(), means, self._es_params
        )
        self._warm_start_applied = True

    def get_samples(self) -> Array:
        if self._first_it:
            # Apply warm-start mean to evosax state if one was provided
            if not self._warm_start_applied and np.any(self.state.mean != 0.0):
                self._apply_warm_start()

            fixed = self._load_fixed_population()
            if fixed is not None:
                fixed = fixed[: self._N]
                self._last_pop = np.asarray(fixed, dtype=np.float64)
                return self._last_pop

        pop_jax, self._es_state = self._strategy.ask(
            self._split_key(), self._es_state, self._es_params
        )
        # Convert to numpy float64 for SBTO cost evaluation
        self._last_pop = np.asarray(pop_jax, dtype=np.float64)
        return self._last_pop

    def update(self, samples: Array, costs: Array) -> None:
        jnp = self._jnp

        # evosax maximises; negate to convert from cost to fitness
        fitness = jnp.array(-costs[: self._N], dtype=jnp.float32)
        population = jnp.array(samples[: self._N], dtype=jnp.float32)

        self._es_state, _ = self._strategy.tell(
            self._split_key(), population, fitness, self._es_state, self._es_params
        )

        # ---------- update SBTO-compatible state ----------
        arg_min = int(np.argmin(costs[: self._N]))
        best = samples[arg_min].copy()
        min_cost = float(costs[arg_min])
        self.update_min_cost_best(self.state, min_cost, best, best_id=arg_min)

        # Set state.mean to the mean of the best-performing population
        means_np = np.asarray(self._es_state.mean, dtype=np.float64)  # (K, D)
        pop_idx = arg_min // self._pop_size
        self.state.mean = means_np[pop_idx]

        self._first_it = False

    def increment_value(self) -> float:
        # Use median std across all populations as a proxy for convergence
        std_arr = np.asarray(self._es_state.std, dtype=np.float64)
        return float(np.median(std_arr))
