"""
Wishart-mixture population initialisation.

Instead of sampling all particles from a single covariance matrix, this module
draws K distinct positive-definite covariance matrices from the Wishart
distribution and uses each to sample a disjoint sub-population.  The result is
a more diverse initial particle cloud: sub-populations explore different
principal directions in control space, which can help PCBO polarise into more
modes.

Background
----------
If V is the base (scale) covariance matrix (e.g. from a CEM warm start),
then drawing Σ_k ~ Wishart(df, V/df) gives E[Σ_k] = V.

- Large df  → tight spread around V  (near-identical covariances)
- Small df  → high spread  (very different covariances; minimum is df = D + 1)

Usage example
-------------
>>> from sbto.solvers.wishart_population_init import WishartMixInit
>>> from sbto.solvers.population_init import generate_population
>>>
>>> strategy = WishartMixInit(
...     ini_dist_path="datasets/.../solver_state_final.npz",
...     n_components=8,
...     wishart_df=None,      # auto: D + 2
...     cov_scale=1.0,
...     use_full_cov=True,    # use loaded CEM covariance as base
...     seed=0,
... )
>>> pop = generate_population(strategy, N=2048, D=435,
...                           save_path="datasets/wishart_init_pop.npz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import wishart

from sbto.solvers.initial_sampling import load_mean_cov_from_solver_state
from sbto.solvers.population_init import _sample_mvn

Array = npt.NDArray[np.float64]


@dataclass
class WishartMixInit:
    """
    Multi-component Gaussian mixture: shared mean, Wishart-sampled covariances.

    Parameters
    ----------
    ini_dist_path : str
        Path to a solver_state_final.npz containing 'mean' and 'cov'.
    n_components : int
        Number of sub-populations (distinct covariance draws).
        Each component receives approximately N // n_components particles.
    wishart_df : int or None
        Degrees of freedom for the Wishart distribution.
        Must be >= D (dimensionality).  If None, uses D + 2 (minimum
        non-degenerate, maximum diversity).  Increase to concentrate
        draws closer to the base covariance.
    cov_scale : float
        Multiplicative factor applied to the base covariance before
        passing to the Wishart.  >1 broadens the initial cloud; <1 tightens.
    use_full_cov : bool
        True  → use the loaded CEM covariance as the base (non-diagonal,
                captures learned correlations).
        False → use a scaled identity matrix as the base (isotropic draws).
    seed : int
        RNG seed for reproducibility.
    """

    ini_dist_path: str
    n_components: int = 8
    wishart_df: Optional[int] = None
    cov_scale: float = 1.0
    use_full_cov: bool = True
    seed: int = 0

    def __call__(self, N: int, D: int) -> Array:
        mean, cov_loaded = load_mean_cov_from_solver_state(self.ini_dist_path)

        # Base covariance — shared across all Wishart draws
        base_cov = (cov_loaded if self.use_full_cov else np.eye(D)) * self.cov_scale

        df = self.wishart_df if self.wishart_df is not None else D + 2
        if df < D:
            raise ValueError(
                f"wishart_df={df} must be >= D={D}. "
                "Increase wishart_df or reduce dimensionality."
            )

        # Wishart scale matrix: E[Σ_k] = df * (base_cov / df) = base_cov
        scale_matrix = base_cov / df

        rng = np.random.default_rng(self.seed)
        K = self.n_components

        # Distribute N particles across K components (last component gets remainder)
        n_per_comp = N // K
        sizes = [n_per_comp] * K
        sizes[-1] += N - n_per_comp * K  # add remainder to last component

        pop = np.empty((N, D))
        idx = 0
        for k, n_k in enumerate(sizes):
            # Draw a random covariance matrix for this sub-population
            cov_k = wishart.rvs(df=df, scale=scale_matrix, random_state=rng)
            pop[idx: idx + n_k] = _sample_mvn(rng, mean, cov_k, n_k)
            idx += n_k

        return pop

    def component_covariances(self, D: int) -> list[Array]:
        """
        Return the K covariance matrices that *would* be drawn for a given D.
        Useful for inspection / visualisation without generating the full population.
        """
        _, cov_loaded = load_mean_cov_from_solver_state(self.ini_dist_path)
        base_cov = (cov_loaded if self.use_full_cov else np.eye(D)) * self.cov_scale
        df = self.wishart_df if self.wishart_df is not None else D + 2
        scale_matrix = base_cov / df
        rng = np.random.default_rng(self.seed)
        return [wishart.rvs(df=df, scale=scale_matrix, random_state=rng)
                for _ in range(self.n_components)]


@dataclass
class RandomWishartMixInit:
    """
    Zero-mean Wishart mixture — no warm start required.

    Matches CEM's default initialisation (mean=0, cov=sigma0²·I) in
    expectation, but gives each sub-population a *distinct* covariance
    drawn from Wishart(df, sigma0²·cov_scale·I / df).

    This creates a more diverse initial cloud than pure N(0, sigma0²·I)
    while remaining free of any warm-start dependency.

    Parameters
    ----------
    sigma0 : float
        Base standard deviation (same meaning as ConfigSolver.sigma0).
    cov_scale : float
        Scale multiplier on sigma0².  >1 gives a broader initial cloud.
        cov_scale=1.0 matches CEM's default spread in expectation.
    n_components : int
        Number of sub-populations (distinct Wishart draws).
    wishart_df : int or None
        Degrees of freedom.  None → D + 2 (maximum diversity for given D).
    seed : int
    """

    sigma0: float = 0.25
    cov_scale: float = 1.0
    n_components: int = 8
    wishart_df: Optional[int] = None
    seed: int = 0

    def __call__(self, N: int, D: int) -> Array:
        rng = np.random.default_rng(self.seed)
        mean = np.zeros(D)
        base_cov = np.eye(D) * (self.sigma0 ** 2) * self.cov_scale

        df = self.wishart_df if self.wishart_df is not None else D + 2
        if df < D:
            raise ValueError(f"wishart_df={df} must be >= D={D}.")
        scale_matrix = base_cov / df

        K = self.n_components
        n_per_comp = N // K
        sizes = [n_per_comp] * K
        sizes[-1] += N - n_per_comp * K

        pop = np.empty((N, D))
        idx = 0
        for n_k in sizes:
            cov_k = wishart.rvs(df=df, scale=scale_matrix, random_state=rng)
            pop[idx: idx + n_k] = _sample_mvn(rng, mean, cov_k, n_k)
            idx += n_k
        return pop


@dataclass
class WishartMixMeanInit:
    """
    Variant: each sub-population also perturbs the mean slightly.

    In addition to distinct Wishart-drawn covariances, the mean of each
    component is jittered by a small isotropic Gaussian (jitter_std).
    This allows the initial cloud to explore multiple basins even before
    the first optimisation step.

    Parameters
    ----------
    jitter_std : float
        Standard deviation of the mean perturbation per component.
        Set to 0.0 to recover plain WishartMixInit behaviour.
    All other parameters are the same as WishartMixInit.
    """

    ini_dist_path: str
    n_components: int = 8
    wishart_df: Optional[int] = None
    cov_scale: float = 1.0
    use_full_cov: bool = True
    jitter_std: float = 0.05
    seed: int = 0

    def __call__(self, N: int, D: int) -> Array:
        mean, cov_loaded = load_mean_cov_from_solver_state(self.ini_dist_path)

        base_cov = (cov_loaded if self.use_full_cov else np.eye(D)) * self.cov_scale
        df = self.wishart_df if self.wishart_df is not None else D + 2
        if df < D:
            raise ValueError(f"wishart_df={df} must be >= D={D}.")
        scale_matrix = base_cov / df

        rng = np.random.default_rng(self.seed)
        K = self.n_components
        n_per_comp = N // K
        sizes = [n_per_comp] * K
        sizes[-1] += N - n_per_comp * K

        pop = np.empty((N, D))
        idx = 0
        for k, n_k in enumerate(sizes):
            cov_k = wishart.rvs(df=df, scale=scale_matrix, random_state=rng)
            mean_k = mean + rng.normal(0.0, self.jitter_std, D)
            pop[idx: idx + n_k] = _sample_mvn(rng, mean_k, cov_k, n_k)
            idx += n_k

        return pop
