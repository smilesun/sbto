"""
Population initialization strategies for sampling-based solvers.

Each class is callable: strategy(N, D) -> ndarray of shape (N, D).

Classes
-------
RandomGaussianInit       — pure N(0, σ₀²·I) random draw
WarmStartMeanOnlyInit    — CEM mean + isotropic noise (scaled identity cov)
WarmStartFullCovInit     — CEM mean + correlated noise (loaded cov × scale)
MixedInit                — frac particles from warm start, rest random Gaussian
SavedPopulationInit      — load a previously saved population from disk

Utilities
---------
generate_population(strategy, N, D, save_path="")
    Generate population and optionally persist to disk.

Persisting to disk makes cross-solver comparisons reproducible: generate
once, then pass the .npz path to both CEM and PCBO via population_init_path.

Memory note: set save_to_disk=False (default) to skip disk I/O.
A population of N=2048 particles × D=345 dims uses ~5.6 MB on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sbto.solvers.initial_sampling import load_mean_cov_from_solver_state

Array = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_mvn(rng: np.random.Generator, mean: Array, cov: Array, N: int) -> Array:
    """Sample N vectors from N(mean, cov) via eigendecomposition (handles near-singular cov)."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    z = rng.standard_normal((N, len(mean)))
    return mean + (z * np.sqrt(eigvals)) @ eigvecs.T


# ---------------------------------------------------------------------------
# Init strategy classes
# ---------------------------------------------------------------------------

@dataclass
class RandomGaussianInit:
    """Pure isotropic Gaussian centred at zero."""
    sigma0: float = 0.25
    seed: int = 0

    def __call__(self, N: int, D: int) -> Array:
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, self.sigma0, (N, D))


@dataclass
class WarmStartMeanOnlyInit:
    """Load mean from solver state; use scaled identity covariance."""
    ini_dist_path: str
    cov_scale: float = 1.0
    seed: int = 0

    def __call__(self, N: int, D: int) -> Array:
        mean, _ = load_mean_cov_from_solver_state(self.ini_dist_path)
        cov = np.eye(D) * self.cov_scale
        rng = np.random.default_rng(self.seed)
        return _sample_mvn(rng, mean, cov, N)


@dataclass
class WarmStartFullCovInit:
    """Load mean + full (non-diagonal) covariance from solver state, optionally scaled."""
    ini_dist_path: str
    cov_scale: float = 1.0
    seed: int = 0

    def __call__(self, N: int, D: int) -> Array:
        mean, cov = load_mean_cov_from_solver_state(self.ini_dist_path)
        rng = np.random.default_rng(self.seed)
        return _sample_mvn(rng, mean, cov * self.cov_scale, N)


@dataclass
class MixedInit:
    """
    mixed_init_frac particles from warm start, remainder from broad random Gaussian.
    Mirrors the mixed initialisation used inside the PCBO solver.
    """
    ini_dist_path: str
    mixed_init_frac: float = 0.5
    use_full_cov: bool = False
    cov_scale: float = 1.0
    sigma0: float = 0.25
    seed: int = 0

    def __call__(self, N: int, D: int) -> Array:
        mean, cov = load_mean_cov_from_solver_state(self.ini_dist_path)
        rng = np.random.default_rng(self.seed)

        n_warm = int(round(self.mixed_init_frac * N))
        n_random = N - n_warm

        warm_cov = (cov if self.use_full_cov else np.eye(D)) * self.cov_scale
        pop = np.empty((N, D))
        pop[:n_warm] = _sample_mvn(rng, mean, warm_cov, n_warm)
        if n_random > 0:
            pop[n_warm:] = rng.normal(0.0, self.sigma0, (n_random, D))
        return pop


@dataclass
class SavedPopulationInit:
    """
    Load a pre-generated population from an .npz file (key: 'population').
    Use this to give CEM and PCBO identical starting particles.
    """
    population_path: str

    def __call__(self, N: int, D: int) -> Array:
        data = np.load(self.population_path)
        pop = np.asarray(data["population"], dtype=np.float64)
        if pop.shape != (N, D):
            raise ValueError(
                f"Saved population shape {pop.shape} does not match requested ({N}, {D}). "
                "Regenerate the population with matching N and D."
            )
        return pop


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def generate_population(
    strategy,
    N: int,
    D: int,
    save_path: str = "",
) -> Array:
    """
    Generate an initial population and optionally persist it to disk.

    Parameters
    ----------
    strategy : callable(N, D) -> ndarray
        Any of the init classes above.
    N : int
        Number of particles.
    D : int
        Dimensionality.
    save_path : str
        If non-empty, save the population as an .npz file at this path.
        Pass the same path to SavedPopulationInit to reuse it across solvers.

    Returns
    -------
    population : ndarray of shape (N, D)
    """
    pop = strategy(N, D)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, population=pop)
    return pop
