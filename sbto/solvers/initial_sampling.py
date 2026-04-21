import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def load_mean_cov_from_solver_state(path: str) -> tuple[Array, Array]:
    """
    Load mean and covariance arrays from a saved solver_state_final.npz file.
    """
    data = np.load(path)
    mean = np.asarray(data["mean"], dtype=np.float64)
    cov = np.asarray(data["cov"], dtype=np.float64)
    return mean, cov
