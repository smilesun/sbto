import numpy as np


def get_top_samples(
    costs: np.ndarray,
    samples: np.ndarray,
    N_top_samples: float
    ) -> np.ndarray:
    assert costs.ndim == 2, "Expected costs of shape (N_it, N)."
    assert samples.ndim == 3, "Expected samples of shape (N_it, N, D)."
    assert costs.shape[:2] == samples.shape[:2], "Mismatched iteration/sample dimensions."

    D = samples.shape[2]
    # Flatten across all iterations
    costs_flat = costs.reshape(-1)
    samples_flat = samples.reshape(-1, D)

    # Remove samples in double (in case keep elites for instance)
    costs_flat_unique, arg_unique = np.unique(costs_flat, return_index=True, sorted=True)
    samples_flat_unique = samples_flat[arg_unique]

    return samples_flat_unique[:N_top_samples], costs_flat_unique[:N_top_samples]


def get_top_samples_per_population(
    costs: np.ndarray,
    samples: np.ndarray,
    num_populations: int,
    k_per_pop: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the best k_per_pop samples from each SV-CMA-ES subpopulation.

    Samples for population p occupy indices [p*pop_size : (p+1)*pop_size] in
    the N axis of both `costs` and `samples`.

    Returns
    -------
    top_samples : (num_populations * k_per_pop, D)
    top_costs   : (num_populations * k_per_pop,)
    pop_labels  : (num_populations * k_per_pop,)  integer population index
    """
    assert costs.ndim == 2, "Expected costs of shape (N_it, N)."
    assert samples.ndim == 3, "Expected samples of shape (N_it, N, D)."
    N_it, N, D = samples.shape
    assert N % num_populations == 0, (
        f"N_samples={N} not divisible by num_populations={num_populations}"
    )
    pop_size = N // num_populations

    all_top_samples, all_top_costs, all_pop_labels = [], [], []
    for p in range(num_populations):
        costs_p = costs[:, p * pop_size : (p + 1) * pop_size]
        samples_p = samples[:, p * pop_size : (p + 1) * pop_size, :]
        top_s, top_c = get_top_samples(costs_p, samples_p, k_per_pop)
        all_top_samples.append(top_s)
        all_top_costs.append(top_c)
        all_pop_labels.append(np.full(len(top_c), p, dtype=np.int32))

    return (
        np.concatenate(all_top_samples, axis=0),
        np.concatenate(all_top_costs, axis=0),
        np.concatenate(all_pop_labels, axis=0),
    )