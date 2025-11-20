import os
import numpy as np
from typing import List

from sbto.data.utils import ALL_SAMPLES_COSTS_FILENAME, save_rollout

def get_all_costs_and_samples_paths(data_dir: str) -> List[str]:
    all_costs_samples_paths = []
    for exp_dir in os.listdir(data_dir):
        exp_path = os.path.join(data_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        costs_samples_path = os.path.join(
            exp_path,
            f"{ALL_SAMPLES_COSTS_FILENAME}.npz"
        )
        all_costs_samples_paths.append(costs_samples_path)
    
    return all_costs_samples_paths

def get_top_samples(
    costs: np.ndarray,
    samples: np.ndarray,
    N_top_samples: float
    ) -> np.ndarray:
    assert costs.ndim == 2, "Expected costs of shape (N, T)."
    assert samples.ndim == 3, "Expected samples of shape (N, T, D)."
    assert costs.shape[:2] == samples.shape[:2], "Mismatched iteration/sample dimensions."

    D = samples.shape[2]
    # Flatten across all iterations
    costs_flat = costs.reshape(-1)
    samples_flat = samples.reshape(-1, D)
    
    N_total = costs_flat.shape[0]
    top_percent = N_top_samples / N_total * 100.
    print(f"Aggregating top {top_percent:.2f}%")

    # Remove samples in double (in case keep elites for instance)
    costs_flat_unique, arg_unique = np.unique(costs_flat, return_index=True, sorted=True)
    samples_flat_unique = samples_flat[arg_unique]

    return samples_flat_unique[:N_top_samples], costs_flat_unique[:N_top_samples]

def aggregate_top_samples(
    sim,
    task,
    data_dir: str,
    N_top_samples: int,
    min_iteration: int = 0,
    ):
    all_costs_samples_paths = get_all_costs_and_samples_paths(data_dir)
    all_costs = []
    all_samples = []

    for path in all_costs_samples_paths:
        file = np.load(path)
        samples, costs = file["samples"], file["costs"]

        all_costs.append(costs)
        all_samples.append(samples)

    all_samples_arr = np.vstack(all_samples)[min_iteration:, :]
    all_costs_arr = np.vstack(all_costs)[min_iteration:, :]

    top_samples, top_costs = get_top_samples(all_costs_arr, all_samples_arr, N_top_samples)
    # Rollout to get data
    t, state_traj, u_traj, obs_traj = sim.rollout(top_samples)
    ## Sanity check (should have the same cost as during the optimization)
    # c = task.cost(state_traj, u_traj, obs_traj)
    # print(top_costs.min(), top_costs.max())
    # print(c.min(), c.max())

    save_rollout(
        data_dir,
        time=t,
        x_traj=state_traj,
        u_traj=u_traj,
        obs_traj=obs_traj,
        costs=top_costs
    )
