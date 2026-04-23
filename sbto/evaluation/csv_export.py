"""
CSV counterparts for every evaluation plot.

Each function receives the same data as the corresponding plot function and
writes a CSV with the same base path (pdf → csv).  This lets ablation studies
load all trials and plot them together without re-running simulations.
"""

import csv
import os
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _csv_path(pdf_path: str) -> str:
    return os.path.splitext(pdf_path)[0] + ".csv"


# ---------------------------------------------------------------------------
# cluster_diversity
# ---------------------------------------------------------------------------

def save_cluster_costs_csv(pdf_path: str, best_costs: Array, cluster_sizes: Array) -> None:
    with open(_csv_path(pdf_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "best_cost", "size"])
        for i, (c, s) in enumerate(zip(best_costs, cluster_sizes)):
            w.writerow([i, float(c), int(s)])


def save_cluster_representatives_csv(
    pdf_path: str,
    time: Array,
    traj: Array,               # (N, T, D)
    best_idx_per_cluster: Array,
    best_costs_per_cluster: Array,
    *,
    prefix: str,
) -> None:
    """One row per (cluster, timestep).  All D dimensions are saved."""
    rep = traj[best_idx_per_cluster]   # (K, T, D)
    K, T, D = rep.shape
    t_arr = np.arange(T) if len(time) < T else time[:T]
    with open(_csv_path(pdf_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "best_cost", "timestep"]
                   + [f"{prefix}_{d}" for d in range(D)])
        for k in range(K):
            cost = float(best_costs_per_cluster[k])
            for t in range(T):
                w.writerow([k, cost, float(t_arr[t])] + rep[k, t].tolist())


# ---------------------------------------------------------------------------
# diversity (last-iteration)
# ---------------------------------------------------------------------------

def save_cost_distance_csv(pdf_path: str, costs: Array, dist_to_best: Array) -> None:
    with open(_csv_path(pdf_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "cost", "dist_to_best"])
        for i, (c, d) in enumerate(zip(costs, dist_to_best)):
            w.writerow([i, float(c), float(d)])


def save_topk_controls_csv(
    pdf_path: str,
    time: Array,
    u_traj: Array,    # (N, T, Nu)
    costs: Array,
    top_idx: Array,
) -> None:
    top_u = u_traj[top_idx]
    top_costs = costs[top_idx]
    K, T, Nu = top_u.shape
    t_arr = np.arange(T) if len(time) < T else time[:T]
    with open(_csv_path(pdf_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "cost", "timestep"] + [f"u_{d}" for d in range(Nu)])
        for rank, (traj, cost) in enumerate(zip(top_u, top_costs)):
            for t in range(T):
                w.writerow([rank, float(cost), float(t_arr[t])] + traj[t].tolist())


def save_topk_states_csv(
    pdf_path: str,
    time: Array,
    x_traj: Array,    # (N, T, Nx)
    costs: Array,
    top_idx: Array,
) -> None:
    top_x = x_traj[top_idx]
    top_costs = costs[top_idx]
    K, T, Nx = top_x.shape
    t_arr = np.arange(T) if len(time) < T else time[:T]
    with open(_csv_path(pdf_path), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "cost", "timestep"] + [f"x_{d}" for d in range(Nx)])
        for rank, (traj, cost) in enumerate(zip(top_x, top_costs)):
            for t in range(T):
                w.writerow([rank, float(cost), float(t_arr[t])] + traj[t].tolist())
