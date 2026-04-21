import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from sbto.evaluation.diversity import avg_joint_variance

Array = npt.NDArray[np.float64]

DIVERSITY_ANALYSIS_FILENAME = "last_iteration_diversity"


def _pairwise_distances(data: Array) -> Array:
    diff = data[:, None, :] - data[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _greedy_mode_count(data: Array, threshold: float) -> int:
    if data.shape[0] == 0:
        return 0

    centers = [data[0]]
    for sample in data[1:]:
        distances = [np.linalg.norm(sample - center) for center in centers]
        if min(distances) > threshold:
            centers.append(sample)
    return len(centers)


def analyze_last_iteration_trajectories(
    samples: Array,
    costs: Array,
    sim,
    *,
    top_k: int = 10,
) -> tuple[dict, dict]:
    _, x_traj, u_traj, obs_traj = sim.rollout(samples, with_x0=True)

    x_std_avg, _, _ = avg_joint_variance(x_traj)
    u_std_avg, _, _ = avg_joint_variance(u_traj)
    obs_std_avg, _, _ = avg_joint_variance(obs_traj)

    flat_u = u_traj.reshape(u_traj.shape[0], -1)
    best_idx = int(np.argmin(costs))
    dist_to_best = np.linalg.norm(flat_u - flat_u[best_idx], axis=1)

    n_top = max(1, min(int(top_k), costs.shape[0]))
    top_idx = np.argsort(costs)[:n_top]
    top_u = flat_u[top_idx]
    pairwise_top = _pairwise_distances(top_u)
    upper = pairwise_top[np.triu_indices(n_top, k=1)]
    mode_threshold = float(np.median(upper)) if upper.size > 0 else 0.0

    metrics = {
        "n_samples": int(costs.shape[0]),
        "top_k": int(n_top),
        "cost_best": float(np.min(costs)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "u_traj_std_avg": float(u_std_avg),
        "x_traj_std_avg": float(x_std_avg),
        "obs_traj_std_avg": float(obs_std_avg),
        "pairwise_u_dist_mean": float(np.mean(upper)) if upper.size > 0 else 0.0,
        "pairwise_u_dist_std": float(np.std(upper)) if upper.size > 0 else 0.0,
        "distance_to_best_mean": float(np.mean(dist_to_best)),
        "distance_to_best_std": float(np.std(dist_to_best)),
        "top_k_mode_threshold": float(mode_threshold),
        "top_k_mode_count": int(_greedy_mode_count(top_u, mode_threshold)),
    }

    analysis_data = {
        "costs": costs,
        "x_traj": x_traj,
        "u_traj": u_traj,
        "obs_traj": obs_traj,
        "top_idx": top_idx,
        "dist_to_best": dist_to_best,
    }
    return metrics, analysis_data


def _save_metrics(dir_path: str, metrics: dict) -> None:
    file_path = os.path.join(dir_path, f"{DIVERSITY_ANALYSIS_FILENAME}.yaml")
    with open(file_path, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)


def _plot_cost_distance(dir_path: str, costs: Array, dist_to_best: Array) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(dist_to_best, costs, alpha=0.7)
    ax.set_xlabel("Distance To Best Control Trajectory")
    ax.set_ylabel("Cost")
    ax.set_title("Last Iteration Cost vs Diversity")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "last_iteration_cost_vs_distance.pdf"), format="pdf")


def _plot_topk_controls(dir_path: str, time: Array, u_traj: Array, costs: Array, top_idx: Array) -> None:
    top_u = u_traj[top_idx]
    top_costs = costs[top_idx]
    nu = top_u.shape[-1]

    plt.close("all")
    fig, axs = plt.subplots(nu, 1, figsize=(10, max(2.5 * nu, 4)), sharex=True)
    if nu == 1:
        axs = [axs]

    for dim in range(nu):
        ax = axs[dim]
        for rank, (traj, cost) in enumerate(zip(top_u, top_costs), start=1):
            ax.plot(time[:traj.shape[0]], traj[:, dim], alpha=0.8, label=f"top{rank} cost={cost:.3g}")
        ax.set_ylabel(f"u[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.5)

    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("Time step")
    fig.suptitle("Top-K Control Trajectories From Last Iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "last_iteration_topk_controls.pdf"), format="pdf")


def _plot_topk_states(dir_path: str, time: Array, x_traj: Array, costs: Array, top_idx: Array, max_dims: int = 6) -> None:
    top_x = x_traj[top_idx]
    top_costs = costs[top_idx]
    nx = top_x.shape[-1]
    n_plot = min(nx, max_dims)

    plt.close("all")
    fig, axs = plt.subplots(n_plot, 1, figsize=(10, max(2.5 * n_plot, 4)), sharex=True)
    if n_plot == 1:
        axs = [axs]

    for dim in range(n_plot):
        ax = axs[dim]
        for rank, (traj, cost) in enumerate(zip(top_x, top_costs), start=1):
            ax.plot(time[:traj.shape[0]], traj[:, dim], alpha=0.8, label=f"top{rank} cost={cost:.3g}")
        ax.set_ylabel(f"x[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.5)

    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("Time step")
    fig.suptitle("Top-K State Trajectories From Last Iteration")
    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, "last_iteration_topk_states.pdf"), format="pdf")


def save_last_iteration_diversity_analysis(
    dir_path: str,
    sim,
    samples_last: Array,
    costs_last: Array,
    *,
    top_k: int = 10,
) -> None:
    metrics, analysis_data = analyze_last_iteration_trajectories(
        samples_last,
        costs_last,
        sim,
        top_k=top_k,
    )
    _save_metrics(dir_path, metrics)
    _plot_cost_distance(dir_path, analysis_data["costs"], analysis_data["dist_to_best"])
    _plot_topk_controls(
        dir_path,
        np.arange(analysis_data["u_traj"].shape[1]),
        analysis_data["u_traj"],
        analysis_data["costs"],
        analysis_data["top_idx"],
    )
    _plot_topk_states(
        dir_path,
        np.arange(analysis_data["x_traj"].shape[1]),
        analysis_data["x_traj"],
        analysis_data["costs"],
        analysis_data["top_idx"],
    )
