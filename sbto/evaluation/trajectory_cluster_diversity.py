import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _pairwise_distances(data: Array) -> Array:
    diff = data[:, None, :] - data[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _default_threshold(data: Array) -> float:
    pairwise = _pairwise_distances(data)
    upper = pairwise[np.triu_indices(pairwise.shape[0], k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.median(upper))


def _cluster_sorted_by_cost(data: Array, costs: Array, threshold: float) -> list[dict]:
    order = np.argsort(costs)
    clusters: list[dict] = []

    for idx in order:
        sample = data[idx]
        assigned = False
        for cluster in clusters:
            dist = np.linalg.norm(sample - cluster["center"])
            if dist <= threshold:
                cluster["indices"].append(int(idx))
                assigned = True
                break
        if not assigned:
            clusters.append(
                {
                    "center": sample,
                    "indices": [int(idx)],
                }
            )
    return clusters


def analyze_cluster_diversity(
    samples: Array,
    costs: Array,
    sim,
    *,
    cluster_on: str = "u_traj",
    cluster_threshold: float | None = None,
    max_samples: int = 100,
) -> tuple[dict, dict]:
    n_keep = min(int(max_samples), costs.shape[0])
    top_idx = np.argsort(costs)[:n_keep]
    samples = samples[top_idx]
    costs = costs[top_idx]

    _, x_traj, u_traj, obs_traj = sim.rollout(samples, with_x0=True)

    if cluster_on == "u_traj":
        cluster_data = u_traj.reshape(u_traj.shape[0], -1)
    elif cluster_on == "x_traj":
        cluster_data = x_traj.reshape(x_traj.shape[0], -1)
    elif cluster_on == "obs_traj":
        cluster_data = obs_traj.reshape(obs_traj.shape[0], -1)
    else:
        raise ValueError(f"Unsupported cluster_on value: {cluster_on}")

    threshold = _default_threshold(cluster_data) if cluster_threshold is None else float(cluster_threshold)
    clusters = _cluster_sorted_by_cost(cluster_data, costs, threshold)

    best_idx_per_cluster = np.array([cluster["indices"][0] for cluster in clusters], dtype=np.int32)
    best_costs_per_cluster = costs[best_idx_per_cluster]
    cluster_sizes = np.array([len(cluster["indices"]) for cluster in clusters], dtype=np.int32)

    metrics = {
        "cluster_on": cluster_on,
        "n_clustered_samples": int(n_keep),
        "cluster_threshold": float(threshold),
        "n_clusters": int(len(clusters)),
        "cluster_sizes": cluster_sizes.tolist(),
        "best_costs_per_cluster": [float(c) for c in best_costs_per_cluster],
        "best_indices_per_cluster": best_idx_per_cluster.tolist(),
    }

    analysis_data = {
        "costs": costs,
        "x_traj": x_traj,
        "u_traj": u_traj,
        "obs_traj": obs_traj,
        "best_idx_per_cluster": best_idx_per_cluster,
        "best_costs_per_cluster": best_costs_per_cluster,
        "cluster_sizes": cluster_sizes,
    }
    return metrics, analysis_data


def _save_yaml(file_path: str, data: dict) -> None:
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _plot_cluster_costs(file_path: str, best_costs: Array, cluster_sizes: Array) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(best_costs.shape[0])
    ax.bar(x, best_costs)
    for i, size in enumerate(cluster_sizes):
        ax.text(i, best_costs[i], f"n={int(size)}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Best Cost")
    ax.set_title("Best Cost Per Cluster")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(file_path, format="pdf")


def _plot_cluster_representatives(
    file_path: str,
    time: Array,
    traj: Array,
    best_idx_per_cluster: Array,
    best_costs_per_cluster: Array,
    cluster_sizes: Array,
    *,
    prefix: str,
    max_dims: int = 6,
) -> None:
    rep_traj = traj[best_idx_per_cluster]
    n_dim = min(rep_traj.shape[-1], max_dims)

    plt.close("all")
    fig, axs = plt.subplots(n_dim, 1, figsize=(10, max(4, 2.5 * n_dim)), sharex=True)
    if n_dim == 1:
        axs = [axs]

    for dim in range(n_dim):
        ax = axs[dim]
        for i, (traj_i, cost_i, size_i) in enumerate(zip(rep_traj, best_costs_per_cluster, cluster_sizes), start=1):
            ax.plot(
                time[:traj_i.shape[0]],
                traj_i[:, dim],
                alpha=0.85,
                label=f"cluster{i} cost={cost_i:.3g} n={int(size_i)}",
            )
        ax.set_ylabel(f"{prefix}[{dim}]")
        ax.grid(True, linestyle="--", alpha=0.5)

    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("Time step")
    fig.suptitle(f"Best Trajectory From Each Cluster ({prefix})")
    fig.tight_layout()
    fig.savefig(file_path, format="pdf")


def save_cluster_diversity_analysis(
    dir_path: str,
    samples: Array,
    costs: Array,
    sim,
    *,
    cluster_on: str = "u_traj",
    cluster_threshold: float | None = None,
    max_samples: int = 100,
) -> None:
    metrics, analysis_data = analyze_cluster_diversity(
        samples,
        costs,
        sim,
        cluster_on=cluster_on,
        cluster_threshold=cluster_threshold,
        max_samples=max_samples,
    )

    prefix = f"cluster_diversity_{cluster_on}"
    yaml_path = os.path.join(dir_path, f"{prefix}.yaml")
    cost_plot_path = os.path.join(dir_path, f"{prefix}_costs.pdf")
    u_plot_path = os.path.join(dir_path, f"{prefix}_controls.pdf")
    x_plot_path = os.path.join(dir_path, f"{prefix}_states.pdf")

    _save_yaml(yaml_path, metrics)
    _plot_cluster_costs(
        cost_plot_path,
        analysis_data["best_costs_per_cluster"],
        analysis_data["cluster_sizes"],
    )
    _plot_cluster_representatives(
        u_plot_path,
        np.arange(analysis_data["u_traj"].shape[1]),
        analysis_data["u_traj"],
        analysis_data["best_idx_per_cluster"],
        analysis_data["best_costs_per_cluster"],
        analysis_data["cluster_sizes"],
        prefix="u",
    )
    _plot_cluster_representatives(
        x_plot_path,
        np.arange(analysis_data["x_traj"].shape[1]),
        analysis_data["x_traj"],
        analysis_data["best_idx_per_cluster"],
        analysis_data["best_costs_per_cluster"],
        analysis_data["cluster_sizes"],
        prefix="x",
    )

    print(
        f"Saved cluster diversity analysis to {dir_path} "
        f"({os.path.basename(yaml_path)}, "
        f"{os.path.basename(cost_plot_path)}, "
        f"{os.path.basename(u_plot_path)}, "
        f"{os.path.basename(x_plot_path)})"
    )


def save_every_n_iteration_cluster_diversity_analysis(
    dir_path: str,
    sim,
    all_samples: Array,
    all_costs: Array,
    every_n: int,
    *,
    cluster_on: str = "u_traj",
    cluster_threshold: float | None = None,
    max_samples: int = 100,
) -> None:
    if every_n <= 0:
        return

    iteration_dir = os.path.join(dir_path, "iteration_cluster_diversity")
    os.makedirs(iteration_dir, exist_ok=True)
    print(
        f"Saving cluster diversity analysis every {every_n} iterations to "
        f"{iteration_dir}"
    )

    for iteration, (samples_it, costs_it) in enumerate(zip(all_samples, all_costs)):
        if iteration % every_n != 0:
            continue
        save_cluster_diversity_analysis(
            iteration_dir,
            samples_it,
            costs_it,
            sim,
            cluster_on=cluster_on,
            cluster_threshold=cluster_threshold,
            max_samples=max_samples,
        )
