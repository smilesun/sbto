"""
Plot the distribution of raw control knots for trajectories below a cost threshold.

Loads raw knot vectors from `all_samples_costs.npz` (if saved) or falls back to
`best_samples_it.npz`.  Reshapes each sample (D,) → (Nknots, Nu), then plots a
jittered scatter of values at each knot position — NO connecting lines.

Configuration is read from config/plot_knot_config.yaml (cost_threshold, etc.).

Usage:
    uv run python scripts/plot_knot_distribution.py <result_dir> [result_dir2 ...]

    # single trial
    uv run python scripts/plot_knot_distribution.py outputs/G1RobotObjRef/2026_04_22__*__hpsearch_7

    # compare multiple trials
    uv run python scripts/plot_knot_distribution.py outputs/G1RobotObjRef/*hpsearch_{6,7}

    # override threshold at CLI
    uv run python scripts/plot_knot_distribution.py ... --cost-threshold 500
"""

import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_CFG = {
    "cost_threshold": 1000.0,
    "max_joints": 29,
    "jitter_frac": 0.35,
    "point_size": 8,
    "point_alpha": 0.5,
}

def _load_cfg(path: str) -> dict:
    cfg = dict(_DEFAULT_CFG)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_nu(result_dir: str) -> int:
    xml_path = os.path.join(result_dir, "mj_model.xml")
    assets_dir = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "sbto", "models", "unitree_g1", "assets")
    )
    with open(xml_path) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets/"', f'meshdir="{assets_dir}/"')
    return int(mujoco.MjModel.from_xml_string(xml).nu)


def _load_knot_times(result_dir: str, nknots: int) -> np.ndarray:
    cfg_paths = glob.glob(os.path.join(result_dir, ".hydra", "config.yaml"))
    if not cfg_paths:
        return np.arange(nknots)
    with open(cfg_paths[0]) as f:
        cfg = yaml.safe_load(f)
    T = cfg["task"]["sim"]["cfg"]["T"]
    step_knots = cfg["task"]["sim"]["cfg"]["step_knots"]
    nknots_full = T // step_knots
    t_full = np.int32(np.ceil(np.linspace(0, 1, nknots_full, endpoint=True) * T))
    return t_full[:nknots]


def _load_all_samples(result_dir: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (samples, costs_or_None).

    If all_samples_costs.npz is present (save_samples_costs=true), returns the
    last iteration's full population with proper per-sample costs → cost threshold
    filtering works.

    Falls back to best_samples_it.npz (best sample per iteration).  The cost
    array there is per-sample-slot minimum across iterations and does NOT align
    with the stored per-iteration samples, so we return None for costs and skip
    cost filtering.
    """
    all_path = os.path.join(result_dir, "all_samples_costs.npz")
    if os.path.exists(all_path):
        d = np.load(all_path)
        samples = d["samples"]
        costs = d["costs"]
        if samples.ndim == 3:
            samples = samples[-1]
            costs = costs[-1]
        return samples, costs

    bp = os.path.join(result_dir, "best_samples_it.npz")
    if not os.path.exists(bp):
        raise FileNotFoundError(f"No sample data found in {result_dir}")
    d = np.load(bp)
    keys = sorted([k for k in d.files if k != "cost"], key=lambda x: int(x))
    samples = np.stack([d[k] for k in keys])
    # costs in this file are per-sample-slot mins, not per-iteration-best costs
    # → return None so the caller skips cost filtering
    return samples, None


def _parse_dir(result_dir: str, cost_threshold: float):
    """Return (knots, t_knots, Nu, label) keeping only samples below threshold."""
    samples, costs = _load_all_samples(result_dir)

    if costs is not None:
        mask = costs < cost_threshold
        n_before = samples.shape[0]
        samples = samples[mask]
        costs = costs[mask]
        if samples.shape[0] == 0:
            raise ValueError(
                f"No samples below cost_threshold={cost_threshold} "
                f"(min cost among {n_before} samples: "
                f"{np.load(os.path.join(result_dir, 'all_samples_costs.npz'))['costs'][-1].min():.1f})"
            )
        note = f"{samples.shape[0]} samples below cost {cost_threshold} (best={costs.min():.2f})"
    else:
        note = (f"{samples.shape[0]} samples (best-per-iteration; "
                f"cost filtering requires save_samples_costs=true)")

    Nu = _load_nu(result_dir)
    Nknots = samples.shape[1] // Nu
    knots = samples.reshape(-1, Nknots, Nu)
    t_knots = _load_knot_times(result_dir, Nknots)
    label = os.path.basename(result_dir)
    print(f"  {label}: {note}, {Nknots} knots, Nu={Nu}")
    return knots, t_knots, Nu, label


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_knot_distribution(
    result_dirs: list[str],
    cost_threshold: float,
    max_joints: int,
    jitter_frac: float,
    point_size: float,
    point_alpha: float,
    out_path: str | None = None,
) -> None:
    datasets = []
    for d in result_dirs:
        try:
            datasets.append(_parse_dir(d, cost_threshold))
        except Exception as e:
            print(f"Skipping {d}: {e}")

    if not datasets:
        print("No valid result directories.")
        return

    Nu = datasets[0][2]
    n_joints = min(Nu, max_joints)
    fig, axs = plt.subplots(
        n_joints, 1,
        figsize=(12, max(4, 1.8 * n_joints)),
        sharex=False,
    )
    if n_joints == 1:
        axs = [axs]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    rng = np.random.default_rng(0)
    n_datasets = len(datasets)

    for joint in range(n_joints):
        ax = axs[joint]
        for i, (knots, t_knots, _, label) in enumerate(datasets):
            color = colors[i % len(colors)]
            M, Nknots = knots.shape[0], knots.shape[1]
            gaps = np.diff(t_knots)
            jitter_width = float(gaps.min()) * jitter_frac if gaps.size > 0 else 1.0
            slot_width = jitter_width / max(n_datasets, 1)
            center_offset = (i - (n_datasets - 1) / 2.0) * slot_width

            for ki in range(Nknots):
                jitter = rng.uniform(-jitter_width / 2, jitter_width / 2, size=M)
                ax.scatter(
                    t_knots[ki] + center_offset + jitter,
                    knots[:, ki, joint],
                    color=color,
                    s=point_size,
                    alpha=point_alpha,
                    linewidths=0,
                    label=label if (joint == 0 and ki == 0) else None,
                )
        ax.set_ylabel(f"joint {joint}", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=7)

    axs[0].legend(loc="upper right", fontsize=7)
    axs[-1].set_xlabel("Simulation timestep (knot)")
    fig.suptitle(
        "Control Knot Distribution — jittered scatter, no interpolation\n"
        f"(cost < {cost_threshold} where available)"
    )
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(result_dirs[0], "knot_distribution.pdf")
    fig.savefig(out_path, format="pdf")

    csv_path = os.path.splitext(out_path)[0] + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "sample_rank", "knot_timestep", "joint", "value"])
        for knots, t_knots, _, label in datasets:
            for m in range(knots.shape[0]):
                for ki, t in enumerate(t_knots):
                    for j in range(knots.shape[2]):
                        w.writerow([label, m, int(t), j, float(knots[m, ki, j])])

    print(f"Saved: {out_path}")
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = _load_cfg(os.path.join(repo_root, "config", "plot_knot_config.yaml"))

    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", help="One or more result directories")
    parser.add_argument("--cost-threshold", type=float, default=cfg["cost_threshold"],
                        help=f"Only plot samples with cost < threshold (default: {cfg['cost_threshold']})")
    parser.add_argument("--max-joints", type=int, default=cfg["max_joints"])
    parser.add_argument("--out", default=None, help="Output PDF path")
    args = parser.parse_args()

    dirs = []
    for pattern in args.result_dirs:
        expanded = sorted(glob.glob(pattern))
        dirs.extend(expanded if expanded else [pattern])

    plot_knot_distribution(
        dirs,
        cost_threshold=args.cost_threshold,
        max_joints=args.max_joints,
        jitter_frac=cfg["jitter_frac"],
        point_size=cfg["point_size"],
        point_alpha=cfg["point_alpha"],
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
