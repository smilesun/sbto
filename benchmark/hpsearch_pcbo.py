"""
Overnight hyperparameter search for PCBO multi-modal trajectory tracking.

Searches over the parameters that most affect polarization / multi-modal behaviour:
  - scalar_reg_loss_weight_neighborhood_kernel
  - lambda_
  - flag_auto_weight
  - mixed_init_frac

Metric: combined score = n_clusters * exp(-best_cost / cost_scale)
        (more clusters AND lower cost is better)

Run from the repo root:
    uv run python benchmark/hpsearch_pcbo.py [--dry-run]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Search grid — 6 focused trials, ordered from most to least promising
# ---------------------------------------------------------------------------

# Hand-picked combinations that span the key axes without blowing up into 40 runs.
# scalar_reg is the dominant knob for polarization; lambda_ and flag_auto_weight
# modulate convergence speed vs. diversity; mixed_init_frac controls initial spread.
TRIALS = [
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "false", "mixed_init_frac": 0.5},
    {"scalar_reg": 1e4, "lambda_": 10.0, "flag_auto_weight": "false", "mixed_init_frac": 0.5},
    {"scalar_reg": 1e8, "lambda_": 10.0, "flag_auto_weight": "false", "mixed_init_frac": 0.5},
    {"scalar_reg": 1e6, "lambda_":  1.0, "flag_auto_weight": "false", "mixed_init_frac": 0.5},
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "true",  "mixed_init_frac": 0.5},
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "false", "mixed_init_frac": 1.0},
    # follow-up: combine the two best multi-modal settings
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "true",  "mixed_init_frac": 1.0},
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "true",  "mixed_init_frac": 0.75},
    # non-diagonal CEM covariance init, scaled up for broad but correlated exploration
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "false", "mixed_init_frac": 0.5,
     "use_loaded_mean_only": "false", "ini_cov_scale": 10.0},
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "false", "mixed_init_frac": 0.5,
     "use_loaded_mean_only": "false", "ini_cov_scale": 100.0},
    {"scalar_reg": 1e6, "lambda_": 10.0, "flag_auto_weight": "true",  "mixed_init_frac": 0.5,
     "use_loaded_mean_only": "false", "ini_cov_scale": 10.0},
]
for i, t in enumerate(TRIALS):
    t["trial_id"] = str(i)

# Seconds to sleep between trials so the GPU can cool down.
COOLDOWN_S = 300

# Base command fragments (non-searched settings taken from run.sh)
BASE_OVERRIDES = [
    "solver=pcbo",
    "solver.cfg.min_it_per_knot=10",
    "warm_start.start_knots=15",
    "warm_start.N_max_incr=100",
    "solver.cfg.N_samples=2048",
    "solver.cfg.adaptive_kappa=true",
    "solver.cfg.kappa_subsample=128",
    "task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz",
    "data_processing.save_video=true",
    "data_processing.save_top=5",
    "data_processing.save_fig=true",
    "data_processing.diversity_plot_every=0",
    "data_processing.save_samples_costs=false",
]

RESULTS_FILE = "benchmark_rst_json_pointers/hpsearch_pcbo_results.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _trial_overrides(params: dict) -> list[str]:
    overrides = [
        f"solver.cfg.scalar_reg_loss_weight_neighborhood_kernel={params['scalar_reg']}",
        f"solver.cfg.lambda_={params['lambda_']}",
        f"solver.cfg.flag_auto_weight={params['flag_auto_weight']}",
        f"solver.cfg.mixed_init_frac={params['mixed_init_frac']}",
        f"description=hpsearch_{params['trial_id']}",
    ]
    if "use_loaded_mean_only" in params:
        overrides.append(f"solver.cfg.use_loaded_mean_only={params['use_loaded_mean_only']}")
    if "ini_cov_scale" in params:
        overrides.append(f"solver.cfg.ini_cov_scale={params['ini_cov_scale']}")
    return overrides


def _parse_result_dir(stdout: str) -> str | None:
    """Extract the result directory from the printed output."""
    for line in stdout.splitlines():
        m = re.search(r"Result directory:\s*(.+)", line)
        if m:
            return m.group(1).strip()
    return None


def _load_cluster_metrics(result_dir: str) -> dict:
    """Read cluster diversity YAML written by save_cluster_diversity_analysis."""
    import yaml
    path = os.path.join(result_dir, "cluster_diversity_u_traj.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_best_cost(result_dir: str) -> float | None:
    """Read best cost from the saved npz trajectory."""
    import numpy as np
    candidates = [
        os.path.join(result_dir, "best_trajectory.npz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            if "cost" in d:
                return float(np.min(d["cost"]))
    # Fall back: parse from cluster metrics
    return None


def _score(n_clusters: int, best_cost: float, cost_scale: float = 1.0) -> float:
    """Combined multi-modal quality score (higher is better)."""
    import math
    if best_cost is None or best_cost <= 0:
        return float(n_clusters)
    return n_clusters * math.exp(-best_cost / cost_scale)


def run_trial(params: dict, dry_run: bool = False) -> dict:
    overrides = BASE_OVERRIDES + _trial_overrides(params)
    cmd = ["uv", "run", "sbto/main.py"] + overrides

    print(f"\n{'='*70}")
    print(f"Trial {params['trial_id']}: {params}")
    print("CMD:", " ".join(cmd))
    print(f"{'='*70}")

    if dry_run:
        return {"trial_id": params["trial_id"], "params": params, "dry_run": True}

    t0 = time.time()
    try:
        # Stream to terminal AND capture for parsing.
        lines: list[str] = []
        proc_obj = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc_obj.stdout is not None
        for line in proc_obj.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
        proc_obj.wait(timeout=7200)
        elapsed = time.time() - t0

        class _Proc:
            returncode = proc_obj.returncode

        proc = _Proc()
        stdout = "".join(lines)

        result_dir = _parse_result_dir(stdout)
        n_clusters = 0
        best_cost = None

        if result_dir:
            cm = _load_cluster_metrics(result_dir)
            n_clusters = cm.get("n_clusters", 0)
            best_cost = _load_best_cost(result_dir)
            if best_cost is None:
                # Try reading from cluster metrics
                costs = cm.get("best_costs_per_cluster", [])
                best_cost = float(min(costs)) if costs else None

        record = {
            "trial_id": params["trial_id"],
            "params": params,
            "result_dir": result_dir,
            "n_clusters": n_clusters,
            "best_cost": best_cost,
            "score": _score(n_clusters, best_cost),
            "elapsed_s": round(elapsed, 1),
            "returncode": proc.returncode,
        }

    except subprocess.TimeoutExpired:
        try:
            proc_obj.kill()
        except Exception:
            pass
        record = {
            "trial_id": params["trial_id"],
            "params": params,
            "result_dir": None,
            "n_clusters": 0,
            "best_cost": None,
            "score": 0.0,
            "elapsed_s": 7200,
            "returncode": -1,
            "error": "timeout",
        }
    except Exception as e:
        record = {
            "trial_id": params["trial_id"],
            "params": params,
            "result_dir": None,
            "n_clusters": 0,
            "best_cost": None,
            "score": 0.0,
            "elapsed_s": time.time() - t0,
            "returncode": -2,
            "error": str(e),
        }

    print(f"  → n_clusters={record['n_clusters']}, best_cost={record['best_cost']}, "
          f"score={record['score']:.4f}, elapsed={record['elapsed_s']:.0f}s")
    return record


def load_existing_results() -> dict[str, dict]:
    if not os.path.exists(RESULTS_FILE):
        return {}
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    return {r["trial_id"]: r for r in results}


def save_results(all_results: list[dict]) -> None:
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


def print_summary(all_results: list[dict]) -> None:
    completed = [r for r in all_results if r.get("returncode", -1) == 0]
    if not completed:
        print("No completed trials yet.")
        return

    best = max(completed, key=lambda r: r["score"])
    print(f"\n{'='*70}")
    print(f"SUMMARY  ({len(completed)}/{len(all_results)} trials completed)")
    print(f"{'='*70}")
    header = f"{'ID':>4}  {'scalar_reg':>12}  {'lambda':>8}  {'auto_w':>6}  "
    header += f"{'mix_frac':>8}  {'n_clust':>8}  {'best_cost':>12}  {'score':>10}"
    print(header)
    print("-" * len(header))
    for r in sorted(completed, key=lambda x: -x["score"]):
        p = r["params"]
        print(
            f"{r['trial_id']:>4}  {p['scalar_reg']:>12.1e}  {p['lambda_']:>8.1f}  "
            f"{str(p['flag_auto_weight']):>6}  {p['mixed_init_frac']:>8.2f}  "
            f"{r['n_clusters']:>8}  {str(r['best_cost']):>12}  {r['score']:>10.4f}"
        )
    print(f"\nBest trial: {best['trial_id']}")
    print(f"  Params: {best['params']}")
    print(f"  n_clusters={best['n_clusters']}, best_cost={best['best_cost']}, "
          f"score={best['score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="PCBO hyperparameter search")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed trials")
    args = parser.parse_args()

    print(f"Total trials: {len(TRIALS)}  |  cooldown between trials: {COOLDOWN_S}s")

    existing = load_existing_results() if args.resume else {}
    all_results = list(existing.values())

    try:
        for i, params in enumerate(TRIALS):
            tid = params["trial_id"]
            if args.resume and tid in existing:
                print(f"Skipping trial {tid} (already completed)")
                continue

            record = run_trial(params, dry_run=args.dry_run)
            all_results.append(record)

            if not args.dry_run:
                save_results(all_results)

            # Cool-down between trials (skip after the last one).
            if not args.dry_run and i < len(TRIALS) - 1:
                print(f"\nCooling down for {COOLDOWN_S}s before next trial …")
                time.sleep(COOLDOWN_S)

    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results.")
        if not args.dry_run:
            save_results(all_results)

    print_summary(all_results)


if __name__ == "__main__":
    main()
