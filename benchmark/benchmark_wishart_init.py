"""
Benchmark: PCBO with zero-mean Wishart-mixture initialisation (no warm start).

Tests whether a diverse initial population generated via RandomWishartMixInit
(zero mean, identity-based covariance, K Wishart draws) lets PCBO find good
solutions without any warm start — addressing the question of whether the
warm-start mean or the distribution diversity is the key ingredient.

CEM baseline for comparison
---------------------------
CEM default init: mean=0, cov=sigma0²·I (single Gaussian, no mixture).
This is equivalent to RandomWishartMixInit with n_components=1.

PCBO settings (trial 8 — best cost+diversity from hpsearch):
    scalar_reg=1e6, lambda=10, auto_w=false, N_samples=2048

Grid explored
-------------
  n_components : 4, 8
  cov_scale    : 1.0 (=CEM default spread), 4.0, 16.0
  wishart_df   : D+2 (default, max diversity)

Run from repo root:
    uv run python benchmark/benchmark_wishart_init.py [--dry-run] [--resume]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = "benchmark_rst_json_pointers/benchmark_wishart_init.json"
LOG_FILE = "benchmark_logs/benchmark_wishart_init.log"
POP_DIR = "benchmark_volatile_data/wishart_pops"

N_SAMPLES = 2048
SIGMA0 = 0.25
COOLDOWN_S = 120  # shorter cooldown than hpsearch — these runs are faster

TRIALS = [
    # CEM-equivalent baseline: single Gaussian, zero mean, sigma0²·I
    {"n_components": 1, "cov_scale": 1.0,  "label": "baseline_cem_equiv"},
    # Wishart mixtures with CEM-equivalent spread
    {"n_components": 4,  "cov_scale": 1.0,  "label": "wishart_k4_s1"},
    {"n_components": 8,  "cov_scale": 1.0,  "label": "wishart_k8_s1"},
    # Broader initial cloud (4× and 16× sigma0²)
    {"n_components": 4,  "cov_scale": 4.0,  "label": "wishart_k4_s4"},
    {"n_components": 8,  "cov_scale": 4.0,  "label": "wishart_k8_s4"},
    {"n_components": 4,  "cov_scale": 16.0, "label": "wishart_k4_s16"},
    {"n_components": 8,  "cov_scale": 16.0, "label": "wishart_k8_s16"},
]
for i, t in enumerate(TRIALS):
    t["trial_id"] = str(i)

PCBO_OVERRIDES = [
    "solver=pcbo",
    f"solver.cfg.N_samples={N_SAMPLES}",
    "solver.cfg.min_it_per_knot=10",
    "warm_start.start_knots=15",
    "warm_start.N_max_incr=100",
    "solver.cfg.load_initial_sampling_state=false",
    "solver.cfg.adaptive_kappa=true",
    "solver.cfg.kappa_subsample=128",
    "solver.cfg.scalar_reg_loss_weight_neighborhood_kernel=1000000.0",
    "solver.cfg.lambda_=10.0",
    "solver.cfg.flag_auto_weight=false",
    "solver.cfg.mixed_init_frac=0.5",
    "task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz",
    "data_processing.save_video=true",
    "data_processing.save_top=5",
    "data_processing.save_fig=true",
    "data_processing.save_samples_costs=false",
    "data_processing.knot_cost_threshold=1000",
    "data_processing.fall_floor_height=0.40",
    "data_processing.fall_drop_threshold=0.25",
]


def _generate_population(trial: dict, D: int, dry_run: bool) -> str:
    """Generate and save the initial population for this trial, return path."""
    label = trial["label"]
    pop_path = os.path.join(POP_DIR, f"{label}.npz")
    if dry_run:
        return pop_path
    if os.path.exists(pop_path):
        print(f"  Population already exists: {pop_path}")
        return pop_path

    sys.path.insert(0, str(REPO_ROOT))
    from sbto.solvers.wishart_population_init import RandomWishartMixInit
    from sbto.solvers.population_init import generate_population

    strategy = RandomWishartMixInit(
        sigma0=SIGMA0,
        cov_scale=trial["cov_scale"],
        n_components=trial["n_components"],
        seed=0,
    )
    Path(pop_path).parent.mkdir(parents=True, exist_ok=True)
    pop = generate_population(strategy, N=N_SAMPLES, D=D, save_path=pop_path)
    print(f"  Generated population {pop.shape}, "
          f"σ_eff={SIGMA0 * trial['cov_scale']**0.5:.3f}, "
          f"K={trial['n_components']}")
    return pop_path


def _get_D() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    # Load D from a known solver state
    from sbto.solvers.initial_sampling import load_mean_cov_from_solver_state
    mean, _ = load_mean_cov_from_solver_state(
        str(REPO_ROOT / "outputs/G1RobotObjRef/2026_04_21__09_07_04/solver_state_final.npz")
    )
    return mean.shape[0]


def run_trial(trial: dict, D: int, dry_run: bool) -> dict:
    pop_path = _generate_population(trial, D, dry_run)

    overrides = PCBO_OVERRIDES + [
        f"solver.cfg.population_init_path={pop_path}",
        f"description=wishart_{trial['label']}",
    ]
    cmd = ["uv", "run", "sbto/main.py"] + overrides

    print(f"\n{'='*70}")
    print(f"Trial {trial['trial_id']}: {trial['label']}  "
          f"K={trial['n_components']}  cov_scale={trial['cov_scale']}")
    print("CMD:", " ".join(cmd))
    print(f"{'='*70}")

    if dry_run:
        return {"trial_id": trial["trial_id"], "params": trial, "dry_run": True}

    t0 = time.time()
    try:
        lines: list[str] = []
        import re
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert proc.stdout
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
        proc.wait(timeout=7200)
        stdout = "".join(lines)

        result_dir = None
        for line in stdout.splitlines():
            m = re.search(r"Result directory:\s*(.+)", line)
            if m:
                result_dir = m.group(1).strip()

        n_clusters, best_cost = 0, None
        if result_dir:
            import yaml
            yaml_path = os.path.join(result_dir, "cluster_diversity_u_traj.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path) as f:
                    cm = yaml.safe_load(f) or {}
                n_clusters = cm.get("n_clusters", 0)
                costs = cm.get("best_costs_per_cluster", [])
                best_cost = float(min(costs)) if costs else None

        record = {
            "trial_id": trial["trial_id"],
            "params": trial,
            "result_dir": result_dir,
            "n_clusters": n_clusters,
            "best_cost": best_cost,
            "elapsed_s": round(time.time() - t0, 1),
            "returncode": proc.returncode,
        }
    except Exception as e:
        record = {
            "trial_id": trial["trial_id"],
            "params": trial,
            "result_dir": None,
            "n_clusters": 0,
            "best_cost": None,
            "elapsed_s": round(time.time() - t0, 1),
            "returncode": -2,
            "error": str(e),
        }

    print(f"  → n_clusters={record['n_clusters']}, best_cost={record['best_cost']}, "
          f"elapsed={record['elapsed_s']:.0f}s")
    return record


def load_existing(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        results = json.load(f)
    return {r["trial_id"]: r for r in results}


def save_results(all_results: list[dict]) -> None:
    os.makedirs(os.path.dirname(RESULTS_FILE) or ".", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {RESULTS_FILE}")


def print_summary(all_results: list[dict]) -> None:
    completed = [r for r in all_results if r.get("returncode", -1) == 0]
    if not completed:
        print("No completed trials.")
        return
    print(f"\n{'='*70}")
    print(f"SUMMARY  ({len(completed)}/{len(all_results)} done)")
    print(f"{'='*70}")
    hdr = f"{'ID':>3}  {'label':<22}  {'K':>3}  {'scale':>6}  {'n_clust':>8}  {'best_cost':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(completed, key=lambda x: (-(x['n_clusters'] or 0), x['best_cost'] or 1e9)):
        p = r["params"]
        print(f"{r['trial_id']:>3}  {p['label']:<22}  {p['n_components']:>3}  "
              f"{p['cov_scale']:>6.1f}  {r['n_clusters']:>8}  {str(r['best_cost']):>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    D = 435 if args.dry_run else _get_D()
    print(f"D={D}, {len(TRIALS)} trials, cooldown={COOLDOWN_S}s between runs")

    existing = load_existing(RESULTS_FILE) if args.resume else {}
    all_results = list(existing.values())

    try:
        for i, trial in enumerate(TRIALS):
            tid = trial["trial_id"]
            if args.resume and tid in existing:
                print(f"Skipping trial {tid} ({trial['label']}) — already done")
                continue

            record = run_trial(trial, D, args.dry_run)
            all_results.append(record)
            if not args.dry_run:
                save_results(all_results)

            if not args.dry_run and i < len(TRIALS) - 1:
                print(f"\nCooling down {COOLDOWN_S}s …")
                time.sleep(COOLDOWN_S)

    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results.")
        if not args.dry_run:
            save_results(all_results)

    print_summary(all_results)


if __name__ == "__main__":
    main()
