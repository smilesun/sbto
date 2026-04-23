"""
Benchmark: PCBO vs SV-CMA-ES with CEM warm-start.

Both solvers start from the mean of the pre-trained CEM solver state
(outputs/G1RobotObjRef/2026_04_21__09_07_04/).  The warm-start sets
solver.state.mean = CEM mean; SVCMAESSolver detects this on the first
get_samples() call and initialises all num_populations evosax CMA-ES means
at that point.

PCBO settings match hpsearch trial 8 (best cost+diversity):
  scalar_reg=1e6, lambda=10, auto_w=false

SV-CMA-ES grid:
  num_populations : 4, 8
  kernel_std      : 1.0, 10.0
  alpha           : 1.0

Run from repo root:
    uv run python benchmark/benchmark_sv_vs_pcbo.py [--dry-run] [--resume]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = "benchmark_rst_json_pointers/benchmark_sv_vs_pcbo.json"

CEM_RUNDIR = str(REPO_ROOT / "outputs/G1RobotObjRef/2026_04_21__09_07_04")
N_SAMPLES = 2048
COOLDOWN_S = 120

COMMON_OVERRIDES = [
    f"solver.cfg.N_samples={N_SAMPLES}",
    "warm_start.rundir=" + CEM_RUNDIR,
    "warm_start.start_knots=15",
    "warm_start.N_max_incr=100",
    "task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz",
    "data_processing.save_video=true",
    "data_processing.save_top=5",
    "data_processing.save_fig=true",
    "data_processing.save_samples_costs=false",
    "data_processing.knot_cost_threshold=1000",
    "data_processing.fall_floor_height=0.40",
    "data_processing.fall_drop_threshold=0.25",
]

TRIALS = [
    # ---- PCBO reference (trial 8 settings) ----
    # Uses PCBO's internal warm-start loader (load_initial_sampling_state=true)
    # which reads ini_dist_path from pcbo.yaml and applies ini_cov_scale=10.
    # warm_start.rundir is set in COMMON_OVERRIDES but PCBO's internal load
    # overrides the covariance, so the collapsed CEM cov is not used.
    {
        "label": "pcbo_trial8",
        "overrides": [
            "solver=pcbo",
            "solver.cfg.min_it_per_knot=10",
            "solver.cfg.load_initial_sampling_state=true",
            "solver.cfg.use_loaded_mean_only=false",
            "solver.cfg.ini_cov_scale=10.0",
            "solver.cfg.adaptive_kappa=true",
            "solver.cfg.kappa_subsample=128",
            "solver.cfg.scalar_reg_loss_weight_neighborhood_kernel=1000000.0",
            "solver.cfg.lambda_=10.0",
            "solver.cfg.flag_auto_weight=false",
            "solver.cfg.mixed_init_frac=0.5",
        ],
    },
    # ---- SV-CMA-ES grid ----
    # warm_start.rundir sets solver.state.mean = CEM mean.
    # SVCMAESSolver._apply_warm_start() initialises all CMA-ES population
    # means at that point; evosax CMA-ES uses its own std_init=0.25.
    {
        "label": "sv_cma_k4_kstd1",
        "overrides": [
            "solver=sv_cma_es",
            "solver.cfg.num_populations=4",
            "solver.cfg.kernel_std=1.0",
            "solver.cfg.alpha=1.0",
            "solver.cfg.sigma0=0.25",
        ],
    },
    {
        "label": "sv_cma_k8_kstd1",
        "overrides": [
            "solver=sv_cma_es",
            "solver.cfg.num_populations=8",
            "solver.cfg.kernel_std=1.0",
            "solver.cfg.alpha=1.0",
            "solver.cfg.sigma0=0.25",
        ],
    },
    {
        "label": "sv_cma_k4_kstd10",
        "overrides": [
            "solver=sv_cma_es",
            "solver.cfg.num_populations=4",
            "solver.cfg.kernel_std=10.0",
            "solver.cfg.alpha=1.0",
            "solver.cfg.sigma0=0.25",
        ],
    },
    {
        "label": "sv_cma_k8_kstd10",
        "overrides": [
            "solver=sv_cma_es",
            "solver.cfg.num_populations=8",
            "solver.cfg.kernel_std=10.0",
            "solver.cfg.alpha=1.0",
            "solver.cfg.sigma0=0.25",
        ],
    },
]
for i, t in enumerate(TRIALS):
    t["trial_id"] = str(i)


def run_trial(trial: dict, dry_run: bool) -> dict:
    overrides = COMMON_OVERRIDES + trial["overrides"] + [
        f"description=sv_vs_pcbo_{trial['label']}",
    ]
    cmd = ["uv", "run", "sbto/main.py"] + overrides

    print(f"\n{'='*70}")
    print(f"Trial {trial['trial_id']}: {trial['label']}")
    print("CMD:", " ".join(cmd))
    print(f"{'='*70}")

    if dry_run:
        return {"trial_id": trial["trial_id"], "params": trial, "dry_run": True}

    t0 = time.time()
    try:
        lines: list[str] = []
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
    print(f"SV-CMA-ES vs PCBO SUMMARY  ({len(completed)}/{len(all_results)} done)")
    print(f"{'='*70}")
    hdr = f"{'ID':>3}  {'label':<24}  {'n_clust':>8}  {'best_cost':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(completed, key=lambda x: (-(x['n_clusters'] or 0), x['best_cost'] or 1e9)):
        print(f"{r['trial_id']:>3}  {r['params']['label']:<24}  {r['n_clusters']:>8}  "
              f"{str(r['best_cost']):>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"{len(TRIALS)} trials: 1 PCBO reference + {len(TRIALS)-1} SV-CMA-ES variants")
    print(f"Warm start from: {CEM_RUNDIR}")

    existing = load_existing(RESULTS_FILE) if args.resume else {}
    all_results = list(existing.values())

    try:
        for i, trial in enumerate(TRIALS):
            tid = trial["trial_id"]
            if args.resume and tid in existing:
                print(f"Skipping trial {tid} ({trial['label']}) — already done")
                continue

            record = run_trial(trial, args.dry_run)
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
