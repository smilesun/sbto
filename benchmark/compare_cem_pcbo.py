"""
Fair comparison: CEM vs PCBO starting from the same initial particle population.

Population is generated once using WarmStartFullCovInit (CEM mean + non-diagonal
CEM covariance × cov_scale) and saved to disk. Both solvers then load that exact
population as their first-iteration particles, so any diversity difference is
purely algorithmic.

Settings match hpsearch trial 8 (best cost+diversity tradeoff):
  scalar_reg=1e6, lambda=10, auto_w=false, N_samples=2048, cov_scale=10

Run from repo root:
    uv run python benchmark/compare_cem_pcbo.py [--dry-run]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
INI_DIST_PATH = "outputs/G1RobotObjRef/2026_04_21__09_07_04/solver_state_final.npz"
SHARED_POP_PATH = "datasets/shared_init_pop_trial8.npz"

N_SAMPLES = 2048
COV_SCALE = 10.0
SEED = 0


def generate_shared_population(dry_run: bool = False) -> None:
    """Generate and save the shared initial population."""
    print(f"\nGenerating shared population (N={N_SAMPLES}, cov_scale={COV_SCALE}x) …")
    print(f"  ini_dist: {INI_DIST_PATH}")
    print(f"  save to: {SHARED_POP_PATH}")
    if dry_run:
        print("  [dry-run] skipping")
        return

    sys.path.insert(0, str(REPO_ROOT))
    from sbto.solvers.population_init import WarmStartFullCovInit, generate_population
    from sbto.solvers.initial_sampling import load_mean_cov_from_solver_state

    mean, _ = load_mean_cov_from_solver_state(str(REPO_ROOT / INI_DIST_PATH))
    D = mean.shape[0]

    strategy = WarmStartFullCovInit(
        ini_dist_path=str(REPO_ROOT / INI_DIST_PATH),
        cov_scale=COV_SCALE,
        seed=SEED,
    )
    pop = generate_population(strategy, N=N_SAMPLES, D=D,
                              save_path=str(REPO_ROOT / SHARED_POP_PATH))
    print(f"  Saved population shape: {pop.shape}  mean_norm={np.linalg.norm(pop.mean(0)):.3f}")


COMMON_OVERRIDES = [
    f"solver.cfg.N_samples={N_SAMPLES}",
    "warm_start.start_knots=15",
    "warm_start.N_max_incr=100",
    f"solver.cfg.population_init_path={SHARED_POP_PATH}",
    "task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz",
    "data_processing.save_video=true",
    "data_processing.save_top=5",
    "data_processing.save_fig=true",
    "data_processing.save_samples_costs=false",
    "data_processing.knot_cost_threshold=1000",
    "data_processing.fall_floor_height=0.40",
    "data_processing.fall_drop_threshold=0.25",
]

RUNS = [
    {
        "description": "cem_shared_init",
        "overrides": [
            "solver=cem",
        ],
    },
    {
        "description": "pcbo_shared_init_trial8",
        "overrides": [
            "solver=pcbo",
            "solver.cfg.min_it_per_knot=10",
            "solver.cfg.adaptive_kappa=true",
            "solver.cfg.kappa_subsample=128",
            "solver.cfg.scalar_reg_loss_weight_neighborhood_kernel=1000000.0",
            "solver.cfg.lambda_=10.0",
            "solver.cfg.flag_auto_weight=false",
            "solver.cfg.mixed_init_frac=0.5",
            "solver.cfg.use_loaded_mean_only=false",
            f"solver.cfg.ini_cov_scale={COV_SCALE}",
            "solver.cfg.load_initial_sampling_state=true",
        ],
    },
]


def run_experiment(run: dict, dry_run: bool = False) -> str | None:
    desc = run["description"]
    overrides = COMMON_OVERRIDES + run["overrides"] + [f"description={desc}"]
    cmd = ["uv", "run", "sbto/main.py"] + overrides

    print(f"\n{'='*70}")
    print(f"Run: {desc}")
    print("CMD:", " ".join(cmd))
    print(f"{'='*70}")

    if dry_run:
        return None

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=False)
    if result.returncode != 0:
        print(f"  [FAILED] returncode={result.returncode}")
    return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    generate_shared_population(dry_run=args.dry_run)

    for run in RUNS:
        run_experiment(run, dry_run=args.dry_run)

    print("\nBoth runs complete. Compare knot_distribution.pdf in each result dir.")


if __name__ == "__main__":
    main()
