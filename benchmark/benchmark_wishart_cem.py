"""
Benchmark: CEM with zero-mean Wishart-mixture initialisation (no warm start).

Companion to benchmark_wishart_init.py (which runs PCBO).
Uses the same pre-generated population files from benchmark_volatile_data/wishart_pops/
so the comparison is perfectly fair — identical starting particles.

Run AFTER benchmark_wishart_init.py has finished (populations must exist):
    uv run python benchmark/benchmark_wishart_cem.py [--dry-run] [--resume]
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
RESULTS_FILE = "benchmark_rst_json_pointers/benchmark_wishart_cem.json"
POP_DIR = "benchmark_volatile_data/wishart_pops"

N_SAMPLES = 2048
COOLDOWN_S = 120

# Same grid as PCBO benchmark — reuses the same population files
TRIALS = [
    {"n_components": 1,  "cov_scale": 1.0,  "label": "baseline_cem_equiv"},
    {"n_components": 4,  "cov_scale": 1.0,  "label": "wishart_k4_s1"},
    {"n_components": 8,  "cov_scale": 1.0,  "label": "wishart_k8_s1"},
    {"n_components": 4,  "cov_scale": 4.0,  "label": "wishart_k4_s4"},
    {"n_components": 8,  "cov_scale": 4.0,  "label": "wishart_k8_s4"},
    {"n_components": 4,  "cov_scale": 16.0, "label": "wishart_k4_s16"},
    {"n_components": 8,  "cov_scale": 16.0, "label": "wishart_k8_s16"},
]
for i, t in enumerate(TRIALS):
    t["trial_id"] = str(i)

CEM_OVERRIDES = [
    "solver=cem",
    f"solver.cfg.N_samples={N_SAMPLES}",
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


def run_trial(trial: dict, dry_run: bool) -> dict:
    pop_path = os.path.join(POP_DIR, f"{trial['label']}.npz")
    if not dry_run and not os.path.exists(pop_path):
        print(f"  [SKIP] Population not found: {pop_path} — run benchmark_wishart_init.py first")
        return {"trial_id": trial["trial_id"], "params": trial, "returncode": -3,
                "error": "population missing"}

    overrides = CEM_OVERRIDES + [
        f"solver.cfg.population_init_path={pop_path}",
        f"description=cem_wishart_{trial['label']}",
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
    print(f"CEM WISHART SUMMARY  ({len(completed)}/{len(all_results)} done)")
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

    print(f"{len(TRIALS)} CEM trials, reusing populations from {POP_DIR}/")

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
