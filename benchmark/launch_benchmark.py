#!/usr/bin/env -S uv run python
"""
Generic multi-solver benchmark launcher — reads trials from a config file.

Each trial specifies a `solver` field (pcbo or sv_cma_es); defaults to pcbo.

Usage (from repo root):
    uv run python benchmark/launch_benchmark.py benchmark_config/pcbo_search.yaml
    uv run python benchmark/launch_benchmark.py benchmark_config/overnight_sv_vs_pcbo.yaml --resume
    uv run python benchmark/launch_benchmark.py benchmark_config/pcbo_search.yaml --dry-run

The results file is derived from the config filename:
    benchmark_config/pcbo_search.yaml  →  benchmark_rst_json_pointers/pcbo_search.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_file: Path) -> dict:
    with open(config_file) as f:
        return yaml.safe_load(f)


def results_file_for(config_file: Path) -> str:
    return f"benchmark_rst_json_pointers/{config_file.stem}.json"


def _bool(v) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def build_base_overrides(cfg: dict) -> list[str]:
    """Overrides common to all solvers (task, data_processing, N_samples, warm_start geometry)."""
    b = cfg["base"]
    return [
        f"solver.cfg.N_samples={cfg['n_samples']}",
        f"warm_start.start_knots={b['start_knots']}",
        f"warm_start.N_max_incr={b['N_max_incr']}",
        f"task.cfg_ref.motion_path={b['motion_path']}",
        "data_processing.save_video=true",
        "data_processing.save_top=5",
        "data_processing.save_fig=true",
        "data_processing.save_samples_costs=true",
        "data_processing.n_last_it=1",
        f"data_processing.knot_cost_threshold={b['knot_cost_threshold']}",
        f"data_processing.fall_floor_height={b['fall_floor_height']}",
        f"data_processing.fall_drop_threshold={b['fall_drop_threshold']}",
    ]


def build_solver_overrides(trial: dict, b: dict) -> list[str]:
    """Solver-specific hydra overrides derived from the trial config and base settings."""
    solver = trial.get("solver", "pcbo")
    N_it = trial.get("N_it", 100)

    if solver == "pcbo":
        ovr = [
            "solver=pcbo",
            f"solver.cfg.N_it={N_it}",
            f"solver.cfg.min_it_per_knot={b['min_it_per_knot']}",
            f"solver.cfg.load_initial_sampling_state={_bool(b['load_initial_sampling_state'])}",
            f"solver.cfg.adaptive_kappa={_bool(b['adaptive_kappa'])}",
            f"solver.cfg.kappa_subsample={b['kappa_subsample']}",
            f"solver.cfg.mixed_init_frac={b['mixed_init_frac']}",
            f"solver.cfg.scalar_reg_loss_weight_neighborhood_kernel={float(trial['scalar_reg'])}",
            f"solver.cfg.lambda_={trial['lambda_']}",
            f"solver.cfg.flag_auto_weight={_bool(trial['flag_auto_weight'])}",
            f"solver.cfg.ini_cov_scale={trial['ini_cov_scale']}",
            f"solver.cfg.use_loaded_mean_only={_bool(trial['use_loaded_mean_only'])}",
        ]
        if "beta" in trial:
            ovr.append(f"solver.cfg.beta={float(trial['beta']):.6e}")
        return ovr

    elif solver == "sv_cma_es":
        ovr = [
            "solver=sv_cma_es",
            f"solver.cfg.N_it={N_it}",
            f"solver.cfg.num_populations={trial.get('num_populations', 8)}",
            f"solver.cfg.kernel_std={trial.get('kernel_std', 1.0)}",
            f"solver.cfg.alpha={trial.get('alpha', 1.0)}",
            f"solver.cfg.sigma0={trial.get('sigma0', 0.25)}",
        ]
        wr = b.get("warm_start_rundir", "")
        if wr:
            ovr.append(f"warm_start.rundir={wr}")
        return ovr

    else:
        raise ValueError(f"Unknown solver: {solver!r}. Supported: pcbo, sv_cma_es")


def build_trial_list(cfg: dict) -> list[dict]:
    trials = []
    for i, t in enumerate(cfg["trials"]):
        trials.append({**t, "trial_id": str(i)})
    return trials


def _trial_summary(trial: dict) -> str:
    solver = trial.get("solver", "pcbo")
    if solver == "pcbo":
        beta_s = f"  beta={float(trial['beta']):.0e}" if "beta" in trial else ""
        return (
            f"  scalar_reg={float(trial['scalar_reg']):.0e}  lambda={trial['lambda_']}  "
            f"auto_w={trial['flag_auto_weight']}  ini_cov_scale={trial['ini_cov_scale']}  "
            f"mean_only={trial['use_loaded_mean_only']}  N_it={trial['N_it']}{beta_s}"
        )
    elif solver == "sv_cma_es":
        return (
            f"  num_pops={trial.get('num_populations', 8)}  "
            f"kernel_std={trial.get('kernel_std', 1.0)}  "
            f"alpha={trial.get('alpha', 1.0)}  "
            f"sigma0={trial.get('sigma0', 0.25)}  N_it={trial['N_it']}"
        )
    return ""


def run_trial(
    trial: dict,
    base_overrides: list[str],
    b: dict,
    bench_name: str,
    cooldown_s: int,
    dry_run: bool,
) -> dict:
    solver_overrides = build_solver_overrides(trial, b)
    overrides = base_overrides + solver_overrides + [
        f"description={bench_name}_{trial['label']}",
    ]
    cmd = ["uv", "run", "sbto/main.py"] + overrides

    solver = trial.get("solver", "pcbo")
    print(f"\n{'='*70}")
    print(f"Trial {trial['trial_id']}: {trial['label']}  [solver={solver}]")
    print(_trial_summary(trial))
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


def save_results(all_results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {path}")


def run_comparison_plot(all_results: list[dict], bench_name: str) -> None:
    dirs = [r["result_dir"] for r in all_results if r.get("result_dir")]
    if len(dirs) < 2:
        print("\nSkipping comparison plot — fewer than 2 completed trials with result dirs.")
        return

    out_path = str(REPO_ROOT / "benchmark_rst_json_pointers" / f"{bench_name}_comparison.pdf")
    cmd = [
        "uv", "run", "python", "scripts/plot_knot_comparison.py",
        *dirs,
        "--output", out_path,
        "--title", bench_name.replace("_", " "),
    ]
    print(f"\nGenerating comparison plot → {out_path}")
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] Comparison plot failed (returncode={e.returncode})")
    except Exception as e:
        print(f"  [WARN] Comparison plot error: {e}")


def print_summary(all_results: list[dict]) -> None:
    completed = [r for r in all_results if r.get("best_cost") is not None]
    if not completed:
        print("No completed trials.")
        return
    print(f"\n{'='*70}")
    print(f"SUMMARY  ({len(completed)}/{len(all_results)} done)")
    print(f"{'='*70}")
    hdr = (f"{'ID':>3}  {'solver':<10}  {'label':<28}  {'rc':>3}  {'n_clust':>7}  "
           f"{'best_cost':>12}  {'<1000?':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(completed, key=lambda x: x["best_cost"] or 1e9):
        flag = "YES" if (r["best_cost"] or 1e9) < 1000 else "---"
        solver = r["params"].get("solver", "pcbo")
        print(f"{r['trial_id']:>3}  {solver:<10}  {r['params']['label']:<28}  "
              f"{r.get('returncode','?'):>3}  {r['n_clusters']:>7}  "
              f"{r['best_cost']:>12.1f}  {flag:>6}")


def main():
    parser = argparse.ArgumentParser(description="Launch benchmark from a config file.")
    parser.add_argument("config", help="Path to benchmark config YAML")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    if not config_file.exists():
        print(f"Error: config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_file)
    base_overrides = build_base_overrides(cfg)
    b = cfg["base"]
    bench_name = config_file.stem
    trials = build_trial_list(cfg)
    cooldown_s = cfg.get("cooldown_s", 120)
    results_file = results_file_for(config_file)

    print(f"Config:  {config_file}")
    print(f"Results: {results_file}")
    print(f"{len(trials)} trials")

    existing = load_existing(results_file) if args.resume else {}
    all_results = list(existing.values())

    try:
        for i, trial in enumerate(trials):
            tid = trial["trial_id"]
            if args.resume and tid in existing:
                print(f"Skipping trial {tid} ({trial['label']}) — already done")
                continue

            record = run_trial(trial, base_overrides, b, bench_name, cooldown_s, args.dry_run)
            all_results.append(record)
            if not args.dry_run:
                save_results(all_results, results_file)

            if not args.dry_run and i < len(trials) - 1:
                print(f"\nCooling down {cooldown_s}s …")
                time.sleep(cooldown_s)

    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results.")
        if not args.dry_run:
            save_results(all_results, results_file)
        print_summary(all_results)
        return

    print_summary(all_results)
    if not args.dry_run:
        run_comparison_plot(all_results, bench_name)


if __name__ == "__main__":
    main()
