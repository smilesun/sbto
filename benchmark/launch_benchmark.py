#!/usr/bin/env -S uv run python
"""
Generic multi-solver benchmark launcher — reads trials from a config file.

Each trial specifies a `solver` field (pcbo, cem, or sv_cma_es); defaults to pcbo.

Usage (from repo root):
    uv run python benchmark/launch_benchmark.py benchmark_recipes/pcbo_search.yaml
    uv run python benchmark/launch_benchmark.py benchmark_recipes/overnight_sv_vs_pcbo.yaml --resume
    uv run python benchmark/launch_benchmark.py benchmark_recipes/pcbo_search.yaml --dry-run

The results file is derived from the config filename:
    benchmark_recipes/pcbo_search.yaml  →  benchmark_rst_json_pointers/pcbo_search.json
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

from common_config import apply_common_policies, load_yaml_with_common_configs

REPO_ROOT = Path(__file__).resolve().parent.parent
WARM_START_INI = REPO_ROOT / "benchmark_volatile_data" / "warm_start_ini.yaml"


def _read_warm_start_ini_dist_path() -> str:
    if not WARM_START_INI.exists():
        return ""
    with open(WARM_START_INI) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("ini_dist_path", "")


def load_config(config_file: Path) -> dict:
    cfg = load_yaml_with_common_configs(config_file)
    return apply_common_policies(cfg)


def results_file_for(config_file: Path) -> str:
    return f"benchmark_rst_json_pointers/{config_file.stem}.json"


def _bool(v) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def _trial_n_samples(cfg: dict, trial: dict | None = None) -> int:
    if trial is None:
        return int(cfg["n_samples"])
    return int(trial.get("n_samples", cfg["n_samples"]))


def build_base_overrides(cfg: dict, trial: dict | None = None) -> list[str]:
    """Overrides common to all solvers (task, data_processing, N_samples, warm_start geometry)."""
    b = cfg["base"]
    return [
        f"solver.cfg.N_samples={_trial_n_samples(cfg, trial)}",
        f"warm_start.start_knots={b['start_knots']}",
        f"warm_start.N_max_incr={b['N_max_incr']}",
        f"task.cfg_ref.motion_path={b['motion_path']}",
        f"data_processing.save_video={_bool(b.get('save_video', True))}",
        f"data_processing.save_top={b.get('save_top', 5)}",
        f"data_processing.save_fig={_bool(b.get('save_fig', True))}",
        f"data_processing.save_samples_costs={_bool(b.get('save_samples_costs', True))}",
        f"data_processing.n_last_it={b.get('n_last_it', 1)}",
        f"data_processing.knot_cost_threshold={b['knot_cost_threshold']}",
        f"data_processing.fall_floor_height={b['fall_floor_height']}",
        f"data_processing.fall_drop_threshold={b['fall_drop_threshold']}",
    ]


def _resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _population_state_path(cfg: dict, trial: dict | None = None) -> Path:
    base = cfg["base"]
    state_path = None
    if trial is not None:
        state_path = _resolve_repo_path(trial.get("population_init_solver_state"))
    if state_path is None:
        state_path = _resolve_repo_path(base.get("population_init_solver_state"))
    if state_path is None:
        state_path = _resolve_repo_path(base.get("population_dim_solver_state"))
    if state_path is None:
        warm_start_rundir = _resolve_repo_path(base.get("warm_start_rundir"))
        if warm_start_rundir is not None:
            state_path = warm_start_rundir / "solver_state_final.npz"
    if state_path is None or not state_path.exists():
        raise FileNotFoundError(
            "Need population_init_solver_state, population_dim_solver_state, or "
            "warm_start_rundir/solver_state_final.npz to generate saved populations."
        )
    return state_path


def _load_population_dim(cfg: dict, trial: dict | None = None) -> int:
    state_path = _population_state_path(cfg, trial)

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from sbto.solvers.initial_sampling import load_mean_cov_from_solver_state

    mean, _ = load_mean_cov_from_solver_state(str(state_path))
    return int(mean.shape[0])


def _fmt_path_token(value: object) -> str:
    text = str(value)
    return text.replace(".", "p").replace("-", "m")


def _population_output_path(cfg: dict, trial: dict, bench_name: str) -> Path:
    explicit = _resolve_repo_path(trial.get("population_init_output"))
    if explicit is not None:
        return explicit

    pop_dir = _resolve_repo_path(
        cfg["base"].get("population_init_dir", "benchmark_volatile_data/generated_populations")
    )
    assert pop_dir is not None

    kind = trial["population_init"]
    n_samples = _trial_n_samples(cfg, trial)
    key = trial.get("population_init_key")
    if key:
        stem = f"{bench_name}_{key}"
    elif kind == "wishart":
        stem = (
            f"{bench_name}_{kind}"
            f"_N{n_samples}"
            f"_K{int(trial['wishart_n_components'])}"
            f"_S{_fmt_path_token(float(trial['wishart_cov_scale']))}"
            f"_seed{int(trial.get('wishart_seed', cfg['base'].get('wishart_seed', 0)))}"
        )
    elif kind == "warm_start_full_cov":
        stem = (
            f"{bench_name}_{kind}"
            f"_N{n_samples}"
            f"_S{_fmt_path_token(float(trial.get('population_init_cov_scale', 1.0)))}"
            f"_seed{int(trial.get('population_init_seed', 0))}"
        )
    else:
        raise ValueError(
            f"Unsupported population_init {kind!r}. "
            "Supported: wishart, warm_start_full_cov."
        )
    return pop_dir / f"{stem}.npz"


def _ensure_population_init(cfg: dict, trial: dict, bench_name: str, dry_run: bool) -> str:
    kind = trial.get("population_init")
    if not kind:
        path = _resolve_repo_path(trial.get("population_init_path"))
        return str(path) if path is not None else ""

    pop_path = _population_output_path(cfg, trial, bench_name)
    if dry_run or pop_path.exists():
        return str(pop_path)

    D = _load_population_dim(cfg, trial)

    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from sbto.solvers.population_init import WarmStartFullCovInit, generate_population
    from sbto.solvers.wishart_population_init import RandomWishartMixInit

    if kind == "wishart":
        sigma0 = float(trial.get("wishart_sigma0", cfg["base"].get("wishart_sigma0", 0.25)))
        cov_scale = float(trial["wishart_cov_scale"])
        n_components = int(trial["wishart_n_components"])
        seed = int(trial.get("wishart_seed", cfg["base"].get("wishart_seed", 0)))
        strategy = RandomWishartMixInit(
            sigma0=sigma0,
            cov_scale=cov_scale,
            n_components=n_components,
            seed=seed,
        )
        detail = f"sigma0={sigma0}, cov_scale={cov_scale}, K={n_components}, seed={seed}"
    elif kind == "warm_start_full_cov":
        state_path = _population_state_path(cfg, trial)
        cov_scale = float(trial.get("population_init_cov_scale", 1.0))
        seed = int(trial.get("population_init_seed", 0))
        strategy = WarmStartFullCovInit(
            ini_dist_path=str(state_path),
            cov_scale=cov_scale,
            seed=seed,
        )
        detail = f"ini_dist_path={state_path}, cov_scale={cov_scale}, seed={seed}"
    else:
        raise ValueError(f"Unsupported population_init {kind!r}")

    pop_path.parent.mkdir(parents=True, exist_ok=True)
    pop = generate_population(strategy, N=_trial_n_samples(cfg, trial), D=D, save_path=str(pop_path))
    print(f"Generated {kind} population {pop.shape} -> {pop_path}  ({detail})")
    return str(pop_path)


def build_solver_overrides(cfg: dict, trial: dict, b: dict, bench_name: str, dry_run: bool) -> list[str]:
    """Solver-specific hydra overrides derived from the trial config and base settings."""
    solver = trial.get("solver", "pcbo")
    N_it = trial.get("N_it", 100)
    pop_path = _ensure_population_init(cfg, trial, bench_name, dry_run)

    if solver == "pcbo":
        ovr = [
            "solver=pcbo",
            f"solver.cfg.N_it={N_it}",
            f"solver.cfg.min_it_per_knot={b['min_it_per_knot']}",
            f"solver.cfg.load_initial_sampling_state={_bool(b['load_initial_sampling_state'])}",
            f"solver.cfg.adaptive_kappa={_bool(b['adaptive_kappa'])}",
            f"solver.cfg.kappa_subsample={b['kappa_subsample']}",
            f"solver.cfg.mixed_init_frac={trial.get('mixed_init_frac', b['mixed_init_frac'])}",
            f"solver.cfg.scalar_reg_loss_weight_neighborhood_kernel={float(trial['scalar_reg'])}",
            f"solver.cfg.lambda_={trial['lambda_']}",
            f"solver.cfg.flag_auto_weight={_bool(trial['flag_auto_weight'])}",
            f"solver.cfg.ini_cov_scale={trial['ini_cov_scale']}",
            f"solver.cfg.use_loaded_mean_only={_bool(trial['use_loaded_mean_only'])}",
        ]
        ini_path = _read_warm_start_ini_dist_path()
        if ini_path:
            ovr.append(f"solver.cfg.ini_dist_path={ini_path}")
        if "beta" in trial:
            ovr.append(f"solver.cfg.beta={float(trial['beta']):.6e}")
        if "sigma0" in trial:
            ovr.append(f"solver.cfg.sigma0={trial['sigma0']}")
        if pop_path:
            ovr.append(f"solver.cfg.population_init_path={pop_path}")
        return ovr

    elif solver == "cem":
        ovr = [
            "solver=cem",
            f"solver.cfg.N_it={N_it}",
        ]
        for key in [
            "elite_frac",
            "keep_frac",
            "min_std_collapsed",
            "alpha_mean",
            "alpha_cov",
            "std_incr",
            "sigma0",
        ]:
            value = trial.get(key, b.get(key))
            if value is not None:
                ovr.append(f"solver.cfg.{key}={value}")
        if pop_path:
            ovr.append(f"solver.cfg.population_init_path={pop_path}")
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
        if pop_path:
            ovr.append(f"solver.cfg.population_init_path={pop_path}")
        return ovr

    else:
        raise ValueError(f"Unknown solver: {solver!r}. Supported: pcbo, cem, sv_cma_es")


def build_trial_list(cfg: dict) -> list[dict]:
    trials = []
    for i, t in enumerate(cfg["trials"]):
        trials.append({**t, "trial_id": str(i)})
    return trials


def _trial_summary(trial: dict) -> str:
    solver = trial.get("solver", "pcbo")
    if solver == "pcbo":
        beta_s = f"  beta={float(trial['beta']):.0e}" if "beta" in trial else ""
        ns_s = f"  N_samples={trial['n_samples']}" if "n_samples" in trial else ""
        init_s = ""
        if trial.get("population_init") == "wishart":
            init_s = (
                f"  wishart(K={trial['wishart_n_components']},"
                f" scale={trial['wishart_cov_scale']})"
            )
        return (
            f"  scalar_reg={float(trial['scalar_reg']):.0e}  lambda={trial['lambda_']}  "
            f"auto_w={trial['flag_auto_weight']}  ini_cov_scale={trial['ini_cov_scale']}  "
            f"mean_only={trial['use_loaded_mean_only']}  N_it={trial['N_it']}{ns_s}{beta_s}{init_s}"
        )
    elif solver == "cem":
        ns_s = f"  N_samples={trial['n_samples']}" if "n_samples" in trial else ""
        init_s = ""
        if trial.get("population_init") == "wishart":
            init_s = (
                f"  wishart(K={trial['wishart_n_components']},"
                f" scale={trial['wishart_cov_scale']})"
            )
        elif trial.get("population_init") == "warm_start_full_cov":
            init_s = f"  shared_pop(scale={trial.get('population_init_cov_scale', 1.0)})"
        elif trial.get("population_init_path"):
            init_s = "  saved_pop"
        return f"  N_it={trial['N_it']}{ns_s}{init_s}"
    elif solver == "sv_cma_es":
        return (
            f"  num_pops={trial.get('num_populations', 8)}  "
            f"kernel_std={trial.get('kernel_std', 1.0)}  "
            f"alpha={trial.get('alpha', 1.0)}  "
            f"sigma0={trial.get('sigma0', 0.25)}  N_it={trial['N_it']}"
        )
    return ""


def run_trial(
    cfg: dict,
    trial: dict,
    b: dict,
    bench_name: str,
    cooldown_s: int,
    dry_run: bool,
) -> dict:
    base_overrides = build_base_overrides(cfg, trial)
    solver_overrides = build_solver_overrides(cfg, trial, b, bench_name, dry_run)
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
            "stdout_tail": stdout[-4000:],
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
            "stdout_tail": "",
        }

    print(f"  → n_clusters={record['n_clusters']}, best_cost={record['best_cost']}, "
          f"elapsed={record['elapsed_s']:.0f}s")
    return record


def is_memory_failure(record: dict) -> bool:
    text_parts = [
        str(record.get("error", "")),
        str(record.get("stdout_tail", "")),
    ]
    text = "\n".join(text_parts).lower()
    patterns = [
        "out of memory",
        "cuda out of memory",
        "resourceexhaustederror",
        "memoryerror",
        "std::bad_alloc",
        "failed to allocate",
        "oom",
        "cublas_status_alloc_failed",
        "xla runtime error",
    ]
    return any(p in text for p in patterns)


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


def _ensure_warm_start(dry_run: bool = False) -> None:
    script = REPO_ROOT / "scripts" / "ensure_warm_start.py"
    cmd = ["uv", "run", "python", str(script)]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print("ensure_warm_start.py failed — aborting.", file=sys.stderr)
        sys.exit(result.returncode)


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
    b = cfg["base"]
    bench_name = config_file.stem
    trials = build_trial_list(cfg)
    cooldown_s = cfg.get("cooldown_s", 120)
    stop_on_memory_failure = bool(cfg.get("stop_on_memory_failure", False))
    results_file = cfg.get("results_file", results_file_for(config_file))

    print(f"Config:  {config_file}")
    print(f"Results: {results_file}")
    print(f"{len(trials)} trials")

    _ensure_warm_start(dry_run=args.dry_run)

    existing = load_existing(results_file) if args.resume else {}
    all_results = list(existing.values())

    try:
        for i, trial in enumerate(trials):
            tid = trial["trial_id"]
            if args.resume and tid in existing:
                print(f"Skipping trial {tid} ({trial['label']}) — already done")
                continue

            record = run_trial(cfg, trial, b, bench_name, cooldown_s, args.dry_run)
            all_results.append(record)
            if not args.dry_run:
                save_results(all_results, results_file)

            if (
                not args.dry_run
                and stop_on_memory_failure
                and record.get("returncode", 0) != 0
                and is_memory_failure(record)
            ):
                print("\nStopping early after memory-capacity failure.")
                break

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
