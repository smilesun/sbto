# Benchmark Guide

All commands are run from the **repository root** (`/home/sunxd/sbto`).

---

## Quick start — launch any benchmark

```bash
# Fresh run
bash benchmark/launch.sh benchmark_config/pcbo_search.yaml

# Resume — skip trials that already have results
bash benchmark/launch.sh benchmark_config/pcbo_search.yaml --resume

# Overnight SV-CMA-ES vs PCBO search
bash benchmark/launch.sh benchmark_config/overnight_sv_vs_pcbo.yaml
```

Or run the Python script directly (output goes to terminal instead of log file):

```bash
uv run python benchmark/launch_benchmark.py benchmark_config/pcbo_search.yaml --resume
```

Results are saved incrementally to `benchmark_rst_json_pointers/pcbo_search.json`
as each trial finishes.

---

## Check benchmark status

```bash
bash benchmark/status.sh
```

This prints a table for every benchmark (completed trials, cost, cluster count)
and lists any currently running benchmark processes plus the last line of each
log file.

---

## Edit the PCBO search configuration

Open **`benchmark_config/pcbo_search.yaml`** and edit the `trials` list.

### Common settings (top of the file)

| Key | Default | What it does |
|-----|---------|--------------|
| `n_samples` | 2048 | Particles drawn per CBO iteration |
| `cooldown_s` | 120 | Seconds to wait between trials |
| `base.N_max_incr` | 100 | Max additional knots grown from warm-start |
| `base.knot_cost_threshold` | 1000 | Only plot trajectories with cost below this |

### Per-trial knobs

| Field | Effect |
|-------|--------|
| `label` | Unique name — used in the result directory and JSON output |
| `scalar_reg` | Polarization strength. **Key knob.** Try `1e5`, `1e6`, `1e7`. Higher → stronger multi-modal spread. |
| `lambda_` | CBO step size. `10.0` = fast convergence, `1.0` = slower/smoother. |
| `flag_auto_weight` | `true` = auto-scale polarization; `false` = use `scalar_reg` as-is. |
| `ini_cov_scale` | Scale applied to the loaded CEM covariance for initial sampling spread. `10` = 10× broader. |
| `use_loaded_mean_only` | `true` = use only the CEM mean (fresh identity covariance); `false` = also load the CEM covariance. |
| `N_it` | CBO iterations per knot batch. `200` gives more refinement at 2× cost. |

**Example — add a new trial:**

```yaml
trials:
  ...
  - label:               my_new_trial
    scalar_reg:          5.0e6
    lambda_:             10.0
    flag_auto_weight:    false
    ini_cov_scale:       10.0
    use_loaded_mean_only: false
    N_it:                150
```

After saving the file, run:

```bash
bash benchmark/launch_pcbo_search.sh --resume
```

`--resume` skips already-completed trials, so only `my_new_trial` will run.

---

## File locations

| Path | Contents |
|------|----------|
| `benchmark_config/pcbo_search.yaml` | Editable trial configuration |
| `benchmark_rst_json_pointers/pcbo_search.json` | JSON results (written after each trial) |
| `benchmark_logs/pcbo_search.log` | Full stdout/stderr from the running benchmark |
| `outputs/G1RobotObjRef/<timestamp>_pcbo_search_<label>/` | Per-trial output: videos, plots, solver state |

---

## Archived benchmarks (already completed, results on disk)

These ran once and are kept for reference. Results are in `benchmark_rst_json_pointers/`.

| Script | What it tested | Results file |
|--------|---------------|--------------|
| `benchmark/hpsearch_pcbo.py` | PCBO hyperparameter grid (11 trials) | `hpsearch_pcbo_results.json` |
| `benchmark/benchmark_sv_vs_pcbo.py` | SV-CMA-ES vs PCBO with CEM warm-start | `benchmark_sv_vs_pcbo.json` |
| `benchmark/benchmark_wishart_init.py` | PCBO with zero-mean Wishart init (no warm-start) | `benchmark_wishart_init.json` |
| `benchmark/benchmark_wishart_cem.py` | CEM with zero-mean Wishart init (no warm-start) | `benchmark_wishart_cem.json` |

To re-run any of them: `uv run python <script> --resume`

---

## Understanding the results

Each entry in the JSON has:

- `best_cost` — lowest trajectory cost found across all diversity clusters. **Below 1000 = robot succeeded.** ~316 = excellent.
- `n_clusters` — number of distinct trajectory modes found. Higher = more diverse solutions.
- `returncode` — exit code of `sbto/main.py`. Both `0` and `1` are normal; `-2` means a Python exception.

The per-trial result directory contains:
- `cluster_diversity_u_traj.yaml` — cluster summary with per-cluster best costs
- `knot_distribution.pdf` — scatter of control knots for trajectories with cost < threshold
- `best_trajectory.npz` + videos — top-5 trajectories
- `cost_over_iterations.pdf` — convergence curve
