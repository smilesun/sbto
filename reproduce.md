# SBTO Reproduction Guide

Everything runs from the **repository root** (`/home/sunxd/sbto` or wherever you cloned).

---

## 1. Environment setup

SBTO uses [uv](https://github.com/astral-sh/uv) for dependency management — no conda required.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and enter the repo
git clone https://github.com/Atarilab/sbto.git
cd sbto

# Install all dependencies (creates a virtual env automatically)
uv sync
```

All subsequent commands use `uv run <script>` — this automatically activates the environment.

> **GPU note**: JAX is configured for CUDA 12 (`jax[cuda12]`).  
> On CPU-only machines remove that extra and install plain `jax` instead.

---

## 2. Download motion data

```bash
mkdir -p datasets && cd datasets
wget "https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset/resolve/main/robot-object.zip"
unzip robot-object.zip
cd ..
```

The benchmark uses `datasets/robot-object/sub10_largebox_000_original.npz` (box-picking reference).

---

## 3. Verify paths and imports

Run the CI path-check script to confirm all critical files and config paths are correct:

```bash
uv run python ci/check_paths.py
```

All 37 checks should pass before running any benchmark.

---

## 4. Step 1 — CEM warm-start

PCBO and SV-CMA-ES benefit from a CEM warm-start: CEM finds a single good trajectory first, and its mean/covariance seed the multi-modal solver.

```bash
bash run_cem.sh
```

This runs:
```
uv run sbto/main.py solver=cem \
    task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
    task/g1/sim/mj_scene@task.sim.mj_scene=box
```

Output lands in `outputs/G1RobotObjRef/<timestamp>/`.  
The warm-start state is `solver_state_final.npz` inside that directory.

**Update the warm-start path** in `sbto/conf/solver/pcbo.yaml` (field `ini_dist_path`) to point to the new run before launching PCBO.

> CEM typically runs 15–45 minutes on a single GPU.  
> A known-good state is already committed at  
> `outputs/G1RobotObjRef2Delete/2026_04_21__09_07_04/solver_state_final.npz`.

---

## 5. Step 2 — Edit the PCBO trial configuration

Open `benchmark_recipes/pcbo_search.yaml`.

**Common settings** (top of file, apply to all trials):

| Key | Default | Description |
|-----|---------|-------------|
| `n_samples` | 2048 | Particles per CBO iteration |
| `cooldown_s` | 120 | Seconds between trials |
| `base.N_max_incr` | 100 | Max extra knots grown from warm-start |
| `base.knot_cost_threshold` | 1000 | Only plot trajectories below this cost |

**Per-trial knobs** (in the `trials` list):

| Field | Effect |
|-------|--------|
| `label` | Unique name — appears in result dir and JSON |
| `scalar_reg` | Polarization strength. Key knob. `1e6` is a good baseline; try `1e5`–`1e7`. |
| `lambda_` | CBO step size. `10.0` = fast, `1.0` = slower/smoother. |
| `flag_auto_weight` | `true` = auto-scale polarization; `false` = use `scalar_reg` as-is. |
| `ini_cov_scale` | Multiplier on the loaded CEM covariance. `10` = 10× broader initial spread. |
| `use_loaded_mean_only` | `true` = use only CEM mean (identity covariance); `false` = also load CEM covariance. |
| `N_it` | CBO iterations per knot batch. `200` = more refinement, ~2× cost. |

Add a new trial at the bottom of the `trials` list:

```yaml
- label:               my_trial
  scalar_reg:          5.0e6
  lambda_:             10.0
  flag_auto_weight:    false
  ini_cov_scale:       10.0
  use_loaded_mean_only: false
  N_it:                100
```

---

## 6. Step 3 — Launch the PCBO benchmark

### Background (recommended)

```bash
bash benchmark/launch_pcbo_search.sh
```

Logs go to `benchmark_logs/pcbo_search.log`. The process keeps running after you close the terminal.

Resume (skip already-completed trials):

```bash
bash benchmark/launch_pcbo_search.sh --resume
```

### Foreground (output to terminal)

```bash
uv run python benchmark/launch_benchmark.py benchmark_recipes/pcbo_search.yaml
uv run python benchmark/launch_benchmark.py benchmark_recipes/pcbo_search.yaml --resume
```

Dry-run (prints commands without executing):

```bash
uv run python benchmark/launch_benchmark.py benchmark_recipes/pcbo_search.yaml --dry-run
```

---

## 7. Monitor progress

```bash
# Live log stream
tail -f benchmark_logs/pcbo_search.log

# Summary table (completed trials, cost, cluster count)
bash benchmark/status.sh
```

Status output shows: trial ID, label, return code, number of clusters, best cost found.  
**Cost < 1000 = robot succeeded. ~316 = excellent.**

---

## 8. Stop a running benchmark

```bash
bash benchmark/kill_all.sh
```

---

## 9. Results

| Path | Contents |
|------|----------|
| `benchmark_rst_json_pointers/pcbo_search.json` | Per-trial JSON: `best_cost`, `n_clusters`, `result_dir`, `elapsed_s` |
| `outputs/G1RobotObjRef/<timestamp>_pcbo_search_<label>/` | Full trial output |
| `benchmark_logs/pcbo_search.log` | Stdout/stderr from the benchmark process |

Inside each trial output directory:

| File | What it shows |
|------|---------------|
| `cluster_diversity_u_traj.yaml` | Cluster summary: per-cluster best costs, n_clusters |
| `knot_distribution.pdf` | Jittered scatter of control knots (samples with cost < threshold) |
| `knot_distribution.txt` | Written instead of PDF when no samples pass the cost threshold |
| `best_trajectory.npz` + `.mp4` | Top-5 trajectories (video + data) |
| `cost_over_iterations.pdf` | Cost convergence curve over CBO iterations |

---

## 10. Run a single trial manually

To iterate quickly without the benchmark launcher:

```bash
uv run sbto/main.py \
    solver=pcbo \
    solver.cfg.N_samples=2048 \
    solver.cfg.scalar_reg_loss_weight_neighborhood_kernel=1e6 \
    solver.cfg.lambda_=10.0 \
    solver.cfg.flag_auto_weight=false \
    solver.cfg.ini_cov_scale=10.0 \
    solver.cfg.use_loaded_mean_only=false \
    solver.cfg.N_it=100 \
    warm_start.start_knots=15 \
    warm_start.N_max_incr=100 \
    task.cfg_ref.motion_path=datasets/robot-object/sub10_largebox_000_original.npz \
    data_processing.save_video=true \
    data_processing.save_top=5 \
    data_processing.save_fig=true \
    description=my_manual_run
```

---

## 11. Visualize knot distributions across trials

```bash
# Single trial
uv run python scripts/plot_knot_distribution.py outputs/G1RobotObjRef/*my_trial*

# Compare two trials side-by-side
uv run python scripts/plot_knot_distribution.py \
    outputs/G1RobotObjRef/*trial_a* \
    outputs/G1RobotObjRef/*trial_b*

# Override cost threshold
uv run python scripts/plot_knot_distribution.py outputs/G1RobotObjRef/*my_trial* \
    --cost-threshold 500
```

---

## 12. Archived benchmarks

These ran once and results are on disk in `benchmark_rst_json_pointers/`:

| Script | What it tested |
|--------|----------------|
| `benchmark_recipes/hpsearch_pcbo.yaml` | PCBO hyperparameter grid (11 trials) |
| `benchmark/benchmark_sv_vs_pcbo.py` | SV-CMA-ES vs PCBO with CEM warm-start |
| `benchmark/benchmark_wishart_init.py` | PCBO with zero-mean Wishart init (no warm-start) |
| `benchmark_recipes/benchmark_wishart_cem.yaml` | CEM with zero-mean Wishart init (no warm-start) |
| `benchmark_recipes/compare_cem_pcbo.yaml` | CEM vs PCBO from the same saved initial population |

Re-run any config-backed benchmark:
`uv run python benchmark/launch_benchmark.py benchmark_recipes/<config>.yaml --resume`

---

## 13. File layout

```
sbto/
├── benchmark/                  # Benchmark scripts and launchers
│   ├── launch_benchmark.py     # Generic launcher (takes config YAML as arg)
│   ├── launch_pcbo_search.sh   # Background launcher for pcbo_search config
│   ├── status.sh               # Print progress table for all benchmarks
│   ├── kill_all.sh             # Stop all running benchmark processes
│   └── README.md               # Benchmark-specific quick-start guide
├── benchmark_common_config/    # Shared policies applied across benchmarks
├── benchmark_recipes/          # Runnable sequential benchmark recipes
│   └── pcbo_search.yaml        # PCBO hyperparameter search recipe
├── benchmark_logs/             # Log files from background benchmark runs
├── benchmark_rst_json_pointers/ # JSON result files (one per benchmark)
├── benchmark_volatile_data/    # Wishart population cache files
├── ci/
│   └── check_paths.py          # Path and import smoke test (37 checks)
├── config/
│   └── plot_knot_config.yaml   # Defaults for knot distribution plots
├── datasets/                   # OmniRetarget motion data
├── outputs/                    # All solver run outputs (videos, plots, npz)
├── sbto/                       # Main package
│   ├── conf/                   # Hydra config files
│   ├── solvers/                # CEM, PCBO, SV-CMA-ES solver implementations
│   └── main.py                 # Entry point
├── scripts/                    # Analysis and visualization scripts
│   └── plot_knot_distribution.py
├── run_cem.sh                  # Quick CEM warm-start launcher
└── reproduce.md                # This file
```
