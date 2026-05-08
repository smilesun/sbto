"""
Smoke-test: verify all relocated paths and config references are consistent.

Run from repo root:
    uv run python test/check_paths.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = []


def check(label: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail and not ok else ""
    print(f"  [{status}] {label}{suffix}")
    if not ok:
        failures.append(label)


def section(title: str) -> None:
    print(f"\n── {title} {'─' * (60 - len(title))}")


# ── Critical data files ────────────────────────────────────────────────────────
section("Critical data files")

WARM_START = REPO / "outputs/G1RobotObjRef2Delete/2026_04_21__09_07_04/solver_state_final.npz"
check("Warm-start solver state (outputs/G1RobotObjRef/...)", WARM_START.exists())

MOTION = REPO / "datasets/robot-object/sub10_largebox_000_original.npz"
check("Motion reference file (datasets/robot-object/...)", MOTION.exists())

for label in ["baseline_cem_equiv", "wishart_k4_s1", "wishart_k8_s1",
              "wishart_k4_s4", "wishart_k8_s4", "wishart_k4_s16", "wishart_k8_s16"]:
    p = REPO / f"benchmark_volatile_data/wishart_pops/{label}.npz"
    check(f"Wishart pop: {label}.npz", p.exists())

# ── Config file path references ────────────────────────────────────────────────
section("Config file path references")

import yaml

with open(REPO / "sbto/conf/data_processing/default.yaml") as f:
    dp = yaml.safe_load(f)
check("data_processing.data_dir == './outputs'",
      dp.get("data_dir") == "./outputs",
      f"got '{dp.get('data_dir')}'")

WARM_START_INI = REPO / "benchmark_volatile_data" / "warm_start_ini.yaml"
if WARM_START_INI.exists():
    with open(WARM_START_INI) as f:
        ws_cfg = yaml.safe_load(f) or {}
    ini_path = ws_cfg.get("ini_dist_path", "")
    resolved = (REPO / ini_path) if ini_path and not Path(ini_path).is_absolute() else Path(ini_path)
    exists = resolved.exists() if ini_path else False
    check("warm_start_ini.yaml: ini_dist_path exists", exists, ini_path)
    check("warm_start_ini.yaml: ini_dist_path is relative",
          not Path(ini_path).is_absolute() if ini_path else True, ini_path)
    check("warm_start_ini.yaml: ini_dist_path not in datasets/",
          "datasets/G1RobotObjRef" not in ini_path, ini_path)
else:
    check("benchmark_volatile_data/warm_start_ini.yaml present", False,
          "run: uv run python scripts/ensure_warm_start.py")

# ── Benchmark launcher/config paths ───────────────────────────────────────────
section("Benchmark launcher/config paths")

launcher_text = (REPO / "benchmark/launch_benchmark.py").read_text()
check("launch_benchmark.py: writes benchmark_rst_json_pointers/",
      "benchmark_rst_json_pointers/" in launcher_text)
check("launch_benchmark.py: supports wishart population init",
      "wishart" in launcher_text)
check("launch_benchmark.py: supports warm_start_full_cov population init",
      "warm_start_full_cov" in launcher_text)
check("launch_benchmark.py: supports solver=cem",
      "solver == \"cem\"" in launcher_text)

for cfg in [
    "benchmark_recipes/pcbo_search.yaml",
    "benchmark_recipes/hpsearch_pcbo.yaml",
    "benchmark_recipes/benchmark_wishart_cem.yaml",
    "benchmark_recipes/compare_cem_pcbo.yaml",
    "benchmark_common_config/shared_initial_population.yaml",
]:
    check(f"{cfg} present", (REPO / cfg).exists())

for cfg in ["benchmark_recipes/benchmark_wishart_cem.yaml", "benchmark_recipes/compare_cem_pcbo.yaml"]:
    text = (REPO / cfg).read_text()
    check(f"{cfg}: no datasets/G1RobotObjRef reference",
          "datasets/G1RobotObjRef" not in text)

# ── Benchmark result JSON files ────────────────────────────────────────────────
section("Benchmark result JSON files (benchmark_rst_json_pointers/)")

for fname in ["pcbo_search.json", "benchmark_sv_vs_pcbo.json",
              "benchmark_wishart_cem.json", "benchmark_wishart_init.json"]:
    p = REPO / "benchmark_rst_json_pointers" / fname
    check(f"{fname} present", p.exists())

# ── Config files ──────────────────────────────────────────────────────────────
section("Config files")

check("config/plot_knot_config.yaml present", (REPO / "config/plot_knot_config.yaml").exists())
check("benchmark_recipes/pcbo_search.yaml present", (REPO / "benchmark_recipes/pcbo_search.yaml").exists())

text = (REPO / "scripts/plot_knot_distribution.py").read_text()
check("plot_knot_distribution.py references config/ not scripts/",
      "config/plot_knot_config.yaml" in text and "scripts/plot_knot_config" not in text)

# ── Dry-run benchmark script ───────────────────────────────────────────────────
section("Benchmark dry-run (reads config, builds commands)")

result = subprocess.run(
    ["uv", "run", "python", "benchmark/launch_benchmark.py", "benchmark_recipes/pcbo_search.yaml", "--dry-run"],
    cwd=REPO, capture_output=True, text=True, timeout=60,
)
check("launch_benchmark.py --dry-run exits cleanly",
      result.returncode == 0, result.stderr[-200:] if result.returncode != 0 else "")

# ── Status script ──────────────────────────────────────────────────────────────
section("Status script")

result = subprocess.run(
    ["bash", "benchmark/status.sh"],
    cwd=REPO, capture_output=True, text=True, timeout=30,
)
check("benchmark/status.sh runs without error", result.returncode == 0,
      result.stderr[-200:] if result.returncode != 0 else "")
check("status.sh shows PCBO Search section",
      "PCBO Search" in result.stdout)

# ── Module imports ─────────────────────────────────────────────────────────────
section("Python module imports")

for module in [
    "sbto.solvers.cbox_polar",
    "sbto.solvers.cem",
    "sbto.solvers.solver_base",
    "sbto.data.save",
    "sbto.utils.plotting",
    "sbto.evaluation.fall_detection",
]:
    r = subprocess.run(
        ["uv", "run", "python", "-c", f"import {module}"],
        cwd=REPO, capture_output=True, text=True, timeout=30,
    )
    check(f"import {module}", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode != 0 else "")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
if failures:
    print(f"\033[31m{len(failures)} check(s) FAILED:\033[0m")
    for f in failures:
        print(f"  • {f}")
    sys.exit(1)
else:
    print(f"\033[32mAll checks passed.\033[0m")
