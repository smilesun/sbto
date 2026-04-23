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

WARM_START = REPO / "outputs/G1RobotObjRef/2026_04_21__09_07_04/solver_state_final.npz"
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

for solver_yaml in ["sbto/conf/solver/pcbo.yaml", "sbto/conf/solver/cbo.yaml"]:
    with open(REPO / solver_yaml) as f:
        cfg = yaml.safe_load(f)
    ini_path = cfg["cfg"].get("ini_dist_path", "")
    exists = Path(ini_path).exists() if ini_path else False
    check(f"{solver_yaml}: ini_dist_path exists", exists, ini_path)
    check(f"{solver_yaml}: ini_dist_path not in datasets/",
          "datasets/G1RobotObjRef" not in ini_path, ini_path)

# ── Benchmark script RESULTS_FILE paths ───────────────────────────────────────
section("Benchmark script RESULTS_FILE paths")

SCRIPTS = {
    "benchmark/launch_benchmark.py":  "benchmark_rst_json_pointers/pcbo_search.json",
    "benchmark/benchmark_sv_vs_pcbo.py":   "benchmark_rst_json_pointers/benchmark_sv_vs_pcbo.json",
    "benchmark/benchmark_wishart_cem.py":  "benchmark_rst_json_pointers/benchmark_wishart_cem.json",
    "benchmark/benchmark_wishart_init.py": "benchmark_rst_json_pointers/benchmark_wishart_init.json",
    "benchmark/hpsearch_pcbo.py":          "benchmark_rst_json_pointers/hpsearch_pcbo_results.json",
}
for script, expected_suffix in SCRIPTS.items():
    text = (REPO / script).read_text()
    check(f"{script}: RESULTS_FILE → benchmark_rst_json_pointers/",
          expected_suffix in text, f"expected '{expected_suffix}'")

POP_SCRIPTS = ["benchmark/benchmark_wishart_init.py", "benchmark/benchmark_wishart_cem.py"]
for script in POP_SCRIPTS:
    text = (REPO / script).read_text()
    check(f"{script}: POP_DIR → benchmark_volatile_data/",
          "benchmark_volatile_data/wishart_pops" in text)

for script in ["benchmark/benchmark_sv_vs_pcbo.py", "benchmark/benchmark_wishart_init.py",
               "benchmark/compare_cem_pcbo.py"]:
    text = (REPO / script).read_text()
    check(f"{script}: no datasets/G1RobotObjRef reference",
          "datasets/G1RobotObjRef" not in text)

# ── Benchmark result JSON files ────────────────────────────────────────────────
section("Benchmark result JSON files (benchmark_rst_json_pointers/)")

for fname in ["pcbo_search.json", "benchmark_sv_vs_pcbo.json",
              "benchmark_wishart_cem.json", "benchmark_wishart_init.json",
              "hpsearch_pcbo_results.json"]:
    p = REPO / "benchmark_rst_json_pointers" / fname
    check(f"{fname} present", p.exists())

# ── Config files ──────────────────────────────────────────────────────────────
section("Config files")

check("config/plot_knot_config.yaml present", (REPO / "config/plot_knot_config.yaml").exists())
check("benchmark_config/pcbo_search.yaml present", (REPO / "benchmark_config/pcbo_search.yaml").exists())

text = (REPO / "scripts/plot_knot_distribution.py").read_text()
check("plot_knot_distribution.py references config/ not scripts/",
      "config/plot_knot_config.yaml" in text and "scripts/plot_knot_config" not in text)

# ── Dry-run benchmark script ───────────────────────────────────────────────────
section("Benchmark dry-run (reads config, builds commands)")

result = subprocess.run(
    ["uv", "run", "python", "benchmark/launch_benchmark.py", "benchmark_config/pcbo_search.yaml", "--dry-run"],
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
