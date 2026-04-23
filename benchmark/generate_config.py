#!/usr/bin/env -S uv run python
"""
Generate a benchmark config YAML from a hyperparameter search space definition.

Reads a search-space YAML that lists values for each HP per solver.
Computes the Cartesian product for each solver (only HPs with multiple
values actually vary; single-element lists are fixed). Writes a flat
trial list that launch_benchmark.py can consume directly.

Usage:
    uv run python benchmark/generate_config.py benchmark_config/grid_search_space.yaml
    uv run python benchmark/generate_config.py benchmark_config/grid_search_space.yaml --output benchmark_config/my_run.yaml
    uv run python benchmark/generate_config.py benchmark_config/grid_search_space.yaml --dry-run

The output path can also be set via the `output` key in the search space YAML.
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Short abbreviations used in auto-generated trial labels
_ABBREV = {
    "beta":                "b",
    "scalar_reg":          "sr",
    "lambda_":             "la",
    "flag_auto_weight":    "aw",
    "ini_cov_scale":       "ic",
    "use_loaded_mean_only": "mo",
    "N_it":                "Ni",
    "num_populations":     "np",
    "kernel_std":          "ks",
    "alpha":               "al",
    "sigma0":              "s0",
}


def _fmt_val(v) -> str:
    """Format a value for use in a trial label."""
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        log = math.log10(abs(v))
        if abs(log - round(log)) < 1e-9:
            return f"1e{int(round(log))}"
        if v == int(v):
            return str(int(v))
        return str(v).replace(".", "p")
    return str(v)


def _make_label(solver: str, combo: dict, varying_keys: list[str]) -> str:
    """Build a short unique label from only the parameters that actually vary."""
    prefix = {"pcbo": "pcbo", "sv_cma_es": "sv"}.get(solver, solver[:4])
    parts = [prefix]
    for k in varying_keys:
        abbr = _ABBREV.get(k, k[:3])
        parts.append(f"{abbr}{_fmt_val(combo[k])}")
    return "_".join(parts)


def _coerce(v):
    """Convert numeric strings to float/int (PyYAML parses '1.0e2' as str in flow seqs)."""
    if not isinstance(v, str):
        return v
    try:
        f = float(v)
        return int(f) if f == int(f) and "e" not in v.lower() else f
    except ValueError:
        return v


def _ensure_list(v):
    raw = v if isinstance(v, list) else [v]
    return [_coerce(x) for x in raw]


def cartesian_trials(solver: str, hp_space: dict) -> list[dict]:
    """Generate all Cartesian-product trial dicts for one solver."""
    keys = list(hp_space.keys())
    value_lists = [_ensure_list(hp_space[k]) for k in keys]

    # Keys that actually vary (multiple values) — used for label generation
    varying = [k for k, vl in zip(keys, value_lists) if len(vl) > 1]

    trials = []
    seen_labels: set[str] = set()

    for combo_vals in itertools.product(*value_lists):
        combo = dict(zip(keys, combo_vals))
        label = _make_label(solver, combo, varying)

        # Guard against accidental duplicates
        suffix = 1
        base_label = label
        while label in seen_labels:
            label = f"{base_label}_{suffix}"
            suffix += 1
        seen_labels.add(label)

        trials.append({"solver": solver, "label": label, **combo})

    return trials


def _float_representer(dumper, data):
    """Represent floats in scientific notation to avoid YAML precision issues."""
    if data != data:  # nan
        return dumper.represent_scalar("tag:yaml.org,2002:float", ".nan")
    if data == int(data) and abs(data) < 1e5:
        return dumper.represent_int(int(data))
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data:.6e}")


def generate(space_file: Path, output_override: Path | None, dry_run: bool) -> None:
    with open(space_file) as f:
        space = yaml.safe_load(f)

    out_path = output_override or REPO_ROOT / space.get(
        "output", "benchmark_config/grid_generated.yaml"
    )
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    all_trials: list[dict] = []
    search = space.get("search", {})

    print(f"Search space: {space_file.name}")
    for solver, hp_space in search.items():
        trials = cartesian_trials(solver, hp_space)
        all_trials.extend(trials)
        keys = list(hp_space.keys())
        value_lists = [_ensure_list(hp_space[k]) for k in keys]
        varying = [(k, _ensure_list(hp_space[k])) for k in keys if len(_ensure_list(hp_space[k])) > 1]
        print(f"\n  [{solver}]  {len(trials)} trials")
        if varying:
            print(f"    Varying:")
            for k, vals in varying:
                print(f"      {k}: {vals}")
        else:
            print(f"    (all parameters fixed — 1 trial)")

    print(f"\nTotal: {len(all_trials)} trials")
    est_h = len(all_trials) * 40 / 60  # ~35 min/trial + 5 min cooldown
    print(f"Estimated runtime: ~{est_h:.1f} hours")
    print(f"Output: {out_path}")

    if dry_run:
        print("\n[dry-run] Not writing file.")
        return

    yaml.add_representer(float, _float_representer)

    cfg = {
        "n_samples": space.get("n_samples", 2048),
        "cooldown_s": space.get("cooldown_s", 300),
        "base": space["base"],
        "trials": all_trials,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# AUTO-GENERATED — do not edit by hand.\n"
        f"# Source: {space_file}\n"
        f"# Trials: {len(all_trials)}  (~{est_h:.1f} h)\n"
        f"# Launch: bash benchmark/launch.sh {out_path.relative_to(REPO_ROOT)}\n\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a benchmark config from a search-space YAML."
    )
    parser.add_argument("space", help="Search space YAML (e.g. benchmark_config/grid_search_space.yaml)")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--dry-run", action="store_true", help="Print trial count without writing")
    args = parser.parse_args()

    space_file = Path(args.space)
    if not space_file.is_absolute():
        space_file = REPO_ROOT / space_file
    if not space_file.exists():
        print(f"Error: {space_file} not found", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else None
    if output and not output.is_absolute():
        output = REPO_ROOT / output

    generate(space_file, output, args.dry_run)


if __name__ == "__main__":
    main()
