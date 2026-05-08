"""
Ensure the CEM warm-start file referenced in benchmark_volatile_data/warm_start_ini.yaml exists.

If the file is missing or the path it contains does not exist, runs run_cem.sh to generate
a new warm-start, then writes the path of the newest result back to warm_start_ini.yaml.

warm_start_ini.yaml is gitignored (lives in benchmark_volatile_data/) and can be deleted and
regenerated at any time.

Usage (from repo root):
    uv run python scripts/ensure_warm_start.py
    uv run python scripts/ensure_warm_start.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
VOLATILE_FILE = REPO / "benchmark_volatile_data" / "warm_start_ini.yaml"
RUN_CEM_SCRIPT = REPO / "run_cem.sh"
OUTPUTS_DIR = REPO / "outputs"


def read_ini_dist_path() -> str:
    if not VOLATILE_FILE.exists():
        return ""
    with open(VOLATILE_FILE) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("ini_dist_path", "")


def write_ini_dist_path(new_rel: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would write to {VOLATILE_FILE.relative_to(REPO)}:")
        print(f"    ini_dist_path: {new_rel}")
        return
    VOLATILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    VOLATILE_FILE.write_text(f"ini_dist_path: {new_rel}\n")
    print(f"  Updated {VOLATILE_FILE.relative_to(REPO)}: ini_dist_path -> {new_rel}")


def find_newest_solver_state() -> Path | None:
    candidates = sorted(
        OUTPUTS_DIR.rglob("solver_state_final.npz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ini_path_str = read_ini_dist_path()
    if ini_path_str:
        resolved = (REPO / ini_path_str) if not Path(ini_path_str).is_absolute() else Path(ini_path_str)
        if resolved.exists():
            print(f"Warm-start file exists: {ini_path_str}")
            sys.exit(0)

    if ini_path_str:
        print(f"Warm-start file missing: {ini_path_str}")
    else:
        print(f"No warm-start configured in {VOLATILE_FILE.relative_to(REPO)}")

    print(f"Running {RUN_CEM_SCRIPT.name} to generate warm-start ...")
    if not args.dry_run:
        result = subprocess.run(["bash", str(RUN_CEM_SCRIPT)], cwd=str(REPO))
        if result.returncode != 0:
            print("run_cem.sh failed — aborting.", file=sys.stderr)
            sys.exit(result.returncode)

    newest = find_newest_solver_state()
    if newest is None:
        print("Could not find any solver_state_final.npz in outputs/ — aborting.", file=sys.stderr)
        sys.exit(1)

    write_ini_dist_path(str(newest.relative_to(REPO)), dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
