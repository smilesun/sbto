from __future__ import annotations

from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_common_configs(path: Path) -> dict:
    return _load_yaml_with_common_configs(path.resolve(), stack=[])


def _load_yaml_with_common_configs(path: Path, stack: list[Path]) -> dict:
    if path in stack:
        cycle = " -> ".join(str(p) for p in [*stack, path])
        raise ValueError(f"Cycle in common_configs: {cycle}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    common_paths = raw.get("common_configs", [])
    if not common_paths:
        return raw

    merged: dict = {}
    for rel in common_paths:
        common_path = Path(rel)
        if not common_path.is_absolute():
            common_path = path.parent / common_path
        common_cfg = _load_yaml_with_common_configs(common_path.resolve(), [*stack, path])
        merged = deep_merge(merged, common_cfg)

    raw_without_common = dict(raw)
    raw_without_common.pop("common_configs", None)
    return deep_merge(merged, raw_without_common)


def apply_common_policies(cfg: dict) -> dict:
    cfg = dict(cfg)
    trials = [dict(t) for t in cfg.get("trials", [])]
    policies = cfg.get("policies", {}) or {}

    shared = policies.get("shared_initial_population", {}) or {}
    if shared.get("enabled", False):
        unique_solvers = {t.get("solver", "pcbo") for t in trials}
        min_solver_count = int(shared.get("min_solver_count", 2))
        if len(unique_solvers) >= min_solver_count:
            enforced = {
                "population_init": shared.get("population_init", "warm_start_full_cov"),
                "population_init_key": shared.get("population_init_key", "shared_initial_population"),
                "population_init_cov_scale": float(shared.get("population_init_cov_scale", 1.0)),
                "population_init_seed": int(shared.get("population_init_seed", 0)),
            }
            if "population_init_solver_state" in shared:
                enforced["population_init_solver_state"] = shared["population_init_solver_state"]
            if "population_init_output" in shared:
                enforced["population_init_output"] = shared["population_init_output"]

            apply_to = set(shared.get("solvers", []))
            new_trials = []
            for trial in trials:
                solver = trial.get("solver", "pcbo")
                if apply_to and solver not in apply_to:
                    new_trials.append(trial)
                    continue
                new_trials.append({**trial, **enforced})
            trials = new_trials

    cfg["trials"] = trials
    return cfg
