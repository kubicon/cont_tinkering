"""Hyperparameter sweeps over `train.py` run configs.

Takes a single YAML file: everything a normal `train.py` config has
(`game`/`network`/`optimizer`/`ppo`/`train`), plus a `sweep` section:

  sweep:
    cartesian:
      optimizer.learning_rate: [0.001, 0.0003]
      network.num_components: [2, 3]
    zip:
      - ppo.category_entropy_coef: [0.1, 0.05, 0.0]
        ppo.gaussian_entropy_coef: [0.1, 0.05, 0.0]

`cartesian` keys are dotted `section.field` paths, each with its own list of
values; every combination across them is a separate run. `zip` is a list of
groups -- within a group, all lists must be the same length and are stepped
through together (like `zip()`) rather than cartesian-producted against each
other; each group counts as one axis, itself cartesian-producted against
everything else.

For every combination, this writes a full run config (the base config with
that combination's overrides applied) to
`{train.checkpoint_dir}/{param=value__param=value...}/config.yaml`, points
that run's own `train.checkpoint_dir` at the same folder, then runs
`train.py` on each config in turn. A run that raises is logged and skipped;
the rest of the sweep continues.

Example:
  python sweep.py configs/multi_point_sweep.yaml
  python sweep.py configs/multi_point_sweep.yaml --dry-run  # only write configs
"""

from __future__ import annotations

import argparse
import copy
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from training.run_config import run_config_from_dict

CONFIG_SECTIONS = {"game", "network", "optimizer", "ppo", "train"}


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "-".join(_format_value(v) for v in value)
    return str(value).replace("/", "_").replace(" ", "")


def _set_path(raw: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _build_axes(sweep_raw: dict) -> list[list[dict[str, Any]]]:
    """Each axis is a list of "choices"; each choice is an override dict
    (one entry for a `cartesian` key, several for a `zip` group). The sweep
    is the cartesian product across axes.
    """
    unknown = set(sweep_raw) - {"cartesian", "zip"}
    if unknown:
        raise ValueError(f"unknown sweep section(s): {sorted(unknown)}")

    all_keys: list[str] = []
    axes: list[list[dict[str, Any]]] = []

    for key, values in (sweep_raw.get("cartesian") or {}).items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.cartesian[{key!r}] must be a non-empty list")
        all_keys.append(key)
        axes.append([{key: v} for v in values])

    for i, group in enumerate(sweep_raw.get("zip") or []):
        keys = list(group.keys())
        if not keys:
            raise ValueError(f"sweep.zip[{i}] must not be empty")
        lengths = {key: len(group[key]) for key in keys}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"sweep.zip[{i}] lists must all have the same length, got {lengths}")
        all_keys.extend(keys)
        axes.append([dict(zip(keys, values)) for values in zip(*(group[key] for key in keys))])

    duplicates = {key for key in all_keys if all_keys.count(key) > 1}
    if duplicates:
        raise ValueError(f"key(s) swept more than once across sweep.cartesian/sweep.zip: {sorted(duplicates)}")

    if not axes:
        raise ValueError("sweep must define at least one key under 'cartesian' or 'zip'")

    return axes


def build_run_configs(sweep_path: str | Path) -> list[tuple[dict, Path]]:
    """Returns `(merged_raw_config, config_path)` for every combination in the sweep."""
    with open(sweep_path) as f:
        raw = yaml.safe_load(f) or {}

    unknown_sections = set(raw) - CONFIG_SECTIONS - {"sweep"}
    if unknown_sections:
        raise ValueError(f"unknown top-level config section(s): {sorted(unknown_sections)}")
    if "sweep" not in raw:
        raise ValueError("config has no 'sweep' section")

    base_raw = {k: v for k, v in raw.items() if k != "sweep"}
    axes = _build_axes(raw["sweep"])

    checkpoint_root = (base_raw.get("train") or {}).get("checkpoint_dir")
    if not checkpoint_root:
        raise ValueError("train.checkpoint_dir is required in a sweep config (run folders live under it)")

    results = []
    for choices in itertools.product(*axes):
        overrides: dict[str, Any] = {}
        for choice in choices:
            overrides.update(choice)

        merged = copy.deepcopy(base_raw)
        for dotted_key, value in overrides.items():
            _set_path(merged, dotted_key, value)

        run_name = "__".join(f"{key}={_format_value(value)}" for key, value in sorted(overrides.items()))
        run_dir = Path(checkpoint_root) / run_name
        _set_path(merged, "train.checkpoint_dir", str(run_dir))

        run_config_from_dict(merged)  # validate eagerly, before writing anything to disk
        results.append((merged, run_dir / "config.yaml"))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="path to a YAML sweep config")
    parser.add_argument("--dry-run", action="store_true", help="only write the per-run configs, don't train")
    args = parser.parse_args()

    runs = build_run_configs(args.config)
    print(f"sweep: {len(runs)} run(s)")

    for _, config_path in runs:
        config_path.parent.mkdir(parents=True, exist_ok=True)

    for merged, config_path in runs:
        with open(config_path, "w") as f:
            yaml.safe_dump(merged, f, sort_keys=False)
        print(f"  wrote {config_path}")

    if args.dry_run:
        return

    failures = []
    for i, (_, config_path) in enumerate(runs, start=1):
        print(f"\n=== run {i}/{len(runs)}: {config_path} ===")
        result = subprocess.run([sys.executable, "train.py", str(config_path)])
        if result.returncode != 0:
            print(f"  run {i} FAILED (exit code {result.returncode}), continuing with the rest of the sweep")
            failures.append(config_path)

    print(f"\nsweep done: {len(runs) - len(failures)}/{len(runs)} succeeded")
    if failures:
        print("failed runs:")
        for config_path in failures:
            print(f"  {config_path}")


if __name__ == "__main__":
    main()
