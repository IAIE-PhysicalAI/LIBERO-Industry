#!/usr/bin/env python3
"""Build the Industry environment configuration map.

The output keys are live BDDL filename stems, not logical task IDs.  This is
the key convention used by ``inject_industry_config`` at runtime.

In addition to the canonical 216 tasks, live BDDLs in ``fractional_tuning_set``
are included when their ``SUITE_LV#_###`` task identity maps unambiguously to a
canonical task with the same suite and level.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero.libero.benchmark.industry_benchmark_config import (
    bddl_stem_for_task,
    parse_industry_task,
    sampling_mode_for_task,
    scene_for_task,
    takt_steps_for_bddl,
)
from libero.libero.benchmark.libero_suite_task_map import libero_task_map


SUITES = {
    "D2D": "libero_industry_d2d",
    "D2S": "libero_industry_d2s",
    "S2D": "libero_industry_s2d",
}
BDDL_ROOT = REPO_ROOT / "libero" / "libero" / "bddl_files"
FRACTIONAL_TUNING_DIR = BDDL_ROOT / "fractional_tuning_set"
DEFAULT_OUTPUT = REPO_ROOT / "libero" / "libero" / "benchmark" / "industry_env_args.json"
_FRACTIONAL_TASK_ID_RE = re.compile(
    r"^(?P<task_id>(?:D2D|D2S|S2D)_LV[123]_\d{3})(?:_|$)", re.IGNORECASE
)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _canonical_task_records(seed):
    records = []
    expected_stems = set()
    level_counts = Counter()
    for suite, suite_dir in SUITES.items():
        task_names = libero_task_map[suite_dir]
        if len(task_names) != 72:
            raise RuntimeError("{} has {} tasks; expected 72".format(suite_dir, len(task_names)))
        for task_name in task_names:
            parsed_suite, level = parse_industry_task(task_name)
            if parsed_suite != suite:
                raise RuntimeError("{} is not a {} task".format(task_name, suite))
            stem = bddl_stem_for_task(task_name)
            if stem in expected_stems:
                raise RuntimeError("Duplicate canonical BDDL stem: {}".format(stem))
            bddl_path = BDDL_ROOT / suite_dir / "{}.bddl".format(stem)
            if not bddl_path.is_file():
                raise RuntimeError("Missing canonical BDDL: {}".format(bddl_path))
            expected_stems.add(stem)
            level_counts[(suite, level)] += 1
            records.append(
                (
                    stem,
                    {
                        "env_args": {"object_sampling": sampling_mode_for_task(task_name)},
                        "set_scene": scene_for_task(task_name, seed=seed),
                        "takt_steps": takt_steps_for_bddl(task_name, bddl_path),
                    },
                )
            )

    expected_counts = {(suite, level): 24 for suite in SUITES for level in (1, 2, 3)}
    if level_counts != expected_counts or len(records) != 216:
        raise RuntimeError("Invalid Industry level coverage: {}".format(dict(level_counts)))

    live_stems = {
        path.stem
        for suite_dir in SUITES.values()
        for path in (BDDL_ROOT / suite_dir).glob("*.bddl")
    }
    if live_stems != expected_stems:
        missing = sorted(expected_stems - live_stems)
        extra = sorted(live_stems - expected_stems)
        raise RuntimeError(
            "Live BDDL set differs from canonical 216 tasks; missing={}, extra={}".format(
                missing, extra
            )
        )
    return dict(records)


def _fractional_task_id(stem):
    """Return the canonical identity prefix required for a fractional BDDL."""
    match = _FRACTIONAL_TASK_ID_RE.match(stem)
    if match is None:
        raise RuntimeError(
            "Fractional BDDL {!r} must start with SUITE_LV#_###".format(stem)
        )
    return match.group("task_id").upper()


def _fractional_task_records(canonical):
    """Generate aligned records, reusing the canonical scene per task identity."""
    if not FRACTIONAL_TUNING_DIR.is_dir():
        raise RuntimeError("Missing fractional tuning directory: {}".format(FRACTIONAL_TUNING_DIR))

    canonical_by_task_id = {}
    for canonical_stem, canonical_record in canonical.items():
        task_id = _fractional_task_id(canonical_stem)
        if task_id in canonical_by_task_id:
            raise RuntimeError("Ambiguous canonical task identity: {}".format(task_id))
        canonical_by_task_id[task_id] = canonical_record

    records = {}
    for bddl_path in sorted(FRACTIONAL_TUNING_DIR.glob("*.bddl")):
        stem = bddl_path.stem
        task_id = _fractional_task_id(stem)
        canonical_record = canonical_by_task_id.get(task_id)
        if canonical_record is None:
            raise RuntimeError(
                "Fractional BDDL {} has no matching canonical task {}".format(stem, task_id)
            )

        env_args = {"object_sampling": sampling_mode_for_task(stem)}
        if canonical_record["env_args"] != env_args:
            raise RuntimeError(
                "Fractional BDDL {} does not align with canonical task {}".format(
                    stem, task_id
                )
            )
        records[stem] = {
            "env_args": env_args,
            "set_scene": canonical_record["set_scene"],
            "takt_steps": takt_steps_for_bddl(stem, bddl_path),
        }
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Optional output path for non-canonical records.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    canonical = _canonical_task_records(args.seed)
    fractional = _fractional_task_records(canonical)
    generated = dict(canonical)
    duplicate_stems = set(generated).intersection(fractional)
    if duplicate_stems:
        raise RuntimeError("Duplicate fractional BDDL stems: {}".format(sorted(duplicate_stems)))
    generated.update(fractional)
    existing = {}
    if args.output.is_file():
        with open(args.output, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    extras = {key: value for key, value in existing.items() if key not in generated}
    print(
        "Industry config: canonical={}, fractional={}, total={}; existing extras to archive: {}".format(
            len(canonical), len(fractional), len(generated), len(extras)
        )
    )
    if args.dry_run:
        return

    _write_json(args.output, generated)
    if extras and args.diagnostics_output is not None:
        _write_json(args.diagnostics_output, extras)
    print("Wrote {}".format(args.output))
    if extras and args.diagnostics_output is not None:
        print("Archived {} non-canonical records to {}".format(len(extras), args.diagnostics_output))


if __name__ == "__main__":
    main()
