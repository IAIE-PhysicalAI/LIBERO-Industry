#!/usr/bin/env python3
"""Stage the original 216 Industry BDDLs using the canonical task and object maps."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from industry_object_map import industry_object_map
from libero.libero.benchmark.libero_suite_language_map import libero_language_map
from libero.libero.benchmark.libero_suite_task_map import libero_task_map


BDDL_ROOT = REPO_ROOT / "libero" / "libero" / "bddl_files"
RUNS_ROOT = BDDL_ROOT / ".industry-runs"
SUITES = {
    "D2D": {
        "task_map": "libero_industry_d2d",
        "source_dir": "libero_industry_d2d-old",
        "destination_dir": "libero_industry_d2d",
    },
    "D2S": {
        "task_map": "libero_industry_d2s",
        "source_dir": "libero_industry_d2s-old",
        "destination_dir": "libero_industry_d2s",
    },
    "S2D": {
        "task_map": "libero_industry_s2d",
        "source_dir": "libero_industry_s2d-old",
        "destination_dir": "libero_industry_s2d",
    },
}


def validate_task_id(task_id, suite):
    match = re.match(r"^(D2D|D2S|S2D)_LV([123])_(\d{3})_", task_id)
    if not match or match.group(1) != suite:
        raise ValueError("Invalid {} task-map ID: {}".format(suite, task_id))
    return int(match.group(2)), int(match.group(3))


def canonical_tasks():
    records = []
    seen = set()
    for suite, config in SUITES.items():
        task_ids = libero_task_map[config["task_map"]]
        if len(task_ids) != 72:
            raise ValueError("{} must contain 72 task IDs".format(config["task_map"]))
        for task_id in task_ids:
            level, ordinal = validate_task_id(task_id, suite)
            if task_id in seen:
                raise ValueError("Duplicate task-map ID: {}".format(task_id))
            seen.add(task_id)
            entry = industry_object_map.get(task_id)
            if entry is None:
                raise ValueError("Object map is missing {}".format(task_id))
            if entry["suite"] != suite:
                raise ValueError("Object map suite does not match {}".format(task_id))
            records.append(
                {
                    "suite": suite,
                    "level": level,
                    "ordinal": ordinal,
                    "task_name": task_id,
                    "source_dir": config["source_dir"],
                    "destination_dir": config["destination_dir"],
                    "has_manual_language": task_id in libero_language_map,
                }
            )
    if len(records) != 216:
        raise ValueError("Expected 216 canonical tasks, found {}".format(len(records)))
    return records


def stage(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id may contain only letters, digits, '.', '_' and '-'")
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError("Staging run already exists: {}".format(run_dir))

    records = canonical_tasks()
    for record in records:
        source = BDDL_ROOT / record["source_dir"] / "{}.bddl".format(record["task_name"])
        if not source.is_file():
            raise FileNotFoundError("Missing original BDDL: {}".format(source))

    run_dir.mkdir(parents=True)
    try:
        for record in records:
            source = BDDL_ROOT / record["source_dir"] / "{}.bddl".format(record["task_name"])
            destination = run_dir / "bddl_files" / record["destination_dir"] / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            record["bddl_path"] = str(destination.relative_to(run_dir))
            record["object_map_key"] = record["task_name"]
        manifest = {
            "version": 1,
            "run_id": run_id,
            "status": "ready",
            "expected_task_count": 216,
            "source": "legacy_bddls_via_industry_object_map",
            "manual_language_map_missing": [
                record["task_name"] for record in records if not record["has_manual_language"]
            ],
            "tasks": records,
            "failures": [],
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(run_dir)
        raise
    return run_dir, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="new staged run identifier")
    args = parser.parse_args()
    run_dir, records = stage(args.run_id)
    print("Staged {} original BDDLs in {}".format(len(records), run_dir))
    for suite in SUITES:
        print("  {}: {}".format(suite, sum(record["suite"] == suite for record in records)))
    missing_language = [record for record in records if not record["has_manual_language"]]
    if missing_language:
        print("Manual language-map entries missing: {} (left unchanged)".format(len(missing_language)))


if __name__ == "__main__":
    main()
