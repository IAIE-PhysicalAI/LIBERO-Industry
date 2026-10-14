#!/usr/bin/env python3
"""Build a reviewable object/composition map from the original Industry BDDLs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
BDDL_ROOT = REPO_ROOT / "libero" / "libero" / "bddl_files"
SOURCE_SUITES = {
    "libero_industry_d2d-old": "D2D",
    "libero_industry_d2s-old": "D2S",
    "libero_industry_s2d-old": "S2D",
}
OUTPUT_PATH = Path(__file__).with_name("industry_object_map.py")
PREDICATES = "On|ReleasedOn|AlignedOn|Routed|DynamicOn"
SUITE_ORDER = {"D2D": 0, "D2S": 1, "S2D": 2}
ENTRY_FIELDS = (
    "suite",
    "source_objects",
    "static_source_objects",
    "destination_objects",
    "target_objects",
    "target_grooves",
    "source_confounders",
    "destination_confounders",
    "prealigned_objects",
)


def section(text: str, name: str, next_name: Optional[str] = None) -> str:
    start = re.search(r"\(\s*:%s\b" % re.escape(name), text)
    if not start:
        raise ValueError(f"Missing :{name} section")
    if next_name:
        end = re.search(r"\(\s*:%s\b" % re.escape(next_name), text[start.end() :])
        if end:
            return text[start.end() : start.end() + end.start()]
    return text[start.end() :]


def declared_assets(text: str) -> tuple[dict[str, str], set[str]]:
    """Return instance-to-asset types and the fixture instance names."""

    assets: dict[str, str] = {}
    fixtures: set[str] = set()
    for name, next_name, is_fixture in (
        ("fixtures", "objects", True),
        ("objects", "obj_of_interest", False),
    ):
        for names, asset in re.findall(
            r"^\s*([^\n-]+?)\s+-\s+([^\s\)]+)\s*$",
            section(text, name, next_name),
            re.MULTILINE,
        ):
            for instance in names.split():
                assets[instance] = asset
                if is_fixture:
                    fixtures.add(instance)
    return assets, fixtures


def facts(
    text: str, section_name: str, next_name: Optional[str] = None
) -> list[tuple[str, str, str]]:
    return re.findall(
        rf"\(({PREDICATES})\s+([^\s\)]+)\s+([^\s\)]+)\)",
        section(text, section_name, next_name),
    )


def extract_task(path: Path, suite: str) -> dict:
    text = path.read_text(encoding="utf-8")
    assets, fixtures = declared_assets(text)
    init_facts = facts(text, "init", "goal")
    goal_facts = facts(text, "goal")
    movable_instances = set(assets) - fixtures
    source_instances = [
        instance
        for predicate, instance, region in init_facts
        if predicate == "On" and region.startswith("conveyor_belt_1_")
    ]
    static_source_instances = [
        instance
        for predicate, instance, region in init_facts
        if predicate == "On"
        and instance in movable_instances
        and not region.startswith("conveyor_belt_1_")
        and region != "conveyor_ghost_1"
    ]
    destination_instances = [
        instance
        for predicate, instance, region in init_facts
        if predicate == "On" and instance in movable_instances and region == "conveyor_ghost_1"
    ]
    prealigned_instances = [
        instance
        for predicate, instance, _region in init_facts
        if predicate == "AlignedOn" and instance in movable_instances
    ]
    goal_instances = [instance for _predicate, instance, _region in goal_facts]
    target_objects = [assets[name] for name in goal_instances]
    target_grooves = (
        [f"{asset}_groove" for asset in target_objects]
        if suite == "D2S" and "_LV2_" in path.stem
        else []
    )
    unknown = set(source_instances + goal_instances) - set(assets)
    if unknown:
        raise ValueError(f"{path.name} references undeclared instances: {sorted(unknown)}")

    source_set = set(source_instances)
    goal_set = set(goal_instances)
    return {
        "suite": suite,
        "source_objects": [assets[name] for name in source_instances],
        "static_source_objects": [assets[name] for name in static_source_instances],
        "destination_objects": [assets[name] for name in destination_instances],
        "target_objects": target_objects,
        "target_grooves": target_grooves,
        "source_confounders": [assets[name] for name in source_instances if name not in goal_set],
        "destination_confounders": [
            assets[name] for name in sorted(movable_instances - source_set - goal_set)
        ],
        "prealigned_objects": [assets[instance] for instance in prealigned_instances],
    }


def build_object_map() -> dict[str, dict]:
    object_map: dict[str, dict] = {}
    for directory_name, suite in SOURCE_SUITES.items():
        directory = BDDL_ROOT / directory_name
        paths = sorted(directory.glob("*.bddl"))
        if len(paths) != 72:
            raise ValueError(f"{directory} must contain 72 BDDL files, found {len(paths)}")
        for path in paths:
            if path.stem in object_map:
                raise ValueError(f"Duplicate task ID: {path.stem}")
            object_map[path.stem] = extract_task(path, suite)
    if len(object_map) != 216:
        raise ValueError(f"Expected 216 task entries, found {len(object_map)}")
    return object_map


def task_sort_key(task_id: str) -> tuple[int, int, int, str]:
    """Sort entries by suite, level, and canonical ordinal for review."""

    match = re.match(r"^(D2D|D2S|S2D)_LV([123])_(\d{3})_", task_id)
    if not match:
        raise ValueError(f"Object-map task ID has an invalid canonical format: {task_id}")
    suite, level, ordinal = match.groups()
    return SUITE_ORDER[suite], int(level), int(ordinal), task_id


def format_list(values: list, indent: str) -> list[str]:
    """Render one mapped asset per line for quick review."""

    if not values:
        return ["[]"]

    lines = ["["]
    for value in values:
        lines.append(f"{indent}{value!r},")
    lines.append(indent[:-4] + "]")
    return lines


def format_entry(task_id: str, entry: dict) -> list[str]:
    """Render one flat-map entry with stable, review-friendly field ordering."""

    lines = [f"    {task_id!r}: {{"]
    for field in ENTRY_FIELDS:
        value = entry[field]
        prefix = f"        {field!r}: "
        if not isinstance(value, list):
            lines.append(f"{prefix}{value!r},")
            continue

        formatted = format_list(value, "            ")
        if len(formatted) == 1:
            lines.append(f"{prefix}{formatted[0]},")
            continue
        lines.append(f"{prefix}{formatted[0]}")
        lines.extend(formatted[1:-1])
        lines.append(f"{formatted[-1]},")
    lines.append("    },")
    return lines


def format_object_map(object_map: dict[str, dict]) -> str:
    """Return the import-compatible flat map in a grouped source format."""

    lines = [
        "# Auto-generated from the three libero_industry_*-old BDDL suites.",
        "# Run generate_industry_object_map.py after changing the original BDDLs.",
        "# Keep the flat task-ID mapping: stage_industry_from_object_map.py imports it.",
        "",
        "industry_object_map = {",
    ]
    current_group = None
    for task_id in sorted(object_map, key=task_sort_key):
        entry = object_map[task_id]
        match = re.match(r"^(D2D|D2S|S2D)_LV([123])_", task_id)
        group = (match.group(1), match.group(2))
        if group != current_group:
            lines.extend(["", f"    # {group[0]} · Level {group[1]}"])
            current_group = group
        lines.extend(format_entry(task_id, entry))
    lines.extend(["}", ""])
    return "\n".join(lines)


def write_object_map(object_map: dict[str, dict]) -> None:
    OUTPUT_PATH.write_text(format_object_map(object_map), encoding="utf-8")


def main() -> None:
    object_map = build_object_map()
    write_object_map(object_map)
    by_suite = {suite: sum(entry["suite"] == suite for entry in object_map.values()) for suite in SOURCE_SUITES.values()}
    print(f"Wrote {len(object_map)} object-map entries to {OUTPUT_PATH}")
    for suite in sorted(by_suite):
        print(f"  {suite}: {by_suite[suite]}")


if __name__ == "__main__":
    main()
