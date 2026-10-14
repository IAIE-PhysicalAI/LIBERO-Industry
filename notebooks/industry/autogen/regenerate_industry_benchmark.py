#!/usr/bin/env python3
"""Stage, preview, and reset-smoke the LIBERO Industry benchmark.

This tool deliberately never writes into a live benchmark suite.  ``build``
creates a self-contained staging run below ``bddl_files/.industry-runs``;
``preview`` and ``verify`` use only BDDLs from that run and always call a
fresh ``env.reset()``.  They do not read or write any ``.init`` file.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gen_d2d
import gen_d2s
import gen_s2d
import configs
import perturbations
from industry_object_map import industry_object_map
from libero.libero.benchmark.industry_benchmark_config import takt_steps_for_bddl
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
import libero.libero.utils.task_generation_utils as task_utils
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.task_generation_utils import generate_bddl_from_task_info


RUNS_ROOT = (
    REPO_ROOT / "libero" / "libero" / "bddl_files" / ".industry-runs"
)
SUITE_DIRS = configs.get_suite_dirs()
SPEEDS = {
    "V3": 0.03,
    "V5": 0.05,
    "V8": 0.08,
    "M3": -0.03,
    "M5": -0.05,
    "M8": -0.08,
}
TRANSPOSE_DEGREES = (0, 90, 180, 270)
D2D_TRANSPOSE_DEGREES = TRANSPOSE_DEGREES
D2D_SPEED_ROTATION_OFFSETS = {
    "V3": 0,
    "V5": 1,
    "V8": 2,
    "M3": 3,
    "M5": 1,
    "M8": 2,
}
TASK_MAP_SUITES = {
    "D2D": "libero_industry_d2d",
    "D2S": "libero_industry_d2s",
    "S2D": "libero_industry_s2d",
}
OBJECT_MAP_FIELDS = (
    "suite",
    "source_objects",
    "static_source_objects",
    "destination_objects",
    "target_objects",
    "source_confounders",
    "destination_confounders",
    "prealigned_objects",
)


@dataclass(frozen=True)
class TaskSpec:
    suite: str
    level: int
    ordinal: int
    target: str
    speed_code: str
    base_speed: float
    slot_speed_code: Optional[str]
    slot_base_speed: Optional[float]
    task_name: str
    seed: int
    transpose_deg: int
    track: str


def _asset_counter(assets):
    """Count asset types while preserving repeated object roles."""
    return Counter(assets)


def _subtract_assets(assets, selected_assets):
    """Return the multiset difference of two ordered asset lists."""
    remaining = _asset_counter(selected_assets)
    result = []
    for asset in assets:
        if remaining[asset]:
            remaining[asset] -= 1
        else:
            result.append(asset)
    return result


def _format_asset_counts(assets):
    return dict(sorted(_asset_counter(assets).items()))


def _validate_object_map_entry(task_name, entry, suite):
    """Reject incomplete or internally inconsistent canonical map records."""
    missing = [field for field in OBJECT_MAP_FIELDS if field not in entry]
    if missing:
        raise AssertionError("{} is missing object-map fields: {}".format(task_name, missing))
    if entry["suite"] != suite:
        raise AssertionError(
            "{} has suite {} instead of {}".format(task_name, entry["suite"], suite)
        )
    for field in OBJECT_MAP_FIELDS[1:]:
        if not isinstance(entry[field], list) or not all(
            isinstance(asset, str) and asset for asset in entry[field]
        ):
            raise AssertionError("{} has invalid {}".format(task_name, field))

    if suite == "S2D":
        expected_source_confounders = []
        # S2D names static plate distractors as destination confounders for
        # historical compatibility; derive that redundant field from the
        # authoritative static-source and target lists.
        expected_destination_confounders = _subtract_assets(
            entry["static_source_objects"], entry["target_objects"]
        )
    else:
        moving_targets = entry["target_objects"]
        if suite == "D2S" and "_LV3_" in task_name:
            moving_targets = _subtract_assets(
                entry["target_objects"], entry["prealigned_objects"]
            )
        expected_source_confounders = _subtract_assets(
            entry["source_objects"], moving_targets
        )
        expected_destination_confounders = _subtract_assets(
            entry["destination_objects"], entry["target_objects"]
        )

    for field, expected in (
        ("source_confounders", expected_source_confounders),
        ("destination_confounders", expected_destination_confounders),
    ):
        if _asset_counter(entry[field]) != _asset_counter(expected):
            raise AssertionError(
                "{} has inconsistent {}: expected {}, found {}".format(
                    task_name,
                    field,
                    _format_asset_counts(expected),
                    _format_asset_counts(entry[field]),
                )
            )

    if _asset_counter(entry["prealigned_objects"]) - _asset_counter(
        entry["target_objects"]
    ):
        raise AssertionError("{} prealigned objects must be targets".format(task_name))


def _stable_seed(seed, task_name):
    digest = hashlib.sha256("{}:{}".format(seed, task_name).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _parse_task_id(task_name):
    match = re.match(r"^(D2D|D2S|S2D)_LV([123])_(\d{3})_", task_name)
    if not match:
        raise ValueError("Invalid canonical Industry task ID: {}".format(task_name))
    suite, level, ordinal = match.groups()
    return suite, int(level), int(ordinal)


def _object_map_key(task_name):
    """Resolve a current task ID to its legacy object-map source."""

    if task_name in industry_object_map:
        return task_name
    suite, level, ordinal = _parse_task_id(task_name)
    prefix = "{}_LV{}_{:03d}_".format(suite, level, ordinal)
    matches = [name for name in industry_object_map if name.startswith(prefix)]
    if len(matches) != 1:
        raise AssertionError(
            "Object map has {} entries for {}".format(len(matches), prefix)
        )
    return matches[0]


def _raw_object_map_entry(task_name):
    return industry_object_map[_object_map_key(task_name)]


def _task_names_for_track(suite, track):
    if track == configs.BIMANUAL:
        return sorted(
            (name for name in industry_object_map if name.startswith("{}_LV".format(suite))),
            key=lambda name: _parse_task_id(name)[1:],
        )
    return libero_task_map[TASK_MAP_SUITES[suite]]


def _speed_pair_from_task_id(task_name, suite, level):
    speed_codes = re.findall(r"(?<![A-Z0-9])([VM][358])(?![A-Z0-9])", task_name)
    expected = 2 if suite == "D2D" and level in (2, 3) else 1
    if len(speed_codes) != expected:
        raise ValueError(
            "{} must contain {} speed code(s), found {}".format(
                task_name, expected, speed_codes
            )
        )
    if suite == "D2D":
        slot_speed_code = speed_codes[1] if len(speed_codes) == 2 else speed_codes[0]
        slot_base_speed = SPEEDS[slot_speed_code]
    else:
        slot_speed_code = None
        slot_base_speed = None
    return (
        speed_codes[0],
        SPEEDS[speed_codes[0]],
        slot_speed_code,
        slot_base_speed,
    )


def _speed_from_task_id(task_name, suite, level):
    """Return the source speed pair retained by legacy schedule callers."""
    speed_code, base_speed, _slot_speed_code, _slot_base_speed = (
        _speed_pair_from_task_id(task_name, suite, level)
    )
    return speed_code, base_speed


def _target_from_task_id(task_name, entry, suite, level, track=configs.SINGLE_ARM):
    """Keep existing fixture logic supplied with the canonical target key."""
    if suite == "S2D" and level == 2:
        # Level-2 BDDL stems intentionally omit the old FLAT/UPRIGHT token.
        # Its mismatch-map key is therefore derived from the canonical
        # object-map target's initial pose, not reconstructed from a filename.
        target_assets = _asset_counter(entry["target_objects"])
        if len(target_assets) != 1:
            raise ValueError(
                "{} must have one S2D Level-2 target asset, found {}".format(
                    task_name, _format_asset_counts(entry["target_objects"])
                )
            )
        initial_asset = next(iter(target_assets))
        matches = [
            target_key
            for target_key, target_spec in configs.get_suite_config(
                "S2D", 2, track
            )["mismatch_map"].items()
            if target_spec["init"] == initial_asset
        ]
        if len(matches) != 1:
            raise ValueError(
                "{} cannot resolve S2D Level-2 target from {}: {}".format(
                    task_name, initial_asset, matches
                )
            )
        return matches[0]

    if suite == "S2D" and level == 3:
        prefix = "{}_LV{}_{:03d}_".format(suite, level, _parse_task_id(task_name)[2])
        remainder = task_name[len(prefix) :]
        target = re.split(r"_(?:V|M)[358](?:_|$)", remainder, maxsplit=1)[0]
        return target.lower()

    target = entry["target_objects"][0]
    if suite == "D2D" and level == 3:
        return re.sub(r"_(upright|flat|inverted)$", "", target)
    return target


def _transpose_schedule(seed, suite, task_names):
    """Return balanced Level-3 transposes aligned with ``task_names``."""
    rng = random.Random(_stable_seed(seed, "{}:level3-transposes".format(suite)))
    transposes_by_task = {}
    for _speed_code in SPEEDS:
        speed_group = []
        for task_name in task_names:
            task_suite, level, _ordinal = _parse_task_id(task_name)
            if task_suite != suite or level != 3:
                continue
            speed_code, _base_speed = _speed_from_task_id(task_name, suite, level)
            if speed_code == _speed_code:
                speed_group.append(task_name)
        expected_group_size = len(TRANSPOSE_DEGREES)
        if len(speed_group) != expected_group_size:
            raise AssertionError(
                "{} Level-3 {} group contains {} tasks instead of {}".format(
                    suite, _speed_code, len(speed_group), expected_group_size
                )
            )
        if suite == "D2D":
            # Rotate the four targets as a Latin schedule. Every speed gets
            # one task at each angle, while every target appears at all four
            # angles with the optimal 1/1/2/2 split across its six tasks.
            target_order = tuple(gen_d2d.configs.D2D_BASE_OBJS)
            tasks_by_target = {}
            for task_name in speed_group:
                target = _target_from_task_id(
                    task_name, _raw_object_map_entry(task_name), suite, 3
                )
                if target in tasks_by_target:
                    raise AssertionError(
                        "D2D Level-3 {} group repeats target {}".format(
                            _speed_code, target
                        )
                    )
                tasks_by_target[target] = task_name
            if set(tasks_by_target) != set(target_order):
                raise AssertionError(
                    "D2D Level-3 {} group has invalid targets: {}".format(
                        _speed_code, sorted(tasks_by_target)
                    )
                )
            rotation_offset = D2D_SPEED_ROTATION_OFFSETS[_speed_code]
            for target_index, target in enumerate(target_order):
                angle = TRANSPOSE_DEGREES[
                    (target_index + rotation_offset) % len(TRANSPOSE_DEGREES)
                ]
                transposes_by_task[tasks_by_target[target]] = angle
        else:
            group = list(TRANSPOSE_DEGREES)
            rng.shuffle(group)
            transposes_by_task.update(zip(speed_group, group))
    return tuple(transposes_by_task.get(task_name, 0) for task_name in task_names)


def _transpose_candidates(spec, usage):
    """Return the assigned transpose and any suite-approved alternatives."""
    if spec.level != 3:
        return None
    if spec.suite in {"D2D", "D2S"}:
        # D2D and D2S have feasible cardinal anchors. Resample position/yaw at
        # the assigned angle instead of changing their balanced task matrix.
        return (spec.transpose_deg,)
    allowed = TRANSPOSE_DEGREES
    alternatives = [angle for angle in allowed if angle != spec.transpose_deg]
    alternatives.sort(
        key=lambda angle: (usage[(spec.suite, angle)], (angle - spec.transpose_deg) % 360)
    )
    return tuple([spec.transpose_deg] + alternatives)


def build_catalog(seed, track=configs.SINGLE_ARM):
    """Return the 216 canonical task IDs with map-fixed compositions."""
    configs.get_suite_dirs(track)
    catalog = []
    seen = set()
    for suite in ("D2D", "D2S", "S2D"):
        task_names = _task_names_for_track(suite, track)
        if len(task_names) != 72:
            raise AssertionError("{} does not contain 72 tasks".format(suite))
        level_counts = {1: 0, 2: 0, 3: 0}
        transpose_counts = Counter()
        transpose_schedule = _transpose_schedule(seed, suite, task_names)
        transpose_by_speed = {speed_code: Counter() for speed_code in SPEEDS}
        for task_name, scheduled_transpose_deg in zip(task_names, transpose_schedule):
            task_suite, level, ordinal = _parse_task_id(task_name)
            if task_suite != suite or task_name in seen:
                raise AssertionError("Invalid or duplicate task ID: {}".format(task_name))
            raw_entry = _raw_object_map_entry(task_name)
            entry = configs.object_map_entry_for_track(task_name, raw_entry, track)
            _validate_object_map_entry(task_name, entry, suite)
            (
                speed_code,
                base_speed,
                slot_speed_code,
                slot_base_speed,
            ) = _speed_pair_from_task_id(task_name, suite, level)
            catalog.append(
                TaskSpec(
                    suite=suite,
                    level=level,
                    ordinal=ordinal,
                    target=_target_from_task_id(
                        task_name, entry, suite, level, track=track
                    ),
                    speed_code=speed_code,
                    base_speed=base_speed,
                    slot_speed_code=slot_speed_code,
                    slot_base_speed=slot_base_speed,
                    task_name=task_name,
                    seed=_stable_seed(seed, task_name),
                    transpose_deg=scheduled_transpose_deg if level == 3 else 0,
                    track=track,
                )
            )
            seen.add(task_name)
            level_counts[level] += 1
            if level == 3:
                transpose_counts[catalog[-1].transpose_deg] += 1
                transpose_by_speed[speed_code][catalog[-1].transpose_deg] += 1
        if level_counts != {1: 24, 2: 24, 3: 24}:
            raise AssertionError("{} has invalid level coverage: {}".format(suite, level_counts))
        expected_transpose_counts = Counter(
            {angle: 6 for angle in TRANSPOSE_DEGREES}
        )
        if transpose_counts != expected_transpose_counts:
            raise AssertionError(
                "{} has invalid Level-3 transpose coverage: {}".format(
                    suite, dict(transpose_counts)
                )
            )
        if suite in {"D2D", "D2S", "S2D"}:
            expected_speed_coverage = Counter(
                {angle: 1 for angle in TRANSPOSE_DEGREES}
            )
            invalid_speed_coverage = {
                speed_code: dict(counts)
                for speed_code, counts in transpose_by_speed.items()
                if counts != expected_speed_coverage
            }
            if invalid_speed_coverage:
                raise AssertionError(
                    "{} has invalid per-speed transpose coverage: {}".format(
                        suite, invalid_speed_coverage
                    )
                )
    if len(catalog) != 216:
        raise AssertionError("Catalog contains {} tasks instead of 216".format(len(catalog)))
    return catalog


def _build_scene(spec, transpose_degrees=None):
    """Register one task scene using its deterministic task seed."""
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    task_utils.TASK_INFO.clear()
    object_map_entry = configs.object_map_entry_for_track(
        spec.task_name, _raw_object_map_entry(spec.task_name), spec.track
    )
    if spec.suite == "D2D":
        target = spec.target
        return gen_d2d.build_and_register_d2d(
            spec.ordinal,
            spec.level,
            target,
            spec.base_speed,
            object_map_entry=object_map_entry,
            transpose_degrees=transpose_degrees,
            slot_base_speed=spec.slot_base_speed,
            track=spec.track,
        )
    if spec.suite == "D2S":
        target = None if spec.level == 1 else spec.target
        return gen_d2s.build_and_register_d2s(
            spec.ordinal,
            spec.level,
            spec.base_speed,
            target,
            object_map_entry=object_map_entry,
            transpose_degrees=transpose_degrees,
            track=spec.track,
        )
    if spec.suite == "S2D":
        return gen_s2d.build_and_register_s2d(
            spec.ordinal,
            spec.level,
            spec.target,
            spec.base_speed,
            object_map_entry=object_map_entry,
            transpose_degrees=transpose_degrees,
            track=spec.track,
        )
    raise ValueError("Unsupported suite: {}".format(spec.suite))


def _extract_language(bddl_text):
    match = re.search(r"\(:language\s+([^\n\)]+)\)", bddl_text)
    if not match:
        raise ValueError("Generated BDDL has no language declaration")
    return match.group(1).strip()


def _bddl_object_assets(text):
    """Return the instance-to-asset mapping declared by one BDDL."""
    objects_start = text.find("(:objects")
    objects_end = text.find("(:obj_of_interest", objects_start)
    if objects_start < 0 or objects_end < 0:
        raise ValueError("Generated BDDL has no object declaration section")
    objects_text = text[objects_start:objects_end]
    object_assets = {}
    for names, asset in re.findall(
        r"^\s*([^\n-]+?)\s+-\s+([^\s\n]+)\s*$", objects_text, re.MULTILINE
    ):
        for instance_name in names.split():
            if instance_name in object_assets:
                raise ValueError("BDDL declares {} more than once".format(instance_name))
            object_assets[instance_name] = asset
    if not object_assets:
        raise ValueError("Generated BDDL declares no task objects")
    return object_assets


def _asset_counts_for_instances(object_assets, instances):
    unknown = sorted(set(instances) - set(object_assets))
    if unknown:
        raise ValueError("BDDL predicate references undeclared objects: {}".format(unknown))
    return Counter(object_assets[instance_name] for instance_name in instances)


def _validate_map_composition(spec, text):
    """Check serialized inventory, placement roles, and goals against the map."""
    entry = configs.object_map_entry_for_track(
        spec.task_name, _raw_object_map_entry(spec.task_name), spec.track
    )
    object_assets = _bddl_object_assets(text)
    init_start = text.find("(:init")
    goal_start = text.find("(:goal")
    if init_start < 0 or goal_start < 0:
        return "missing init or goal section"
    init_text = text[init_start:goal_start]
    goal_text = text[goal_start:]
    if spec.suite == "D2D":
        expected_assets = entry["source_objects"] + entry["destination_objects"]
    elif spec.suite == "D2S":
        expected_assets = entry["source_objects"] + entry["prealigned_objects"]
    else:
        expected_assets = entry["static_source_objects"] + entry["destination_objects"]
    actual_assets = Counter(object_assets.values())
    if actual_assets != _asset_counter(expected_assets):
        return "object inventory differs from object map: expected {}, found {}".format(
            _format_asset_counts(expected_assets), dict(sorted(actual_assets.items()))
        )

    on_facts = re.findall(r"\(On\s+([^\s()]+)\s+([^\s()]+)\)", init_text)
    aligned_facts = re.findall(
        r"\(AlignedOn\s+([^\s()]+)\s+([^\s()]+)\)", init_text
    )

    def assets_on(location_prefix):
        return _asset_counts_for_instances(
            object_assets,
            [
                instance_name
                for instance_name, location in on_facts
                if instance_name in object_assets and location.startswith(location_prefix)
            ],
        )

    if spec.suite == "D2D":
        role_checks = (
            ("conveyor source", entry["source_objects"], assets_on("conveyor_belt_1_")),
            ("ghost destination", entry["destination_objects"], assets_on("conveyor_ghost_1")),
        )
    elif spec.suite == "D2S":
        role_checks = (
            ("conveyor source", entry["source_objects"], assets_on("conveyor_belt_1_")),
            (
                "prealigned destination",
                entry["prealigned_objects"],
                _asset_counts_for_instances(
                    object_assets,
                    [
                        instance_name
                        for instance_name, _goal_region in aligned_facts
                        if instance_name in object_assets
                    ],
                ),
            ),
        )
    else:
        role_checks = (
            ("static source", entry["static_source_objects"], assets_on("grey_plate_1_")),
            ("ghost destination", entry["destination_objects"], assets_on("conveyor_ghost_1")),
        )
    for role, expected, actual in role_checks:
        if _asset_counter(expected) != actual:
            return "{} placement differs from object map: expected {}, found {}".format(
                role, _format_asset_counts(expected), dict(sorted(actual.items()))
            )

    goal_instances = re.findall(
        r"\((?:DynamicOn|Routed|AlignedOn|ReleasedOn)\s+([^\s()]+)\s+[^\s()]+\)",
        goal_text,
    )
    goal_assets = _asset_counts_for_instances(object_assets, goal_instances)
    if goal_assets != _asset_counter(entry["target_objects"]):
        return "goal objects differ from object map: expected {}, found {}".format(
            _format_asset_counts(entry["target_objects"]),
            dict(sorted(goal_assets.items())),
        )
    return None


def _conveyor_goal_predicate(spec):
    """Return the canonical conveyor predicate for the affected task suites."""
    if spec.suite not in {"D2D", "S2D"}:
        return None
    return "DynamicOn" if spec.level == 1 else "Routed"


def _normalize_conveyor_goal_predicates(spec, bddl_text):
    """Keep D2D/S2D goal predicates level-specific after BDDL serialization."""
    expected_predicate = _conveyor_goal_predicate(spec)
    if expected_predicate is None:
        return bddl_text

    goal_start = bddl_text.find("(:goal")
    if goal_start < 0:
        raise ValueError("Generated BDDL has no goal section")
    prefix, goal_text = bddl_text[:goal_start], bddl_text[goal_start:]
    goal_text, replacements = re.subn(
        r"\((?:DynamicOn|Routed)\s+([^\s()]+)\s+conveyor_ghost_1\)",
        r"({} \1 conveyor_ghost_1)".format(expected_predicate),
        goal_text,
    )
    if replacements == 0:
        raise ValueError("Missing DynamicOn/Routed conveyor goal")
    return prefix + goal_text


def _validate_d2s_goal_policy(spec, text, goal_start):
    """Enforce D2S's level-specific destination and goal predicates."""
    if spec.suite != "D2S":
        return None

    goal_text = text[goal_start:] if goal_start >= 0 else ""
    goal_facts = re.findall(
        r"\((On|ReleasedOn|AlignedOn)\s+([^\s()]+)\s+([^\s()]+)\)",
        goal_text,
    )
    if not goal_facts:
        return "missing D2S placement goals"

    if spec.level == 1:
        if "grey_plate" in text:
            return "D2S Level 1 must not use grey_plate"
        if not re.search(r"\btoy_tray_1\s+-\s+toy_tray\b", text):
            return "D2S Level 1 is missing toy_tray_1"
        if any(
            predicate != "ReleasedOn" or target != "toy_tray_1_goal_region"
            for predicate, _object_name, target in goal_facts
        ):
            return "D2S Level 1 requires ReleasedOn goals on toy_tray_1_goal_region"
        return None

    if any(predicate != "AlignedOn" for predicate, _object_name, _target in goal_facts):
        return "D2S Levels 2 and 3 require AlignedOn goals"
    if any(not target.endswith("_goal_region") for _predicate, _object_name, target in goal_facts):
        return "D2S Levels 2 and 3 require inlay goal regions"
    if spec.level == 3:
        goal_regions = [target for _predicate, _object_name, target in goal_facts]
        if len(goal_regions) != len(set(goal_regions)):
            return "D2S Level 3 requires one goal object per inlay region"
    return None


def _validate_bddl_text(spec, text):
    if "_Ablation" in text:
        return "scene suffix still contains _Ablation"
    # XML compatibility aliases remain declared as regions, but new task
    # predicates must never select them.
    init_start = text.find("(:init")
    goal_start = text.find("(:goal")
    init_text = text[init_start:goal_start] if init_start >= 0 and goal_start >= 0 else text
    if "_region_d2d_" in init_text:
        return "legacy _d2d conveyor region was emitted"
    if spec.suite in {"D2D", "D2S"}:
        if not re.search(
            r"\(AtSpeed conveyor_belt_1 [-0-9.]+ steady takt_[-0-9.]+ offset_[-0-9.]+\)",
            text,
        ):
            return "missing strict AtSpeed predicate"
    if spec.suite in {"D2D", "S2D"} and not re.search(
        r"\(LoadSlot conveyor_ghost_1 [^\s]+ [xy] pulse [-0-9.]+ service_[-0-9.]+\)",
        text,
    ):
        return "missing strict LoadSlot predicate"
    if spec.suite == "D2D":
        source_match = re.search(
            r"\(AtSpeed conveyor_belt_1 ([-0-9.]+) steady ", text
        )
        slot_match = re.search(
            r"\(LoadSlot conveyor_ghost_1 [^\s]+ [xy] pulse ([-0-9.]+) service_",
            text,
        )
        if source_match is None or slot_match is None:
            return "missing D2D conveyor speed predicates"
        source_speed = float(source_match.group(1))
        slot_speed = float(slot_match.group(1))
        for actual_speed, base_speed, label in (
            (source_speed, spec.base_speed, "source"),
            (slot_speed, spec.slot_base_speed, "slot"),
        ):
            lower, upper = perturbations.d2d_speed_band(base_speed)
            if (
                actual_speed * base_speed <= 0.0
                or not lower <= abs(actual_speed) <= upper
            ):
                return "D2D {} speed is outside its filename watershed".format(label)
        if spec.level == 1 and not np.isclose(source_speed, slot_speed):
            return "D2D Level 1 conveyors must have equal signed speeds"
        if spec.level == 2 and (
            source_speed * slot_speed <= 0.0
            or np.isclose(abs(source_speed), abs(slot_speed))
        ):
            return "D2D Level 2 requires same-direction unequal speeds"
        if spec.level == 3 and source_speed * slot_speed >= 0.0:
            return "D2D Level 3 requires opposite-direction speeds"
    expected_predicate = _conveyor_goal_predicate(spec)
    if expected_predicate is not None:
        goal_text = text[goal_start:] if goal_start >= 0 else ""
        predicates = re.findall(
            r"\((DynamicOn|Routed)\s+[^\s()]+\s+conveyor_ghost_1\)",
            goal_text,
        )
        if not predicates:
            return "missing conveyor goal predicate"
        if set(predicates) != {expected_predicate}:
            return "expected {} conveyor goals, found {}".format(
                expected_predicate, sorted(set(predicates))
            )
    d2s_goal_error = _validate_d2s_goal_policy(spec, text, goal_start)
    if d2s_goal_error:
        return d2s_goal_error
    composition_error = _validate_map_composition(spec, text)
    if composition_error:
        return composition_error
    return None


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _run_dir(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id may contain only letters, digits, '.', '_' and '-'")
    return RUNS_ROOT / run_id


def _scene_visuals(spec):
    return perturbations.sample_vis(spec.level, rng=random.Random(spec.seed))


def command_build(args):
    run_dir = _run_dir(args.run_id)
    if run_dir.exists():
        raise RuntimeError(
            "Staging run already exists: {}. Choose a new --run-id.".format(run_dir)
        )
    all_suite_dirs = configs.get_suite_dirs(args.track)
    catalog = build_catalog(args.seed, track=args.track)
    if args.suite:
        catalog = [spec for spec in catalog if spec.suite == args.suite]
    suite_dirs = {
        suite: all_suite_dirs[suite]
        for suite in dict.fromkeys(spec.suite for spec in catalog)
    }
    expected_task_count = len(catalog)
    records = []
    failures = []
    transpose_usage = Counter()
    run_dir.mkdir(parents=True)
    try:
        for index, spec in enumerate(catalog, 1):
            suite_dir = run_dir / "bddl_files" / suite_dirs[spec.suite]
            suite_dir.mkdir(parents=True, exist_ok=True)
            print("[{}/{}] {}".format(index, len(catalog), spec.task_name))
            try:
                transpose_degrees = _transpose_candidates(spec, transpose_usage)
                scene_name, sampling_mode, layout_error, metadata = _build_scene(
                    spec, transpose_degrees=transpose_degrees
                )
                if layout_error or scene_name is None:
                    raise RuntimeError(layout_error or "scene builder returned no scene")
                with tempfile.TemporaryDirectory(dir=str(run_dir)) as temp_dir:
                    generated, serializer_failures = generate_bddl_from_task_info(folder=temp_dir)
                    if serializer_failures or len(generated) != 1:
                        raise RuntimeError(
                            "BDDL serializer failures: {}".format(serializer_failures)
                        )
                    generated_path = Path(generated[0])
                    content = generated_path.read_text()
                content = content.replace("_Ablation", "_Manipulation")
                content = _normalize_conveyor_goal_predicates(spec, content)
                validation_error = _validate_bddl_text(spec, content)
                if validation_error:
                    raise RuntimeError(validation_error)

                bddl_path = suite_dir / "{}.bddl".format(spec.task_name)
                with bddl_path.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(content)
                metadata = metadata or {}
                record = asdict(spec)
                record.update(
                    {
                        "bddl_path": str(bddl_path.relative_to(run_dir)),
                        "language": _extract_language(content),
                        "sampling_mode": sampling_mode,
                        "metadata": metadata or {},
                        "set_scene": _scene_visuals(spec),
                        "takt_steps": takt_steps_for_bddl(spec.task_name, bddl_path),
                    }
                )
                if spec.level == 3:
                    actual_transpose_deg = metadata.get("Transpose Degrees")
                    if actual_transpose_deg not in TRANSPOSE_DEGREES:
                        raise RuntimeError(
                            "Level-3 builder did not report a cardinal transpose: {}".format(
                                actual_transpose_deg
                            )
                        )
                    record["requested_transpose_deg"] = spec.transpose_deg
                    record["actual_transpose_deg"] = actual_transpose_deg
                    if (
                        spec.suite in {"D2D", "D2S"}
                        and actual_transpose_deg != spec.transpose_deg
                    ):
                        raise RuntimeError(
                            "{} Level-3 changed scheduled transpose {} to {}".format(
                                spec.suite, spec.transpose_deg, actual_transpose_deg
                            )
                        )
                    transpose_usage[(spec.suite, actual_transpose_deg)] += 1
                records.append(record)
            except Exception as error:
                failures.append({"task_name": spec.task_name, "error": str(error)})
                print("  FAILED: {}".format(error))
    finally:
        manifest = {
            "version": 4,
            "run_id": args.run_id,
            "seed": args.seed,
            "track": args.track,
            "suite_dirs": suite_dirs,
            "expected_task_count": expected_task_count,
            "tasks": records,
            "failures": failures,
            "status": (
                "ready"
                if not failures and len(records) == expected_task_count
                else "failed"
            ),
            "actual_transpose_distribution": {
                suite: {
                    str(angle): transpose_usage[(suite, angle)]
                    for angle in TRANSPOSE_DEGREES
                }
                for suite in suite_dirs
            },
        }
        _write_json(run_dir / "manifest.json", manifest)

    if failures:
        raise RuntimeError(
            "Build staged {} valid tasks and {} failures; inspect {}".format(
                len(records), len(failures), run_dir / "manifest.json"
            )
        )
    print("Staged {} BDDLs in {}".format(len(records), run_dir))


def _task_env(record, render):
    kwargs = {
        "bddl_file_name": str(_run_dir(record["run_id"]) / record["bddl_path"]),
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": ["agentview"],
        "has_renderer": False,
        "has_offscreen_renderer": render,
        "use_camera_obs": render,
        "control_freq": 20,
        "renderer": "mujoco",
        "robots": ["Panda"],
        "gripper_types": "default",
        "object_sampling": record["sampling_mode"],
        "set_scene": record["set_scene"],
        "takt_steps": record["takt_steps"],
        "ignore_done": True,
    }
    try:
        return OffScreenRenderEnv(**kwargs)
    except TypeError:
        # Robosuite 1.4 does not accept the native-MuJoCo renderer argument.
        kwargs.pop("renderer")
        return OffScreenRenderEnv(**kwargs)


def _reset_and_step(record, steps, render, video_path=None):
    random.seed(record["seed"])
    np.random.seed(record["seed"])
    env = None
    try:
        env = _task_env(record, render=render)
        seed_fn = getattr(env, "seed", None)
        if callable(seed_fn):
            try:
                seed_fn(record["seed"])
            except Exception:
                pass
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        frames = []
        if render and "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])
        action_dim = getattr(env, "action_dim", None)
        if action_dim is None:
            action_dim = getattr(getattr(env, "env", None), "action_dim", 7)
        action = np.zeros(action_dim)
        for _ in range(steps):
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, _, terminated, truncated, _ = step_result
                done = bool(terminated or truncated)
            else:
                obs, _, done, _ = step_result
            if render and "agentview_image" in obs:
                frames.append(obs["agentview_image"][::-1])
            if done:
                break
        state = np.asarray(env.get_init_state())
        if not np.isfinite(state).all():
            raise RuntimeError("simulator state contains non-finite values")
        if render and video_path is not None:
            if not frames:
                raise RuntimeError("agentview did not produce any preview frames")
            import imageio

            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(video_path), frames, fps=20)
        return {"status": "ok", "frames": len(frames), "steps": steps}
    finally:
        if env is not None:
            env.close()


def _load_manifest_for_run(run_id):
    run_dir = _run_dir(run_id)
    manifest = _read_json(run_dir / "manifest.json")
    expected = manifest.get("expected_task_count")
    if manifest.get("status") != "ready" or len(manifest.get("tasks", [])) != expected:
        raise RuntimeError("Run {} is not a complete staged build".format(run_id))
    for record in manifest["tasks"]:
        record["run_id"] = run_id
    return run_dir, manifest


def command_preview(args):
    run_dir, manifest = _load_manifest_for_run(args.run_id)
    selected = []
    suites = (args.suite,) if args.suite else tuple(manifest["suite_dirs"])
    levels = (args.level,) if args.level else (1, 2, 3)
    for suite in suites:
        for level in levels:
            tasks = [
                record
                for record in manifest["tasks"]
                if record["suite"] == suite and record["level"] == level
            ]
            selected.extend(sorted(tasks, key=lambda item: item["ordinal"])[: args.per_level])
    report = {"run_id": args.run_id, "uses_init_files": False, "tasks": []}
    for index, record in enumerate(selected, 1):
        video = run_dir / "previews" / record["suite"] / "LV{}".format(record["level"]) / "{}.mp4".format(record["task_name"])
        print("[{}/{}] {}".format(index, len(selected), record["task_name"]))
        try:
            result = _reset_and_step(record, args.steps, render=True, video_path=video)
            result.update({"task_name": record["task_name"], "video": str(video.relative_to(run_dir))})
        except Exception as error:
            result = {"task_name": record["task_name"], "status": "failed", "error": str(error)}
            print(traceback.format_exc())
        report["tasks"].append(result)
    _write_json(run_dir / "preview_report.json", report)
    failed = [record for record in report["tasks"] if record["status"] != "ok"]
    if failed:
        raise RuntimeError("{} preview tasks failed; inspect preview_report.json".format(len(failed)))
    print("Saved {} reset-only preview videos under {}".format(len(selected), run_dir / "previews"))


def command_verify(args):
    run_dir, manifest = _load_manifest_for_run(args.run_id)
    records = manifest["tasks"] if args.reset_all else []
    report = {"run_id": args.run_id, "uses_init_files": False, "tasks": []}
    for index, record in enumerate(records, 1):
        print("[{}/{}] {}".format(index, len(records), record["task_name"]))
        try:
            result = _reset_and_step(record, args.steps, render=False)
            result["task_name"] = record["task_name"]
        except Exception as error:
            result = {"task_name": record["task_name"], "status": "failed", "error": str(error)}
            print(traceback.format_exc())
        report["tasks"].append(result)
    _write_json(run_dir / "verification_report.json", report)
    failed = [record for record in report["tasks"] if record["status"] != "ok"]
    if failed:
        raise RuntimeError("{} verification tasks failed; inspect verification_report.json".format(len(failed)))
    print("Verified {} staged tasks without init files".format(len(records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="stage benchmark BDDLs")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--run-id", required=True)
    build.add_argument("--suite", type=str.upper, choices=tuple(SUITE_DIRS))
    build.add_argument(
        "--track",
        choices=configs.GENERATION_TRACKS,
        default=configs.SINGLE_ARM,
    )
    build.set_defaults(handler=command_build)

    preview = subparsers.add_parser("preview", help="render reset-only staged previews")
    preview.add_argument("--run-id", required=True)
    preview.add_argument("--suite", type=str.upper, choices=tuple(SUITE_DIRS))
    preview.add_argument("--level", type=int, choices=(1, 2, 3))
    preview.add_argument("--per-level", type=int, default=3)
    preview.add_argument("--steps", type=int, default=100)
    preview.set_defaults(handler=command_preview)

    verify = subparsers.add_parser("verify", help="reset-smoke staged tasks without init files")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--reset-all", action="store_true")
    verify.add_argument("--steps", type=int, default=100)
    verify.set_defaults(handler=command_verify)

    args = parser.parse_args()
    try:
        args.handler(args)
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
