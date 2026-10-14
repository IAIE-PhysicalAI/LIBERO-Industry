"""Canonical configuration policy for the 216-task LIBERO Industry benchmark.

This module intentionally depends only on the Python standard library.  It is
shared by live ``industry_env_args.json`` generation and staged benchmark
regeneration so that a task has one visual policy regardless of its entry
point.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_AGENTVIEW_ORBIT = {
    "azimuth_deg": 0.0,
    "elevation_deg": 38.922340,
    "distance_m": 1.130651,
}
CAMERA_ROTATION_MAX_DEG = {1: 0.0, 2: 1.0, 3: 2.0}
WRIST_CAMERA_POSITION_MAX_M = {1: 0.0, 2: 0.00229, 3: 0.00457}
LIGHTING_SCALE_MAX_DELTA = {1: 0.0, 2: 0.5, 3: 1.0}

# These are renderer color profiles, not correlated color temperatures or
# photometric measurements.  The generator resolves the selected profile to
# exact RGB coefficients so runtime rendering contains no hidden randomness.
LIGHTING_COLOR_PROFILES = {
    "neutral": (1.0, 0.81, 0.65),
    "warm": (1.0, 0.72, 0.50),
    "cool": (0.82, 0.90, 1.0),
}
_TASK_ID_RE = re.compile(r"^(D2D|D2S|S2D)_LV([123])(?:_|$)", re.IGNORECASE)
_S2D_LV2_POSE_RE = re.compile(r"_(?:FLAT|UPRIGHT)(?=_(?:V|M)[358](?:_|$))")
_INDUSTRY_ASSET_ROOT = (
    Path(__file__).resolve().parents[1] / "assets" / "industry_objects"
)
_GROOVE_MARGIN_M = 0.01


def parse_industry_task(task_name):
    """Return ``(suite, level)`` from an Industry task ID or BDDL filename."""
    stem = Path(task_name).stem.upper()
    match = _TASK_ID_RE.match(stem)
    if match is None:
        raise ValueError("Expected an Industry task ID, got {!r}".format(task_name))
    return match.group(1), int(match.group(2))


def groove_pitch_for_asset(groove_type, axis):
    """Return the runtime-equivalent dynamic-slot pitch from asset metadata."""
    normalized_axis = str(axis).strip().lower()
    if normalized_axis not in {"x", "y"}:
        raise ValueError("LoadSlot axis must be 'x' or 'y', got {!r}".format(axis))

    normalized_type = str(groove_type).strip().lower()
    xml_path = (
        _INDUSTRY_ASSET_ROOT
        / normalized_type
        / "{}.xml".format(normalized_type)
    )
    if not xml_path.is_file():
        raise FileNotFoundError(
            "Missing LoadSlot asset XML for '{}': {}".format(
                normalized_type, xml_path
            )
        )

    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError as error:
        raise ValueError("Invalid LoadSlot asset XML {}: {}".format(xml_path, error))

    radius_site = next(
        (
            site
            for site in root.iter("site")
            if "horizontal_radius_site" in site.attrib.get("name", "")
        ),
        None,
    )
    if radius_site is None or not radius_site.attrib.get("pos"):
        raise ValueError(
            "LoadSlot asset '{}' has no horizontal_radius_site position"
            .format(normalized_type)
        )

    values = [
        abs(float(value)) for value in radius_site.attrib["pos"].split()
    ]
    if len(values) < 2:
        raise ValueError(
            "LoadSlot asset '{}' has an invalid horizontal_radius_site position"
            .format(normalized_type)
        )
    if values[0] > 0.0 and values[1] > 0.0:
        dimensions = {"x": 2.0 * values[0], "y": 2.0 * values[1]}
    else:
        diameter = 2.0 * max(values)
        dimensions = {"x": diameter, "y": diameter}
    return dimensions[normalized_axis] + _GROOVE_MARGIN_M


def stable_task_seed(seed, task_name):
    """Return a deterministic, process-independent seed for one task."""
    digest = hashlib.sha256("{}:{}".format(seed, task_name).encode("utf-8")).digest()
    # Keep this identical to regenerate_industry_benchmark._stable_seed().
    # The manifest and the checked-in JSON must derive an identical scene for
    # the same logical task and benchmark seed.
    return int.from_bytes(digest[:4], "little", signed=False)


def sampling_mode_for_task(task_name):
    """Return the canonical object sampling mode for an Industry task."""
    _suite, level = parse_industry_task(task_name)
    return "uniform" if level == 1 else "random"


def bddl_stem_for_task(task_name):
    """Return the filename stem emitted for a canonical task ID.

    S2D Level-2 task IDs contain a logical ``FLAT`` / ``UPRIGHT`` pose token,
    but the live BDDL filename intentionally omits it.
    """
    suite, level = parse_industry_task(task_name)
    stem = Path(task_name).stem
    if suite == "S2D" and level == 2:
        return _S2D_LV2_POSE_RE.sub("", stem, count=1)
    return stem


def _sample_bounded_gaussian_vector(rng, max_magnitude):
    """Sample an isotropic Gaussian vector with a hard Euclidean bound."""
    if max_magnitude == 0.0:
        return [0.0, 0.0, 0.0]
    sigma = max_magnitude / 3.0
    while True:
        vector = [rng.gauss(0.0, sigma) for _ in range(3)]
        if math.sqrt(sum(component * component for component in vector)) <= max_magnitude:
            return vector


def _sample_bounded_gaussian_scalar(rng, max_magnitude):
    """Sample a zero-mean Gaussian scalar with a hard absolute bound."""
    if max_magnitude == 0.0:
        return 0.0
    sigma = max_magnitude / 3.0
    while True:
        value = rng.gauss(0.0, sigma)
        if abs(value) <= max_magnitude:
            return value


def _rounded_vector(vector, digits):
    return [round(component, digits) for component in vector]


def _sample_camera_config(level, orbit, rng):
    rotation_max_deg = CAMERA_ROTATION_MAX_DEG[level]
    cameras = {
        "agentview": {
            "orbit": {
                "azimuth_deg": round(orbit["azimuth_deg"], 6),
                "elevation_deg": round(orbit["elevation_deg"], 6),
                "distance_m": round(orbit["distance_m"], 6),
            },
            "extrinsic_offset": {
                "position_m": [0.0, 0.0, 0.0],
                "rotation_vector_deg": _rounded_vector(
                    _sample_bounded_gaussian_vector(rng, rotation_max_deg), 6
                ),
            },
        },
    }
    if level > 1:
        cameras["robot0_eye_in_hand"] = {
            "extrinsic_offset": {
                "position_m": _rounded_vector(
                    _sample_bounded_gaussian_vector(
                        rng, WRIST_CAMERA_POSITION_MAX_M[level]
                    ),
                    9,
                ),
                "rotation_vector_deg": _rounded_vector(
                    _sample_bounded_gaussian_vector(rng, rotation_max_deg), 6
                ),
            }
        }
    return cameras


def _sample_lighting(level, rng):
    profile_name = (
        "neutral"
        if level == 1
        else rng.choice(tuple(sorted(LIGHTING_COLOR_PROFILES)))
    )
    color = LIGHTING_COLOR_PROFILES[profile_name]
    task_scale = 2.5 + _sample_bounded_gaussian_scalar(
        rng, LIGHTING_SCALE_MAX_DELTA[level]
    )
    ambient_scale = 0.3
    return {
        "task_diffuse_rgb": _rounded_vector(
            [channel * task_scale for channel in color], 6
        ),
        "ambient_diffuse_rgb": _rounded_vector(
            [channel * ambient_scale for channel in color], 6
        ),
        "cast_shadow": level > 1,
    }


def sample_scene(level, rng=None):
    """Sample the canonical visual policy for one task level.

    ``rng`` may be a ``random.Random`` instance.  Passing it explicitly keeps
    benchmark construction deterministic without mutating global RNG state.
    """
    rng = random if rng is None else rng
    if level == 1:
        orbit = dict(BASE_AGENTVIEW_ORBIT)
    elif level == 2:
        orbit = {
            "azimuth_deg": rng.choice((45, 315)) + rng.uniform(-5.0, 5.0),
            "elevation_deg": BASE_AGENTVIEW_ORBIT["elevation_deg"]
            + rng.uniform(-5.0, 5.0),
            "distance_m": BASE_AGENTVIEW_ORBIT["distance_m"]
            + rng.uniform(-0.1, 0.1),
        }
    elif level == 3:
        while True:
            azimuth = rng.uniform(0.0, 360.0)
            if not 135.0 <= azimuth <= 230.0:
                break
        orbit = {
            "azimuth_deg": azimuth,
            "elevation_deg": BASE_AGENTVIEW_ORBIT["elevation_deg"]
            + rng.uniform(-10.0, 10.0),
            "distance_m": BASE_AGENTVIEW_ORBIT["distance_m"]
            + rng.uniform(-0.2, 0.2),
        }
    else:
        raise ValueError("Unsupported Industry level: {}".format(level))

    cameras = _sample_camera_config(level, orbit, rng)
    lighting = _sample_lighting(level, rng)
    table_style = (
        {"color": "green", "finish": "matte"}
        if level == 1
        else {
            "color": rng.choice(("green", "blue", "black")),
            "finish": rng.choice(("matte", "glossy")),
        }
    )
    return {
        "cameras": cameras,
        "lighting": lighting,
        "table_style": table_style,
    }


def scene_for_task(task_name, seed=42):
    """Return the deterministic canonical scene configuration for one task."""
    _suite, level = parse_industry_task(task_name)
    return sample_scene(level, rng=random.Random(stable_task_seed(seed, task_name)))


def count_task_objects(bddl_path):
    """Return actionable target and source-conveyor-confounder counts."""
    path = Path(bddl_path)
    text = path.read_text(encoding="utf-8")
    init_match = re.search(
        r"\(\s*:init\b(.*?)(?=\(\s*:goal\b)", text, flags=re.DOTALL | re.IGNORECASE
    )
    goal_match = re.search(r"\(\s*:goal\b(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if init_match is None or goal_match is None:
        raise ValueError("{} is missing an :init or :goal section".format(path))

    predicate_pattern = r"On|ReleasedOn|AlignedOn|Routed|DynamicOn"
    goal_instances = set(
        re.findall(
            r"\(\s*(?:{})\s+([^\s\)]+)\s+[^\s\)]+\s*\)".format(predicate_pattern),
            goal_match.group(1),
            flags=re.IGNORECASE,
        )
    )
    initial_on = re.findall(
        r"\(\s*On\s+([^\s\)]+)\s+([^\s\)]+)\s*\)",
        init_match.group(1),
        flags=re.IGNORECASE,
    )
    source_instances = {
        instance for instance, region in initial_on
        if region.startswith("conveyor_belt_1_")
    }
    destination_instances = {
        instance for instance, region in initial_on
        if region == "conveyor_ghost_1"
    }
    target_instances = (
        goal_instances & source_instances
        if source_instances
        else goal_instances - destination_instances
    )
    source_confounders = source_instances - target_instances
    if not goal_instances:
        raise ValueError("{} has no target objects in :goal".format(path))
    return len(target_instances), len(source_confounders)


def get_industry_timing(bddl_path):
    """Read strict conveyor timing predicates from an Industry BDDL."""
    text = Path(bddl_path).read_text(encoding="utf-8")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    at_speed = re.search(
        r"\(\s*AtSpeed\s+conveyor_belt_1\s+({0})\s+steady\s+"
        r"takt_({0})\s+offset_({0})\s*\)".format(number),
        text,
        flags=re.IGNORECASE,
    )
    load_slot = re.search(
        r"\(\s*LoadSlot\s+\S+\s+([^\s\)]+)\s+([xy])\s+"
        r"pulse\s+({0})\s+service_({0})\s*\)".format(number),
        text,
        flags=re.IGNORECASE,
    )
    return {
        "at_speed": (
            {
                "speed": float(at_speed.group(1)),
                "takt": float(at_speed.group(2)),
                "offset": float(at_speed.group(3)),
            }
            if at_speed is not None
            else None
        ),
        "load_slot": (
            {
                "groove_type": load_slot.group(1).lower(),
                "axis": load_slot.group(2).lower(),
                "speed": float(load_slot.group(3)),
                "service": float(load_slot.group(4)),
            }
            if load_slot is not None
            else None
        ),
    }


def takt_steps_for_bddl(task_name, bddl_path):
    """Compute an episode horizon from the BDDL's actual timing predicates."""
    suite, level = parse_industry_task(task_name)
    timing = get_industry_timing(bddl_path)
    num_targets, num_source_confounders = count_task_objects(bddl_path)
    num_objects = num_targets + num_source_confounders
    if suite in {"D2D", "D2S"} and level in {2, 3}:
        if not 0 < num_objects <= 4:
            raise ValueError(
                "{} must contain between one and four source-conveyor objects"
                .format(bddl_path)
            )
        # LV2/LV3 deadlines use the four-position source-conveyor capacity.
        num_objects = 4
    elif suite == "S2D":
        num_objects = num_targets
    if num_objects == 0:
        raise ValueError("{} has no timed source objects".format(bddl_path))

    at_speed = timing["at_speed"]
    load_slot = timing["load_slot"]
    control_freq = 20
    t_initial_reach = 1.0
    t_robot_cycle = 3.0
    t_buffer = 1.5
    if suite == "S2D":
        if load_slot is None or load_slot["speed"] == 0.0:
            raise ValueError("{} requires a non-zero LoadSlot speed".format(bddl_path))
        groove_pitch = groove_pitch_for_asset(
            load_slot["groove_type"], load_slot["axis"]
        )
        t_slot = (groove_pitch / abs(load_slot["speed"])) + load_slot["service"]
        t_max = t_initial_reach + (num_objects * max(t_robot_cycle, t_slot)) + t_buffer
    elif suite == "D2S":
        if at_speed is None:
            raise ValueError("{} requires AtSpeed".format(bddl_path))
        # The authored offset already includes the loading / startup lead time.
        t_max = at_speed["offset"] + ((num_objects - 1) * at_speed["takt"]) + t_robot_cycle + t_buffer
    else:
        if at_speed is None or load_slot is None or load_slot["speed"] == 0.0:
            raise ValueError("{} requires AtSpeed and non-zero LoadSlot".format(bddl_path))
        groove_pitch = groove_pitch_for_asset(
            load_slot["groove_type"], load_slot["axis"]
        )
        t_slot = (groove_pitch / abs(load_slot["speed"])) + load_slot["service"]
        t_final_arrival = at_speed["offset"] + ((num_objects - 1) * at_speed["takt"])
        t_max = t_final_arrival + 1.5 + t_slot + t_buffer
    return math.ceil(t_max * control_freq)
