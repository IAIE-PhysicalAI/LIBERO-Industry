import copy
import os
import re
from pathlib import Path
CURRENT_FILE_DIR = Path(__file__).resolve().parent
LIBERO_INDUSTRY_BASE = CURRENT_FILE_DIR.parents[2]
BASE_DIR = str(LIBERO_INDUSTRY_BASE / "libero" / "libero" / "bddl_files" / "libero_industry_draft" / "modular_test")
ASSET_DIR = str(LIBERO_INDUSTRY_BASE / "libero" / "libero" / "assets" / "industry_objects")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "bddl_files")
VIDEO_OUT_DIR = os.path.join(BASE_DIR, "videos")
EXCEL_PATH = os.path.join(BASE_DIR, "Suite_Tracker.xlsx")

GENERATE_BDDL_ONLY = False
SAVE_MP4_VIDEO = True        # Set to False to run physics without rendering MP4s

BASE_SPEEDS = [0.03, 0.05, 0.08, -0.03, -0.05, -0.08]

# ==========================================
# SUITE 1: D2D (Dynamic-to-Dynamic)
# ==========================================
D2D_BASE_OBJS = ["mouse_black_taipan", "mouse_white_taipan", "mouse_naga", "headset"]
D2D = {
    1: {
        "targets": [f"{o}_upright" for o in D2D_BASE_OBJS],
        "all_objects": [f"{o}_upright" for o in D2D_BASE_OBJS],
        "shift_range": 0.02,
        "language": "put all objects from the conveyor into the moving tray",
        "arrival_takt_by_speed": {0.03: 4.0, 0.05: 5.0, 0.08: 5.0},
        "service_time": 1.5  # LV1: Most forgiving service time
    },
    2: {
        "targets": [f"{o}_upright" for o in D2D_BASE_OBJS],
        "all_objects": [f"{o}_upright" for o in D2D_BASE_OBJS],
        "shift_range": 0.02, "yaw_range": 5,
        "grooves": ["upright_groove"],
        "arrival_takt": 6.0,
        "service_time": 2.0,  # LV2: Moderate service time
        "composition": {
            # Legacy: two or three targets plus source-only distractors,
            # always filling the four conveyor placement regions.
            "source_total": 4,
            "target_counts": [2, 3],
        },
    },
    3: {
        "targets": D2D_BASE_OBJS,
        "all_objects": [f"{o}_upright" for o in D2D_BASE_OBJS],
        # D2D L3 owns its own parallel-transport anchor convention.  Each
        # cardinal transpose has a workspace-safe anchor pair in gen_d2d.
        # This is deliberately independent of the D2S/S2D symmetric layout.
        "shift_range": 0.05, "shift_x_range": 0.02, "shift_y_range": 0.05,
        "yaw_range": 10,
        "transpositions": [0, 90, 180, 270],
        "arrival_takt_by_speed": {0.03: 5.0, 0.05: 6.0, 0.08: 6.0},
        "service_time": 2.5,  # LV3: Aggressive service time
        "composition": {
            "source_total": 4,
            "target_counts": [2, 3],
            # The legacy split always leaves exactly three confounders across
            # belt and ghost, with one or two already on the ghost.
            "confounder_total": 3,
            "poses": {"inversion_probability": 0.0},
        },
    }
}

# ==========================================
# SUITE 2: D2S (Dynamic-to-Static)
# ==========================================
# [USER MISTAKE MARKER]: Removed hardcoded "_1" from LV1 base objects to allow dynamic scaling.
D2S_BASE_OBJS = ["school_bus", "mario", "thomas", "mini_fire_engine", "mini_roller", "porter"]
D2S = {
    1: {
        "targets": ["school_bus_upright", "mario_flat", "thomas_upright"],
        "shift_range": 0.05, "speed_noise": 0.05,
        "takt": 4.0,
        "language": "pick the objects from the conveyor and place them on the plate",
    },
    2: {
        "targets": ["school_bus_upright", "mario_flat", "thomas_upright", "mini_fire_engine_upright"],
        "all_objects": [f"{o}_upright" if o != "mario" else f"{o}_flat" for o in D2S_BASE_OBJS],
        "shift_range": 0.05, "speed_noise": 0.005, "yaw_range": 10,
        "forward_takt_by_speed": {0.03: 4, 0.05: 5, 0.08: 6},
        "composition": {
            # Legacy: four belt objects, three matching empty grooves, and
            # one unmatched source-only confounder.
            "source_total": 4,
            "goal_groove_count": 3,
            "grooves": {"match_pose": True},
            "large_bases": ["mini_fire_engine", "mini_roller"],
            "inlay_spacing": {"small": 0.16, "one_large": 0.20, "two_large": 0.25},
        },
    },
    3: {
        "targets": ["school_bus", "mario", "thomas", "mini_fire_engine"],
        "all_objects": D2S_BASE_OBJS,
        "shift_range": 0.04, "transpositions": [0, 90, 180, 270], "yaw_range": 10,
        "composition": {
            "target_count": 2,
            "source_confounder_counts": [1, 2],
            "destination_confounder_count": 1,
            "mutually_exclusive_huge_bases": ["mini_fire_engine", "mini_roller"],
            "poses": {
                "forced_flat_bases": ["mario"],
                "movable_poses": ["upright", "flat"],
            },
            "inlay_spacing": {"small": 0.15, "large": 0.22},
        },
    }
}

# ==========================================
# SUITE 3: S2D (Static-to-Dynamic)
# ==========================================
S2D_TARGETS = ["pump_bottle", "serum_bottle", "cream_bottle", "cream_jar"]
S2D_POSE_MATCH_TARGETS = {"pump_bottle", "serum_bottle", "cream_bottle"}
S2D = {
    1: {
        "targets": [f"{o}_flat" for o in S2D_TARGETS],
        "shift_range": 0.05, "speed_noise": 0.05,  # Relative ±5%
        "service_time": 1.5  
    },
    2: {
        # Keep the historical key name for generator compatibility.  In the
        # single-arm track each source already matches its destination pose.
        "mismatch_map": {
            "pump_bottle_flat": {"init": "pump_bottle_flat", "inlay": "pump_bottle_flat_groove"},
            "serum_bottle_flat": {"init": "serum_bottle_flat", "inlay": "serum_bottle_flat_groove"},
            "cream_bottle_flat": {"init": "cream_bottle_flat", "inlay": "cream_bottle_flat_groove"},
            "cream_jar_upright": {"init": "cream_jar_upright", "inlay": "cream_jar_upright_inlay"}
        },
        "all_objects": [f"{o}_flat" for o in S2D_TARGETS] + [f"{o}_upright" for o in S2D_TARGETS],
        "shift_range": 0.05, "speed_noise": 0.005, "yaw_range": 10,  # Absolute ±0.005 m/s
        "service_time": 2.0,
        "composition": {
            "target_count": 3,
            "source_confounder_counts": [1, 2],
        },
    },
    3: {
        "target_map": {
            "pump_bottle": {"inlay": "pump_bottle_upright_inlay", "pose": "upright"},
            "serum_bottle": {"inlay": "serum_bottle_flat_inlay", "pose": "flat"},
            "cream_bottle": {"inlay": "cream_bottle_flat_inlay", "pose": "flat"},
            "cream_jar": {"inlay": "cream_jar_upright_inlay", "pose": "upright"}
        },
        "all_objects": [f"{o}_flat" for o in S2D_TARGETS] + [f"{o}_upright" for o in S2D_TARGETS],
        "shift_range": 0.05, "speed_range": (0.03, 0.08), "yaw_range": 20, "transpositions": [0, 90, 180, 270],
        "service_time": 2.5,
        "composition": {
            "target_count": 3,
            "target_poses": ["upright", "flat"],
            "source_confounder_counts": [1, 2],
            "destination_confounder_min": 1,
            "destination_confounder_max": 2,
            "confounder_total_max": 3,
        },
    }
}


# The unsuffixed suites remain the single-arm-compatible benchmark.  The
# bimanual track preserves the earlier out-of-plane pose requirements without
# duplicating the three scene generators.
SINGLE_ARM = "single_arm"
BIMANUAL = "bimanual"
GENERATION_TRACKS = (SINGLE_ARM, BIMANUAL)

SUITE_DIRS_BY_TRACK = {
    SINGLE_ARM: {
        "D2D": "libero_industry_d2d",
        "D2S": "libero_industry_d2s",
        "S2D": "libero_industry_s2d",
    },
    BIMANUAL: {
        "D2D": "libero_industry_d2d_bimanual",
        "D2S": "libero_industry_d2s_bimanual",
        "S2D": "libero_industry_s2d_bimanual",
    },
}

BIMANUAL_D2D = copy.deepcopy(D2D)
BIMANUAL_D2D[2]["grooves"] = ["upright_groove", "v_groove"]
BIMANUAL_D2D[3]["all_objects"] = [
    f"{obj}_{pose}"
    for obj in D2D_BASE_OBJS
    for pose in ("upright", "inverted")
]
BIMANUAL_D2D[3]["composition"]["poses"]["inversion_probability"] = 0.25

BIMANUAL_D2S = copy.deepcopy(D2S)
BIMANUAL_D2S[2]["composition"]["grooves"] = {
    "default": ["upright_groove", "flat_groove", "v_groove"],
    "allowed_by_base": {
        "mario": ["flat_groove", "flat_groove", "v_groove"],
        "mini_fire_engine": ["flat_groove", "flat_groove", "flat_groove"],
    },
}

# The legacy map and bimanual configuration retain Mini Fire Engine.  The
# unsuffixed single-arm Level-3 suite uses the five remaining graspable toys.
D2S[3]["targets"] = [
    "school_bus",
    "mario",
    "thomas",
    "porter",
    "mini_roller",
]
D2S[3]["all_objects"] = list(D2S[3]["targets"])
D2S[3]["composition"]["mutually_exclusive_huge_bases"] = []

BIMANUAL_S2D = copy.deepcopy(S2D)
BIMANUAL_S2D[2]["mismatch_map"] = {
    "pump_bottle_flat": {
        "init": "pump_bottle_upright",
        "inlay": "pump_bottle_flat_groove",
    },
    "serum_bottle_flat": {
        "init": "serum_bottle_upright",
        "inlay": "serum_bottle_flat_groove",
    },
    "cream_bottle_flat": {
        "init": "cream_bottle_upright",
        "inlay": "cream_bottle_flat_groove",
    },
    "cream_jar_upright": {
        "init": "cream_jar_flat",
        "inlay": "cream_jar_flat_groove",
    },
}

_TRACK_CONFIGS = {
    SINGLE_ARM: {"D2D": D2D, "D2S": D2S, "S2D": S2D},
    BIMANUAL: {"D2D": BIMANUAL_D2D, "D2S": BIMANUAL_D2S, "S2D": BIMANUAL_S2D},
}

# Exact V-groove allocation from the pre-single-arm D2S Level-2 suite.
_BIMANUAL_D2S_LV2_V_GROOVE_BASES = {
    1: {"school_bus"},
    2: {"school_bus"},
    3: {"thomas", "mini_roller"},
    7: {"school_bus"},
    9: {"mini_roller"},
    10: {"thomas", "porter"},
    15: {"thomas"},
    18: {"thomas"},
    19: {"thomas", "porter"},
    22: {"porter"},
    23: {"school_bus"},
}


# (moving targets, prealigned target, source-only confounders).  Across all
# 24 tasks every remaining toy appears on the conveyor exactly 17 times;
# moving goals are split 10/10/10/9/9 and total goals 14/15/15/14/14.
_SINGLE_ARM_D2S_LV3_COMPOSITIONS = {
    1: (("school_bus", "mini_roller"), "mario", ("porter",)),
    2: (("mario", "porter"), "thomas", ("mini_roller",)),
    3: (("school_bus", "mario"), "porter", ("thomas", "mini_roller")),
    4: (("thomas", "porter"), "mini_roller", ("school_bus", "mario")),
    5: (("porter", "mini_roller"), "thomas", ("school_bus", "mario")),
    6: (("school_bus", "mario"), "porter", ("thomas", "mini_roller")),
    7: (("mario", "thomas"), "mini_roller", ("school_bus", "porter")),
    8: (("mario", "mini_roller"), "school_bus", ("thomas",)),
    9: (("school_bus", "mario"), "porter", ("mini_roller",)),
    10: (("mario", "thomas"), "mini_roller", ("school_bus", "porter")),
    11: (("porter", "mini_roller"), "school_bus", ("thomas",)),
    12: (("thomas", "mini_roller"), "mario", ("school_bus", "porter")),
    13: (("school_bus", "thomas"), "mini_roller", ("mario",)),
    14: (("mario", "porter"), "school_bus", ("thomas",)),
    15: (("thomas", "porter"), "mario", ("mini_roller",)),
    16: (("porter", "mini_roller"), "thomas", ("school_bus", "mario")),
    17: (("thomas", "mini_roller"), "school_bus", ("mario", "porter")),
    18: (("school_bus", "porter"), "mario", ("thomas", "mini_roller")),
    19: (("mario", "mini_roller"), "thomas", ("school_bus", "porter")),
    20: (("school_bus", "thomas"), "porter", ("mario", "mini_roller")),
    21: (("school_bus", "mini_roller"), "thomas", ("mario", "porter")),
    22: (("school_bus", "thomas"), "porter", ("mini_roller",)),
    23: (("mario", "thomas"), "mini_roller", ("porter",)),
    24: (("school_bus", "porter"), "mario", ("thomas",)),
}


def get_suite_config(suite, level, track=SINGLE_ARM):
    """Return one shared generator configuration for a benchmark track."""

    try:
        return _TRACK_CONFIGS[track][suite][level]
    except KeyError as error:
        raise ValueError(
            "Unknown generation track/suite/level: {!r}/{!r}/{!r}".format(
                track, suite, level
            )
        ) from error


def get_suite_dirs(track=SINGLE_ARM):
    try:
        return SUITE_DIRS_BY_TRACK[track]
    except KeyError as error:
        raise ValueError(
            "Unknown generation track {!r}; choose {}".format(
                track, ", ".join(GENERATION_TRACKS)
            )
        ) from error


def _strip_pose(asset):
    return re.sub(r"_(?:upright|flat|inverted)$", "", asset)


def _replace_asset(entry, old, new):
    for field, assets in entry.items():
        if isinstance(assets, list):
            entry[field] = [new if asset == old else asset for asset in assets]


def _asset_with_template_pose(base, template_asset):
    match = re.search(r"_(upright|flat|inverted)$", template_asset)
    if match is None:
        raise ValueError("Asset has no pose suffix: {}".format(template_asset))
    pose = "flat" if base == "mario" else match.group(1)
    return "{}_{}".format(base, pose)


def _single_arm_d2s_lv3_entry(task_name, entry):
    match = re.match(r"^D2S_LV3_(\d{3})_", task_name)
    if match is None:
        raise ValueError("Invalid D2S Level-3 task name: {}".format(task_name))
    ordinal = int(match.group(1))
    moving_bases, prealigned_base, confounder_bases = (
        _SINGLE_ARM_D2S_LV3_COMPOSITIONS[ordinal]
    )

    moving_templates = list(entry["target_objects"])
    for asset in entry["prealigned_objects"]:
        moving_templates.remove(asset)
    confounder_templates = list(entry["source_confounders"])
    if len(moving_templates) != len(moving_bases):
        raise ValueError("{} must have two moving targets".format(task_name))
    if len(confounder_templates) != len(confounder_bases):
        raise ValueError("{} changed its source-object count".format(task_name))
    if len(entry["prealigned_objects"]) != 1:
        raise ValueError("{} must have one prealigned target".format(task_name))

    moving_assets = [
        _asset_with_template_pose(base, template)
        for base, template in zip(moving_bases, moving_templates)
    ]
    confounder_assets = [
        _asset_with_template_pose(base, template)
        for base, template in zip(confounder_bases, confounder_templates)
    ]
    prealigned_asset = _asset_with_template_pose(
        prealigned_base, entry["prealigned_objects"][0]
    )
    entry["source_objects"] = moving_assets + confounder_assets
    entry["target_objects"] = moving_assets + [prealigned_asset]
    entry["source_confounders"] = confounder_assets
    entry["prealigned_objects"] = [prealigned_asset]


def object_map_entry_for_track(task_name, entry, track=SINGLE_ARM):
    """Return the legacy object-map entry adapted to one benchmark track."""

    get_suite_dirs(track)  # Validate before copying or changing the entry.
    adapted = copy.deepcopy(entry)

    if track == SINGLE_ARM and task_name.startswith("D2D_LV3_"):
        for field, assets in adapted.items():
            if isinstance(assets, list):
                adapted[field] = [
                    _strip_pose(asset) + "_upright"
                    if asset.endswith("_inverted")
                    else asset
                    for asset in assets
                ]

    if track == SINGLE_ARM and task_name.startswith("D2S_LV3_"):
        _single_arm_d2s_lv3_entry(task_name, adapted)

    if track == SINGLE_ARM and task_name.startswith("D2S_"):
        _replace_asset(
            adapted, "mini_fire_engine_flat", "mini_fire_engine_upright"
        )

    if task_name.startswith("S2D_LV2_"):
        initial_assets = set(adapted["target_objects"])
        if len(initial_assets) != 1:
            raise ValueError(
                "{} must have one S2D Level-2 target asset".format(task_name)
            )
        old = next(iter(initial_assets))
        matches = {
            target
            for cfg in (S2D[2], BIMANUAL_S2D[2])
            for target, spec in cfg["mismatch_map"].items()
            if spec["init"] == old
        }
        if len(matches) != 1:
            raise ValueError(
                "{} cannot resolve its S2D target from {}".format(
                    task_name, old
                )
            )
        target = next(iter(matches))
        replacement = get_suite_config("S2D", 2, track)["mismatch_map"][target][
            "init"
        ]
        _replace_asset(adapted, old, replacement)

    if track == SINGLE_ARM and task_name.startswith("S2D_LV3_"):
        target_bases = {_strip_pose(asset) for asset in adapted["target_objects"]}
        if len(target_bases) != 1:
            raise ValueError(
                "{} must have one S2D Level-3 target base".format(task_name)
            )
        target_base = next(iter(target_bases))
        if target_base in S2D_POSE_MATCH_TARGETS:
            target_pose = S2D[3]["target_map"][target_base]["pose"]
            replacement = "{}_{}".format(target_base, target_pose)
            for pose in ("flat", "upright"):
                _replace_asset(
                    adapted, "{}_{}".format(target_base, pose), replacement
                )

    if task_name.startswith("D2S_LV2_"):
        v_groove_bases = set()
        if track == BIMANUAL:
            match = re.match(r"^D2S_LV2_(\d{3})_", task_name)
            if match is None:
                raise ValueError("Invalid D2S Level-2 task name: {}".format(task_name))
            v_groove_bases = _BIMANUAL_D2S_LV2_V_GROOVE_BASES.get(
                int(match.group(1)), set()
            )
        adapted["target_grooves"] = [
            "{}_v_groove".format(_strip_pose(asset))
            if _strip_pose(asset) in v_groove_bases
            else "{}_groove".format(asset)
            for asset in adapted["target_objects"]
        ]

    return adapted
