import math
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import configs
import perturbations
import task_composition as composition
import utils
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
from libero.libero.utils.mu_utils import InitialSceneTemplates, MU_DICT, SCENE_DICT
from libero.libero.utils.task_generation_utils import register_task_info
import libero.libero.utils.task_generation_utils as task_utils


QUICK_TEST = True
# The default 1.0 m industry workbench places Aubo at -0.16 - 1.0 / 2.
spatial_validator = perturbations.DynamicSpatialValidator(
    configs.ASSET_DIR, base_pos=(-0.66, 0.0, 0.912)
)
# The belt and static destination use one common X shift.  The fixed 24 cm
# nominal gap is certified from the measured belt/inlay envelopes below.
D2S_SHARED_X_PERTURBATION = 0.05
D2S_LV12_BELT_ANCHOR = (0.0, 0.0)
D2S_LV1_TRAY_ANCHOR = (-0.24, 0.0)
D2S_STATIC_Y_LINE_X = -0.24
D2S_LV3_LOCAL_JITTER = 0.005
D2S_LV3_X_PERTURBATION = 0.02
# ``rot_deg`` is relative to the established Level-1/2 D2S belt pose. Thus
# 0/180 keep the Y-parallel side layout; 90/270 use the X-parallel triangle.
D2S_LV3_MIN_BELT_Y = 0.20
D2S_LV3_D2D_BELT_ANCHOR = (-0.105, 0.40)
D2S_LV3_BELT_ANCHORS = {
    0: D2S_LV12_BELT_ANCHOR,
    90: D2S_LV3_D2D_BELT_ANCHOR,
    180: D2S_LV12_BELT_ANCHOR,
    270: D2S_LV3_D2D_BELT_ANCHOR,
}


def _rotated_half_extents(asset, yaw):
    half_x, half_y = spatial_validator._parse_xml_extents(asset)
    cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (
        cosine * half_x + sine * half_y,
        sine * half_x + cosine * half_y,
    )


def _sample_lv2_y_line(cfg, fixture_assets):
    """Place three measured groove envelopes at negative X along Y."""
    yaws = [
        random.choice((0.0, math.pi / 2.0))
        + math.radians(random.uniform(-cfg["yaw_range"], cfg["yaw_range"]))
        for _ in fixture_assets
    ]
    extents = [
        _rotated_half_extents(asset, yaw)
        for asset, yaw in zip(fixture_assets, yaws)
    ]
    y_half_extents = [extent[1] for extent in extents]

    # Three slots share one fixture translation and retain independent local
    # perturbations.  The cap reserves both forms of motion at the workspace
    # boundary and between every neighboring groove.
    available_half_span = min(
        perturbations.S2D_GEOMETRIC_CENTER[1] - perturbations.S2D_BOUNDS["y_min"],
        perturbations.S2D_BOUNDS["y_max"] - perturbations.S2D_GEOMETRIC_CENTER[1],
    )
    local_cap = min(
        cfg["shift_range"],
        (available_half_span - sum(y_half_extents) - perturbations.LAYOUT_CLEARANCE) / 4.0,
    )
    if local_cap <= 0.0:
        raise ValueError("D2S grooves exceed the available negative-X Y-line")

    line_span = (
        2.0 * sum(y_half_extents)
        + 2.0 * perturbations.LAYOUT_CLEARANCE
        + 4.0 * local_cap
    )
    current_y = perturbations.S2D_GEOMETRIC_CENTER[1] - line_span / 2.0
    positions = []
    for index, y_half_extent in enumerate(y_half_extents):
        current_y += y_half_extent
        positions.append((D2S_STATIC_Y_LINE_X, current_y))
        current_y += y_half_extent
        if index < len(y_half_extents) - 1:
            current_y += perturbations.LAYOUT_CLEARANCE + 2.0 * local_cap
    return positions, yaws, local_cap, extents


def _validate_lv2_y_line(belt_pos, fixture_positions, fixture_assets, fixture_yaws):
    belt_half_width, _ = spatial_validator.d2d_lv3_cross_track_half_widths()
    extents = [
        _rotated_half_extents(asset, yaw)
        for asset, yaw in zip(fixture_assets, fixture_yaws)
    ]
    for index, ((x_pos, y_pos), (half_x, half_y)) in enumerate(
        zip(fixture_positions, extents)
    ):
        if (
            x_pos - half_x < perturbations.S2D_BOUNDS["x_min"]
            or x_pos + half_x > perturbations.S2D_BOUNDS["x_max"]
            or y_pos - half_y < perturbations.S2D_BOUNDS["y_min"]
            or y_pos + half_y > perturbations.S2D_BOUNDS["y_max"]
        ):
            return perturbations.LayoutValidationResult(
                False, "groove {} exceeds spatial workspace bounds".format(index)
            )
        if not spatial_validator._circle_is_kinematically_safe(
            x_pos, y_pos, math.hypot(half_x, half_y)
        ):
            return perturbations.LayoutValidationResult(
                False, "groove {} exceeds kinematic workspace bounds".format(index)
            )
        if belt_pos[0] - belt_half_width <= x_pos + half_x + perturbations.LAYOUT_CLEARANCE:
            return perturbations.LayoutValidationResult(
                False, "groove {} collides with the conveyor belt".format(index)
            )

    for first in range(len(fixture_positions)):
        for second in range(first + 1, len(fixture_positions)):
            first_x, first_y = fixture_positions[first]
            second_x, second_y = fixture_positions[second]
            first_half_x, first_half_y = extents[first]
            second_half_x, second_half_y = extents[second]
            x_overlaps = abs(first_x - second_x) < first_half_x + second_half_x
            y_separated = abs(first_y - second_y) >= (
                first_half_y + second_half_y + perturbations.LAYOUT_CLEARANCE
            )
            if x_overlaps and not y_separated:
                return perturbations.LayoutValidationResult(
                    False, "D2S grooves {} and {} collide".format(first, second)
                )
    return perturbations.LayoutValidationResult(True)


def _sample_lv3_y_line(cfg, extents):
    """Pack a side-of-belt inlay line along the full safe Y span."""
    local_cap = D2S_LV3_LOCAL_JITTER
    x_half_extent = max(half_x for half_x, _ in extents)
    y_half_extents = [half_y for _, half_y in extents]
    required_height = (
        2.0 * sum(y_half_extents)
        + 2.0 * perturbations.LAYOUT_CLEARANCE
        + 6.0 * local_cap
    )
    shared_cap = min(
        cfg["shift_range"],
        D2S_STATIC_Y_LINE_X - x_half_extent - local_cap - perturbations.S2D_BOUNDS["x_min"],
        perturbations.S2D_BOUNDS["x_max"] - D2S_STATIC_Y_LINE_X - x_half_extent - local_cap,
        (
            perturbations.S2D_BOUNDS["y_max"]
            - perturbations.S2D_BOUNDS["y_min"]
            - required_height
        ) / 2.0,
    )
    if shared_cap < 0.0:
        raise ValueError("D2S Level-3 Y-line cannot fit safely")

    cursor = perturbations.S2D_BOUNDS["y_min"] + shared_cap + local_cap
    positions = []
    for index, half_y in enumerate(y_half_extents):
        cursor += half_y
        positions.append((D2S_STATIC_Y_LINE_X, cursor))
        cursor += half_y
        if index < len(y_half_extents) - 1:
            cursor += perturbations.LAYOUT_CLEARANCE + 2.0 * local_cap
    return positions, shared_cap


def _sample_lv3_x_belt_triangle(cfg, rot_deg, extents):
    """Pack a measured 2--1 formation below an X-parallel belt."""
    if rot_deg not in (90, 270):
        raise ValueError("D2S Level-3 triangle requires a 90/270 transpose")

    local_cap = D2S_LV3_LOCAL_JITTER
    separation_guard = perturbations.LAYOUT_CLEARANCE + 2.0 * local_cap
    half_x = [extent[0] for extent in extents]
    half_y = [extent[1] for extent in extents]

    # target_init_0 is the lower apex. The other two targets share a compact
    # upper row; 90 degrees mirrors their ordering without changing the side
    # of the belt on which the formation is placed.
    left_index, right_index = ((1, 2) if rot_deg == 270 else (2, 1))
    base_width = (
        2.0 * half_x[left_index]
        + separation_guard
        + 2.0 * half_x[right_index]
    )
    relative_x = [0.0, 0.0, 0.0]
    relative_x[left_index] = -base_width / 2.0 + half_x[left_index]
    relative_x[right_index] = base_width / 2.0 - half_x[right_index]

    row_gap = max(
        half_y[0] + half_y[left_index],
        half_y[0] + half_y[right_index],
    ) + separation_guard
    apex_y = (
        perturbations.S2D_BOUNDS["y_min"]
        + cfg["shift_range"]
        + local_cap
        + half_y[0]
    )
    base_y = apex_y + row_gap

    min_edge = min(x_pos - half_x[index] for index, x_pos in enumerate(relative_x))
    max_edge = max(x_pos + half_x[index] for index, x_pos in enumerate(relative_x))
    envelope_midpoint = (min_edge + max_edge) / 2.0
    x_offset = perturbations.S2D_GEOMETRIC_CENTER[0] - envelope_midpoint
    positions = [
        (
            relative_x[index] + x_offset,
            apex_y if index == 0 else base_y,
        )
        for index in range(3)
    ]

    # Reserve the full configured belt-yaw range. Global XY perturbation is
    # shared by the belt and inlays, so only the independent local jitter must
    # be added to their relative separation proof.
    belt_half_width, _ = spatial_validator.d2d_lv3_cross_track_half_widths()
    belt_x, reference_belt_y = D2S_LV3_BELT_ANCHORS[rot_deg]
    belt_y = max(D2S_LV3_MIN_BELT_Y, reference_belt_y)
    yaw_limit = math.radians(cfg["yaw_range"])
    for yaw_deviation in (-yaw_limit, 0.0, yaw_limit):
        normal_x = -math.sin(yaw_deviation)
        normal_y = math.cos(yaw_deviation)
        for (x_pos, y_pos), (half_x_extent, half_y_extent) in zip(
            positions, extents
        ):
            projected_half_extent = (
                abs(normal_x) * half_x_extent
                + abs(normal_y) * half_y_extent
            )
            jitter_guard = local_cap * (abs(normal_x) + abs(normal_y))
            required_y = y_pos + (
                belt_half_width
                + projected_half_extent
                + perturbations.LAYOUT_CLEARANCE
                + jitter_guard
                - (belt_x - x_pos) * normal_x
            ) / normal_y
            belt_y = max(belt_y, required_y)

    shared_x_cap = D2S_LV3_X_PERTURBATION
    shared_y_cap = cfg["shift_range"]
    for (x_pos, y_pos), (half_x_extent, half_y_extent) in zip(positions, extents):
        shared_x_cap = min(
            shared_x_cap,
            x_pos - half_x_extent - local_cap - perturbations.S2D_BOUNDS["x_min"],
            perturbations.S2D_BOUNDS["x_max"] - x_pos - half_x_extent - local_cap,
        )
        shared_y_cap = min(
            shared_y_cap,
            y_pos - half_y_extent - local_cap - perturbations.S2D_BOUNDS["y_min"],
            perturbations.S2D_BOUNDS["y_max"] - y_pos - half_y_extent - local_cap,
        )
    if shared_x_cap < 0.0 or shared_y_cap < 0.0:
        raise ValueError("D2S Level-3 triangle exceeds spatial workspace bounds")
    return positions, shared_x_cap, shared_y_cap, (belt_x, belt_y)


def _sample_lv3_formation(cfg, rot_deg, fixture_assets):
    """Choose Level-3's formation from its physical loading transpose."""
    yaws = [
        # Flat inlays use their local Y axis as the long axis.  Rotate that
        # axis with the Level-3 transposition while preserving yaw noise.
        math.radians(90.0 + rot_deg + random.uniform(-cfg["yaw_range"], cfg["yaw_range"]))
        for _ in fixture_assets
    ]
    extents = [
        _rotated_half_extents(asset, yaw)
        for asset, yaw in zip(fixture_assets, yaws)
    ]
    if rot_deg in (0, 180):
        positions, shared_cap = _sample_lv3_y_line(cfg, extents)
        return (
            positions,
            yaws,
            D2S_LV3_X_PERTURBATION,
            shared_cap,
            0.0,
            D2S_LV3_LOCAL_JITTER,
            D2S_LV3_BELT_ANCHORS[rot_deg],
        )

    positions, shared_x_cap, shared_y_cap, belt_anchor = _sample_lv3_x_belt_triangle(
        cfg, rot_deg, extents
    )
    # The 90/270-degree physical belt has its width along Y, so it keeps its
    # measured triangle. Its X range remains certified per formation instead
    # of using the side-of-belt Y-line applied at 0/180 degrees.
    return (
        positions,
        yaws,
        shared_x_cap,
        shared_y_cap,
        D2S_LV3_LOCAL_JITTER,
        D2S_LV3_LOCAL_JITTER,
        belt_anchor,
    )


def _validate_lv3_triangle(
    belt_pos,
    rot_deg,
    fixture_positions,
    fixture_assets,
    fixture_yaws,
    belt_yaw_deg=None,
):
    belt_half_width, _ = spatial_validator.d2d_lv3_cross_track_half_widths()
    # ``rot_deg`` is the normalized physical and emitted belt yaw. The 2--1
    # formation is below an X-oriented belt; the Y-line is left of a
    # Y-oriented belt.
    if belt_yaw_deg is None:
        belt_yaw_deg = rot_deg
    if rot_deg in (90, 270):
        apex, second, third = fixture_positions
        if not apex[1] < min(second[1], third[1]):
            return perturbations.LayoutValidationResult(
                False, "target_init_0 is not the lower triangle apex"
            )
        if abs(second[1] - third[1]) > 2.0 * D2S_LV3_LOCAL_JITTER + 1e-9:
            return perturbations.LayoutValidationResult(
                False, "target_init_1/2 do not share the triangle base row"
            )
        mirrored_order_is_valid = (
            second[0] < third[0] if rot_deg == 270 else second[0] > third[0]
        )
        if not mirrored_order_is_valid:
            return perturbations.LayoutValidationResult(
                False, "target_init_1/2 do not match the transpose direction"
            )
    base_normal = (1.0, 0.0) if rot_deg in (0, 180) else (0.0, 1.0)
    # The cardinal formation is chosen from ``rot_deg``.  Rotate its collision
    # normal by the independently sampled belt-yaw deviation before measuring
    # separation, so a noisy belt cannot invalidate a nominally safe layout.
    yaw_deviation = math.radians(belt_yaw_deg - rot_deg)
    cosine, sine = math.cos(yaw_deviation), math.sin(yaw_deviation)
    normal = (
        cosine * base_normal[0] - sine * base_normal[1],
        sine * base_normal[0] + cosine * base_normal[1],
    )
    extents = [
        _rotated_half_extents(asset, yaw)
        for asset, yaw in zip(fixture_assets, fixture_yaws)
    ]
    for index, ((x_pos, y_pos), (half_x, half_y)) in enumerate(
        zip(fixture_positions, extents)
    ):
        if (
            x_pos - half_x < perturbations.S2D_BOUNDS["x_min"]
            or x_pos + half_x > perturbations.S2D_BOUNDS["x_max"]
            or y_pos - half_y < perturbations.S2D_BOUNDS["y_min"]
            or y_pos + half_y > perturbations.S2D_BOUNDS["y_max"]
        ):
            return perturbations.LayoutValidationResult(False, "inlay exceeds spatial workspace bounds")
        if not spatial_validator._circle_is_kinematically_safe(
            x_pos, y_pos, math.hypot(half_x, half_y)
        ):
            return perturbations.LayoutValidationResult(False, "inlay exceeds kinematic workspace bounds")
        static_normal_half_extent = abs(normal[0]) * half_x + abs(normal[1]) * half_y
        separation = (belt_pos[0] - x_pos) * normal[0] + (belt_pos[1] - y_pos) * normal[1]
        if separation < belt_half_width + static_normal_half_extent + perturbations.LAYOUT_CLEARANCE:
            return perturbations.LayoutValidationResult(False, "inlay collides with conveyor belt")

    for first in range(3):
        for second in range(first + 1, 3):
            dx = abs(fixture_positions[first][0] - fixture_positions[second][0])
            dy = abs(fixture_positions[first][1] - fixture_positions[second][1])
            if (
                dx < extents[first][0] + extents[second][0] + perturbations.LAYOUT_CLEARANCE
                and dy < extents[first][1] + extents[second][1] + perturbations.LAYOUT_CLEARANCE
            ):
                return perturbations.LayoutValidationResult(False, "D2S inlays collide")
    return perturbations.LayoutValidationResult(True)


def _validate_lv1_tray(belt_pos, tray_pos, tray_asset, tray_yaw):
    """Validate the fixed-yaw toy tray with its real AABB, not its circle.

    A circumcircle is still used for reachability, but it is unnecessarily
    conservative at the rectangular training boundary for an unrotated tray.
    The exact AABB preserves the requested shared +/-5 cm X range.
    """
    tray_half_x, tray_half_y = _rotated_half_extents(tray_asset, tray_yaw)
    if (
        tray_pos[0] - tray_half_x < perturbations.S2D_BOUNDS["x_min"]
        or tray_pos[0] + tray_half_x > perturbations.S2D_BOUNDS["x_max"]
        or tray_pos[1] - tray_half_y < perturbations.S2D_BOUNDS["y_min"]
        or tray_pos[1] + tray_half_y > perturbations.S2D_BOUNDS["y_max"]
    ):
        return perturbations.LayoutValidationResult(
            False, "toy tray exceeds spatial workspace bounds"
        )
    if not spatial_validator._circle_is_kinematically_safe(
        tray_pos[0], tray_pos[1], spatial_validator.circumradius(tray_asset)
    ):
        return perturbations.LayoutValidationResult(
            False, "toy tray exceeds kinematic workspace bounds"
        )

    belt_half_width, _ = spatial_validator.d2d_lv3_cross_track_half_widths()
    if belt_pos[0] - belt_half_width <= (
        tray_pos[0] + tray_half_x + perturbations.LAYOUT_CLEARANCE
    ):
        return perturbations.LayoutValidationResult(
            False, "toy tray collides with the conveyor belt"
        )
    return perturbations.LayoutValidationResult(True)


def _sample_static_formation(cfg, level, rot_deg, fixture_assets):
    """Return a certified D2S three-fixture formation and jitter limits."""
    if len(fixture_assets) == 1:
        return (
            [D2S_LV1_TRAY_ANCHOR],
            [0.0],
            D2S_SHARED_X_PERTURBATION,
            cfg["shift_range"],
            0.0,
            0.0,
            D2S_LV12_BELT_ANCHOR,
        )

    if len(fixture_assets) != 3:
        raise ValueError("D2S static layouts require one plate or three inlays.")

    if level == 2:
        positions, yaws, local_cap, _ = _sample_lv2_y_line(cfg, fixture_assets)
        return (
            positions,
            yaws,
            D2S_SHARED_X_PERTURBATION,
            local_cap,
            0.0,
            local_cap,
            D2S_LV12_BELT_ANCHOR,
        )
    return _sample_lv3_formation(cfg, rot_deg, fixture_assets)


def _validate_d2s_candidate(
    level,
    layout,
    belt_pos,
    belt_yaw_deg,
    fixture_positions,
    fixture_assets,
    fixture_yaws,
    rot_deg,
):
    """Apply the same centre/inlay checks at every D2S level."""
    centre_result = spatial_validator.validate_conveyor_centre_region(layout["belt"])
    if not centre_result:
        return centre_result
    if level == 1:
        return _validate_lv1_tray(
            belt_pos,
            fixture_positions[0],
            fixture_assets[0],
            fixture_yaws[0],
        )
    if level == 2:
        return _validate_lv2_y_line(
            belt_pos, fixture_positions, fixture_assets, fixture_yaws
        )
    if level == 3:
        return _validate_lv3_triangle(
            belt_pos,
            rot_deg,
            fixture_positions,
            fixture_assets,
            fixture_yaws,
            belt_yaw_deg=belt_yaw_deg,
        )
    return spatial_validator.validate_d2s_layout(layout)


def _d2s_safety_circles(level, belt_pos, fixture_positions, fixture_assets, fixture_yaws):
    """Build exact reach and spatial envelopes for a common negative-X move."""
    kinematic = [("conveyor centre_region", belt_pos, 0.0)]
    spatial = []
    for index, (position, asset, yaw) in enumerate(
        zip(fixture_positions, fixture_assets, fixture_yaws)
    ):
        if level == 1:
            half_x = half_y = spatial_validator.circumradius(asset)
        else:
            half_x, half_y = _rotated_half_extents(asset, yaw)
        kinematic.append(
            ("static fixture {}".format(index + 1), position, math.hypot(half_x, half_y))
        )
        spatial.append(
            ("static fixture {}".format(index + 1), position, half_x, half_y)
        )
    return kinematic, spatial


def generate_safe_d2s_layout(
    cfg,
    level,
    fixture_assets,
    max_attempts=150,
    transpose_degrees=None,
):
    """Sample one complete D2S layout while preserving its scheduled transpose."""
    last_reason = "no candidate was sampled"
    transposes = tuple(transpose_degrees or ())
    if level == 3 and transpose_degrees is not None and len(transposes) != 1:
        raise ValueError("D2S Level 3 requires exactly one scheduled transpose")
    for attempt in range(max_attempts):
        if level == 3:
            rot_deg = (
                transposes[0]
                if transpose_degrees is not None
                else random.choice(cfg["transpositions"])
            )
        else:
            rot_deg = 0

        try:
            (
                base_positions,
                fixture_yaws,
                shared_x_cap,
                shared_y_cap,
                local_x_cap,
                local_y_cap,
                belt_anchor,
            ) = _sample_static_formation(cfg, level, rot_deg, fixture_assets)
        except ValueError as error:
            last_reason = str(error)
            continue

        # The belt is a virtual transport surface, so D2S validates its
        # centre only for reachability. Static fixtures are packed directly in
        # the bounded workspace instead of treating the whole group as a
        # physical counterpart to the unbounded belt.
        shift_x = random.uniform(-shared_x_cap, shared_x_cap)
        shift_y = random.uniform(-shared_y_cap, shared_y_cap)
        belt_yaw = float(rot_deg)
        # Carry the configured yaw perturbation through to ``belt_init``
        # instead of reconstructing a fixed yaw when serializing the BDDL.
        if level in (2, 3):
            belt_yaw += random.uniform(-cfg["yaw_range"], cfg["yaw_range"])
        belt_pos = (belt_anchor[0] + shift_x, belt_anchor[1] + shift_y)
        fixture_positions = [
            (
                base_x + shift_x + random.uniform(-local_x_cap, local_x_cap),
                base_y + shift_y + random.uniform(-local_y_cap, local_y_cap),
            )
            for base_x, base_y in base_positions
        ]
        for correction_attempt in range(2):
            static_fixtures = {
                "static_{}".format(index): {
                    "pos": position,
                    "yaw_deg": math.degrees(fixture_yaws[index]),
                    "xml_name": fixture_assets[index],
                    "safety_radius": spatial_validator.circumradius(fixture_assets[index]),
                }
                for index, position in enumerate(fixture_positions)
            }
            layout = {
                "belt": {
                    # Keep the BDDL root anchor unchanged; centre_region retains
                    # its asset-defined 2 cm local offset.
                    "pos": belt_pos,
                    "yaw_deg": belt_yaw,
                    "xml_name": "conveyor_belt",
                    "safety_radius": 0.0,
                    "skip_spatial_bounds": True,
                    "skip_kinematic_bounds": True,
                },
                "static_fixtures": static_fixtures,
            }
            result = _validate_d2s_candidate(
                level,
                layout,
                belt_pos,
                belt_yaw,
                fixture_positions,
                fixture_assets,
                fixture_yaws,
                rot_deg,
            )
            if result:
                return (
                    shift_x,
                    shift_y,
                    rot_deg,
                    belt_pos,
                    perturbations.S2D_GEOMETRIC_CENTER,
                    fixture_positions,
                    fixture_yaws,
                    belt_yaw,
                )
            last_reason = result.reason
            if correction_attempt or "workspace bounds" not in result.reason:
                break
            kinematic_circles, spatial_circles = _d2s_safety_circles(
                level,
                belt_pos,
                fixture_positions,
                fixture_assets,
                fixture_yaws,
            )
            correction_x = spatial_validator.negative_x_translation_to_safe_region(
                kinematic_circles, spatial_circles
            )
            if correction_x is None or correction_x >= -1e-9:
                break
            # Keep the sampled yaw, Y shift, and local formation intact; only
            # a common negative-X translation is allowed to repair reach.
            shift_x += correction_x
            belt_pos = (belt_pos[0] + correction_x, belt_pos[1])
            fixture_positions = [
                (position[0] + correction_x, position[1])
                for position in fixture_positions
            ]
    raise ValueError(
        "Failed to find bounded D2S layout after {} attempts: {}".format(
            max_attempts, last_reason
        )
    )


def _sample_composition(cfg, level, target, object_map_entry=None):
    if object_map_entry is not None:
        source_assets = list(object_map_entry["source_objects"])
        target_assets = list(object_map_entry["target_objects"])
        if level == 1:
            return {
                "source_assets": source_assets,
                "goal_assets": target_assets,
                "groove_assets": [],
                "destination_confounder_assets": [],
            }
        if level == 2:
            expected_grooves = [f"{asset}_groove" for asset in target_assets]
            groove_assets = object_map_entry.get(
                "target_grooves", expected_grooves
            )
            if len(groove_assets) != len(target_assets):
                raise ValueError("D2S Level-2 target objects and grooves must have equal lengths")
            if (
                cfg["composition"]["grooves"].get("match_pose")
                and list(groove_assets) != expected_grooves
            ):
                raise ValueError(
                    "D2S Level-2 grooves must match each target object's pose"
                )
            return {
                "source_assets": source_assets,
                "goal_assets": target_assets,
                "groove_assets": list(groove_assets),
                "destination_confounder_assets": [],
            }
        prealigned_assets = list(object_map_entry["prealigned_objects"])
        moving_target_assets = composition.remove_assets(
            target_assets, prealigned_assets
        )
        target_bases = [
            composition.strip_pose(asset) for asset in moving_target_assets
        ]
        destination_bases = [
            composition.strip_pose(asset) for asset in prealigned_assets
        ]
        return {
            "source_assets": source_assets,
            "target_bases": target_bases,
            "destination_bases": destination_bases,
            "target_assets": moving_target_assets,
            "source_confounder_assets": composition.remove_assets(
                source_assets, moving_target_assets
            ),
            "destination_confounder_assets": prealigned_assets,
            "target_inlays": [
                composition.d2s_level3_fixture(asset)
                for asset in moving_target_assets
            ],
            "destination_inlays": [
                composition.d2s_level3_fixture(asset)
                for asset in prealigned_assets
            ],
        }
    if level == 1:
        return {
            "source_assets": list(cfg["targets"]),
            "goal_assets": list(cfg["targets"]),
            "groove_assets": [],
            "destination_confounder_assets": [],
        }
    if level == 2:
        return composition.sample_d2s_level2(cfg, target)
    return composition.sample_d2s_level3(cfg, target)


def build_and_register_d2s(
    task_id,
    level,
    base_speed,
    target=None,
    object_map_entry=None,
    transpose_degrees=None,
    track=configs.SINGLE_ARM,
):
    """Register one D2S scene.

    The staged 216-task workflow always supplies ``object_map_entry``.  The
    map-free branch exists only for legacy, exploratory script runs and is
    not a benchmark-generation path.
    """
    cfg = configs.get_suite_config("D2S", level, track)
    scene_composition = _sample_composition(cfg, level, target, object_map_entry)
    source_assets = scene_composition["source_assets"]
    if level == 3:
        # Pre-aligned objects own the canonical ``_1`` instance. A same-asset
        # belt object is a source confounder and receives ``_2``.
        destination_instances, destination_counts = composition.named_instances(
            scene_composition["destination_confounder_assets"]
        )
        source_instances, _ = composition.named_instances(
            source_assets, destination_counts
        )
    else:
        source_instances, source_counts = composition.named_instances(source_assets)
        destination_instances, _ = composition.named_instances(
            scene_composition["destination_confounder_assets"], source_counts
        )
    speed, takt, offset = perturbations.calc_belt_timing(
        base_speed, len(source_assets), cfg.get("speed_noise", 0.0)
    )
    if level == 1:
        # The 1.5 s offset already includes the loading / startup lead time.
        takt = cfg["takt"]
    elif level == 2 and base_speed > 0:
        takt = cfg["forward_takt_by_speed"][round(base_speed, 2)]
    fixture_assets = ["industry_workbench", "conveyor_belt"]
    if level == 1:
        fixture_assets.append("toy_tray")
        fixture_instances = ["toy_tray_1"]
    elif level == 2:
        fixture_assets.extend(scene_composition["groove_assets"])
        fixture_instances, _ = composition.named_instances(scene_composition["groove_assets"])
    else:
        inlay_assets = list(
            dict.fromkeys(
                scene_composition["target_inlays"]
                + scene_composition["destination_inlays"]
            )
        )
        fixture_assets.extend(inlay_assets)
        fixture_instances, _ = composition.named_instances(inlay_assets)
        inlay_instance_by_asset = dict(zip(inlay_assets, fixture_instances))
        target_inlays = [
            inlay_instance_by_asset[asset]
            for asset in scene_composition["target_inlays"]
        ]
        destination_inlay = inlay_instance_by_asset[
            scene_composition["destination_inlays"][0]
        ]
        random.shuffle(fixture_instances)

    # The static formation follows the fixture instances after Level-3's
    # deliberate shuffle, so every generated position is paired with the
    # correct concrete asset and goal region.
    static_fixture_assets = [composition.instance_asset(name) for name in fixture_instances]
    # Level 1's object-map order is its explicit conveyor arrival order.  It
    # uses the three nearest slots so the final arrival agrees with the
    # three-object takt horizon.  Levels 2 and 3 retain their four-slot
    # randomization.
    source_slots = (
        [1, 2, 3]
        if level == 1
        else random.sample([1, 2, 3, 4], len(source_instances))
    )
    try:
        (
            _,
            _,
            rot_deg,
            belt_pos,
            _,
            fixture_positions,
            fixture_yaws,
            belt_yaw_deg,
        ) = generate_safe_d2s_layout(
            cfg,
            level,
            static_fixture_assets,
            transpose_degrees=transpose_degrees,
        )
    except ValueError as error:
        return None, None, str(error), None

    scene_name = "D2S_LV{}_{}".format(level, str(task_id).zfill(3))

    if level == 3:
        target_instances = composition.mapped_instances(
            source_instances, scene_composition["target_assets"]
        )
    else:
        source_instance_by_asset = {
            composition.instance_asset(instance_name): instance_name
            for instance_name in source_instances
        }

    class D2SScene(InitialSceneTemplates):
        def __init__(self):
            super().__init__(
                "industry_workbench",
                composition.asset_counts(fixture_assets),
                composition.asset_counts(source_assets + scene_composition["destination_confounder_assets"]),
            )

        def define_regions(self):
            belt_yaw = math.radians(belt_yaw_deg)
            self.regions.update(
                self.get_region_dict(
                    belt_pos,
                    "belt_init",
                    self.workspace_name,
                    0.001,
                    (belt_yaw, belt_yaw),
                )
            )
            for index, (position, yaw) in enumerate(zip(fixture_positions, fixture_yaws)):
                self.regions.update(
                    self.get_region_dict(
                        position,
                        "target_init_{}".format(index),
                        self.workspace_name,
                        0.001,
                        (yaw, yaw),
                    )
                )
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        def init_states(self):
            states = [
                ("On", "conveyor_belt_1", "industry_workbench_belt_init"),
                ("AtSpeed", "conveyor_belt_1", str(speed), "steady", "takt_{}".format(takt), "offset_{}".format(offset)),
            ]
            for index, fixture_instance in enumerate(fixture_instances):
                states.append(("On", fixture_instance, "industry_workbench_target_init_{}".format(index)))
            if level == 3:
                # D2S Level 3's one destination confounder is intentionally
                # pre-aligned and remains an already-satisfied goal.
                states.append(("AlignedOn", destination_instances[0], "{}_goal_region".format(destination_inlay)))

            prefix = "tail_region" if speed < 0 else "offset_region"
            for instance_name, slot in zip(source_instances, source_slots):
                states.append(("On", instance_name, "conveyor_belt_1_{}_{}".format(prefix, slot)))
            return states

    MU_DICT[scene_name.lower()] = D2SScene
    SCENE_DICT["industry_workbench"] = [D2SScene]

    if level == 1:
        goals = [
            ("ReleasedOn", instance_name, "toy_tray_1_goal_region")
            for instance_name in source_instances
        ]
    elif level == 2:
        goals = [
            ("AlignedOn", source_instance_by_asset[asset], "{}_goal_region".format(inlay_instance))
            for asset, inlay_instance in zip(scene_composition["goal_assets"], fixture_instances)
        ]
    else:
        goals = [
            ("AlignedOn", instance_name, "{}_goal_region".format(inlay_instance))
            for instance_name, inlay_instance in zip(target_instances, target_inlays)
        ]
        goals.append(("AlignedOn", destination_instances[0], "{}_goal_region".format(destination_inlay)))
    register_task_info(
        cfg.get("language", "place toys"),
        scene_name=scene_name,
        objects_of_interest=[state[1] for state in goals],
        goal_states=goals,
    )
    metadata = {
        "Target Instances": str(
            source_instances if level == 1 else (
                [source_instance_by_asset[asset] for asset in scene_composition["goal_assets"]]
                if level == 2 else target_instances
            )
        ),
        "Source Confounders": str(
            [] if level == 1 else (
                list(set(source_instances) - set(source_instance_by_asset[asset] for asset in scene_composition["goal_assets"]))
                if level == 2 else list(set(source_instances) - set(target_instances))
            )
        ),
        "Destination Confounders": str(destination_instances),
    }
    if level == 3:
        metadata["Transpose Degrees"] = rot_deg
    return scene_name, ("random" if level > 1 else "uniform"), None, metadata


def _targets_for_level(level, quick_test):
    if level == 1:
        targets = [None]
    else:
        targets = configs.D2S[level]["targets"]
    return targets[:1] if quick_test else targets


def main():
    os.makedirs(configs.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(configs.VIDEO_OUT_DIR, exist_ok=True)
    tracker = utils.TaskTracker(
        configs.EXCEL_PATH, ["Task ID", "Suite", "Level", "Status", "Video", "Error"]
    )
    print("RUNNING D2S GENERATOR (Quick Test: {})".format(QUICK_TEST))
    task_id = 1
    for level in (1, 2, 3):
        speeds = configs.BASE_SPEEDS[:1] if QUICK_TEST else configs.BASE_SPEEDS
        for target in _targets_for_level(level, QUICK_TEST):
            for base_speed in speeds:
                task_utils.TASK_INFO.clear()
                scene_name, sampling_mode, layout_error, metadata = build_and_register_d2s(
                    task_id, level, base_speed, target
                )
                log_data = {"Task ID": "D2S_{}".format(task_id), "Suite": "D2S", "Level": level}
                if metadata:
                    log_data.update(metadata)
                if layout_error:
                    print("Skipping task {}: {}".format(task_id, layout_error))
                    log_data.update({"Status": "Fail", "Error": layout_error})
                    tracker.log_task(log_data)
                else:
                    print("Generating {}...".format(scene_name))
                    utils.execute_and_log(
                        scene_name,
                        configs.OUTPUT_FOLDER,
                        configs.GENERATE_BDDL_ONLY,
                        configs.SAVE_MP4_VIDEO,
                        configs.VIDEO_OUT_DIR,
                        task_id,
                        perturbations.sample_vis(level),
                        sampling_mode,
                        tracker,
                        log_data,
                    )
                task_id += 1
    tracker.print_paths()


if __name__ == "__main__":
    main()
