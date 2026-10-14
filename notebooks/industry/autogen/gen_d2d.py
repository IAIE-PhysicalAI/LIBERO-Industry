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
spatial_validator = perturbations.DynamicSpatialValidator(configs.ASSET_DIR)

# Level 1 and 2 retain the D2D transport directions and opposing X roots.
# Their compact belt Y anchors rely on bounded rejection sampling to keep the
# 3.2 m physical rail outside the robot-base keep-out under configured noise.
# This is *not* an S2D bounded-anchor calculation.
# Canonical D2D pickup centre.  With the existing +/- 2 cm L1/L2 scene
# translation, the belt root samples across x=[-0.125, -0.085].
D2D_LV12_BELT_ANCHOR = (-0.105, 0.0)
D2D_LV12_GHOST_ANCHOR = (0.03, 0.05)
# The 3.2 m contact rail runs along global X at the normalized yaw=90.
# The validator rejects the small fraction of XY/yaw draws that enter the
# robot-base keep-out.
D2D_LV1_BELT_Y = 0.30
D2D_LV2_BELT_Y = 0.30
D2D_LV3_TRANSPOSE_DEGREES = (0, 90, 180, 270)
# These common, target-independent anchors keep every Level-3 target inlay
# feasible without coupling object assignment to a fallback orientation.  The
# 90/270 roots include the inward correction needed by the largest headset
# slot; the 0-degree belt retains Level 2's nominal pickup anchor and the
# 180-degree layout mirrors its Y coordinates across the robot base.
D2D_LV3_BASE_ANCHORS = {
    0: ((-0.105, 0.30), (-0.055, 0.05)),
    90: ((-0.272025, 0.07), (0.03, 0.07)),
    180: ((-0.105, -0.30), (-0.15, -0.05)),
    270: ((0.07, -0.02), (-0.232025, -0.02)),
}


def _d2d_lv3_base_anchors(transpose_deg):
    """Return D2D L3's workspace-safe cardinal anchor pair."""
    if transpose_deg not in D2D_LV3_TRANSPOSE_DEGREES:
        raise ValueError(
            "D2D Level 3 requires a cardinal transpose, got {}"
            .format(transpose_deg)
        )
    return D2D_LV3_BASE_ANCHORS[transpose_deg]


def _contract_ghost_anchor(belt, ghost, yaw_deg, minimum_separation, attempt):
    """Move the ghost toward the belt while preserving its along-track offset."""
    normal = perturbations._rotate_xy((1.0, 0.0), yaw_deg)
    tangent = perturbations._rotate_xy((0.0, 1.0), yaw_deg)
    delta = (ghost[0] - belt[0], ghost[1] - belt[1])
    along = delta[0] * tangent[0] + delta[1] * tangent[1]
    cross = delta[0] * normal[0] + delta[1] * normal[1]
    if abs(cross) + 1e-9 < minimum_separation:
        raise ValueError("D2D reference anchors cannot clear the selected groove")
    separation = min(
        abs(cross),
        minimum_separation + attempt * perturbations.LAYOUT_CLEARANCE,
    )
    signed_separation = math.copysign(separation, cross)
    return (
        belt[0] + along * tangent[0] + signed_separation * normal[0],
        belt[1] + along * tangent[1] + signed_separation * normal[1],
    )


def generate_safe_d2d_layout(
    cfg,
    level,
    ghost_target,
    max_attempts=150,
    transpose_degrees=None,
    load_axis="y",
):
    """Sample safe D2D anchors while preserving the scheduled transpose."""
    last_reason = "no candidate was sampled"
    transposes = tuple(transpose_degrees or ())
    if level == 3 and transpose_degrees is not None and len(transposes) != 1:
        raise ValueError("D2D Level 3 requires exactly one scheduled transpose")
    for attempt in range(max_attempts):
        if level == 3:
            rot_deg = (
                transposes[0]
                if transpose_degrees is not None
                else random.choice(cfg["transpositions"])
            )
            # Keep the far-side anchor near x=0 with D2D-only limits. Both
            # tracks receive the same translation; the validator remains the
            # acceptance gate for collision and reachability.
            shift_x_range = cfg["shift_x_range"]
            shift_y_range = (
                cfg["shift_x_range"]
                if rot_deg in (0, 180)
                else cfg["shift_y_range"]
            )
        else:
            rot_deg = 0.0
            shift_x_range = shift_y_range = cfg["shift_range"]
        shift_x = random.uniform(-shift_x_range, shift_x_range)
        shift_y = random.uniform(-shift_y_range, shift_y_range)
        if level in (1, 2):
            # Both normalized conveyor frames use local +Y as the transport
            # axis. Level 2 shares one yaw perturbation so speed signs have an
            # exact same/opposite world-direction meaning.
            belt_yaw = ghost_yaw = 90.0
            rot_deg = 0.0
            if level == 2:
                shared_yaw_noise = random.uniform(
                    -cfg["yaw_range"], cfg["yaw_range"]
                )
                # Preserve the established random stream for every later
                # position, groove, and source-slot sample.
                random.uniform(-cfg["yaw_range"], cfg["yaw_range"])
                belt_yaw += shared_yaw_noise
                ghost_yaw += shared_yaw_noise
        else:
            # The normalized belt and ghost both expose local +Y transport.
            # A shared deviation keeps their world axes exactly parallel.
            yaw_range = 5.0 if rot_deg in (0, 180) else cfg["yaw_range"]
            shared_yaw_noise = random.uniform(
                -yaw_range, yaw_range
            )
            # Consume the former independent ghost draw so later layout
            # samples remain deterministic for the same benchmark seed.
            random.uniform(-cfg["yaw_range"], cfg["yaw_range"])
            belt_yaw = ghost_yaw = rot_deg - 90.0 + shared_yaw_noise

        if level in (1, 2):
            belt_y = D2D_LV1_BELT_Y if level == 1 else D2D_LV2_BELT_Y
            belt_anchor_base = (D2D_LV12_BELT_ANCHOR[0], belt_y)
            ghost_base = D2D_LV12_GHOST_ANCHOR
        else:
            try:
                belt_anchor_base, ghost_base = _d2d_lv3_base_anchors(rot_deg)
            except ValueError as error:
                last_reason = str(error)
                continue

        if level == 3 and rot_deg not in (0, 180):
            try:
                minimum_separation = spatial_validator.d2d_required_track_separation(
                    ghost_target, load_axis=load_axis
                )
                ghost_base = _contract_ghost_anchor(
                    belt_anchor_base,
                    ghost_base,
                    belt_yaw,
                    minimum_separation,
                    attempt,
                )
            except (FileNotFoundError, ValueError) as error:
                last_reason = str(error)
                continue

        belt_pos = (belt_anchor_base[0] + shift_x, belt_anchor_base[1] + shift_y)
        ghost_pos = (ghost_base[0] + shift_x, ghost_base[1] + shift_y)
        layout = {
            "belt": {
                # The root anchor is used as the existing centre-region reach
                # proxy. Preserve the asset's small local centre-site offset.
                "pos": belt_pos,
                "yaw_deg": belt_yaw,
                "xml_name": "conveyor_belt",
                "safety_radius": 0.0,
                "skip_spatial_bounds": True,
                # The virtual rail is intentionally unbounded.  D2D checks
                # only its centre-region reach below.
                "skip_kinematic_bounds": True,
            },
            "ghost": {
                "pos": ghost_pos,
                "yaw_deg": ghost_yaw,
                "xml_name": "conveyor_ghost",
                "safety_radius": 0.0,
                "skip_spatial_bounds": True,
                # Ghost reach is defined by its active groove envelopes, not
                # by the anchor at the middle of the virtual track.
                "skip_kinematic_bounds": True,
            },
        }
        result = spatial_validator.validate_d2d_layout(
            layout,
            ghost_target,
            # The reference L1/L2 ghost rail deliberately crosses the
            # workspace boundary. Its task targets are the two reachable
            # groove goal CoMs; full inlay envelopes remain required for L3.
            slot_radius=0.0 if level in (1, 2) else None,
            load_axis=load_axis,
        )
        if result:
            return shift_x, shift_y, rot_deg, belt_pos, ghost_pos, belt_yaw, ghost_yaw
        last_reason = result.reason
    raise ValueError(
        "Failed to find bounded D2D layout after {} attempts: {}".format(
            max_attempts, last_reason
        )
    )


def _sample_composition(cfg, level, target, object_map_entry=None):
    if object_map_entry is not None:
        return {
            "target_assets": list(object_map_entry["target_objects"]),
            "source_assets": list(object_map_entry["source_objects"]),
            "destination_confounder_assets": list(
                object_map_entry["destination_objects"]
            ),
        }
    if level == 1:
        source_assets = [target, target] + random.sample(
            [asset for asset in cfg["all_objects"] if asset != target], 2
        )
        return {
            "target_assets": [target, target],
            "source_assets": source_assets,
            "destination_confounder_assets": [],
        }
    if level == 2:
        return composition.sample_d2d_level2(cfg, target)
    return composition.sample_d2d_level3(cfg, target)


def _target_instance_names(level, target, source_instances, destination_instances):
    if level in (1, 2):
        target_instances = [
            instance_name
            for instance_name in source_instances
            if composition.instance_asset(instance_name) == target
        ]
    else:
        target_instances = [
            instance_name
            for instance_name in source_instances
            if composition.strip_pose(composition.instance_asset(instance_name)) == target
        ]
        # Legacy D2D Level 3 ghost confounders are already routed goals.
        target_instances.extend(destination_instances)
    return target_instances


def build_and_register_d2d(
    task_id,
    level,
    target,
    base_speed,
    object_map_entry=None,
    transpose_degrees=None,
    slot_base_speed=None,
    track=configs.SINGLE_ARM,
):
    """Register one D2D scene.

    The staged 216-task workflow always supplies ``object_map_entry``.  The
    map-free branch exists only for legacy, exploratory script runs and is
    not a benchmark-generation path.
    """
    cfg = configs.get_suite_config("D2D", level, track)
    if slot_base_speed is None:
        if object_map_entry is not None and level in (2, 3):
            raise ValueError(
                "Canonical D2D Level {} requires the filename's second speed"
                .format(level)
            )
        if level == 1:
            slot_base_speed = base_speed
        elif level == 2:
            magnitudes = (0.03, 0.05, 0.08)
            source_magnitude = abs(base_speed)
            next_magnitude = magnitudes[
                (magnitudes.index(source_magnitude) + 1) % len(magnitudes)
            ]
            slot_base_speed = math.copysign(next_magnitude, base_speed)
        else:
            slot_base_speed = -base_speed

    if level == 1 and not math.isclose(slot_base_speed, base_speed, abs_tol=1e-9):
        raise ValueError("D2D Level 1 requires equal signed conveyor speeds")
    if level == 2 and (
        base_speed * slot_base_speed <= 0.0
        or math.isclose(abs(base_speed), abs(slot_base_speed), abs_tol=1e-9)
    ):
        raise ValueError(
            "D2D Level 2 requires same-direction, different-magnitude speeds"
        )
    if level == 3 and base_speed * slot_base_speed >= 0.0:
        raise ValueError("D2D Level 3 requires opposite-direction speeds")

    scene_composition = _sample_composition(cfg, level, target, object_map_entry)
    source_assets = scene_composition["source_assets"]
    source_instances, counts = composition.named_instances(source_assets)
    destination_instances, _ = composition.named_instances(
        scene_composition["destination_confounder_assets"], counts
    )
    speed_rng = random
    if level == 3:
        speed_rng = random.Random()
        speed_rng.setstate(random.getstate())
    speed = perturbations.sample_d2d_speed(base_speed, rng=speed_rng)
    speed, takt, offset = perturbations.calc_belt_timing(
        speed, len(source_assets)
    )
    takt_by_speed = cfg.get("arrival_takt_by_speed")
    takt = (
        takt_by_speed[abs(base_speed)]
        if takt_by_speed is not None
        else cfg["arrival_takt"]
    )
    if level == 1:
        ghost_target, load_axis = "electronics_tray", "y"
    elif level == 2:
        target_base = composition.strip_pose(target)
        ghost_target = "{}_{}".format(target_base, random.choice(cfg["grooves"]))
        load_axis = random.choice(["x", "y"])
    else:
        ghost_target = "{}_upright_inlay".format(target)
        load_axis = random.choice(["x", "y"])

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
        _, _, actual_transpose_deg, belt_pos, ghost_pos, belt_yaw, ghost_yaw = generate_safe_d2d_layout(
            cfg,
            level,
            ghost_target,
            load_axis=load_axis,
            transpose_degrees=transpose_degrees,
        )
    except ValueError as error:
        return None, None, str(error), None

    # Sample the slot speed only after spatial generation so adding the second
    # filename speed cannot perturb groove, placement, or layout coordinates.
    slot_speed = (
        speed
        if level == 1
        else perturbations.sample_d2d_speed(slot_base_speed, rng=speed_rng)
    )

    scene_name = "D2D_LV{}_{}".format(level, str(task_id).zfill(3))
    all_assets = source_assets + scene_composition["destination_confounder_assets"]

    class D2DScene(InitialSceneTemplates):
        def __init__(self):
            super().__init__(
                "industry_workbench",
                {"industry_workbench": 1, "conveyor_belt": 1, "conveyor_ghost": 1},
                composition.asset_counts(all_assets),
            )

        def define_regions(self):
            self.regions.update(
                self.get_region_dict(
                    belt_pos,
                    "belt_init",
                    self.workspace_name,
                    0.01,
                    (math.radians(belt_yaw), math.radians(belt_yaw)),
                )
            )
            self.regions.update(
                self.get_region_dict(
                    ghost_pos,
                    "ghost_init",
                    self.workspace_name,
                    0.01,
                    (math.radians(ghost_yaw), math.radians(ghost_yaw)),
                )
            )
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        def init_states(self):
            service_time = cfg.get("service_time", 1.0)
            states = [
                ("On", "conveyor_belt_1", "industry_workbench_belt_init"),
                ("AtSpeed", "conveyor_belt_1", str(speed), "steady", "takt_{}".format(takt), "offset_{}".format(offset)),
                ("On", "conveyor_ghost_1", "industry_workbench_ghost_init"),
                (
                    "LoadSlot",
                    "conveyor_ghost_1",
                    ghost_target,
                    load_axis,
                    "pulse",
                    str(slot_speed),
                    "service_{}".format(service_time),
                ),
            ]
            region_prefix = "tail_region" if speed < 0 else "offset_region"
            for instance_name, slot in zip(source_instances, source_slots):
                states.append(("On", instance_name, "conveyor_belt_1_{}_{}".format(region_prefix, slot)))
            for instance_name in destination_instances:
                states.append(("On", instance_name, "conveyor_ghost_1"))
            return states

    MU_DICT[scene_name.lower()] = D2DScene
    SCENE_DICT["industry_workbench"] = [D2DScene]
    if object_map_entry is not None:
        target_instances = composition.mapped_instances(
            source_instances + destination_instances,
            object_map_entry["target_objects"],
        )
        if level == 1:
            goals = [
                ("DynamicOn", instance_name, "conveyor_ghost_1")
                for instance_name in target_instances
            ]
        else:
            goals = [
                ("Routed", instance_name, "conveyor_ghost_1")
                for instance_name in target_instances
            ]
    else:
        target_instances = _target_instance_names(
            level, target, source_instances, destination_instances
        )
        if level == 1:
            goals = [("DynamicOn", target_instances[0], "conveyor_ghost_1")]
        else:
            goals = [
                ("Routed", instance_name, "conveyor_ghost_1")
                for instance_name in target_instances
            ]
    register_task_info(
        cfg.get("language", "route targets"),
        scene_name=scene_name,
        objects_of_interest=[state[1] for state in goals],
        goal_states=goals,
    )
    metadata = {
        "Source Speed": speed,
        "Slot Speed": slot_speed,
        "Target Instances": str(target_instances),
        "Source Confounders": str(
            [
                instance_name for instance_name in source_instances
                if instance_name not in target_instances
            ]
        ),
        "Destination Confounders": str(destination_instances),
    }
    if level == 3:
        metadata["Transpose Degrees"] = actual_transpose_deg
    # Keep the staged reset policy identical to the shared Industry contract
    # used when initial-state metadata is prepared: Level 1 is fixed/uniform,
    # while the perturbation levels are randomized.
    sampling_mode = "uniform" if level == 1 else cfg.get("sampling", "random")
    return scene_name, sampling_mode, None, metadata


def _targets_for_level(level, quick_test):
    if level == 3:
        targets = configs.D2D[3]["targets"]
    else:
        targets = ["{}_upright".format(base) for base in configs.D2D_BASE_OBJS]
    return targets[:1] if quick_test else targets


def main():
    os.makedirs(configs.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(configs.VIDEO_OUT_DIR, exist_ok=True)
    tracker = utils.TaskTracker(
        configs.EXCEL_PATH, ["Task ID", "Suite", "Level", "Status", "Video", "Error"]
    )
    print("RUNNING D2D GENERATOR (Quick Test: {})".format(QUICK_TEST))
    task_id = 1
    for level in (1, 2, 3):
        speeds = configs.BASE_SPEEDS[:1] if QUICK_TEST else configs.BASE_SPEEDS
        for target in _targets_for_level(level, QUICK_TEST):
            for base_speed in speeds:
                task_utils.TASK_INFO.clear()
                scene_name, sampling_mode, layout_error, metadata = build_and_register_d2d(
                    task_id, level, target, base_speed
                )
                log_data = {"Task ID": "D2D_{}".format(task_id), "Suite": "D2D", "Level": level}
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
