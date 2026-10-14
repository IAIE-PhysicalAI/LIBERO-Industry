import math
import os
import random
import sys
from pathlib import Path

# LIBERO-industry is installed in the Python 3.8 generation environment but is
# used directly from this checkout in the Python 3.10 deploy environment.
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
DEBUG_PARTIAL_COMPLETION = True
spatial_validator = perturbations.DynamicSpatialValidator(configs.ASSET_DIR)


def record_partial_completion(env, log_data):
    """Record the workbench's goal-level debug metrics after the rollout."""
    industry_env = env
    if not hasattr(industry_env, "get_partial_success"):
        industry_env = getattr(env, "env", None)
    if industry_env is None:
        print("[S2D DEBUG] Partial completion unavailable: industry environment not found")
        return

    partial_success = industry_env.get_partial_success()
    full_success = bool(industry_env._check_success())
    transfer = float(partial_success["pcr_transfer"])
    slotting = float(partial_success["pcr_slotting"])
    log_data.update(
        {
            "PCR Transfer": transfer,
            "PCR Slotting": slotting,
            "Goal Success": full_success,
        }
    )
    print(
        "[S2D DEBUG] Partial completion: transfer={:.1%}, slotting={:.1%}, "
        "success={}".format(transfer, slotting, full_success)
    )


def generate_safe_s2d_layout(
    cfg,
    level,
    inlay_name,
    max_attempts=150,
    transpose_degrees=None,
    source_plate_yaw_offset_deg=None,
):
    """Sample S2D anchors with reserved plate-to-track collision clearance."""
    last_reason = "no candidate was sampled"
    # Retry candidates on a clone of the task RNG.  The legacy feasibility
    # gate below determines the shared-stream commit point, preserving seeded
    # composition and source-slot draws outside this spatial fix.
    layout_rng = random.Random()
    layout_rng.setstate(random.getstate())
    legacy_commit_state = None
    load_axis = "y"
    require_track_clearance = level >= 2
    if require_track_clearance:
        plate_half_extents = spatial_validator.collision_half_extents("grey_plate")
        slot_half_extents = spatial_validator.collision_half_extents(inlay_name)
    transposes = tuple(transpose_degrees or ())
    if level == 3 and transpose_degrees is not None and not transposes:
        raise ValueError("S2D Level 3 requires at least one transpose degree")
    attempts_per_transpose = max(1, max_attempts // len(transposes or (0,)))

    def validate_anchor_pair(
        tray_anchor,
        ghost_anchor,
        shift_x,
        shift_y,
        tray_yaw_deg,
        ghost_yaw_deg,
        required_track_clearance,
        allow_x_correction,
    ):
        """Validate one deterministic anchor pair and optional common X repair."""
        result = perturbations.LayoutValidationResult(False, "not validated")
        for correction_attempt in range(2 if allow_x_correction else 1):
            tray_pos = (
                tray_anchor[0] + shift_x,
                tray_anchor[1] + shift_y,
            )
            ghost_pos = (
                ghost_anchor[0] + shift_x,
                ghost_anchor[1] + shift_y,
            )
            layout = {
                "ghost": {
                    "pos": ghost_pos,
                    "yaw_deg": ghost_yaw_deg,
                    "xml_name": "conveyor_ghost",
                    "safety_radius": perturbations.S2D_GHOST_FIXTURE_RADIUS,
                },
                "plate": {
                    "pos": tray_pos,
                    "yaw_deg": tray_yaw_deg,
                    "xml_name": "grey_plate",
                    "safety_radius": perturbations.S2D_TRAY_SPAWN_RADIUS,
                },
            }
            result = spatial_validator.validate_s2d_layout(
                layout,
                inlay_name,
                load_axis=load_axis,
                required_track_clearance=required_track_clearance,
            )
            if result:
                return result, shift_x, tray_pos, ghost_pos
            if (
                correction_attempt
                or "workspace bounds" not in result.reason
                or not allow_x_correction
            ):
                break
            kinematic_circles, spatial_circles = (
                spatial_validator.s2d_safety_circles(
                    layout, inlay_name, load_axis=load_axis
                )
            )
            correction_x = spatial_validator.negative_x_translation_to_safe_region(
                kinematic_circles, spatial_circles
            )
            if correction_x is None or correction_x >= -1e-9:
                break
            shift_x += correction_x
        return result, shift_x, tray_pos, ghost_pos

    for attempt in range(max_attempts):
        shift_x = layout_rng.uniform(-cfg["shift_range"], cfg["shift_range"])
        shift_y = layout_rng.uniform(-cfg["shift_range"], cfg["shift_range"])
        if level == 3:
            transpose_deg = (
                transposes[min(attempt // attempts_per_transpose, len(transposes) - 1)]
                if transpose_degrees is not None
                else layout_rng.choice(cfg["transpositions"])
            )
            # Plate and moving slot are independently perturbed around the
            # same scheduled cardinal setup.  The validator sees both yaws.
            tray_yaw_deg = transpose_deg + layout_rng.uniform(
                -cfg["yaw_range"], cfg["yaw_range"]
            )
            ghost_yaw_deg = transpose_deg + layout_rng.uniform(
                -cfg["yaw_range"], cfg["yaw_range"]
            )
        elif level == 2:
            transpose_deg = 0
            # Independently perturb the plate and moving slot fixture while
            # keeping both around the Level-2 zero-degree baseline.
            tray_yaw_deg = layout_rng.uniform(
                -cfg.get("yaw_range", 0.0), cfg.get("yaw_range", 0.0)
            )
            ghost_yaw_deg = layout_rng.uniform(
                -cfg.get("yaw_range", 0.0), cfg.get("yaw_range", 0.0)
            )
        else:
            # Preserve Level 1's shared yaw draw (its configured range is 0).
            transpose_deg = 0
            shared_yaw_deg = layout_rng.uniform(
                -cfg.get("yaw_range", 0.0), cfg.get("yaw_range", 0.0)
            )
            tray_yaw_deg = shared_yaw_deg
            ghost_yaw_deg = shared_yaw_deg
        if source_plate_yaw_offset_deg is not None:
            tray_yaw_deg = ghost_yaw_deg + source_plate_yaw_offset_deg

        # Reproduce the old spawn-envelope acceptance point before changing
        # anchors.  This keeps the deterministic benchmark's yaws, shifts, and
        # downstream RNG contract while the physical check evolves.
        legacy_tray_anchor, legacy_ghost_anchor, _ = (
            perturbations.get_s2d_safe_base_anchors()
        )
        (
            legacy_result,
            legacy_shift_x,
            legacy_tray_pos,
            legacy_ghost_pos,
        ) = validate_anchor_pair(
            legacy_tray_anchor,
            legacy_ghost_anchor,
            shift_x,
            shift_y,
            tray_yaw_deg,
            ghost_yaw_deg,
            required_track_clearance=None,
            allow_x_correction=True,
        )
        if not legacy_result:
            last_reason = legacy_result.reason
            continue
        if legacy_commit_state is None:
            legacy_commit_state = layout_rng.getstate()

        if not require_track_clearance:
            random.setstate(legacy_commit_state)
            return (
                legacy_shift_x,
                shift_y,
                transpose_deg,
                legacy_tray_pos,
                legacy_ghost_pos,
                tray_yaw_deg,
                ghost_yaw_deg,
            )

        tray_anchor, ghost_anchor, _ = (
            perturbations.get_s2d_collision_clearance_anchors(
                plate_half_extents,
                slot_half_extents,
                tray_yaw_deg,
                ghost_yaw_deg,
                load_axis=load_axis,
            )
        )
        track_normal = perturbations._rotate_xy((1.0, 0.0), ghost_yaw_deg)
        legacy_delta = (
            legacy_ghost_pos[0] - legacy_tray_pos[0],
            legacy_ghost_pos[1] - legacy_tray_pos[1],
        )
        preferred_sign = (
            1.0
            if legacy_delta[0] * track_normal[0]
            + legacy_delta[1] * track_normal[1]
            >= 0.0
            else -1.0
        )
        anchor_pairs = (
            ((tray_anchor, ghost_anchor), (ghost_anchor, tray_anchor))
            if preferred_sign > 0.0
            else ((ghost_anchor, tray_anchor), (tray_anchor, ghost_anchor))
        )
        for physical_tray_anchor, physical_ghost_anchor in anchor_pairs:
            result, physical_shift_x, tray_pos, ghost_pos = validate_anchor_pair(
                physical_tray_anchor,
                physical_ghost_anchor,
                legacy_shift_x,
                shift_y,
                tray_yaw_deg,
                ghost_yaw_deg,
                required_track_clearance=perturbations.S2D_TRACK_REQUIRED_CLEARANCE,
                allow_x_correction=False,
            )
            if result:
                random.setstate(legacy_commit_state)
                return (
                    physical_shift_x,
                    shift_y,
                    transpose_deg,
                    tray_pos,
                    ghost_pos,
                    tray_yaw_deg,
                    ghost_yaw_deg,
                )
            last_reason = result.reason

    random.setstate(legacy_commit_state or layout_rng.getstate())
    raise ValueError(
        "Failed to find safe S2D layout after {} attempts: {}".format(
            max_attempts, last_reason
        )
    )


def _sample_composition(cfg, level, target, object_map_entry=None):
    if object_map_entry is not None:
        destination_assets = list(object_map_entry["destination_objects"])
        destination_target_assets = []
        available_destination_assets = list(destination_assets)
        for asset in object_map_entry["target_objects"]:
            if asset in available_destination_assets:
                destination_target_assets.append(asset)
                available_destination_assets.remove(asset)
        target_assets = composition.remove_assets(
            object_map_entry["target_objects"], destination_target_assets
        )
        source_confounders = composition.remove_assets(
            object_map_entry["static_source_objects"], target_assets
        )
        return {
            "target_assets": target_assets,
            "source_confounder_assets": source_confounders,
            "destination_confounder_assets": destination_assets,
            "destination_target_assets": destination_target_assets,
            "inlay": (
                cfg["mismatch_map"][target]["inlay"]
                if level == 2
                else (
                    cfg["target_map"][target]["inlay"]
                    if level == 3
                    else "cosmetics_tray"
                )
            ),
        }
    if level == 1:
        return {
            "target_assets": [target] * 3,
            "source_confounder_assets": [],
            "destination_confounder_assets": [],
            "inlay": "cosmetics_tray",
        }
    if level == 2:
        return composition.sample_s2d_level2(cfg, target)
    return composition.sample_s2d_level3(cfg, target)


def build_and_register_s2d(
    task_id,
    level,
    target,
    base_speed,
    object_map_entry=None,
    transpose_degrees=None,
    track=configs.SINGLE_ARM,
):
    """Register one S2D scene.

    The staged 216-task workflow always supplies ``object_map_entry``.  The
    map-free branch exists only for legacy, exploratory script runs and is
    not a benchmark-generation path.
    """
    cfg = configs.get_suite_config("S2D", level, track)
    scene_composition = _sample_composition(cfg, level, target, object_map_entry)
    target_assets = scene_composition["target_assets"]
    source_assets = target_assets + scene_composition["source_confounder_assets"]
    source_instances, counts = composition.named_instances(source_assets)
    destination_instances, _ = composition.named_instances(
        scene_composition["destination_confounder_assets"], counts
    )
    target_instances = source_instances[: len(target_assets)]
    if object_map_entry is not None:
        target_instances.extend(
            composition.mapped_instances(
                destination_instances,
                scene_composition["destination_target_assets"],
            )
        )

    if level == 1:
        # Level 1 speed_noise is a relative ±5% perturbation.
        noise = cfg.get("speed_noise", 0.0)
        speed = round(base_speed * (1 + random.uniform(-noise, noise)), 4)
    elif level == 2:
        # Level 2 speed_noise is an absolute ±0.005 m/s perturbation.
        noise = cfg.get("speed_noise", 0.0)
        speed = round(base_speed + random.uniform(-noise, noise), 4)
    else:
        # Level 3 samples a magnitude independently of the filename's speed.
        speed = round(math.copysign(random.uniform(*cfg["speed_range"]), base_speed), 4)
    try:
        (
            _,
            _,
            actual_transpose_deg,
            tray_pos,
            ghost_pos,
            tray_yaw_deg,
            ghost_yaw_deg,
        ) = generate_safe_s2d_layout(
            cfg,
            level,
            scene_composition["inlay"],
            transpose_degrees=transpose_degrees,
            source_plate_yaw_offset_deg=(
                90.0
                if track == configs.SINGLE_ARM
                and target_assets[0].endswith("_flat")
                and composition.strip_pose(target_assets[0])
                in configs.S2D_POSE_MATCH_TARGETS
                else None
            ),
        )
    except ValueError as error:
        return None, None, str(error), None

    scene_name = "S2D_LV{}_{}".format(level, str(task_id).zfill(3))
    all_assets = source_assets + scene_composition["destination_confounder_assets"]

    class S2DScene(InitialSceneTemplates):
        def __init__(self):
            super().__init__(
                "industry_workbench",
                {"industry_workbench": 1, "conveyor_ghost": 1, "grey_plate": 1},
                composition.asset_counts(all_assets),
            )

        def define_regions(self):
            # The anchors already include their sampled shift. Keep this
            # positive but effectively point-sized for Robosuite 1.4 and 1.5.
            anchor_half_len = 1e-6
            self.regions.update(
                self.get_region_dict(
                    ghost_pos,
                    "ghost_init",
                    self.workspace_name,
                    anchor_half_len,
                    (math.radians(ghost_yaw_deg), math.radians(ghost_yaw_deg)),
                )
            )
            self.regions.update(
                self.get_region_dict(
                    tray_pos,
                    "tray_init",
                    self.workspace_name,
                    anchor_half_len,
                    (math.radians(tray_yaw_deg), math.radians(tray_yaw_deg)),
                )
            )
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        def init_states(self):
            service_time = cfg.get("service_time", 1.0)
            states = [
                ("On", "conveyor_ghost_1", "industry_workbench_ghost_init"),
                ("On", "grey_plate_1", "industry_workbench_tray_init"),
                (
                    "LoadSlot",
                    "conveyor_ghost_1",
                    scene_composition["inlay"],
                    "y",
                    "pulse",
                    str(speed),
                    "service_{}".format(service_time),
                ),
            ]
            for instance_name, slot in zip(
                source_instances,
                random.sample([1, 2, 3, 4, 5, 6], len(source_instances)),
            ):
                states.append(("On", instance_name, "grey_plate_1_place_region_{}".format(slot)))
            # Legacy S2D Level 3 target-side confounders are moving-track
            # obstacles, deliberately excluded from the Routed goals.
            for instance_name in destination_instances:
                states.append(("On", instance_name, "conveyor_ghost_1"))
            return states

    MU_DICT[scene_name.lower()] = S2DScene
    SCENE_DICT["industry_workbench"] = [S2DScene]
    goals = [("Routed", instance_name, "conveyor_ghost_1") for instance_name in target_instances]
    register_task_info(
        cfg.get("language", "route cosmetics"),
        scene_name=scene_name,
        objects_of_interest=[],
        goal_states=goals,
    )
    metadata = {
        "Target Instances": str(target_instances),
        "Source Confounders": str(source_instances[len(target_instances) :]),
        "Destination Confounders": str(destination_instances),
    }
    if level == 3:
        metadata["Transpose Degrees"] = actual_transpose_deg
    return scene_name, ("random" if level > 1 else "uniform"), None, metadata


def _targets_for_level(level, quick_test):
    if level == 1:
        targets = configs.S2D[1]["targets"]
    elif level == 2:
        targets = list(configs.S2D[2]["mismatch_map"])
    else:
        targets = configs.S2D_TARGETS
    return targets[:1] if quick_test else targets


def main():
    os.makedirs(configs.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(configs.VIDEO_OUT_DIR, exist_ok=True)
    tracker = utils.TaskTracker(
        configs.EXCEL_PATH,
        [
            "Task ID",
            "Suite",
            "Level",
            "Status",
            "Video",
            "PCR Transfer",
            "PCR Slotting",
            "Goal Success",
            "Error",
        ],
    )
    print("RUNNING S2D GENERATOR (Quick Test: {})".format(QUICK_TEST))
    task_id = 1
    for level in (1, 2, 3):
        speeds = configs.BASE_SPEEDS[:1] if QUICK_TEST else configs.BASE_SPEEDS
        for target in _targets_for_level(level, QUICK_TEST):
            for base_speed in speeds:
                task_utils.TASK_INFO.clear()
                scene_name, sampling_mode, layout_error, metadata = build_and_register_s2d(
                    task_id, level, target, base_speed
                )
                log_data = {"Task ID": "S2D_{}".format(task_id), "Suite": "S2D", "Level": level}
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
                        debug_callback=(
                            record_partial_completion
                            if DEBUG_PARTIAL_COMPLETION
                            else None
                        ),
                    )
                task_id += 1
    tracker.print_paths()


if __name__ == "__main__":
    main()
