from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from libero.libero.envs.regions.base_region_sampler import MultiRegionRandomSampler
from libero.libero.envs.regions.workspace_region_sampler import LiberoIndustrySampler

import numpy as np
import xml.etree.ElementTree as ET
import robosuite.utils.transform_utils as T
import os
import json
import re
import random

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange
from libero.libero.assets.industry_objects.conveyor_physics import ConveyorBeltMixin
from libero.libero.assets.industry_objects.conveyor_slot_physics import ConveyorSlotMixin

class ConveyorPhysicsSampler:
    """Samples physics parameters for the conveyor belt to enable Domain Randomization."""
    def __init__(self, speed_range=(0.02, 0.1), modes=["steady", "pulse"]):
        self.speed_range = speed_range
        self.modes = modes

    def sample(self):
        return {
            "speed": np.random.uniform(*self.speed_range),
            "mode": np.random.choice(self.modes)
        }

@register_problem
class Libero_Industry_Workbench_Endurance(ConveyorBeltMixin, ConveyorSlotMixin, BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.completed_objects = set()
        self.workspace_name = "industry_workbench"
        self.visualization_sites_list = []
        if "industry_workbench_full_size" in kwargs:
            self.industry_workbench_full_size = workbench_full_size
        else:
            self.industry_workbench_full_size = (1.0, 1.2, 0.05)
        self.industry_workbench_offset = (0.0, 0, 0.80)
        self.z_offset = 0.01 - self.industry_workbench_full_size[2]

        self.current_conveyor_params = {"speed": 0.05, "mode": "steady"}
        self.physics_sampler = ConveyorPhysicsSampler()
        self._groove_info_cache = None

        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.industry_workbench_offset})
        kwargs.update({"arena_type": "industry_workbench"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {"scene_xml": "scenes/libero_industry_workbench_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "gray-ceramic",
                        "wall_style": "yellow-linen",
                }
            }
        )
        # =========================================================================
        # 🚨 THE "SMART FLATTENER" & DEBUG BLOCK
        # =========================================================================
        print("\n" + "="*60)
        print(f"[DEBUG WORKBENCH] Initial kwargs keys: {list(kwargs.keys())}")

        # 1. Pop the nested dictionary entirely so Robosuite's super() never sees it
        nested_env_args = kwargs.pop("env_args", {})
        if nested_env_args:
             print(f"[DEBUG WORKBENCH] Extracted nested env_args: {nested_env_args}")

        # 2. Extract scene
        self.custom_set_scene = kwargs.pop("set_scene", None)

        # Fallback: if industry configs were not injected by ControlEnv (e.g. direct
        # instantiation for demo collection), load them from the JSON ourselves.
        if self.custom_set_scene is None:
            try:
                current_dir = os.path.abspath(os.path.dirname(bddl_file_name))
                json_path = None
                while current_dir != "/":
                    potential_path = os.path.join(current_dir, "benchmark", "industry_env_args.json")
                    if os.path.exists(potential_path):
                        json_path = potential_path
                        break
                    current_dir = os.path.dirname(current_dir)
                if os.path.exists(json_path):
                    with open(json_path, "r") as _f:
                        master_scene_dict = json.load(_f)
                    task_name = os.path.basename(bddl_file_name).replace(".bddl", "")
                    task_cfg = master_scene_dict.get(task_name) or master_scene_dict.get(task_name.lower())
                    if task_cfg:
                        if "set_scene" in task_cfg:
                            self.custom_set_scene = task_cfg["set_scene"]
                        if "env_args" in task_cfg:
                            nested_env_args = task_cfg["env_args"]
                        if "takt_steps" in task_cfg and "takt_steps" not in kwargs:
                            kwargs["takt_steps"] = task_cfg["takt_steps"]
                        print(f"[DEBUG WORKBENCH] Loaded industry configs from fallback JSON for {task_name}")
            except Exception:
                pass

        # 3. Extract object sampling (Check nested first, then flat, default to random)
        raw_sampling = nested_env_args.get("object_sampling", kwargs.pop("object_sampling", "random"))
        self.object_sampling = str(raw_sampling).lower()
        if self.object_sampling not in {"uniform", "random"}:
            self.object_sampling = "random"

        # 4. Extract takt_steps (Check nested first, then flat, default to 400)
        # We ensure it converts to an integer dynamically based on the JSON
        self.takt_steps = int(nested_env_args.get("takt_steps", kwargs.pop("takt_steps", 400)))
        
        self.current_step = 0
        self._task_name = kwargs.pop("task_name", None)

        # --- Industrial metric tracking variables ---
        self._failure_steps = []       # step indices of failures (for MTBF)
        self._first_failure_step = None  # step index of first failure (for MTTF)

        print(f"[DEBUG WORKBENCH] Applied Sampling: {self.object_sampling}")
        print(f"[DEBUG WORKBENCH] Applied Takt Steps: {self.takt_steps}")
        print(f"[DEBUG WORKBENCH] Applied Set Scene: {self.custom_set_scene is not None}")

        # 5. Flatten the rest of the standard Robosuite args back into kwargs
        for k, v in nested_env_args.items():
            if k not in ["object_sampling", "takt_steps", "set_scene", "robots"]:
                kwargs[k] = v
                print(f"[DEBUG WORKBENCH] Passed standard arg '{k}' = {v} to Robosuite.")
        print("="*60 + "\n")
        # =========================================================================

        super().__init__(bddl_file_name, *args, **kwargs)


    def _load_model(self):
        super()._load_model()
        # Apply the scene modifications to the FINAL merged task tree
        if self.custom_set_scene is not None:
            if hasattr(self.model.mujoco_arena, "apply_set_scene"):
                self.model.mujoco_arena.apply_set_scene(self.model.root, self.custom_set_scene)
            else:
                print("Warning: mujoco_arena does not have apply_set_scene method.")

    def _reset_internal(self):
        self.current_step = 0
        self._achieved_success = False  # Track if success has been achieved
        self._track_initialized = False
        self._failure_steps = []
        self._first_failure_step = None
        super()._reset_internal()
        self.sim.forward()

        if hasattr(self, 'dynamic_groove_names') and len(self.dynamic_groove_names) > 0:
            self.sim.forward()
            ConveyorSlotMixin._initialize_track(self)
            ConveyorSlotMixin._apply_physical_track(self)
            self.sim.forward()

        # Pre-compute groove info cache for success checking
        self._groove_info_cache = {}
        for g_name in self.dynamic_groove_names:
            site_name = f"{g_name}_goal_region"
            if site_name in self.sim.model.site_names:
                self._groove_info_cache[g_name] = self.sim.model.site_name2id(site_name)

        found_running_state = False
        for state in self.parsed_problem["initial_state"]:
            if state[0].lower() == "atspeed":
                self.current_conveyor_params["speed"] = float(state[2])
                self.current_conveyor_params["mode"] = state[3]
                found_running_state = True
                break

        if not found_running_state:
            self.current_conveyor_params = self.physics_sampler.sample()

        self.setup_conveyor_belt(
            velocity=self.current_conveyor_params["speed"],
            local_axis=(0, 1, 0)
        )

        groove_objs = [name for name in self.objects_dict.keys() if name.startswith("dyn_")]
        available_grooves = []

        if groove_objs:
            groove_objs.sort(key=lambda x: int(re.search(r'_(\d+)$', x).group(1)) if re.search(r'_(\d+)$', x) else 999)
            max_idx = max(1, int(len(groove_objs) * 0.5))
            front_pool = groove_objs[1:max_idx+1] if len(groove_objs) > 1 else groove_objs
            backup_pool = groove_objs[max_idx+1:] if len(groove_objs) > 1 else []
            random.shuffle(front_pool)
            random.shuffle(backup_pool)
            available_grooves = front_pool + backup_pool

        has_ghost_fixture = "conveyor_ghost" in self.parsed_problem.get("fixtures", {})
        
        if has_ghost_fixture:
            ghost_was_placed = False
            for state in self.parsed_problem["initial_state"]:
                if state[0].lower() == "on" and state[1] == "conveyor_ghost_1":
                    ghost_was_placed = True
                    
            if not ghost_was_placed:
                raise ValueError(
                    "\n[FATAL BDDL ERROR] 'conveyor_ghost' is defined in fixtures but was NOT placed in the initial state! "
                    "You must include an ('On', 'conveyor_ghost_1', '..._ghost_init_region') predicate "
                    "in your BDDL file before loading the slots.\n"
                )
        for state in self.parsed_problem["initial_state"]:
            if state[0].lower() == "on" and "conveyor_ghost" in state[2]:
                obj_to_teleport = state[1]

                if not available_grooves:
                    continue

                chosen_groove = available_grooves.pop(0)

                groove_obj = self.objects_dict[chosen_groove]
                groove_joint = groove_obj.joints[-1]
                groove_qpos = self.sim.data.get_joint_qpos(groove_joint)
                target_quat = groove_qpos[3:7]

                groove_mat = T.quat2mat(T.convert_quat(target_quat, to="xyzw"))

                groove_site_id = self.sim.model.site_name2id(f"{chosen_groove}_goal_region")
                groove_site_global = self.sim.data.site_xpos[groove_site_id]
                floor_xyz = groove_site_global + groove_mat.dot(np.array([0, 0, -0.005]))

                obj = self.objects_dict[obj_to_teleport]
                local_bottom_offset = obj.bottom_offset
                global_bottom_offset = groove_mat.dot(local_bottom_offset)

                target_pos = floor_xyz - global_bottom_offset
                target_pos += groove_mat.dot(np.array([0, 0, 0.01]))

                obj_joint = obj.joints[-1]
                self.sim.data.set_joint_qpos(obj_joint, np.concatenate([target_pos, target_quat]))
                self.sim.data.set_joint_qvel(obj_joint, self.sim.data.get_joint_qvel(groove_joint))

        self.sim.forward()

    def step(self, action):
        # Increment step counter for takt time tracking
        self.current_step += 1

        # Handle "pulse" mode (rhythmic start/stop)
        if self.current_conveyor_params["mode"] == "pulse":
            sim_time = self.sim.data.time
            is_moving = 1.0 if (sim_time % 2.0) < 1.0 else 0.0
            self.conveyor_velocity = self.current_conveyor_params["speed"] * is_moving

        # Run standard simulation logic
        obs, reward, base_done, info = super().step(action)

        # --- Industrial metric: failure detection ---
        # Check for objects falling out of bounds (below workbench surface)
        out_of_bounds = False
        for obj_name, obj in self.objects_dict.items():
            if obj_name.startswith("dyn_"):
                continue
            try:
                obj_joint = obj.joints[-1]
                obj_z = self.sim.data.get_joint_qpos(obj_joint)[2]
                if obj_z < (self.z_offset - 0.15):
                    out_of_bounds = True
                    break
            except Exception:
                pass

        if out_of_bounds and self._first_failure_step is None:
            self._first_failure_step = self.current_step
            self._failure_steps.append(self.current_step)

        # --- Success / Takt logic ---
        is_success = False
        takt_limit_reached = (self.current_step >= self.takt_steps)

        if takt_limit_reached:
            is_success = self._check_success()
            # Takt timeout without full routing delivery is a tracking failure
            if not is_success:
                if self._first_failure_step is None:
                    self._first_failure_step = self.current_step
                self._failure_steps.append(self.current_step)

        if is_success:
            done = True
            reward = 1.0
        else:
            done = base_done
            reward = 0.0

        # --- Compute 5 industrial automation metrics ---
        # 1. Success Rate (成功率)
        info["success"] = is_success

        # 2. Yield Rate (产出率): successfully routed objects / total timesteps
        routed_count = 0
        for state in self.parsed_problem.get("goal_state", []):
            if len(state) == 3 and state[0].lower() == "routed":
                obj_name = state[1]
                routed_now = self._eval_predicate(state)
                was_delivered = hasattr(self, '_delivered_objects') and obj_name in self._delivered_objects
                if routed_now or was_delivered:
                    routed_count += 1
        info["yield_rate"] = routed_count / max(1, self.current_step)

        # 3. Takt Utilization Rate (节拍利用率)
        info["takt_utilization"] = self.current_step / max(1, self.takt_steps)

        # 4. MTBF — Mean Time Between Failures (平均故障间隔时间)
        if len(self._failure_steps) >= 2:
            intervals = [
                self._failure_steps[i] - self._failure_steps[i - 1]
                for i in range(1, len(self._failure_steps))
            ]
            info["mtbf"] = sum(intervals) / len(intervals)
        elif len(self._failure_steps) == 1:
            info["mtbf"] = float(self._failure_steps[0])
        else:
            info["mtbf"] = float(self.current_step)

        # 5. MTTF — Time to First Failure (首次故障时间)
        info["mttf"] = self._first_failure_step if self._first_failure_step is not None else -1

        # Existing auxiliary info fields
        info["takt_limit_reached"] = takt_limit_reached
        info["conveyor_speed"] = self.conveyor_velocity
        info["conveyor_mode"] = self.current_conveyor_params["mode"]

        return obs, reward, done, info
        
    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "industry_workbench":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        from libero.libero.envs.objects import get_object_fn
        self.dynamic_groove_names = []

        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(name=object_name)

        for state in self.parsed_problem["initial_state"]:
            if state[0].lower() == "loadslot":
                expected_format = "('LoadSlot', 'conveyor_name', 'groove_type', 'orientation (x/y)', 'mode', 'speed')"
                assert len(state) == 6, f"\n[BDDL ERROR] Incorrect arguments for LoadSlot.\nExpected: {expected_format}\nGot: {state}\n"

                conveyor_name = state[1]
                groove_type = state[2]
                orientation = state[3].lower()
                mode = state[4].lower()
                speed = float(state[5])

                groove_class = get_object_fn(groove_type)
                dummy = groove_class(name="dummy")

                dim_x = dummy.parsed_size[0] * 2.0
                dim_y = dummy.parsed_size[1] * 2.0
                for site in dummy.worldbody.findall(".//site"):
                    if "horizontal_radius_site" in site.get("name", ""):
                        pos_str = site.get("pos")
                        if pos_str:
                            pos_vals = [abs(float(x)) for x in pos_str.split(" ")]

                            if pos_vals[0] > 0 and pos_vals[1] > 0:
                                dim_x = pos_vals[0] * 2.0
                                dim_y = pos_vals[1] * 2.0
                            else:
                                radius = max(pos_vals)
                                dim_x = radius * 2.0
                                dim_y = radius * 2.0
                        break
                margin = 0.01

                groove_bottom_offset = dummy.bottom_offset[2]

                if orientation == "x":
                    groove_step = dim_x + margin
                    target_quat_arr = [0.7071068, 0, 0, 0.7071068]
                else:
                    groove_step = dim_y + margin
                    target_quat_arr = [1, 0, 0, 0]

                ghost_obj = self.objects_dict.get(conveyor_name) or self.fixtures_dict.get(conveyor_name)
                try:
                    site_xml = None
                    for site in ghost_obj.worldbody.findall(".//site"):
                        if "ghost_active_region" in site.get("name", ""):
                            site_xml = site
                            break

                    if site_xml is not None:
                        size_str = site_xml.get("size")
                        half_length = float(size_str.split(" ")[1])
                    else:
                        half_length = 0.4
                except Exception:
                    half_length = 0.4

                conveyor_length = half_length * 2
                num_grooves = int(conveyor_length / groove_step)

                for i in range(num_grooves):
                    # CRITICAL FIX: If speed is negative, reverse the naming index.
                    # This physically places dyn_groove_0 at the +Y (upstream) end for reverse flow
                    idx = i if speed >= 0 else (num_grooves - 1 - i)
                    g_name = f"dyn_{groove_type}_{idx}"
                    
                    g_obj = groove_class(name=g_name)
                    self.objects_dict[g_name] = g_obj
                    self.dynamic_groove_names.append(g_name)

                self.setup_physical_track(
                    conveyor_name=conveyor_name,
                    speed=speed,
                    mode=mode,
                    groove_quat=target_quat_arr,
                    groove_bottom_offset=groove_bottom_offset,
                    groove_step=groove_step
                )

    def _load_sites_in_arena(self, mujoco_arena):
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]

        for object_region_name in list(region_dict.keys()):

            if "industry_workbench" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2 + self.workspace_offset[0],
                    (ranges[3] + ranges[1]) / 2 + self.workspace_offset[1],
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    z_offset=self.workspace_offset[2],
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone
                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos + np.array([0.0, 0.0, -0.90]),
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )

                            if "quat" not in site.attrib:
                                site_name = site.get("name", "UNKNOWN_SITE")
                                print(f"\n[!] XML BUG DETECTED: Site '{site_name}' is missing a 'quat' attribute! Forcing default '1 0 0 0'.")
                            # -------------------------------------
        self.object_sites_dict = object_sites_dict

        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        super()._add_placement_initializer()
        if hasattr(self.conditional_placement_initializer, "samplers"):
            for sampler_name, sampler in self.conditional_placement_initializer.samplers.items():
                sampler.__class__ = LiberoIndustrySampler

        if hasattr(self.conditional_placement_initializer, "samplers") and hasattr(
            self.conditional_placement_initializer, "sample_args"
        ):
            for sampler_name, sampler in self.conditional_placement_initializer.samplers.items():
                s_args = self.conditional_placement_initializer.sample_args.get(sampler_name, {})
                site_name = s_args.get("site_name", "")
                if site_name in self.object_sites_dict:
                    site_xy_size = self.object_sites_dict[site_name].size[:2]
                    sampler.x_ranges = [[-site_xy_size[0], site_xy_size[0]]]
                    sampler.y_ranges = [[-site_xy_size[1], site_xy_size[1]]]
                    sampler.num_ranges = len(sampler.x_ranges)

        if self.object_sampling == "uniform":
            yaw_rotation = (0.0, 0.0)
            xy_scale = 0.5  # Reduced from 0.95 to avoid belt-wall collisions
        else:
            yaw_rotation = (0.0, 2 * np.pi)
            xy_scale = 0.6  # Reduced from 0.95

        initializers = [
            self.placement_initializer,
            self.conditional_placement_initializer,
            self.conditional_placement_on_objects_initializer
        ]

        for init in initializers:
            if hasattr(init, 'samplers'):
                for sampler_name, sampler in init.samplers.items():
                    obj_name = sampler_name.replace("_sampler", "")

                    if self.is_fixture(obj_name):
                        correct_z_offset = self.fixtures_dict[obj_name].z_offset
                        sampler.z_offset = correct_z_offset
                    else:
                        if hasattr(sampler, "x_ranges") and hasattr(sampler, "y_ranges"):
                            scaled_x_ranges = []
                            for x_min, x_max in sampler.x_ranges:
                                x_center = 0.5 * (x_min + x_max)
                                x_half = 0.5 * (x_max - x_min) * xy_scale
                                scaled_x_ranges.append([x_center - x_half, x_center + x_half])
                            sampler.x_ranges = scaled_x_ranges

                            scaled_y_ranges = []
                            for y_min, y_max in sampler.y_ranges:
                                y_center = 0.5 * (y_min + y_max)
                                y_half = 0.5 * (y_max - y_min) * xy_scale
                                scaled_y_ranges.append([y_center - y_half, y_center + y_half])
                            sampler.y_ranges = scaled_y_ranges

                        sampler.rotation = yaw_rotation
                        sampler.rotation_axis = "z"
                        sampler.z_offset = 0.002
                        sampler.ensure_object_boundary_in_range = False
                        sampler.ensure_valid_placement = False

    def _check_success(self):
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            if len(state) == 3 and state[0].lower() == "routed":
                obj_name = state[1]
                routed_now = self._eval_predicate(state)
                was_delivered = hasattr(self, '_delivered_objects') and obj_name in self._delivered_objects
                result = (routed_now or was_delivered) and result
            else:
                result = self._eval_predicate(state) and result
        return result

    def get_partial_success(self):
        """Return the fraction of satisfied sub-goal predicates (0.0 to 1.0).

        Falls back to binary success when the goal_state is missing or empty.
        """
        goal_state = self.parsed_problem.get("goal_state", [])
        if not goal_state:
            return 1.0 if self._check_success() else 0.0

        satisfied = 0
        for state in goal_state:
            try:
                if self._eval_predicate(state):
                    satisfied += 1
            except Exception:
                pass

        return satisfied / len(goal_state)

    def _eval_predicate(self, state):
        """Custom success checking based on physical distance."""

        if len(state) == 3:
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()
        self.set_visualization()

    def set_visualization(self):
        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6586131746834771, 0.0, 1.6103500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="paperview",
            pos=[2.1, 0.535, 2.075],
            quat=[0.513, 0.353, 0.443, 0.645],
        )
