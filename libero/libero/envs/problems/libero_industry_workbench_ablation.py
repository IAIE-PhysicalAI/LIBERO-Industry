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

#from libero.libero.envs.predicates.ablation_predicates import AblationPhaseTracker
#from libero.libero.envs.predicates import get_predicate_fn_dict
#from libero.libero.envs.predicates.base_predicates import BinaryAtomic
# ======================================================

_DYNAMIC_SOURCE_REGION = re.compile(
    r"_(?P<kind>offset|tail)_(?P<family>region|bowl)(?:_d2d)?_(?P<slot>\d+)$"
)


def _load_slot_runtime_params(state, groove_step):
    """Read the runtime cadence from a parsed LoadSlot predicate."""
    speed = float(state[5])
    service_time = next(
        (
            float(param.lower()[len("service_"):])
            for param in state[6:]
            if param.lower().startswith("service_")
        ),
        1.0,
    )
    return {
        "speed": speed,
        "mode": state[4].lower(),
        "service_time": service_time,
        "transfer_time": groove_step / abs(speed) if speed else 0.0,
    }


def _timed_source_metadata(initial_state, site_name):
    """Translate a current BDDL source region into runtime timing metadata."""
    match = _DYNAMIC_SOURCE_REGION.search(site_name)
    if match is None:
        return None

    at_speed = next(
        (state for state in initial_state if state[0].lower() == "atspeed"),
        None,
    )
    if at_speed is None:
        return None

    tagged_values = {}
    for param in at_speed[3:]:
        lowered = param.lower()
        for tag in ("takt", "offset"):
            prefix = tag + "_"
            if lowered.startswith(prefix):
                tagged_values[tag] = float(lowered[len(prefix):])

    if set(tagged_values) != {"takt", "offset"}:
        return None

    slot = int(match.group("slot"))
    if match.group("family") == "bowl":
        # Bowl XML names run far-to-near; BDDL timing slots run near-to-far.
        slot = 4 - slot
    if slot < 1:
        raise ValueError("Invalid dynamic conveyor slot in {!r}".format(site_name))

    return {
        "speed": float(at_speed[2]),
        "takt": tagged_values["takt"],
        "offset": tagged_values["offset"],
        "slot": slot,
    }


def extract_targets_from_bddl(parsed_problem_dict):
    """
    独立函数：利用 Base Predicates 的多态性，动态提取需要放置的 Object 和 Receptacle。
    由于它在类外部，我们将 parsed_problem 字典作为参数传入。
    """
    goal_states = parsed_problem_dict.get("goal_state", [])
    target_objs = []
    target_receptacles = []
    predicate_dict = get_predicate_fn_dict()
    
    for state in goal_states:
        pred_name = state[0].lower()
        if pred_name in predicate_dict and isinstance(predicate_dict[pred_name], BinaryAtomic):
            if len(state) >= 3:
                obj_name = state[1]
                receptacle_name = state[2]
                if obj_name not in target_objs: target_objs.append(obj_name)
                if receptacle_name not in target_receptacles: target_receptacles.append(receptacle_name)
                
    return target_objs, target_receptacles
        
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
class Libero_Industry_Workbench_Ablation(ConveyorBeltMixin, ConveyorSlotMixin, BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.completed_objects = set()
        self.workspace_name = "industry_workbench"
        self.visualization_sites_list = []
        if "industry_workbench_full_size" in kwargs:
            self.industry_workbench_full_size = workbench_full_size
        else:
            self.industry_workbench_full_size = (1.0, 1.2, 0.05)
        self.industry_workbench_offset = (0.0, 0, 0.90)
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

        # 1. Pop the nested dictionary entirely so Robosuite's super() never sees it
        # Apply your default "uniform" object_sampling if nothing was passed
        nested_env_args = kwargs.pop("env_args", {"object_sampling": "uniform"})
        if nested_env_args:
             print(f"[DEBUG WORKBENCH] Extracted nested env_args: {nested_env_args}")

        # 2. Extract scene or apply the default hardcoded parameters
        self.custom_set_scene = kwargs.pop("set_scene", None)

        if self.custom_set_scene is None:
            print("[DEBUG WORKBENCH] Applying default Cartesian/Euler camera parameters.")
            self.custom_set_scene = {
                "camera": {
                    # Defines a +/- 2cm spatial shift across X, Y, Z
                    "perturb_pos_range": [0,0],   
                    # Defines a +/- 1.5 degree rotational shift across Roll, Pitch, Yaw
                    "perturb_euler_range": [0,0]    
                },
                "lighting": {
                    "tone": "neutral",
                    "noise_level": 0.0,
                    "castshadow": "false"
                },
                "table_style": {
                    "color": "green",
                    "finish": "matte"
                }
            }

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
        
        target_objs, target_receptacles = extract_targets_from_bddl(self.parsed_problem)
        
        self.ablation_tracker = AblationPhaseTracker(
            env=self, 
            target_objs_list=target_objs, 
            target_receptacles_list=target_receptacles
        )

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
        super()._reset_internal()
        if hasattr(self, "ablation_tracker"):
            self.ablation_tracker.reset()
        self.sim.forward()
        for state in self.parsed_problem["initial_state"]:
            if state[0].lower() == "alignedon":
                obj_name = state[1]
                region_name = state[2]
                
                if obj_name in self.objects_dict and region_name in self.object_sites_dict:
                    obj = self.objects_dict[obj_name]
                    site_obj = self.object_sites_dict[region_name]
                    
                    print(f"\n[DEBUG ALIGN] --- Snapping {obj_name} into {region_name} ---")
                    
                    try:
                        site_id = self.sim.model.site_name2id(site_obj.name)
                        site_mat = self.sim.data.site_xmat[site_id].reshape(3, 3)
                        site_quat_xyzw = T.mat2quat(site_mat)
                        site_quat_wxyz = T.convert_quat(site_quat_xyzw, to="wxyz")
                        site_pos = self.sim.data.site_xpos[site_id].copy() 
                        
                        obj_joint = obj.joints[-1]
                        qpos = self.sim.data.get_joint_qpos(obj_joint).copy()
                        
                        print(f"[DEBUG ALIGN] Original COM: {qpos[0:3]}")
                        print(f"[DEBUG ALIGN] Target Site Pos: {site_pos}")
                        
                        # Step 1: Set the rotation first and update kinematics
                        qpos[3:7] = site_quat_wxyz
                        self.sim.data.set_joint_qpos(obj_joint, qpos)
                        self.sim.forward() # Force MuJoCo to calculate new site positions
                        
                        # Step 2: Find the exact vector from bottom_site to COM
                        try:
                            # Safely attempt to use the exact bottom_site from your XML
                            bottom_site_name = f"{obj_name}_bottom_site"
                            bottom_site_id = self.sim.model.site_name2id(bottom_site_name)
                            bottom_pos = self.sim.data.site_xpos[bottom_site_id]
                            print(f"[DEBUG ALIGN] Using explicit {bottom_site_name}")
                        except Exception:
                            # Fallback if no bottom_site is found
                            global_bottom_offset = site_mat.dot(obj.bottom_offset)
                            bottom_pos = qpos[0:3] + global_bottom_offset
                            print(f"[DEBUG ALIGN] Using fallback bottom_offset")
                            
                        # The mathematical vector pointing from the bottom to the COM
                        com_offset_vector = qpos[0:3] - bottom_pos 
                        
                        # Step 3: Target = Site Position + Vector to COM + 2mm safety drop
                        target_xyz = site_pos + com_offset_vector + np.array([0, 0, 0.002])
                        
                        qpos[0:3] = target_xyz
                        self.sim.data.set_joint_qpos(obj_joint, qpos)
                        
                        print(f"[DEBUG ALIGN] Final Target COM: {target_xyz}")
                        print(f"[DEBUG ALIGN] ---------------------------------------------")
                    except Exception as e:
                        print(f"[DEBUG ALIGN] ERROR for {obj_name}: {e}")

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

        if not found_running_state and not getattr(self, "dynamic_groove_names", []):
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
            
        ROTATION_MAP = {
            "v_groove": np.array([[1, 0, 0], 
                                  [0, 0.7071068, -0.7071068], 
                                  [0, 0.7071068, 0.7071068]]),
            "cross_pose": np.array([[0, 0, 1], 
                                    [0, 1, 0], 
                                    [-1, 0, 0]]),
            "inverted": np.array([[1, 0, 0], 
                                  [0, -1, 0], 
                                  [0, 0, -1]])
        }

        for state in self.parsed_problem["initial_state"]:
            if state[0].lower() == "on" and "conveyor_ghost" in state[2]:
                obj_to_teleport = state[1]

                if not available_grooves:
                    continue

                chosen_groove = available_grooves.pop(0)

                groove_obj = self.objects_dict[chosen_groove]
                groove_joint = groove_obj.joints[-1]
                
                # Fetch exact goal region coordinates
                groove_site_id = self.sim.model.site_name2id(f"{chosen_groove}_goal_region")
                site_pos = self.sim.data.site_xpos[groove_site_id].copy()
                site_mat = self.sim.data.site_xmat[groove_site_id].reshape(3, 3)

                obj = self.objects_dict[obj_to_teleport]
                obj_name = obj.name
                n1, n2 = obj_name.lower(), chosen_groove.lower()

                # Apply the static dictionary dispatch
                target_obj_mat = site_mat.copy()
                
                if "v_groove" in n2:
                    target_obj_mat = site_mat.dot(ROTATION_MAP["v_groove"])
                elif ("upright" in n1 and "flat" in n2) or ("flat" in n1 and "upright" in n2):
                    target_obj_mat = site_mat.dot(ROTATION_MAP["cross_pose"])
                elif "inverted" in n1 or "inverted" in n2:
                    target_obj_mat = site_mat.dot(ROTATION_MAP["inverted"])

                target_quat_xyzw = T.mat2quat(target_obj_mat)
                target_quat = T.convert_quat(target_quat_xyzw, to="wxyz")

                # ==========================================================
                # 2. APPLY ROTATION AND RE-CALCULATE OFFSET
                # ==========================================================
                obj_joint = obj.joints[-1]
                qpos = self.sim.data.get_joint_qpos(obj_joint).copy()
                
                # We MUST apply rotation first, so bottom_site coordinates update accurately
                qpos[3:7] = target_quat
                self.sim.data.set_joint_qpos(obj_joint, qpos)
                self.sim.forward() 

                try:
                    bottom_site_id = self.sim.model.site_name2id(f"{obj_name}_bottom_site")
                    bottom_pos = self.sim.data.site_xpos[bottom_site_id]
                except Exception:
                    bottom_pos = qpos[0:3] + target_obj_mat.dot(obj.bottom_offset)

                com_offset = qpos[0:3] - bottom_pos

                # ==========================================================
                # 3. SAFE KINEMATIC DROP
                # ==========================================================
                target_pos = site_pos + com_offset
                
                # Elevate slightly to guarantee no mesh penetration at spawn
                if "tray" in n2:
                    target_pos[2] += 0.03  
                elif "v_groove" in n2:
                    target_pos[2] += 0.02  
                else:
                    target_pos[2] += 0.01  

                qpos[0:3] = target_pos
                self.sim.data.set_joint_qpos(obj_joint, qpos)

                # Match the exact belt velocity so it falls straight down relative to the slot
                self.sim.data.set_joint_qvel(obj_joint, self.sim.data.get_joint_qvel(groove_joint))
                
        self.sim.forward()

    def step(self, action):
        # 1. Increment step counter (keep for tracking)
        self.current_step += 1

        # 2. Handle "pulse" mode (rhythmic start/stop)
        if self.current_conveyor_params["mode"] == "pulse":
            sim_time = self.sim.data.time
            is_moving = 1.0 if (sim_time % 2.0) < 1.0 else 0.0
            self.conveyor_velocity = self.current_conveyor_params["speed"] * is_moving

        # 3. Run standard simulation logic
        obs, reward, base_done, info = super().step(action)

        # 4. ABLATION LOGIC: Check success EVERY frame, regardless of Takt time
        is_success = self._check_success()
        
        # 5. Terminate immediately if success is achieved
        if is_success:
            done = True
            reward = 1.0
        else:
            # If not success, use the base environment's termination state
            done = base_done
            reward = 0.0

        # 6. Update info dictionary for the client (serve_industry.py)
        info["takt_limit_reached"] = False  # Ablations ignore takt limit
        info["conveyor_speed"] = self.conveyor_velocity
        info["conveyor_mode"] = self.current_conveyor_params["mode"]
        info["success"] = is_success  

        # 7. ABLATION TRACKER
        if hasattr(self, 'ablation_tracker'):
            phase_metrics = self.ablation_tracker.evaluate_phases(is_success)
            
            # Defensive check: Ensure the tracker did not return None
            if phase_metrics is not None:
                info["ablation_phase_metrics"] = phase_metrics
                
                # 3. Check for early termination (Irreversible Saturation) safely
                global_faults = phase_metrics.get("global_faults", {})
                if global_faults.get("early_terminate_trigger", False):
                    # Force the episode to end
                    done = True
            else:
                # Fallback to empty structures to prevent downstream logging crashes
                info["ablation_phase_metrics"] = {
                    "suite_type": "UNKNOWN",
                    "global_faults": {},
                    "object_metrics": {}
                }

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
                expected_format = "('LoadSlot', 'conveyor_name', 'groove_type', 'orientation (x/y)', 'mode', 'speed', '[service_X]')"
                assert len(state) >= 6, f"\n[BDDL ERROR] Incorrect arguments for LoadSlot.\nExpected: {expected_format}\nGot: {state}\n"

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

                self.current_conveyor_params.update(
                    _load_slot_runtime_params(state, groove_step)
                )

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
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone
                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
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
                            if "quat" not in site.attrib:
                                site_name = site.get("name", "UNKNOWN_SITE")
                                print(f"\n[!] XML BUG DETECTED: Site '{site_name}' is missing a 'quat' attribute! Forcing default '1 0 0 0'.")
                                site.set("quat", "1 0 0 0") # Actively inject the default
                                
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

                    timing = _timed_source_metadata(
                        self.parsed_problem.get("initial_state", []), site_name
                    )
                    if timing is not None:
                        sampler._conveyor_timing = timing
                        sampler.x_ranges = [[0.0, 0.0]]
                        sampler.y_ranges = [[0.0, 0.0]]
                        sampler.num_ranges = 1

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
        goal_state = self.parsed_problem.get("goal_state", [])
        result = True
    
        for state in goal_state:
            is_state_true = False
            
            if len(state) == 3 and state[0].lower() in ["routed", "dynamicon"]:
                obj_name = state[1]
                routed_now = self._eval_predicate(state)
                was_delivered = hasattr(self, '_delivered_objects') and obj_name in self._delivered_objects
                is_state_true = (routed_now or was_delivered)
            else:
                is_state_true = self._eval_predicate(state)

            if self.current_step % 10 == 0:                
                if not is_state_true and state[0].lower() == "on":
                    try:
                        obj_name, site_name = state[1], state[2]
                        
                        # --- ROBUST BODY ID LOOKUP ---
                        obj_body_id = None
                        for suffix in ["", "_main", "_root"]:
                            try:
                                obj_body_id = self.sim.model.body_name2id(f"{obj_name}{suffix}")
                                break
                            except ValueError:
                                pass
                        
                        if obj_body_id is None:
                            print(f"      [MATH ERROR] Could not locate body for {obj_name} even with suffixes.")
                            continue
                            
                        # Fetch IDs and positions
                        site_id = self.sim.model.site_name2id(site_name)
                        
                        obj_pos = self.sim.data.body_xpos[obj_body_id]
                        site_pos = self.sim.data.site_xpos[site_id]
                        
                        z_diff = obj_pos[2] - site_pos[2]
                        xy_dist = np.linalg.norm(obj_pos[:2] - site_pos[:2])
                        
                        print(f"      [MATH] Found Object as: {self.sim.model.body_id2name(obj_body_id)}")
                        print(f"      [MATH] Z-Difference (Obj_Z - Site_Z): {z_diff:.5f}m")
                        print(f"      [MATH] XY-Distance to Target Center : {xy_dist:.5f}m")
                        
                        if z_diff < 0:
                            print("      [FLAG] Object Z is lower than Site Z (Mesh Penetration or Toppled).")
                    except Exception as e:
                        print(f"      [MATH ERROR] Exception during math extraction: {e}")
                
            result = is_state_true and result
            
        return result

    def get_partial_success(self):
        goal_state = self.parsed_problem.get("goal_state", [])
        if not goal_state:
            val = 1.0 if self._check_success() else 0.0
            return {"pcr_transfer": val, "pcr_slotting": val}

        init_state = self.parsed_problem.get("initial_state", self.parsed_problem.get("init_state", []))

        # 1. Map goal objects to their target destinations
        goal_objects = {}  # {obj_name: target_dest}
        for state in goal_state:
            if len(state) >= 3:
                obj_name = state[1]
                target_dest = state[2]
                goal_objects[obj_name] = target_dest

        # 2. Identify confounders (goal objects that start on their target dest)
        confounder_names = set()
        for state in init_state:
            if len(state) >= 3:
                # Bulletproof check: scan the predicate for both the object and its destination
                for i in range(1, len(state)):
                    for j in range(1, len(state)):
                        if i != j:
                            obj_candidate = state[i]
                            loc_candidate = state[j]
                            if obj_candidate in goal_objects and goal_objects[obj_candidate] == loc_candidate:
                                confounder_names.add(obj_candidate)

        # 3. Categorize
        manipulating_objects = [obj for obj in goal_objects if obj not in confounder_names]

        # 4. Evaluate predicates with latching
        successful_manipulating = 0
        successful_confounders = 0

        for state in goal_state:
            if len(state) >= 3:
                obj_name = state[1]
                is_success = False
                try:
                    # Check current predicate status
                    is_success = self._eval_predicate(state)
                    # Include latching for dynamic targets
                    if not is_success and hasattr(self, '_delivered_objects'):
                        is_success = obj_name in self._delivered_objects
                except Exception:
                    pass

                if is_success:
                    if obj_name in confounder_names:
                        successful_confounders += 1
                    else:
                        successful_manipulating += 1

        # 5. Calculate final PCRs
        total_manip = len(manipulating_objects)
        total_goal = len(goal_state)

        pcr_transfer = float(successful_manipulating / total_manip) if total_manip > 0 else 0.0
        pcr_slotting = float((successful_manipulating + successful_confounders) / total_goal) if total_goal > 0 else 0.0

        return {"pcr_transfer": pcr_transfer, "pcr_slotting": pcr_slotting}

    def _eval_predicate(self, state):
        """
        Custom success checking for the Industry Benchmark.
        Intercepts 'On' to bypass rigid XML thickness constraints while maintaining 
        strict geometric precision in the local reference frame.
        """
        if len(state) == 3:
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            
            # =========================================================
            # PRECISION OVERRIDE: Localized 'On' Evaluation
            # =========================================================
            if predicate_fn_name.lower() == "on":
                try:
                    # 1. Robust ID Lookup (Bypasses the XML suffix mismatch)
                    obj_body_id = None
                    for suffix in ["", "_main", "_root"]:
                        try:
                            obj_body_id = self.sim.model.body_name2id(f"{object_1_name}{suffix}")
                            break
                        except ValueError:
                            pass
                            
                    if obj_body_id is not None:
                        site_id = self.sim.model.site_name2id(object_2_name)
                        
                        # 2. Extract Positions, Matrix, and Strict Size
                        obj_pos = self.sim.data.body_xpos[obj_body_id]
                        site_pos = self.sim.data.site_xpos[site_id]
                        site_mat = self.sim.data.site_xmat[site_id].reshape(3, 3)
                        site_size = self.sim.model.site_size[site_id]
                        
                        # 3. Transform to Local Frame for High-Precision Alignment
                        # This ensures accuracy even if the plate/fixture is rotated
                        rel_pos = obj_pos - site_pos
                        local_rel_pos = site_mat.T.dot(rel_pos)
                        
                        # The COM must fall strictly within the exact X/Y bounds of the XML site
                        is_aligned_x = abs(local_rel_pos[0]) <= site_size[0]
                        is_aligned_y = abs(local_rel_pos[1]) <= site_size[1]
                        
                        # The COM must be above the site's origin (ignoring the 1mm thickness limitation)
                        is_above = local_rel_pos[2] >= 0.0
                        
                        # 4. Strict Topological Contact Check
                        is_contacting = False
                        fixture_name = object_2_name.replace("_goal_region", "")
                        
                        for i in range(self.sim.data.ncon):
                            c = self.sim.data.contact[i]
                            g1 = self.sim.model.geom_id2name(c.geom1) or ""
                            g2 = self.sim.model.geom_id2name(c.geom2) or ""
                            
                            is_obj = (object_1_name in g1) or (object_1_name in g2)
                            is_fixture = (fixture_name in g1) or (fixture_name in g2)
                            
                            if is_obj and is_fixture:
                                is_contacting = True
                                break
                                
                        if is_aligned_x and is_aligned_y and is_above and is_contacting:
                            return True
                except Exception:
                    pass # Silently fallback if math extraction fails
            # =========================================================

            # Fallback to standard Libero evaluation for all other predicates
            # (Routed, DynamicOn, ReleasedOn, AlignedOn will process normally)
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
            
        return False

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



@register_problem
class Libero_Industry_Workbench_D0_Ablation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "industry_workbench" # Overridden from "main_table"
        self.visualization_sites_list = []
        self.industry_workbench_full_size = (1.0, 1.2, 0.05)
        self.industry_workbench_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.industry_workbench_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.industry_workbench_offset})
        
        # Route to the Industry Arena and XMLs
        kwargs.update({"arena_type": "industry_workbench"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_industry_workbench_base_style.xml"})

        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                   "scene_properties": {
                        "floor_style": "gray-ceramic",
                        "wall_style": "yellow-linen",
                    }
                }
            )
        # ==========================================================
        # SCENE CONFIGURATION EXTRACTION & FALLBACK
        # ==========================================================
        self.custom_set_scene = kwargs.pop("set_scene", None)

        if self.custom_set_scene is None:
            print("[DEBUG D0 ABLATION] Applying default Cartesian/Euler camera parameters.")
            self.custom_set_scene = {
                "camera": {
                    "perturb_pos_range": [0, 0],   
                    "perturb_euler_range": [0, 0]    
                },
                "lighting": {
                    "tone": "neutral",
                    "noise_level": 0.0,
                    "castshadow": "false"
                },
                "table_style": {
                    "color": "green",
                    "finish": "matte"
                }
            }
        super().__init__(bddl_file_name, *args, **kwargs)

        target_objs, target_receptacles = extract_targets_from_bddl(self.parsed_problem)
        
        self.ablation_tracker = AblationPhaseTracker(
            env=self, 
            target_objs_list=target_objs, 
            target_receptacles_list=target_receptacles
        )

    def _reset_internal(self):
        super()._reset_internal()
        if hasattr(self, "ablation_tracker"):
            self.ablation_tracker.reset()
        
    def _load_model(self):
        super()._load_model()
        
        # Apply the scene modifications to the FINAL merged task tree
        if self.custom_set_scene is not None:
            if hasattr(self.model.mujoco_arena, "apply_set_scene"):
                # Pass the XML root and the config dictionary to the arena
                self.model.mujoco_arena.apply_set_scene(self.model.root, self.custom_set_scene)
            else:
                print("[WARNING] mujoco_arena does not have apply_set_scene method. Scene configs ignored.")

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if hasattr(self, 'ablation_tracker'):
            info["ablation_phase_metrics"] = self.ablation_tracker.evaluate_phases(
                self._check_success()
            )
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
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "main_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
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
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved. Consider conjunction goals at the moment
        """
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
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
                    # We toggle the alpha value
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

        # For visualization purpose
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
