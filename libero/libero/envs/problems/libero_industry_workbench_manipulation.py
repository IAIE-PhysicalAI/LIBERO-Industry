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
class Libero_Industry_Workbench_Manipulation(ConveyorBeltMixin, ConveyorSlotMixin, BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.completed_objects = set()
        self.workspace_name = "industry_workbench"
        self.visualization_sites_list = []
        if "industry_workbench_full_size" in kwargs:
            self.industry_workbench_full_size = workbench_full_size
        else:
            self.industry_workbench_full_size = (1.0, 1.2, 0.05)
        self.industry_workbench_offset = (0, 0, 0.90)
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
        super()._reset_internal()
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
            self.sim.forward()
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
                # Only update speed and mode. 
                # locked by the LoadSlot kinematic derivation.
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
        if ConveyorSlotMixin._seat_initial_headsets_in_inlays(self):
            ConveyorSlotMixin._apply_physical_track(self)
            self.sim.forward()
        print("\n[DEBUG OBJECT PLACEMENTS] --- POST-RESET WORLD COORDINATES & GEOM IDS ---")
        self.sim.forward() # Ensure kinematics are fully updated
        
        for obj_name, obj in self.objects_dict.items():
            if "dyn_" in obj_name:
                continue
            try:
                body_id = self.obj_body_id.get(obj_name)
                if body_id is not None:
                    # Get world position
                    world_pos = self.sim.data.body_xpos[body_id]
                    
                    # Get all geometry IDs belonging to this body
                    geom_body_ids = np.where(self.sim.model.geom_bodyid == body_id)[0].tolist()
                    
                    print(f"[DEBUG OBJECT] Object: {obj_name}")
                    #print(f"  -> Body ID: {body_id}")
                    #print(f"  -> Geom IDs: {geom_body_ids}")
                    print(f"  -> World Position (X, Y, Z): {world_pos}")
            except Exception as e:
                print(f"[DEBUG OBJECT] Could not fetch data for {obj_name}: {e}")
        print("-" * 70 + "\n")

    def step(self, action):
        # Increment step counter for takt time tracking
        self.current_step += 1

        # Handle "pulse" mode (rhythmic start/stop)
        if self.current_conveyor_params["mode"] == "pulse":
            sim_time = self.sim.data.time
            
            # Fetch the parsed times (defaulting to 1.0s if missing)
            service_t = self.current_conveyor_params.get("service_time", 1.0)
            transfer_t = self.current_conveyor_params.get("transfer_time", 1.0)
            
            # The total cycle is the time spent moving + the time spent stopped
            cycle_time = transfer_t + service_t
            
            # Determine where we are in the current cycle
            cycle_position = sim_time % cycle_time
            
            # Move during the transfer phase, stop during the service phase
            is_moving = 1.0 if cycle_position < transfer_t else 0.0
            
            self.conveyor_velocity = self.current_conveyor_params["speed"] * is_moving

        # Run standard simulation logic
        obs, reward, base_done, info = super().step(action)

        is_success = False
        takt_limit_reached = (self.current_step >= self.takt_steps)

        if takt_limit_reached:
            is_success = self._check_success()

        if is_success:
            done = True
            reward = 1.0
        else:
            done = base_done  
            reward = 0.0

        # Update info dictionary
        info["takt_limit_reached"] = takt_limit_reached
        info["conveyor_speed"] = self.conveyor_velocity
        info["conveyor_mode"] = self.current_conveyor_params["mode"]
        info["success"] = is_success  

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
                # 1. Enforce minimum parameters, expecting optional service time
                expected_format = "('LoadSlot', 'conveyor_name', 'groove_type', 'orientation', 'mode', 'speed', '[service_X]')"
                assert len(state) >= 6, f"\n[BDDL ERROR] Incorrect arguments for LoadSlot.\nExpected: {expected_format}\nGot: {state}\n"

                conveyor_name = state[1]
                groove_type = state[2]
                orientation = state[3].lower()
                mode = state[4].lower()
                speed = float(state[5])
                self.current_conveyor_params.update({"speed": speed, "mode": mode})

                # 2. Base groove instantiation and dimension calculations
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

                # 3. DYNAMIC KINEMATIC CALCULATION & BDDL PARSING
                # Parse ONLY the service time parameter from the BDDL
                found_service = False
                for extra_param in state[6:]:
                    if "service_" in extra_param.lower():
                        self.current_conveyor_params["service_time"] = float(extra_param.lower().replace("service_", ""))
                        found_service = True

                if not found_service:
                    print(f"\n[WARNING] BDDL LoadSlot for '{conveyor_name}': 'service_xxx' is missing. Defaulting to service_1.0")
                    self.current_conveyor_params["service_time"] = 1.0

                # Derive duration from speed magnitude while preserving the
                # signed speed itself for the physical travel direction.
                speed_magnitude = abs(speed)
                if speed_magnitude > 0:
                    calculated_transfer = groove_step / speed_magnitude
                else:
                    calculated_transfer = 0.0
                    
                self.current_conveyor_params["transfer_time"] = calculated_transfer

                # Output the analytical breakdown to the terminal
                service_t = self.current_conveyor_params["service_time"]
                imposed_takt = service_t + calculated_transfer
                
                print(f"\n[INFO] Conveyor '{conveyor_name}' Kinematics:")
                print(f"[INFO] -> Groove Spacing (s_c): {groove_step:.4f} meters")
                print(f"[INFO] -> Target Speed (v): {speed:.4f} m/s")
                if mode == "pulse":
                    print(f"[INFO] -> Derived Transfer Time (T_r): {calculated_transfer:.3f}s")
                    print(f"[INFO] -> Defined Service Time (T_s): {service_t:.3f}s")
                    print(f"[INFO] -> Imposed Takt Time (T_t): {imposed_takt:.3f}s\n")
                else:
                    print(f"[INFO] -> Note: Conveyor is in '{mode}' mode. Intermittent Takt pacing (T_t) does not apply.\n")
                # -------------------------------------
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
                                
                                if not site_name.endswith("_default_site"):
                                    print(f"\n[!] XML BUG DETECTED: Site '{site_name}' is missing a 'quat' attribute! Forcing default '1 0 0 0'.")
                                
                                # Always actively inject the default to prevent MuJoCo errors
                                site.set("quat", "1 0 0 0")
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
        # --- DYNAMIC CONVEYOR REGION CALCULATION (START) ---
        # S2D's conveyor_ghost is configured by LoadSlot rather than AtSpeed.
        # The generic conveyor-belt fallback is still needed below, but its
        # missing-AtSpeed warning is not actionable for that suite.
        is_s2d_ghost_scene = "conveyor_ghost" in self.parsed_problem.get("fixtures", {})
        if hasattr(self.conditional_placement_initializer, "samplers"):
            
            v = None
            takt_time = None
            offset_time = None
            conveyor_name = None
            
            # Parse dynamic parameters securely using explicit BDDL tags
            init_states = self.parsed_problem.get("initial_state", self.parsed_problem.get("init_state", []))
            for state in init_states:
                if state[0].lower() == "atspeed":
                    conveyor_name = state[1]
                    try:
                        v = float(state[2])
                    except ValueError:
                        pass
                    for param in state[3:]:
                        if "takt_" in param.lower():
                            takt_time = float(param.lower().replace("takt_", ""))
                        elif "offset_" in param.lower():
                            offset_time = float(param.lower().replace("offset_", ""))

            if None in (v, takt_time, offset_time, conveyor_name):
                if not is_s2d_ghost_scene:
                    print("[DEBUG CONVEYOR] WARNING: Missing parameters. Falling back to (v=0.05, takt=5.0, offset=2.0).")
                v = v if v is not None else 0.05
                takt_time = takt_time if takt_time is not None else 5.0
                offset_time = offset_time if offset_time is not None else 2.0
                conveyor_name = conveyor_name if conveyor_name is not None else "conveyor_belt_1"

            for sampler_name, sampler in self.conditional_placement_initializer.samplers.items():
                sampler.__class__ = LiberoIndustrySampler

        if hasattr(self.conditional_placement_initializer, "samplers") and hasattr(
            self.conditional_placement_initializer, "sample_args"
        ):
            for sampler_name, sampler in self.conditional_placement_initializer.samplers.items():
                s_args = self.conditional_placement_initializer.sample_args.get(sampler_name, {})
                site_name = s_args.get("site_name", "")
                if site_name in self.object_sites_dict:
                    site_size = self.object_sites_dict[site_name].size
                    if isinstance(site_size, str):
                        site_xy_size = [float(value) for value in site_size.split()[:2]]
                    else:
                        site_xy_size = site_size[:2]
                    sampler.x_ranges = [[-site_xy_size[0], site_xy_size[0]]]
                    sampler.y_ranges = [[-site_xy_size[1], site_xy_size[1]]]
                    sampler.num_ranges = len(sampler.x_ranges)

                    # Apply dynamic timing *after* generic range setup.  The
                    # sampler resolves centre_region and belt orientation at
                    # placement time, once the fixture has its real world pose.
                    dynamic_match = re.search(
                        r"_(offset|tail)_region(?:_d2d)?_(\d+)$", site_name
                    )
                    if dynamic_match:
                        slot = int(dynamic_match.group(2))
                        sampler._conveyor_timing = {
                            "speed": v,
                            "takt": takt_time,
                            "offset": offset_time,
                            "slot": slot,
                        }
                        sampler.x_ranges = [[0.0, 0.0]]
                        sampler.y_ranges = [[0.0, 0.0]]
                        sampler.num_ranges = 1

        # --- DYNAMIC CONVEYOR REGION CALCULATION (END) ---

        if self.object_sampling == "uniform":
            yaw_rotation = (0.0, 0.0)
            xy_scale = 0.5  # Reduced from 0.95 to avoid belt-wall collisions
        else:
            yaw_rotation = (0.0, 2 * np.pi)
            xy_scale = 0.2  # Reduced from 0.95

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

    def _requires_live_routed_state(self, state):
        """Disable delivery latching only for the cream-jar V groove."""
        if (
            len(state) < 3
            or state[0].lower() != "routed"
            or "cream_jar" not in state[1].lower()
        ):
            return False
        target_fixture = state[2]
        init_state = self.parsed_problem.get(
            "initial_state", self.parsed_problem.get("init_state", [])
        )
        return any(
            len(init) >= 3
            and init[0].lower() == "loadslot"
            and init[1] == target_fixture
            and init[2].lower() == "cream_jar_v_groove"
            for init in init_state
        )

    def _check_success(self):
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            if len(state) == 3 and state[0].lower() in ["routed", "dynamicon"]:
                obj_name = state[1]
                routed_now = self._eval_predicate(state)
                was_delivered = (
                    not self._requires_live_routed_state(state)
                    and hasattr(self, '_delivered_objects')
                    and obj_name in self._delivered_objects
                )
                result = (routed_now or was_delivered) and result
            else:
                result = self._eval_predicate(state) and result
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
                    if (
                        not is_success
                        and not self._requires_live_routed_state(state)
                        and hasattr(self, '_delivered_objects')
                    ):
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
