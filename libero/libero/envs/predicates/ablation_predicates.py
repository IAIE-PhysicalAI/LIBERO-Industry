import numpy as np
import robosuite.utils.transform_utils as T

class AblationPhaseTracker:
    def __init__(self, env, target_objs_list, target_receptacles_list):
        self.env = env
        self.target_objs = target_objs_list
        self.target_receptacles = target_receptacles_list
        
        # Industrial Constant Calibration (VDI 2860 / ISO 9283 / IEC 60204-1)
        self.SATURATION_PERSISTENCE_LIMIT = 30  # 1.5 seconds at 20Hz
        self.IDLE_PERSISTENCE_LIMIT = 40        # 2.0 seconds at 20Hz
        self.WORKSPACE_Z_MIN = 0.6

        self.reset()

    def reset(self):
        self.global_faults = {
            "out_of_bounds": False,           
            "kinematic_saturation": False,    
            "policy_freeze_idle": False,        # Track OEE Stalled/Idle state
            "early_terminate_trigger": False,  
            "total_eef_displacement": 0.0,    
        }
        
        # Internal frame-counters
        self._consecutive_saturation_frames = 0
        
        # Jitter-Proof Anchor Variables
        self._idle_anchor_pos = None
        self._consecutive_idle_frames = 0
        self._last_eef_pos = None

        self.history = {}
        self._dynamic_states = {}

        for obj in self.target_objs:
            self.history[obj] = {
                "reached": False, 
                "grasped": False, 
                "lifted": False,
                "transported": False, 
                "placed_attempt": False, 
                "micro_slip": False,
                "dropped_midway": False  # Crucial for distinguishing slip timing
            }
            self._dynamic_states[obj] = {
                "steps_in_air": 0,
                "was_grasped_last_step": False,
                "lift_z_threshold": None,
                "init_local_pose": None,
                "has_terminal_fault": False
            }

    def _get_obj_kinematics(self, obj_name):
        if obj_name in self.env.sim.model.site_names:
            site_id = self.env.sim.model.site_name2id(obj_name)
            site_mat = self.env.sim.data.site_xmat[site_id].reshape(3, 3)
            return self.env.sim.data.site_xpos[site_id], T.mat2quat(site_mat)

        target_obj = self.env.get_object(obj_name)
        actual_name = target_obj.root_body if (target_obj is not None and hasattr(target_obj, 'root_body')) else obj_name
        
        if actual_name not in self.env.sim.model.body_names:
            actual_name = f"{obj_name}_main" if f"{obj_name}_main" in self.env.sim.model.body_names else obj_name
        try:
            body_id = self.env.sim.model.body_name2id(actual_name)
            return self.env.sim.data.body_xpos[body_id], self.env.sim.data.body_xquat[body_id]
        except ValueError:
            return None, None

    def _site_contains_xy(self, site_name, obj_pos):
        if site_name not in self.env.sim.model.site_names:
            return False
        site_id = self.env.sim.model.site_name2id(site_name)
        site_pos = self.env.sim.data.site_xpos[site_id]
        site_mat = self.env.sim.data.site_xmat[site_id].reshape(3, 3)
        local_pos = site_mat.T.dot(obj_pos - site_pos)
        return bool(np.all(np.abs(local_pos[:2]) <= self.env.sim.model.site_size[site_id][:2]))

    def _target_contains_xy(self, target_name, obj_name, obj_pos):
        if self._site_contains_xy(target_name, obj_pos):
            return True
        if target_name in self.env.sim.model.site_names:
            return False

        if "conveyor_ghost" in target_name:
            tray_names = getattr(self.env, "dynamic_groove_names", [])
            lock = getattr(self.env, "_tray_planar_locks", {}).get(obj_name)
            if lock and lock.get("tray") in tray_names:
                return True
            return any(
                self._site_contains_xy(f"{tray_name}_goal_region", obj_pos)
                for tray_name in tray_names
            )

        target_obj = self.env.get_object(target_name)
        target_pos, _ = self._get_obj_kinematics(target_name)
        if target_obj is None or target_pos is None:
            return False

        radius_site = f"{target_obj.naming_prefix}horizontal_radius_site"
        if radius_site in self.env.sim.model.site_names:
            site_id = self.env.sim.model.site_name2id(radius_site)
            radius = np.linalg.norm(self.env.sim.data.site_xpos[site_id][:2] - target_pos[:2])
        else:
            radius = abs(float(target_obj.horizontal_radius))
        return bool(np.linalg.norm(obj_pos[:2] - target_pos[:2]) <= radius)

    def _is_pre_reach_idle(self):
        has_active_object = any(
            metrics["reached"] and not metrics["placed_attempt"]
            and not self._dynamic_states[name]["has_terminal_fault"]
            for name, metrics in self.history.items()
        )
        has_terminal_fault = any(
            state["has_terminal_fault"] for state in self._dynamic_states.values()
        )
        return (
            self._consecutive_idle_frames >= self.IDLE_PERSISTENCE_LIMIT
            and any(not metrics["reached"] for metrics in self.history.values())
            and not has_active_object
            and not has_terminal_fault
        )

    def evaluate_phases(self, is_success=False):
        robot = self.env.robots[0]
        eef_pos = self.env.sim.data.site_xpos[robot.eef_site_id].copy()
        
        # 修正原代码无切片导致维度不匹配的隐患，确保为 3x3 旋转矩阵
        eef_mat = self.env.sim.data.site_xmat[robot.eef_site_id].reshape(3, 3).copy()
        # -------------------------------------------------------------
        # 1. Kinematic Velocity & OEE Idle Tracking (Jitter-Proof)
        # -------------------------------------------------------------
        if self._idle_anchor_pos is None:
            self._idle_anchor_pos = eef_pos.copy()

        if hasattr(self, '_last_eef_pos') and self._last_eef_pos is not None:
            self.global_faults["total_eef_displacement"] += np.linalg.norm(eef_pos - self._last_eef_pos)
        self._last_eef_pos = eef_pos.copy()

        anchor_displacement = np.linalg.norm(eef_pos - self._idle_anchor_pos)
        if anchor_displacement > 0.005:
            self._idle_anchor_pos = eef_pos.copy()
            self._consecutive_idle_frames = 0
        else:
            self._consecutive_idle_frames += 1

        # -------------------------------------------------------------
        # 2. Kinematic Saturation & Safety Interrupt Logic (IEC 60204-1)
        # -------------------------------------------------------------
        is_saturated_this_frame = False
        try:
            qpos = robot._joint_positions
            limits_low, limits_high = robot.joint_limits
            if np.any(qpos - limits_low < 0.03) or np.any(limits_high - qpos < 0.03):
                is_saturated_this_frame = True
                self.global_faults["kinematic_saturation"] = True
        except Exception:
            pass 

        if is_saturated_this_frame:
            self._consecutive_saturation_frames += 1
        else:
            self._consecutive_saturation_frames = 0

        if self._consecutive_saturation_frames >= self.SATURATION_PERSISTENCE_LIMIT:
            self.global_faults["early_terminate_trigger"] = True

        if self._is_pre_reach_idle():
            self.global_faults["policy_freeze_idle"] = True

        # -------------------------------------------------------------
        # 3. Object Phase State Machine
        # -------------------------------------------------------------
        for obj_name in self.target_objs:
            obj_pos, _ = self._get_obj_kinematics(obj_name)
            target_obj = self.env.get_object(obj_name)
            if target_obj is None or obj_pos is None: continue
            
            h = self.history[obj_name]
            d = self._dynamic_states[obj_name]
            is_grasped = self.env._check_grasp(robot.gripper, target_obj.contact_geoms)

            if obj_pos[2] < self.WORKSPACE_Z_MIN:
                self.global_faults["out_of_bounds"] = True

            if not h["reached"] and np.linalg.norm(eef_pos - obj_pos) < 0.05:
                h["reached"] = True

            if is_grasped and not d["has_terminal_fault"]:
                h["grasped"] = True
                h["reached"] = True 
                h["micro_slip"] = False
                h["dropped_midway"] = False

            # --- 修正点 1：复合式起升状态检测 (Hybrid Lift Gate) ---
            if is_grasped and not h["lifted"]:
                if d["lift_z_threshold"] is None:
                    # 动态锚定抓取瞬间的物理坐标
                    d["lift_z_threshold"] = obj_pos[2] + 0.02 
                    d["init_obj_pos_at_grasp"] = obj_pos.copy()
                
                # 规则：满足以下两个条件之一即可断定已脱离初始稳态（进入 Lift 阶段）：
                # 核心依据：Z轴垂直提升满2cm，或者水平位移超过3cm（防止低空直接平移被漏判）
                horizontal_shift = np.linalg.norm(obj_pos[:2] - d["init_obj_pos_at_grasp"][:2])
                if (obj_pos[2] > d["lift_z_threshold"]) or (horizontal_shift > 0.03):
                    h["lifted"] = True
                    d["init_local_pose"] = eef_mat.T.dot(obj_pos - eef_pos) 

            # --- 修正点 2：稳态与时序状态更新 ---
            if h["lifted"] and is_grasped:
                d["steps_in_air"] += 1
                if d["init_local_pose"] is not None:
                    current_local_pose = eef_mat.T.dot(obj_pos - eef_pos)
                    # 只有当物体在夹爪内部发生位移，且处于前 15 帧内，才锁死微滑脱
                    if np.linalg.norm(current_local_pose - d["init_local_pose"]) > 0.02 and d["steps_in_air"] <= 15:
                        h["micro_slip"] = True
                        d["has_terminal_fault"] = True

                if d["steps_in_air"] > 15:
                    h["transported"] = True

            # =================================================================
            # Phase: Release Logic (时空容纳综合判定)
            # =================================================================
            if h["lifted"] and not is_grasped and d["was_grasped_last_step"] and not d["has_terminal_fault"]:
                
                # 提取目标容器的空间包络线边界
                in_receptacle_space = any(
                    self._target_contains_xy(recep_name, obj_name, obj_pos)
                    for recep_name in self.target_receptacles
                )

                # 判定分流核心逻辑：
                if in_receptacle_space:
                    # 无论花了多少步，只要在目标空间内释放，一律计为放置尝试
                    h["placed_attempt"] = True
                else:
                    # 在目标空间外释放：
                    # 规则：如果在短时间内（15帧内）且没有运出安全距离就掉了，则是 micro_slip；否则是运输中途掉落
                    if d["steps_in_air"] <= 15 and not h["transported"]:
                        h["micro_slip"] = True
                    else:
                        h["dropped_midway"] = True
                    d["has_terminal_fault"] = True

            if is_success:
                h["placed_attempt"] = True
                h["micro_slip"] = False
                h["dropped_midway"] = False
                d["has_terminal_fault"] = False

            d["was_grasped_last_step"] = is_grasped

        return {            
            "global_faults": self.global_faults,
            "object_metrics": self.history
        }
