import numpy as np


_QUATERNION_EPS = 1e-8
_POST_GRIPPER_REARM_STEPS = 5


def _normalize_quaternion_wxyz(quat):
    """Return a normalized MuJoCo quaternion, or None when it is degenerate."""
    quat = np.asarray(quat, dtype=float).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm < _QUATERNION_EPS:
        return None
    return quat / norm


def _multiply_quaternions_wxyz(first, second):
    """Hamilton product for MuJoCo's ``[w, x, y, z]`` convention."""
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _split_z_swing_twist(quat):
    """Split a quaternion into swing and twist about the world Z axis.

    The twist is undefined for a 180-degree swing about a horizontal axis. In
    that case callers must leave the live orientation untouched.
    """
    quat = _normalize_quaternion_wxyz(quat)
    if quat is None:
        return None

    twist = np.array([quat[0], 0.0, 0.0, quat[3]], dtype=float)
    twist = _normalize_quaternion_wxyz(twist)
    if twist is None:
        return None

    twist_conjugate = twist * np.array([1.0, -1.0, -1.0, -1.0])
    swing = _multiply_quaternions_wxyz(quat, twist_conjugate)
    swing = _normalize_quaternion_wxyz(swing)
    if swing is None:
        return None
    return swing, twist


def _replace_z_twist(quat, target_twist):
    """Apply a cached Z twist while preserving the live roll / pitch swing."""
    split = _split_z_swing_twist(quat)
    target_twist = _normalize_quaternion_wxyz(target_twist)
    if split is None or target_twist is None:
        return None

    swing, _ = split
    result = _normalize_quaternion_wxyz(
        _multiply_quaternions_wxyz(swing, target_twist)
    )
    if result is None:
        return None

    live_quat = _normalize_quaternion_wxyz(quat)
    if live_quat is not None and np.dot(result, live_quat) < 0.0:
        result = -result
    return result


def _locks_roll_pitch_on_belt(_obj_name):
    return True


def _get_dof_adr(sim, obj_body_id):
    return sim.model.jnt_dofadr[sim.model.body_jntadr[obj_body_id]]

def _get_body_geom_ids(sim, body_id):
    """Get all geom IDs belonging to a body and its descendants."""
    geom_body_ids = sim.model.geom_bodyid
    body_parent_ids = sim.model.body_parentid

    descendant_bodies = np.zeros(sim.model.nbody, dtype=bool)
    for candidate_id in range(sim.model.nbody):
        current_id = candidate_id
        while current_id != 0 and current_id != body_id:
            current_id = body_parent_ids[current_id]
        if current_id == body_id:
            descendant_bodies[candidate_id] = True

    return np.where(descendant_bodies[geom_body_ids])[0].tolist()

def _in_contact_with(sim, obj_geom_ids, belt_geom_ids, obj_name=""):
    """Robust contact check used as a shared workspace utility."""
    if not obj_geom_ids or not belt_geom_ids:
        return False
    
    belt_set = set(belt_geom_ids)
    obj_set = set(obj_geom_ids)

    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if (c.geom1 in belt_set and c.geom2 in obj_set) or \
           (c.geom2 in belt_set and c.geom1 in obj_set):
            return True
            
    return False

class ConveyorBeltMixin:
    def setup_conveyor_belt(self, velocity=0.05, local_axis=(0, 1, 0)):
        self.conveyor_velocity = velocity
        self._conveyor_local_axis = np.array(local_axis, dtype=float)
        n = np.linalg.norm(self._conveyor_local_axis)
        if n < 1e-8:
            raise ValueError("conveyor local_axis must be non-zero")
        self._conveyor_local_axis /= n
        self._belt_geom_ids = None
        self._belt_body_id = None
        # These caches describe one episode's contacts and must not survive reset.
        for cache_name in (
            "_anchored_quats",
            "_contact_steps",
            "_contact_memory",
            "_anchored_pos",
            "_gripper_interaction_active",
            "_gripper_clear_steps",
        ):
            cache = getattr(self, cache_name, None)
            if cache is not None:
                cache.clear()

    def _get_belt_geom_ids(self):
        if self._belt_geom_ids is None:
            self._belt_geom_ids = []
            for gid in range(self.sim.model.ngeom):
                gname = self.sim.model.geom_id2name(gid)
                if gname and ('conveyor_belt' in gname) and ('contact_region' in gname) and ('ghost' not in gname):
                    self._belt_geom_ids.append(gid)
            if self._belt_geom_ids:
                self._belt_body_id = self.sim.model.geom_bodyid[self._belt_geom_ids[0]]
        return self._belt_geom_ids

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'conveyor_velocity'):
            action_array = np.asarray(action, dtype=float).reshape(-1)
            self._conveyor_gripper_closing = bool(
                action_array.size and action_array[-1] > 0.0
            )
            self._apply_conveyor_belt_physics()

    def _apply_conveyor_belt_physics(self):
        belt_geom_ids = self._get_belt_geom_ids()
        if not belt_geom_ids:
            return

        # 1. STATIC CACHING (Runs only once)
        if not hasattr(self, '_static_caches_initialized'):
            self._gripper_geoms = set()
            for gid in range(self.sim.model.ngeom):
                gname = self.sim.model.geom_id2name(gid)
                if gname and ('finger' in gname or 'gripper' in gname):
                    self._gripper_geoms.add(gid)

            self._obj_geom_cache = {}
            for obj_name in self.objects_dict:
                if "dyn_" in obj_name or "ghost" in obj_name:
                    continue
                obj_body_id = self.obj_body_id[obj_name]
                self._obj_geom_cache[obj_name] = set(_get_body_geom_ids(self.sim, obj_body_id))

            self._anchored_quats = {}
            self._contact_steps = {}
            self._contact_memory = {}
            self._anchored_pos = {}
            self._gripper_interaction_active = set()
            self._gripper_clear_steps = {}
            self._static_caches_initialized = True

        R = np.array(self.sim.data.body_xmat[self._belt_body_id]).reshape(3, 3)
        world_axis = R @ self._conveyor_local_axis
        wn = np.linalg.norm(world_axis)
        if wn < 1e-8:
            return
        world_axis /= wn
        target_vel = world_axis * self.conveyor_velocity

        # 2. SINGLE-PASS CONTACT RESOLUTION (O(n) speedup)
        # We loop through MuJoCo contacts exactly once per frame and sort them.
        active_belt_contacts = set()
        active_gripper_contacts = set()
        belt_set = set(belt_geom_ids)

        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            if g1 in belt_set: active_belt_contacts.add(g2)
            elif g2 in belt_set: active_belt_contacts.add(g1)

            if g1 in self._gripper_geoms: active_gripper_contacts.add(g2)
            elif g2 in self._gripper_geoms: active_gripper_contacts.add(g1)

        # 3. APPLY PHYSICS LOOP
        for obj_name, obj_geom_set in self._obj_geom_cache.items():
            
            # Fast set intersections replace the old _in_contact_with loops
            has_gripper_contact = not obj_geom_set.isdisjoint(active_gripper_contacts)
            is_physically_touching = not obj_geom_set.isdisjoint(active_belt_contacts)

            # Any robot contact suspends belt stabilization immediately. A
            # one-sided graze is enough; waiting for a bilateral grasp would
            # let the belt constraint fight the closing fingers.
            if has_gripper_contact:
                self._gripper_interaction_active.add(obj_name)
                self._gripper_clear_steps[obj_name] = 0
                self._anchored_quats.pop(obj_name, None) 
                self._contact_steps.pop(obj_name, None)
                self._contact_memory.pop(obj_name, None)
                self._anchored_pos.pop(obj_name, None)
                continue 

            interaction_active = obj_name in self._gripper_interaction_active
            if interaction_active:
                if (
                    getattr(self, '_conveyor_gripper_closing', False)
                    or not is_physically_touching
                ):
                    self._gripper_clear_steps[obj_name] = 0
                else:
                    clear_steps = self._gripper_clear_steps.get(obj_name, 0) + 1
                    self._gripper_clear_steps[obj_name] = clear_steps
                    if clear_steps >= _POST_GRIPPER_REARM_STEPS:
                        self._gripper_interaction_active.discard(obj_name)
                        self._gripper_clear_steps.pop(obj_name, None)
                        interaction_active = False
                
            # CONTACT GRACE PERIOD LOGIC
            if is_physically_touching:
                self._contact_memory[obj_name] = 5
            else:
                self._contact_memory[obj_name] = max(0, self._contact_memory.get(obj_name, 0) - 1)
                
            is_effectively_touching = self._contact_memory.get(obj_name, 0) > 0

            if not is_effectively_touching:
                self._anchored_quats.pop(obj_name, None) 
                self._contact_steps.pop(obj_name, None)
                self._anchored_pos.pop(obj_name, None)
                continue
            
            self._contact_steps[obj_name] = self._contact_steps.get(obj_name, 0) + 1
                
            obj_body_id = self.obj_body_id[obj_name]
            dof = _get_dof_adr(self.sim, obj_body_id)
            jnt_id = self.sim.model.body_jntadr[obj_body_id]
            qpos_adr = self.sim.model.jnt_qposadr[jnt_id]
            lock_roll_pitch = _locks_roll_pitch_on_belt(obj_name)
            
            if obj_name not in self._anchored_pos:
                self._anchored_pos[obj_name] = self.sim.data.qpos[qpos_adr:qpos_adr+2].copy()

            orientation_stabilized = False
            if interaction_active:
                self._anchored_quats.pop(obj_name, None)
            else:
                # Keep each payload's initial orientation while the belt is its
                # only support; first robot contact releases this stabilization.
                if obj_name not in self._anchored_quats:
                    raw_quat = self.sim.data.qpos[qpos_adr+3 : qpos_adr+7].copy()
                    if lock_roll_pitch:
                        normalized_quat = _normalize_quaternion_wxyz(raw_quat)
                        if normalized_quat is not None:
                            self._anchored_quats[obj_name] = normalized_quat
                    else:
                        split = _split_z_swing_twist(raw_quat)
                        if split is not None:
                            _, self._anchored_quats[obj_name] = split

                anchored_quat = self._anchored_quats.get(obj_name)
                if anchored_quat is not None:
                    live_quat = self.sim.data.qpos[qpos_adr+3 : qpos_adr+7].copy()
                    stabilized_quat = (
                        anchored_quat
                        if lock_roll_pitch
                        else _replace_z_twist(live_quat, anchored_quat)
                    )
                    if stabilized_quat is not None:
                        self.sim.data.qpos[qpos_adr+3 : qpos_adr+7] = stabilized_quat
                        orientation_stabilized = True
            
            if abs(target_vel[0]) < 1e-5:
                self.sim.data.qpos[qpos_adr] = self._anchored_pos[obj_name][0]
            if abs(target_vel[1]) < 1e-5:
                self.sim.data.qpos[qpos_adr+1] = self._anchored_pos[obj_name][1]
            
            # VELOCITY OVERRIDE
            current_z_vel = self.sim.data.qvel[dof + 2]
            if is_physically_touching:
                current_z_vel *= 0.1

            self.sim.data.qvel[dof:dof + 2] = target_vel[:2]
            self.sim.data.qvel[dof + 2] = current_z_vel
            
            # A belt-only payload stays angularly still. Robot contact releases
            # the stabilization above so grasping remains fully physical.
            if orientation_stabilized:
                if lock_roll_pitch:
                    self.sim.data.qvel[dof + 3:dof + 6] = 0.0
                else:
                    self.sim.data.qvel[dof + 5] = 0.0
