import re
import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from .conveyor_physics import _get_body_geom_ids
from .cosmetics_inlay_orientation import cosmetics_inlay_alignment


_HEADSET_INLAY_ORIENTATION_TOLERANCE = np.deg2rad(8.0)
_HEADSET_INLAY_ORIENTATIONS = (
    np.eye(3),
    np.diag([1.0, -1.0, -1.0]),
)
_HEADSET_INLAY_MAX_GAP = 0.003


def headset_inlay_alignment(obj_name, groove_name, relative_rotation):
    """Accept only the two fully constrained, face-up/down Headset poses."""
    if not (
        "headset" in obj_name.lower()
        and "headset" in groove_name.lower()
        and "inlay" in groove_name.lower()
    ):
        return None

    rotation = np.asarray(relative_rotation, dtype=float).reshape(3, 3)
    errors = (
        np.arccos(
            np.clip((np.trace(target.T.dot(rotation)) - 1.0) * 0.5, -1.0, 1.0)
        )
        for target in _HEADSET_INLAY_ORIENTATIONS
    )
    return min(errors) <= _HEADSET_INLAY_ORIENTATION_TOLERANCE + 1e-9


def headset_inlay_depth_alignment(midpoint_gap, penetration_limit=-0.0005):
    """Accept a half-depth seat only at the inlay rim, never far above it."""
    return penetration_limit <= float(midpoint_gap) <= _HEADSET_INLAY_MAX_GAP


def _slot_geom_contact_enabled(groove_name, geom_name):
    """Use the symmetric half-depth base as the Headset inlay collider."""
    return not (
        "headset" in groove_name.lower()
        and "inlay" in groove_name.lower()
        and not geom_name.lower().endswith("half_depth_bottom")
    )


def get_yaw(quat):
    # Convert wxyz to xyzw for robosuite tools
    q = T.convert_quat(quat, to="xyzw")
    # Returns Euler angles (Roll, Pitch, Yaw)
    return T.mat2euler(T.quat2mat(q))[2]

def _get_qpos_adr(sim, obj_body_id):
    return sim.model.jnt_qposadr[sim.model.body_jntadr[obj_body_id]]

def _get_qvel_adr(sim, obj_body_id):
    return sim.model.jnt_dofadr[sim.model.body_jntadr[obj_body_id]]


def _geom_lowest_projection(sim, geom_id, normal):
    """Return one collision geom's lowest world projection on ``normal``."""
    geom_type = int(sim.model.geom_type[geom_id])
    size = np.asarray(sim.model.geom_size[geom_id], dtype=float)
    center = np.asarray(sim.data.geom_xpos[geom_id], dtype=float)
    rotation = np.asarray(sim.data.geom_xmat[geom_id]).reshape(3, 3)
    local_normal = rotation.T.dot(normal)

    if geom_type == 2:  # sphere
        extent = size[0]
    elif geom_type == 3:  # capsule
        extent = size[0] + size[1] * abs(local_normal[2])
    elif geom_type == 4:  # ellipsoid
        extent = np.linalg.norm(size[:3] * local_normal)
    elif geom_type == 5:  # cylinder
        extent = (
            size[0] * np.linalg.norm(local_normal[:2])
            + size[1] * abs(local_normal[2])
        )
    elif geom_type == 6:  # box
        extent = np.dot(np.abs(local_normal), size[:3])
    elif geom_type == 7:  # mesh
        mesh_id = int(sim.model.geom_dataid[geom_id])
        vertex_start = int(sim.model.mesh_vertadr[mesh_id])
        vertex_count = int(sim.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            sim.model.mesh_vert[vertex_start:vertex_start + vertex_count]
        )
        world_vertices = center + vertices.dot(rotation.T)
        return float(np.min(world_vertices.dot(normal)))
    else:
        extent = float(sim.model.geom_rbound[geom_id])
    return float(np.dot(center, normal) - extent)


class ConveyorSlotMixin:
    _SUFFIX_PATTERN = re.compile(r'(_v_groove|_groove|_inlay|_inverted|_upright|_flat)+$')
    _TRAY_FLAT_DOT_MIN = 0.98
    _TRAY_SUPPORT_GAP_MIN = -0.001
    _TRAY_SUPPORT_GAP_MAX = 0.005
    _TRAY_SUPPORT_CLEARANCE = 0.0005
    _TRAY_PENETRATION_LIMIT = -0.0005
    _FLAT_GROOVE_SUPPORT_GAP = -0.0001
    _TRAY_SETTLING_GAP_MIN = -0.010
    _TRAY_SETTLING_ANGULAR_SPEED_MAX = 0.2
    _TRAY_SETTLING_TILT_STIFFNESS = 144.0
    _TRAY_SETTLING_TILT_DAMPING = 24.0
    _TRAY_SETTLING_MAX_TILT_ACCELERATION = 30.0
    _TRAY_SETTLING_MAX_TILT_SPEED = 1.5
    _PUMP_FLAT_SETTLING_MAX_TILT_ACCELERATION = 60.0
    _PUMP_FLAT_SETTLING_MAX_TILT_SPEED = 3.0
    _TRAY_SETTLING_STABLE_STEPS = 3
    _V_GROOVE_SETTLING_LINEAR_SPEED_MAX = 0.03
    _V_GROOVE_SETTLING_ANGULAR_SPEED_MAX = 0.2
    _V_GROOVE_SETTLING_STABLE_STEPS = 3
    _V_GROOVE_SUPPORT_LOAD_RATIO_MIN = 0.8
    _V_GROOVE_SUPPORT_LOAD_RATIO_MAX = 1.2
    _V_GROOVE_SUPPORT_TORQUE_RATIO_MAX = 0.15
    _V_GROOVE_MAXIMUM_PENETRATION = 0.005
    _V_GROOVE_CROSS_DOT_MAX = 0.30
    _V_GROOVE_SERVICE_POSITION_KP = 500.0
    _V_GROOVE_SERVICE_ROTATION_KP = 10.0
    _V_GROOVE_SERVICE_ROTATION_KD = 1.0
    _GROOVE_LOCK_BREAK_FORCE_RATIO = 0.05
    _GROOVE_LOCK_BREAK_FORCE_MIN = 0.02
    # Keep dynamic slots out of collisions with static fixtures while retaining
    # product-to-slot contact for routing. This bit is deliberately outside the
    # default MuJoCo collision group used by fixtures and robots.
    _SLOT_CONTACT_BIT = 1 << 29

    def setup_physical_track(self, conveyor_name, speed=0.02, mode="steady", groove_quat=[1, 0, 0, 0], groove_bottom_offset=-0.005, groove_step=None):
        self.target_conveyor_name = conveyor_name
        self.conveyor_speed = speed
        self.conveyor_mode = mode
        self.local_groove_quat = groove_quat
        self.groove_bottom_offset = groove_bottom_offset
        self.groove_step = groove_step
        self._track_initialized = False
        self.groove_local_ys = []
        # Pre-computed caches
        self._groove_qpos_adrs = None
        self._groove_qvel_adrs = None
        self._groove_site_ids = None
        self._groove_top_offsets = None
        self._groove_site_sizes = None
        self._groove_site_mats = None
        self._object_info_cache = None
        self._groove_geom_ids = None
        self._groove_geom_to_index = None
        self._tray_floor_geom_ids = None
        self._v_groove_support_geom_ids = None
        self._gripper_geom_ids = None
        self._robot_contact_geom_ids = None
        # Delivery tracking (reset on new track setup)
        self._delivered_objects = set()
        self._object_routed_steps = {}
        self._tray_planar_locks = {}
        self._tray_settling = {}
        self._v_groove_pose_locks = {}
        self._v_groove_settling = {}
        self._v_groove_was_service_phase = False
        self._slot_capacity_rejections = set()

    def _initialize_track(self):
        if not hasattr(self, 'dynamic_groove_names') or not self.dynamic_groove_names:
            return

        conv_id = self.obj_body_id[self.target_conveyor_name]
        conv_pos = self.sim.data.body_xpos[conv_id]
        conv_quat = self.sim.data.body_xquat[conv_id]
        conv_mat = T.quat2mat(T.convert_quat(conv_quat, to="xyzw"))

        table_z = self.model.mujoco_arena.table_top_abs[2]
        self.abs_z_height = table_z - self.groove_bottom_offset + 0.005

        site_name = f"{self.target_conveyor_name}_ghost_active_region"
        try:
            site_id = self.sim.model.site_name2id(site_name)
            half_length = self.sim.model.site_size[site_id][1]
        except Exception:
            half_length = 0.4

        self._ghost_half_length = half_length

        self.belt_start_y_local = -half_length
        self.belt_end_y_local = half_length

        local_quat_np = np.array(self.local_groove_quat)
        local_mat = T.quat2mat(T.convert_quat(local_quat_np, to="xyzw"))
        world_groove_mat = conv_mat.dot(local_mat)
        self.world_groove_quat_wxyz = T.convert_quat(T.mat2quat(world_groove_mat), to="wxyz")

        num_grooves = len(self.dynamic_groove_names)
        if hasattr(self, 'groove_step') and self.groove_step is not None:
            spacing = self.groove_step
            perfect_length = num_grooves * spacing
            self.belt_start_y_local = -perfect_length / 2.0
            self.belt_end_y_local = perfect_length / 2.0
        else:
            spacing = (self.belt_end_y_local - self.belt_start_y_local) / num_grooves

        self.groove_local_ys = []
        for i in range(num_grooves):
            self.groove_local_ys.append(self.belt_start_y_local + (i * spacing))

        self.forward_vec = conv_mat.dot(np.array([0, 1, 0]))

        # Pre-compute groove addresses, site IDs, sizes, and matrices
        self._groove_qpos_adrs = []
        self._groove_qvel_adrs = []
        self._groove_site_ids = []
        self._groove_top_offsets = []
        self._groove_site_sizes = []
        self._groove_site_mats = []

        for g_name in self.dynamic_groove_names:
            g_id = self.obj_body_id[g_name]
            self._groove_qpos_adrs.append(_get_qpos_adr(self.sim, g_id))
            self._groove_qvel_adrs.append(_get_qvel_adr(self.sim, g_id))

            site_name = f"{g_name}_goal_region"
            try:
                site_id = self.sim.model.site_name2id(site_name)
                self._groove_site_ids.append(site_id)
                self._groove_site_sizes.append(self.sim.model.site_size[site_id])
            except Exception:
                self._groove_site_ids.append(-1)
                self._groove_site_sizes.append(np.array([0.05, 0.05, 0.05]))

            groove_obj = self.objects_dict.get(g_name)
            self._groove_top_offsets.append(
                np.asarray(groove_obj.top_offset).copy()
                if groove_obj is not None else np.zeros(3)
            )

        # 1. INITIALIZATION FIX: Place their actual physics coordinates
        for i, g_name in enumerate(self.dynamic_groove_names):
            g_id = self.obj_body_id[g_name]
            qpos_adr = self._groove_qpos_adrs[i]
            qvel_adr = self._groove_qvel_adrs[i]

            local_xy = np.array([0, self.groove_local_ys[i], 0])
            world_pos = conv_pos + conv_mat.dot(local_xy)
            world_pos[2] = self.abs_z_height

            self.sim.data.qpos[qpos_adr:qpos_adr+3] = world_pos
            self.sim.data.qpos[qpos_adr+3:qpos_adr+7] = self.world_groove_quat_wxyz
            self.sim.data.qvel[qvel_adr:qvel_adr+6] = 0

        # Pre-compute slot collider IDs. In S2D a static plate can overlap the
        # ghost track. Dynamic inlays must still contact products, but letting
        # their dense mesh colliders contact the static plate saturates MuJoCo's
        # contact buffer and makes every routing check prohibitively expensive.
        self._groove_geom_ids = []
        self._groove_geom_to_index = {}
        self._tray_floor_geom_ids = []
        self._v_groove_support_geom_ids = []
        for slot_index, g_name in enumerate(self.dynamic_groove_names):
            geom_ids = set(_get_body_geom_ids(self.sim, self.obj_body_id[g_name]))
            self._groove_geom_ids.append(geom_ids)
            floor_geom_ids = set()
            support_geom_ids = {"left": set(), "right": set()}
            for geom_id in geom_ids:
                self._groove_geom_to_index[geom_id] = slot_index
                geom_name = (self.sim.model.geom_id2name(geom_id) or "").lower()
                if self.sim.model.geom_contype[geom_id] or self.sim.model.geom_conaffinity[geom_id]:
                    self.sim.model.geom_contype[geom_id] = 0
                    self.sim.model.geom_conaffinity[geom_id] = (
                        self._SLOT_CONTACT_BIT
                        if _slot_geom_contact_enabled(g_name, geom_name)
                        else 0
                    )
                if geom_name.endswith("v_slant_left"):
                    support_geom_ids["left"].add(geom_id)
                elif geom_name.endswith("v_slant_right"):
                    support_geom_ids["right"].add(geom_id)
                if "tray" in g_name.lower() and geom_name.endswith("bottom"):
                    # Level-1 transport is kinematic in-plane.  Keep the real
                    # tray base as a normal-only collision floor so it cannot
                    # add noisy tangential friction to that transport.
                    self.sim.model.geom_condim[geom_id] = 1
                    self.sim.model.geom_friction[geom_id, :] = 0.0
                    self.sim.model.geom_margin[geom_id] = max(
                        float(self.sim.model.geom_margin[geom_id]),
                        self._TRAY_SUPPORT_CLEARANCE,
                    )
                    floor_geom_ids.add(geom_id)
            self._tray_floor_geom_ids.append(floor_geom_ids)
            self._v_groove_support_geom_ids.append(support_geom_ids)

        # Pre-compute product info and collider IDs. A dynamic slot may be an
        # inlay rather than a groove, so membership in dynamic_groove_names is
        # the only reliable way to exclude it from the product cache.
        self._object_info_cache = {}
        for obj_name, obj in self.objects_dict.items():
            if obj_name in self.dynamic_groove_names:
                continue
            obj_id = self.obj_body_id.get(obj_name)
            if obj_id is not None:
                try:
                    obj_joint = obj.joints[-1]
                    geom_ids = set(_get_body_geom_ids(self.sim, obj_id))
                    contact_geom_ids = set()
                    for geom_id in geom_ids:
                        if self.sim.model.geom_contype[geom_id] or self.sim.model.geom_conaffinity[geom_id]:
                            contact_geom_ids.add(geom_id)
                            self.sim.model.geom_contype[geom_id] |= self._SLOT_CONTACT_BIT
                    self._object_info_cache[obj_name] = {
                        'id': obj_id,
                        'joint': obj_joint,
                        'qvel_adr': self.sim.model.jnt_dofadr[self.sim.model.joint_name2id(obj_joint)],
                        'qpos_adr': self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(obj_joint)],
                        'geom_ids': geom_ids,
                        'contact_geom_ids': contact_geom_ids,
                        'bottom_offset': np.asarray(obj.bottom_offset).copy(),
                        'midpoint_offset': 0.5 * (
                            np.asarray(obj.top_offset, dtype=float)
                            + np.asarray(obj.bottom_offset, dtype=float)
                        ),
                        'support_axis': self._object_support_axis(obj),
                    }
                except Exception:
                    pass

        self._gripper_geom_ids = {
            gid
            for gid in range(self.sim.model.ngeom)
            if (gname := self.sim.model.geom_id2name(gid))
            and ('finger' in gname or 'gripper' in gname)
        }
        self._robot_contact_geom_ids = set(self._gripper_geom_ids)
        for body_name in ("robot0_base", "robot0_base_link"):
            try:
                body_id = int(self.sim.model.body_name2id(body_name))
            except (ValueError, KeyError, TypeError):
                continue
            self._robot_contact_geom_ids.update(
                _get_body_geom_ids(self.sim, body_id)
            )
        self._robot_contact_geom_ids = {
            geom_id
            for geom_id in self._robot_contact_geom_ids
            if self.sim.model.geom_contype[geom_id]
            or self.sim.model.geom_conaffinity[geom_id]
        }

        self._setup_end_blocks()

        self._sticky_compat = {g_name: {} for g_name in self.dynamic_groove_names}
        self._object_routed_steps = {}
        self._delivered_objects = set()
        self._tray_planar_locks = {}
        self._tray_settling = {}
        self._v_groove_pose_locks = {}
        self._v_groove_settling = {}
        self._v_groove_was_service_phase = False
        self._slot_capacity_rejections = set()
        self._routed_threshold = 10  # 1 second at 20Hz

        self._track_initialized = True

    def _setup_end_blocks(self):
        """Add visual blocks at both belt ends to hide grooves when they loop around."""
        if not self.dynamic_groove_names:
            return

        first_groove_name = self.dynamic_groove_names[0]
        groove_obj = self.objects_dict.get(first_groove_name)
        if groove_obj is None:
            return

        # 1. Parse groove outer dimensions from horizontal_radius_site
        groove_half_x = 0.1
        groove_half_y = 0.1
        for site in groove_obj.worldbody.findall(".//site"):
            if "horizontal_radius_site" in site.get("name", ""):
                pos_str = site.get("pos")
                if pos_str:
                    pos_vals = [abs(float(x)) for x in pos_str.split(" ")]
                    if pos_vals[0] > 0:
                        groove_half_x = pos_vals[0]
                    if pos_vals[1] > 0:
                        groove_half_y = pos_vals[1]
                break

        # 2. Parse groove Z height from top_site - bottom_site
        top_z = 0.025
        bottom_z = 0.0
        for site in groove_obj.worldbody.findall(".//site"):
            name = site.get("name", "")
            pos_str = site.get("pos")
            if pos_str:
                pos_vals = [float(x) for x in pos_str.split(" ")]
                if "top_site" in name:
                    top_z = pos_vals[2]
                elif "bottom_site" in name:
                    bottom_z = pos_vals[2]

        groove_height = top_z - bottom_z
        block_half_z = max(groove_height / 2.0, 0.015)

        # 3. Determine orientation: belt along X or Y
        local_quat_np = np.array(self.local_groove_quat)
        is_x_orientation = abs(abs(local_quat_np[0]) - 0.7071068) < 0.01

        # Block thickness along belt = 1 full groove width
        if hasattr(self, 'groove_step') and self.groove_step is not None:
            block_half_thickness = max(self.groove_step / 2.0, 0.02)
        else:
            block_half_thickness = 0.03

        if is_x_orientation:
            block_half_size = np.array([groove_half_y, block_half_thickness, block_half_z])
        else:
            block_half_size = np.array([groove_half_x, block_half_thickness, block_half_z])

        # 4. Compute Z position: block bottom sits on belt surface
        conv_id = self.obj_body_id[self.target_conveyor_name]
        conv_pos = self.sim.data.body_xpos[conv_id]
        belt_surface_z = self.abs_z_height + bottom_z
        block_center_world_z = belt_surface_z + block_half_z
        block_local_z = block_center_world_z - conv_pos[2]

        # Block positions at groove-loop boundaries (not ghost extent)
        front_local_y = self.belt_end_y_local
        back_local_y = self.belt_start_y_local

        # 5. Find end block bodies and update geom size + body position
        for geom_name, body_suffix, local_y in [
            ("conveyor_end_block_front", "end_block_front_body", front_local_y),
            ("conveyor_end_block_back", "end_block_back_body", back_local_y),
        ]:
            try:
                prefixed_geom = f"{self.target_conveyor_name}_{geom_name}"
                prefixed_body = f"{self.target_conveyor_name}_{body_suffix}"

                geom_id = self.sim.model.geom_name2id(prefixed_geom)
                body_id = self.sim.model.body_name2id(prefixed_body)

                # Set geom size
                try:
                    self.sim.model.geom_size[geom_id] = block_half_size
                except (ValueError, AttributeError):
                    self.sim.model.geom_size.flags.writeable = True
                    self.sim.model.geom_size[geom_id] = block_half_size

                self.sim.model.geom_rgba[geom_id] = np.array(
                    [0.2, 0.2, 0.2, 1.0]
                )

                # Move the dedicated body (belt always extends along local Y)
                target_body_pos = np.array([0, local_y, block_local_z])

                self.sim.model.body_pos[body_id] = target_body_pos

            
            except ValueError:
                all_bodies = [
                    self.sim.model.body_id2name(i)
                    for i in range(self.sim.model.nbody)
                ]
                end_block = [b for b in all_bodies if b and "end_block" in b]
                ghost = [b for b in all_bodies if b and "ghost" in b]
                print(
                    f"[END_BLOCK] FAILED — 'end_block' bodies: {end_block}, "
                    f"'ghost' bodies: {ghost}"
                )
            except Exception as e:
                print(f"[END_BLOCK] FAILED: {e}")

    def _extract_product(self, category_name):
        """Extract product name using regex. Returns None for trays (universal match)."""
        if "_tray" in category_name:
            return None
            
        return self._SUFFIX_PATTERN.sub('', category_name)

    def _object_matches_groove(self, obj_name, groove_name):
        """Check if object's product matches groove's intended product."""
        groove_obj = self.objects_dict.get(groove_name)
        obj = self.objects_dict.get(obj_name)
        if groove_obj is None or obj is None:
            return False
        groove_product = self._extract_product(groove_obj.category_name)
        if groove_product is None:
            return True  # tray: matches everything
        obj_product = self._extract_product(obj.category_name)
        return obj_product == groove_product

    @staticmethod
    def _is_single_object_carrier(groove_name):
        """Return whether one dynamic carrier has capacity for one product."""
        carrier_name = groove_name.lower()
        return "tray" not in carrier_name and "plate" not in carrier_name

    def _has_confirmed_gripper_grasp(self, obj_name, has_gripper_contact):
        """Distinguish a real re-grasp from a harmless graze."""
        if not has_gripper_contact:
            return False
        try:
            return bool(
                self._check_grasp(
                    self.robots[0].gripper,
                    self.objects_dict[obj_name].contact_geoms,
                )
            )
        except (AttributeError, KeyError, TypeError):
            return True

    @staticmethod
    def _colliding_object_names(object_info_cache, contacts):
        """Return products participating in a current product-product contact."""
        owner_by_geom = {
            geom_id: obj_name
            for obj_name, obj_cache in (object_info_cache or {}).items()
            for geom_id in obj_cache.get("geom_ids", ())
        }
        colliding = set()
        for contact in contacts:
            first = owner_by_geom.get(int(contact.geom1))
            second = owner_by_geom.get(int(contact.geom2))
            if first is not None and second is not None and first != second:
                colliding.update((first, second))
        return colliding

    def _single_carrier_occupants(self, groove_name, contact_geom_ids=None):
        """Return every product physically occupying one single-object carrier."""
        if not self._is_single_object_carrier(groove_name):
            return set()

        pose_locks = getattr(self, "_v_groove_pose_locks", {})
        occupants = {
            obj_name
            for obj_name, lock in pose_locks.items()
            if lock.get("slot") == groove_name
        }
        object_cache = getattr(self, "_object_info_cache", None) or {}
        if contact_geom_ids is None:
            try:
                slot_index = self.dynamic_groove_names.index(groove_name)
                groove_geom_ids = self._groove_geom_ids[slot_index]
            except (AttributeError, IndexError, TypeError, ValueError):
                return occupants
            contact_geom_ids = set()
            for contact_index in range(self.sim.data.ncon):
                contact = self.sim.data.contact[contact_index]
                geom1, geom2 = int(contact.geom1), int(contact.geom2)
                if geom1 in groove_geom_ids:
                    contact_geom_ids.add(geom2)
                if geom2 in groove_geom_ids:
                    contact_geom_ids.add(geom1)

        contact_geom_ids = set(contact_geom_ids)
        occupants.update(
            obj_name
            for obj_name, obj_cache in object_cache.items()
            if not set(obj_cache.get("geom_ids", ())).isdisjoint(
                contact_geom_ids
            )
            and pose_locks.get(obj_name, {}).get("slot", groove_name)
            == groove_name
        )
        return occupants

    def _uses_v_groove_service_lock(self, obj_name, groove_name):
        """Return whether this pair settles freely until a pulse-service boundary."""
        return (
            self.conveyor_mode == "pulse"
            and "cream_jar" in obj_name.lower()
            and "cream_jar" in groove_name.lower()
            and "v_groove" in groove_name.lower()
        )

    def _v_groove_support_state(self, obj_name, groove_name):
        """Return load-bearing state for an object in one authored V groove."""
        obj_cache = (self._object_info_cache or {}).get(obj_name)
        try:
            slot_index = self.dynamic_groove_names.index(groove_name)
            support_ids = self._v_groove_support_geom_ids[slot_index]
            groove_ids = self._groove_geom_ids[slot_index]
        except (AttributeError, IndexError, ValueError):
            return None
        if (
            obj_cache is None
            or not obj_cache['contact_geom_ids']
            or not support_ids['left']
            or not support_ids['right']
        ):
            return None

        sim = self.sim
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        object_ids = obj_cache['contact_geom_ids']
        object_body_ids = {
            int(sim.model.geom_bodyid[geom_id]) for geom_id in object_ids
        }
        object_mass = sum(
            float(sim.model.body_mass[body_id]) for body_id in object_body_ids
        )
        gravity = np.asarray(sim.model.opt.gravity, dtype=float)
        gravity_norm = float(np.linalg.norm(gravity))
        weight = object_mass * gravity_norm
        up = -gravity / max(gravity_norm, 1e-8)
        radius = max(
            (float(sim.model.geom_rbound[geom_id]) for geom_id in object_ids),
            default=0.0,
        )

        left_contact = False
        right_contact = False
        normal_load = 0.0
        net_force = np.zeros(3)
        net_torque = np.zeros(3)
        net_support_force = np.zeros(3)
        contact_count = 0
        minimum_contact_distance = np.inf
        object_com = np.asarray(
            sim.data.body_xpos[obj_cache['id']], dtype=float
        )
        for contact_id in range(sim.data.ncon):
            contact = sim.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            object_is_first = geom1 in object_ids and geom2 in groove_ids
            object_is_second = geom2 in object_ids and geom1 in groove_ids
            if not object_is_first and not object_is_second:
                continue

            groove_geom = geom2 if object_is_first else geom1
            left_contact = left_contact or groove_geom in support_ids['left']
            right_contact = right_contact or groove_geom in support_ids['right']
            contact_wrench = np.zeros(6)
            try:
                mujoco.mj_contactForce(model, data, contact_id, contact_wrench)
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "MuJoCo V-groove contact-force extraction is unavailable."
                ) from exc
            contact_frame = np.asarray(contact.frame, dtype=float).reshape(3, 3)
            force_on_object = contact_frame.T.dot(contact_wrench[:3])
            torque_on_object = contact_frame.T.dot(contact_wrench[3:])
            if object_is_first:
                force_on_object *= -1.0
                torque_on_object *= -1.0
            contact_pos = np.asarray(contact.pos, dtype=float)
            if groove_geom in support_ids['left'] or groove_geom in support_ids['right']:
                support_normal = contact_frame[0].copy()
                if float(np.dot(support_normal, up)) < 0.0:
                    support_normal *= -1.0
                support_force = support_normal * max(
                    float(contact_wrench[0]), 0.0
                )
                net_support_force += support_force
            normal_load += max(float(contact_wrench[0]), 0.0)
            net_force += force_on_object
            net_torque += torque_on_object + np.cross(
                contact_pos - object_com, force_on_object
            )
            contact_count += 1
            minimum_contact_distance = min(
                minimum_contact_distance, float(contact.dist)
            )

        force_scale = max(weight, 1e-8)
        torque_scale = max(weight * radius, 1e-8)
        upward_load = float(np.dot(net_force, up))
        upward_load_ratio = upward_load / force_scale
        torque_ratio = float(np.linalg.norm(net_torque) / torque_scale)
        maximum_penetration = (
            0.0
            if not np.isfinite(minimum_contact_distance)
            else max(0.0, -minimum_contact_distance)
        )
        return {
            "contact_count": contact_count,
            "left_contact": left_contact,
            "right_contact": right_contact,
            "normal_load": float(normal_load),
            "load_ratio": float(normal_load / force_scale),
            "upward_load": upward_load,
            "upward_load_ratio": upward_load_ratio,
            "net_force": net_force,
            "net_torque": net_torque,
            "net_support_force": net_support_force,
            "torque_ratio": torque_ratio,
            "torque_ready": bool(
                torque_ratio <= self._V_GROOVE_SUPPORT_TORQUE_RATIO_MAX
            ),
            "maximum_penetration": float(maximum_penetration),
            "weight": weight,
            "load_ready": bool(
                (left_contact or right_contact)
                and self._V_GROOVE_SUPPORT_LOAD_RATIO_MIN
                <= upward_load_ratio
                <= self._V_GROOVE_SUPPORT_LOAD_RATIO_MAX
                and maximum_penetration
                <= self._V_GROOVE_MAXIMUM_PENETRATION
            ),
        }

    @staticmethod
    def _uses_flat_groove_floor_guard(obj_name, groove_name):
        obj_name = obj_name.lower()
        groove_name = groove_name.lower()
        return (
            "flat_groove" in groove_name
            and (
                "cream_bottle" in obj_name
                or "pump_bottle" in obj_name
            )
        )

    def _is_routed_in_slot_region(
        self, obj_name, groove_name, obj_cache=None,
        site_pos=None, site_mat=None, site_size=None,
    ):
        """Return whether an object's CoM and orientation fit a compatible slot."""
        if not self._object_matches_groove(obj_name, groove_name):
            return False

        if (
            self._is_single_object_carrier(groove_name)
            and self._single_carrier_occupants(groove_name) != {obj_name}
        ):
            return False

        if obj_cache is None:
            obj_cache = (self._object_info_cache or {}).get(obj_name)
        if obj_cache is None:
            return False

        if site_pos is None or site_mat is None or site_size is None:
            try:
                site_id = self.sim.model.site_name2id(f"{groove_name}_goal_region")
                site_pos = self.sim.data.site_xpos[site_id]
                site_mat = self.sim.data.site_xmat[site_id].reshape(3, 3)
                site_size = self.sim.model.site_size[site_id]
            except (ValueError, KeyError, TypeError):
                return False

        obj_body_id = obj_cache['id']
        try:
            com_body_id = self.sim.model.body_name2id(f"{obj_name}_main")
            if com_body_id >= 0:
                obj_body_id = com_body_id
        except (ValueError, KeyError, TypeError):
            pass

        obj_pos = self.sim.data.body_xpos[obj_body_id]
        local_obj_pos = site_mat.T.dot(obj_pos - site_pos)
        if (abs(local_obj_pos[0]) > site_size[0] or
                abs(local_obj_pos[1]) > site_size[1]):
            return False

        obj_mat = self.sim.data.body_xmat[obj_body_id].reshape(3, 3)
        ox, oy, oz = obj_mat[:, 0], obj_mat[:, 1], obj_mat[:, 2]
        sx, sy, sz = site_mat[:, 0], site_mat[:, 1], site_mat[:, 2]
        xx, zz = np.dot(ox, sx), np.dot(oz, sz)
        n1, n2 = obj_name.lower(), groove_name.lower()
        pump_axis = oz
        if "pump_bottle" in n1:
            try:
                contact_site_id = self.sim.model.site_name2id(
                    f"{obj_name}_contact_site"
                )
                pump_axis = self.sim.data.site_xmat[contact_site_id].reshape(
                    3, 3
                )[:, 2]
            except (AttributeError, ValueError, KeyError, TypeError):
                pass

        if "pump_bottle" in n1 and "flat_v_groove" in n2:
            try:
                slot_index = self.dynamic_groove_names.index(groove_name)
            except (AttributeError, ValueError):
                return False
            support_state = self._v_groove_support_state(obj_name, groove_name)
            floor_gap = self._groove_floor_support_gap(
                obj_cache, slot_index, site_mat
            )
            object_min_y = min(
                _geom_lowest_projection(self.sim, geom_id, sy)
                for geom_id in obj_cache["contact_geom_ids"]
            )
            object_max_y = max(
                -_geom_lowest_projection(self.sim, geom_id, -sy)
                for geom_id in obj_cache["contact_geom_ids"]
            )
            site_center_y = float(np.dot(site_pos, sy))
            longitudinally_contained = (
                object_min_y >= site_center_y - site_size[1] - 0.0005
                and object_max_y <= site_center_y + site_size[1] + 0.0005
            )
            if (
                support_state is None
                or support_state["maximum_penetration"] > 0.0005
                or floor_gap < self._TRAY_PENETRATION_LIMIT
                or not longitudinally_contained
            ):
                return False
            projected_axis = pump_axis - np.dot(pump_axis, sz) * sz
            projection_norm = np.linalg.norm(projected_axis)
            return (
                abs(np.dot(pump_axis, sz)) < 0.1
                and projection_norm > 1e-6
                and abs(np.dot(projected_axis / projection_norm, sy)) > 0.9
            )

        if (
            "pump_bottle" in n1
            and self._uses_flat_groove_floor_guard(obj_name, groove_name)
        ):
            try:
                slot_index = self.dynamic_groove_names.index(groove_name)
            except (AttributeError, ValueError):
                return False
            floor_gap = self._groove_floor_support_gap(
                obj_cache, slot_index, site_mat
            )
            object_min_y = min(
                _geom_lowest_projection(self.sim, geom_id, sy)
                for geom_id in obj_cache["contact_geom_ids"]
            )
            object_max_y = max(
                -_geom_lowest_projection(self.sim, geom_id, -sy)
                for geom_id in obj_cache["contact_geom_ids"]
            )
            site_center_y = float(np.dot(site_pos, sy))
            longitudinally_contained = (
                object_min_y >= site_center_y - site_size[1] - 0.0005
                and object_max_y <= site_center_y + site_size[1] + 0.0005
            )
            if (
                floor_gap < self._TRAY_PENETRATION_LIMIT
                or floor_gap > self._TRAY_SUPPORT_GAP_MAX
                or not longitudinally_contained
            ):
                return False
            projected_axis = pump_axis - np.dot(pump_axis, sz) * sz
            projection_norm = np.linalg.norm(projected_axis)
            return (
                abs(np.dot(pump_axis, sz)) < 0.1
                and projection_norm > 1e-6
                and abs(np.dot(projected_axis / projection_norm, sy)) > 0.9
            )

        cosmetics_alignment = cosmetics_inlay_alignment(
            obj_name,
            groove_name,
            site_mat.T.dot(obj_mat),
        )
        if cosmetics_alignment is not None:
            return cosmetics_alignment

        headset_alignment = headset_inlay_alignment(
            obj_name,
            groove_name,
            site_mat.T.dot(obj_mat),
        )
        if headset_alignment is not None:
            if not headset_alignment:
                return False
            try:
                slot_index = self.dynamic_groove_names.index(groove_name)
            except (AttributeError, ValueError):
                return False
            return headset_inlay_depth_alignment(
                self._headset_inlay_midpoint_gap(obj_cache, slot_index, site_mat),
                self._TRAY_PENETRATION_LIMIT,
            )

        if "inverted" in n1 or "inverted" in n2:
            return zz < -0.9 and abs(xx) > 0.9
        if "flat" in n1 and "upright" in n2 and "inlay" in n2:
            # Signed flat-asset conversion: body X -> slot X,
            # body Y -> -slot Z, and body Z -> slot Y.
            return (
                np.dot(ox, sx) > 0.9
                and np.dot(oy, sz) < -0.9
                and np.dot(oz, sy) > 0.9
            )
        if (("upright" in n1 and "flat" in n2) or
                ("flat" in n1 and "upright" in n2)):
            if "upright" in n1 and "flat_groove" in n2:
                projected_oz = oz - np.dot(oz, sz) * sz
                projection_norm = np.linalg.norm(projected_oz)
                return (
                    abs(zz) < 0.1
                    and projection_norm > 1e-6
                    and abs(np.dot(projected_oz / projection_norm, site_mat[:, 1])) > 0.9
                )
            aligned = zz > 0.9 and abs(xx) > 0.9
            converted = abs(zz) < 0.1 and (
                abs(np.dot(ox, sz)) > 0.9 or abs(np.dot(oy, sz)) > 0.9
            )
            return aligned or converted
        if "v_groove" in n2:
            if "cream_jar" in n1:
                # cream_jar_flat.xml rotates the visual mesh +90 degrees
                # about body X, making its semantic lid axis body -Y.
                # The symmetric V supports either upward-facing 45-degree
                # pose, never either lid-down twin.
                semantic_top = -oy
                top_x = np.dot(semantic_top, sx)
                top_y = np.dot(semantic_top, sy)
                top_z = np.dot(semantic_top, sz)
                expected = np.sqrt(0.5)
                return (
                    abs(abs(top_x) - expected) < 0.15
                    and abs(top_y) < self._V_GROOVE_CROSS_DOT_MAX
                    and abs(top_z - expected) < 0.15
                )
            seat_tilt = abs(float(np.arctan2(np.dot(ox, sz), zz)))
            return (
                np.deg2rad(15.0) <= seat_tilt <= np.deg2rad(55.0)
                and abs(np.dot(ox, sy)) < self._V_GROOVE_CROSS_DOT_MAX
            )
        if "groove" in n2:
            return zz > 0.9 and abs(xx) > 0.9
        return zz > 0.9 and xx > 0.9

    def _tray_support_gap(self, obj_cache, slot_index, site_mat):
        """Measure the true collision-bottom gap above one tray's support plane."""
        normal = site_mat[:, 2]
        lowest = min(
            _geom_lowest_projection(self.sim, geom_id, normal)
            for geom_id in obj_cache['contact_geom_ids']
        )
        groove_id = self.obj_body_id[self.dynamic_groove_names[slot_index]]
        groove_pos = self.sim.data.body_xpos[groove_id]
        groove_mat = self.sim.data.body_xmat[groove_id].reshape(3, 3)
        support_pos = groove_pos + groove_mat.dot(
            self._groove_top_offsets[slot_index]
        )
        return float(lowest - np.dot(support_pos, normal))

    def _groove_floor_support_gap(self, obj_cache, slot_index, site_mat):
        """Measure collision-bottom clearance above an authored groove floor."""
        normal = site_mat[:, 2]
        floor_geom_ids = [
            geom_id
            for geom_id in self._groove_geom_ids[slot_index]
            if (self.sim.model.geom_id2name(geom_id) or "")
            .lower().endswith("bottom")
        ]
        object_bottom = min(
            _geom_lowest_projection(self.sim, geom_id, normal)
            for geom_id in obj_cache['contact_geom_ids']
        )
        if not floor_geom_ids:
            return np.inf
        floor_top = max(
            -_geom_lowest_projection(self.sim, geom_id, -normal)
            for geom_id in floor_geom_ids
        )
        return float(object_bottom - floor_top)

    def _headset_inlay_midpoint_gap(self, obj_cache, slot_index, site_mat):
        """Return the Headset midpoint height above the inlay rim plane."""
        groove_name = self.dynamic_groove_names[slot_index]
        groove_id = self.obj_body_id[groove_name]
        groove_pos = self.sim.data.body_xpos[groove_id]
        groove_mat = self.sim.data.body_xmat[groove_id].reshape(3, 3)
        object_midpoint = (
            self.sim.data.body_xpos[obj_cache['id']]
            + self.sim.data.body_xmat[obj_cache['id']].reshape(3, 3).dot(
                obj_cache['midpoint_offset']
            )
        )
        groove_rim = groove_pos + groove_mat.dot(
            self._groove_top_offsets[slot_index]
        )
        return float(np.dot(object_midpoint - groove_rim, site_mat[:, 2]))

    def _seat_initial_headsets_in_inlays(self):
        """Lower reset-time Headsets already centered over half-depth inlays."""
        claimed_slots = set()
        changed = False
        for obj_name, obj_cache in getattr(
            self, "_object_info_cache", {}
        ).items():
            if "headset" not in obj_name.lower():
                continue
            obj_pos = self.sim.data.body_xpos[obj_cache['id']]
            obj_mat = self.sim.data.body_xmat[obj_cache['id']].reshape(3, 3)
            candidates = []
            for slot_index, groove_name in enumerate(self.dynamic_groove_names):
                if (
                    slot_index in claimed_slots
                    or "headset" not in groove_name.lower()
                    or "inlay" not in groove_name.lower()
                ):
                    continue
                site_id = self._groove_site_ids[slot_index]
                if site_id < 0:
                    continue
                site_pos = self.sim.data.site_xpos[site_id]
                site_mat = self.sim.data.site_xmat[site_id].reshape(3, 3)
                local_pos = site_mat.T.dot(obj_pos - site_pos)
                if (
                    np.all(
                        np.abs(local_pos[:2])
                        <= self._groove_site_sizes[slot_index][:2]
                    )
                    and headset_inlay_alignment(
                        obj_name, groove_name, site_mat.T.dot(obj_mat)
                    )
                ):
                    candidates.append(
                        (np.linalg.norm(local_pos[:2]), slot_index, site_mat)
                    )
            if not candidates:
                continue
            _, slot_index, site_mat = min(candidates, key=lambda item: item[0])
            midpoint_gap = self._headset_inlay_midpoint_gap(
                obj_cache, slot_index, site_mat
            )
            correction = self._TRAY_SUPPORT_CLEARANCE - midpoint_gap
            qpos_adr = obj_cache['qpos_adr']
            qvel_adr = obj_cache['qvel_adr']
            self.sim.data.qpos[qpos_adr:qpos_adr+3] += (
                site_mat[:, 2] * correction
            )
            self.sim.data.qvel[qvel_adr:qvel_adr+6] = 0.0
            claimed_slots.add(slot_index)
            changed = True
            print(
                f"[HEADSET INIT SEAT] {obj_name} -> "
                f"{self.dynamic_groove_names[slot_index]} | "
                f"gap={1000.0 * midpoint_gap:.2f}mm -> "
                f"{1000.0 * self._TRAY_SUPPORT_CLEARANCE:.2f}mm"
            )
        if changed:
            self.sim.forward()
        return changed

    def _breakable_lock_contact(self, obj_name, obj_cache):
        """Return a meaningful robot/product impact on a locked flat-groove object."""
        object_geom_ids = obj_cache['contact_geom_ids']
        external_owners = {
            geom_id: other_name
            for other_name, other_cache in self._object_info_cache.items()
            if other_name != obj_name
            for geom_id in other_cache['contact_geom_ids']
        }
        external_owners.update(
            {
                geom_id: "robot"
                for geom_id in (self._robot_contact_geom_ids or ())
                if geom_id not in external_owners
            }
        )
        if not object_geom_ids or not external_owners:
            return None

        sim = self.sim
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        object_body_ids = {
            int(sim.model.geom_bodyid[geom_id]) for geom_id in object_geom_ids
        }
        object_weight = sum(
            float(sim.model.body_mass[body_id]) for body_id in object_body_ids
        ) * float(np.linalg.norm(sim.model.opt.gravity))
        break_force = max(
            self._GROOVE_LOCK_BREAK_FORCE_MIN,
            self._GROOVE_LOCK_BREAK_FORCE_RATIO * object_weight,
        )
        owner_forces = {}
        owner_contacts = {}
        for contact_id in range(sim.data.ncon):
            contact = sim.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in object_geom_ids:
                other_geom = geom2
            elif geom2 in object_geom_ids:
                other_geom = geom1
            else:
                continue
            owner = external_owners.get(other_geom)
            if owner is None:
                continue
            contact_wrench = np.zeros(6)
            try:
                mujoco.mj_contactForce(
                    model, data, contact_id, contact_wrench
                )
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "MuJoCo groove-lock contact-force extraction is unavailable."
                ) from exc
            owner_forces[owner] = owner_forces.get(owner, 0.0) + max(
                float(contact_wrench[0]), 0.0
            )
            owner_contacts.setdefault(
                owner,
                (other_geom, float(contact.dist), np.asarray(contact.pos).copy()),
            )

        if not owner_forces:
            return None
        owner, normal_force = max(owner_forces.items(), key=lambda item: item[1])
        if normal_force <= break_force:
            return None
        other_geom, distance, position = owner_contacts[owner]
        return {
            'owner': owner,
            'other_geom': other_geom,
            'force': normal_force,
            'threshold': break_force,
            'distance': distance,
            'position': position,
        }

    @staticmethod
    def _object_support_axis(obj):
        """Return the object's local bottom-to-top axis without name checks."""
        support_axis = (
            np.asarray(obj.top_offset, dtype=float)
            - np.asarray(obj.bottom_offset, dtype=float)
        )
        axis_norm = np.linalg.norm(support_axis)
        if axis_norm <= 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return support_axis / axis_norm

    def _world_support_axis(self, obj_cache):
        obj_mat = self.sim.data.body_xmat[obj_cache['id']].reshape(3, 3)
        return obj_mat.dot(obj_cache['support_axis'])

    def _apply_tray_tilt_torque(
        self,
        obj_cache,
        tray_normal,
        *,
        max_tilt_acceleration=None,
        max_tilt_speed=None,
    ):
        """Apply bounded settling torque toward the tray normal."""
        if max_tilt_acceleration is None:
            max_tilt_acceleration = self._TRAY_SETTLING_MAX_TILT_ACCELERATION
        if max_tilt_speed is None:
            max_tilt_speed = self._TRAY_SETTLING_MAX_TILT_SPEED
        body_id = obj_cache['id']
        support_axis = self._world_support_axis(obj_cache)
        axis_cross = np.cross(support_axis, tray_normal)
        cross_norm = np.linalg.norm(axis_cross)
        axis_dot = np.clip(np.dot(support_axis, tray_normal), -1.0, 1.0)

        if cross_norm > 1e-9:
            rotation_error = (
                axis_cross / cross_norm
                * np.arctan2(cross_norm, axis_dot)
            )
        elif axis_dot >= 0.0:
            rotation_error = np.zeros(3)
        else:
            # An exactly inverted object has no unique shortest rotation axis.
            reference = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(reference, support_axis)) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            rotation_axis = np.cross(support_axis, reference)
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation_error = np.pi * rotation_axis

        angular_velocity = self.sim.data.qvel[
            obj_cache['qvel_adr'] + 3:obj_cache['qvel_adr'] + 6
        ]
        tilt_velocity = (
            angular_velocity
            - tray_normal * np.dot(angular_velocity, tray_normal)
        )
        angular_acceleration = (
            self._TRAY_SETTLING_TILT_STIFFNESS * rotation_error
            - self._TRAY_SETTLING_TILT_DAMPING * angular_velocity
        )
        acceleration_norm = np.linalg.norm(angular_acceleration)
        if acceleration_norm > max_tilt_acceleration:
            angular_acceleration *= (
                max_tilt_acceleration
                / acceleration_norm
            )

        tilt_speed = np.linalg.norm(tilt_velocity)
        if (
            tilt_speed >= max_tilt_speed
            and np.dot(angular_acceleration, tilt_velocity) > 0.0
        ):
            angular_acceleration -= tilt_velocity * (
                np.dot(angular_acceleration, tilt_velocity)
                / (tilt_speed * tilt_speed)
            )

        try:
            inertia_mat = np.asarray(
                self.sim.data.ximat[body_id], dtype=float
            ).reshape(3, 3)
        except (AttributeError, IndexError, ValueError):
            inertia_mat = self.sim.data.body_xmat[body_id].reshape(3, 3)
        principal_inertia = np.asarray(
            self.sim.model.body_inertia[body_id], dtype=float
        )
        world_inertia = inertia_mat.dot(
            np.diag(principal_inertia)
        ).dot(inertia_mat.T)
        self.sim.data.xfrc_applied[body_id, 3:6] = world_inertia.dot(
            angular_acceleration
        )

    def _is_supported_flat_on_tray(
        self, obj_cache, slot_index, site_pos, site_mat, site_size, slot_contact_geoms
    ):
        """Return whether a released object is physically settled on a Level-1 tray."""
        if obj_cache['geom_ids'].isdisjoint(slot_contact_geoms):
            return False

        obj_pos = self.sim.data.body_xpos[obj_cache['id']]
        local_obj_pos = site_mat.T.dot(obj_pos - site_pos)
        if (abs(local_obj_pos[0]) > site_size[0] or
                abs(local_obj_pos[1]) > site_size[1]):
            return False

        support_axis = self._world_support_axis(obj_cache)
        if np.dot(support_axis, site_mat[:, 2]) < self._TRAY_FLAT_DOT_MIN:
            return False

        support_gap = self._tray_support_gap(obj_cache, slot_index, site_mat)
        return self._TRAY_SUPPORT_GAP_MIN <= support_gap <= self._TRAY_SUPPORT_GAP_MAX

    def _correct_tray_penetration(
        self, obj_cache, slot_index, site_mat, tray_velocity
    ):
        """Enforce a unilateral tray floor without adding tangential friction."""
        support_gap = self._tray_support_gap(obj_cache, slot_index, site_mat)
        if support_gap >= self._TRAY_PENETRATION_LIMIT:
            return support_gap

        normal = site_mat[:, 2]
        correction = self._TRAY_SUPPORT_CLEARANCE - support_gap
        qpos_adr = obj_cache['qpos_adr']
        qvel_adr = obj_cache['qvel_adr']
        self.sim.data.qpos[qpos_adr:qpos_adr+3] += normal * correction

        object_velocity = self.sim.data.qvel[qvel_adr:qvel_adr+3]
        inward_speed = np.dot(object_velocity - tray_velocity, normal)
        if inward_speed < 0.0:
            self.sim.data.qvel[qvel_adr:qvel_adr+3] -= normal * inward_speed
        return self._TRAY_SUPPORT_CLEARANCE

    def _correct_groove_floor_penetration(
        self, obj_name, obj_cache, slot_index, site_mat, groove_velocity
    ):
        """Enforce an authored groove bottom before evaluating its goal pose."""
        support_gap = self._groove_floor_support_gap(
            obj_cache, slot_index, site_mat
        )
        if support_gap >= self._TRAY_PENETRATION_LIMIT:
            return support_gap

        normal = site_mat[:, 2]
        correction = self._FLAT_GROOVE_SUPPORT_GAP - support_gap
        qpos_adr = obj_cache['qpos_adr']
        qvel_adr = obj_cache['qvel_adr']
        self.sim.data.qpos[qpos_adr:qpos_adr+3] += normal * correction
        object_velocity = self.sim.data.qvel[qvel_adr:qvel_adr+3]
        inward_speed = np.dot(object_velocity - groove_velocity, normal)
        if inward_speed < 0.0:
            self.sim.data.qvel[qvel_adr:qvel_adr+3] -= normal * inward_speed
        if correction >= 0.001:
            print(
                f"[GROOVE FLOOR GUARD] {obj_name} -> "
                f"{self.dynamic_groove_names[slot_index]} | "
                f"gap={1000.0 * support_gap:.2f}mm | "
                f"lift={1000.0 * correction:.2f}mm"
            )
        return self._FLAT_GROOVE_SUPPORT_GAP

    def _carry_pump_flat_groove_settling(
        self, obj_cache, site_mat, groove_velocity
    ):
        """Carry a supported Pump tangentially while gravity finishes seating."""
        normal = site_mat[:, 2]
        object_velocity = self.sim.data.qvel[
            obj_cache['qvel_adr']:obj_cache['qvel_adr']+3
        ]
        groove_tangent = groove_velocity - normal * np.dot(
            groove_velocity, normal
        )
        object_normal = normal * np.dot(object_velocity, normal)
        self.sim.data.qvel[
            obj_cache['qvel_adr']:obj_cache['qvel_adr']+3
        ] = groove_tangent + object_normal

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        if hasattr(self, 'conveyor_speed'):
            if not getattr(self, '_track_initialized', False):
                self._initialize_track()
            self._apply_physical_track(policy_step=policy_step)

    def _post_action(self, action):
        result = super()._post_action(action)
        track_velocity = getattr(self, '_track_velocity', None)
        if track_velocity is not None:
            physical_grooves = getattr(self, '_physical_service_grooves', set())
            for index, qvel_adr in enumerate(self._groove_qvel_adrs):
                if index not in physical_grooves:
                    self.sim.data.qvel[qvel_adr:qvel_adr+3] = track_velocity
                    self.sim.data.qvel[qvel_adr+3:qvel_adr+6] = 0.0
            locked_objects = set(self._v_groove_pose_locks) | set(
                self._tray_planar_locks
            )
            for obj_name in locked_objects:
                qvel_adr = self._object_info_cache[obj_name]['qvel_adr']
                self.sim.data.qvel[qvel_adr:qvel_adr+3] = track_velocity
                self.sim.data.qvel[qvel_adr+3:qvel_adr+6] = 0.0
        return result

    def _apply_physical_track(self, policy_step=False):
        if not hasattr(self, 'dynamic_groove_names'):
            return

        self._physical_service_grooves = set()

        current_speed = self.conveyor_speed
        is_service_phase = False
        service_remaining = np.inf
        service_time = 0.0
        if self.conveyor_mode == "pulse":
            params = getattr(self, "current_conveyor_params", {})
            transfer_time = params.get("transfer_time", 1.0)
            service_time = params.get("service_time", 1.0)
            cycle_time = transfer_time + service_time
            phase_time = self.sim.data.time % cycle_time if cycle_time > 0.0 else 0.0
            is_service_phase = (
                transfer_time <= 0.0
                or cycle_time <= 0.0
                or phase_time >= transfer_time
            )
            if is_service_phase:
                current_speed = 0.0
                service_remaining = max(0.0, cycle_time - phase_time)
        service_ended = (
            self._v_groove_was_service_phase
            and not is_service_phase
        )
        control_dt = 1.0 / max(float(getattr(self, 'control_freq', 20.0)), 1.0)
        terminal_service_frame = (
            is_service_phase
            and service_time > 0.0
            # Latch inside the last stationary control interval. The physics
            # hook runs at the MuJoCo timestep, so it can resolve this boundary
            # even though the policy acts only at ``control_freq``.
            and service_remaining <= control_dt + 1e-9
        )
        service_lock_frame = service_ended or terminal_service_frame
        self._v_groove_was_service_phase = is_service_phase
        dt = self.sim.model.opt.timestep

        # External forces persist in MuJoCo until explicitly changed. Clear the
        # previous settling torque before evaluating this action's contacts.
        for obj_cache in self._object_info_cache.values():
            self.sim.data.xfrc_applied[obj_cache['id'], 3:6] = 0.0

        conv_id = self.obj_body_id[self.target_conveyor_name]
        conv_pos = self.sim.data.body_xpos[conv_id]
        conv_mat = T.quat2mat(T.convert_quat(self.sim.data.body_xquat[conv_id], to="xyzw"))

        vel_vec = self.forward_vec * current_speed
        self._track_velocity = vel_vec
        self._groove_site_mats = [self.sim.data.site_xmat[sid].reshape(3, 3) if sid >= 0 else None for sid in self._groove_site_ids]
        site_xpos = [self.sim.data.site_xpos[sid] if sid >= 0 else None for sid in self._groove_site_ids]
        belt_range = self.belt_end_y_local - self.belt_start_y_local

        # Reuse one physics-frame contact snapshot for every product and slot.
        gripper_contacts = set()
        slot_contacts = [set() for _ in self.dynamic_groove_names]
        tray_floor_contacts = [set() for _ in self.dynamic_groove_names]
        contacts = self.sim.data.contact[:self.sim.data.ncon]
        colliding_objects = self._colliding_object_names(
            self._object_info_cache, contacts
        )
        for contact in contacts:
            geom1, geom2 = contact.geom1, contact.geom2
            if geom1 in self._gripper_geom_ids:
                gripper_contacts.add(geom2)
            elif geom2 in self._gripper_geom_ids:
                gripper_contacts.add(geom1)

            slot_index = self._groove_geom_to_index.get(geom1)
            if slot_index is not None:
                slot_contacts[slot_index].add(geom2)
                if geom1 in self._tray_floor_geom_ids[slot_index]:
                    tray_floor_contacts[slot_index].add(geom2)
            slot_index = self._groove_geom_to_index.get(geom2)
            if slot_index is not None:
                slot_contacts[slot_index].add(geom1)
                if geom2 in self._tray_floor_geom_ids[slot_index]:
                    tray_floor_contacts[slot_index].add(geom1)

        carrier_occupants = {
            groove_name: self._single_carrier_occupants(
                groove_name, slot_contacts[slot_index]
            )
            for slot_index, groove_name in enumerate(self.dynamic_groove_names)
            if self._is_single_object_carrier(groove_name)
        }
        capacity_rejections = {
            (obj_name, groove_name)
            for groove_name, occupants in carrier_occupants.items()
            if len(occupants) > 1
            for obj_name in occupants
        }
        previous_rejections = getattr(
            self, "_slot_capacity_rejections", set()
        )
        for obj_name, groove_name in sorted(
            capacity_rejections - previous_rejections
        ):
            print(
                f"[SLOT CAPACITY REJECT] {obj_name} -> {groove_name} | "
                f"occupants={sorted(carrier_occupants[groove_name])}"
            )
        self._slot_capacity_rejections = capacity_rejections
        over_capacity_objects = {
            obj_name for obj_name, _ in capacity_rejections
        }
        self._delivered_objects.difference_update(over_capacity_objects)
        for obj_name in over_capacity_objects:
            self._object_routed_steps[obj_name] = 0

        routed_objects = set()
        grasped_objects = set()
        for i, g_name in enumerate(self.dynamic_groove_names):
            g_id = self.obj_body_id[g_name]
            qpos_adr = self._groove_qpos_adrs[i]
            qvel_adr = self._groove_qvel_adrs[i]
            self.sim.data.qfrc_applied[qvel_adr:qvel_adr+6] = 0.0

            old_groove_pos = self.sim.data.qpos[qpos_adr:qpos_adr+3].copy()
            self.groove_local_ys[i] += current_speed * dt

            # Handle looping
            did_loop = False
            if self.groove_local_ys[i] > self.belt_end_y_local:
                self.groove_local_ys[i] -= belt_range
                did_loop = True
            elif self.groove_local_ys[i] < self.belt_start_y_local:
                self.groove_local_ys[i] += belt_range
                did_loop = True

            local_xy = np.array([0, self.groove_local_ys[i], 0])
            exact_world_pos = conv_pos + conv_mat.dot(local_xy)
            exact_world_pos[2] = self.abs_z_height
            delta_pos = exact_world_pos - old_groove_pos

            site_pos = site_xpos[i]
            site_mat = self._groove_site_mats[i]
            site_size = self._groove_site_sizes[i]
            physical_v_service_support = False

            if site_pos is not None and site_mat is not None:
                for obj_name, obj_cache in self._object_info_cache.items():
                    obj_geom_ids = obj_cache['geom_ids']
                    is_grasped = not obj_geom_ids.isdisjoint(gripper_contacts)
                    actually_grasped = self._has_confirmed_gripper_grasp(
                        obj_name, is_grasped
                    )
                    if is_grasped:
                        grasped_objects.add(obj_name)
                        # Preserve a placed object's independent tray lock when
                        # another placement merely grazes it. A confirmed
                        # re-grasp still returns that object to robot ownership.
                        if actually_grasped:
                            self._tray_planar_locks.pop(obj_name, None)
                            self._tray_settling.pop(obj_name, None)
                        groove_lock = self._v_groove_pose_locks.get(obj_name)
                        headset_inlay_lock = bool(
                            groove_lock is not None
                            and groove_lock.get('headset_inlay', False)
                        )
                        if (
                            groove_lock is not None
                            and actually_grasped
                            and not headset_inlay_lock
                        ):
                            print(
                                f"[GROOVE POSE UNLOCK] {obj_name} -> "
                                f"{groove_lock.get('slot')} | reason=grasp"
                            )
                            self._v_groove_pose_locks.pop(obj_name, None)
                            self._v_groove_settling.pop(obj_name, None)

                    obj_pos = self.sim.data.body_xpos[obj_cache['id']]
                    local_obj_pos = site_mat.T.dot(obj_pos - site_pos)

                    # B & C. POSITION, TYPE, AND ORIENTATION COMPATIBILITY
                    if obj_name not in self._sticky_compat.get(g_name, {}):
                        self._sticky_compat[g_name][obj_name] = self._object_matches_groove(obj_name, g_name)
                    
                    is_correct_type = self._sticky_compat[g_name][obj_name]
                    is_tray = "tray" in g_name.lower()
                    is_inside_xyz = False
                    should_stick = False

                    if is_correct_type:
                        if is_tray:
                            inside_tray_xy = (
                                abs(local_obj_pos[0]) <= site_size[0]
                                and abs(local_obj_pos[1]) <= site_size[1]
                            )
                            if not is_grasped and inside_tray_xy:
                                self._correct_tray_penetration(
                                    obj_cache, i, site_mat, vel_vec
                                )
                            support_gap = self._tray_support_gap(
                                obj_cache, i, site_mat
                            )
                            settling = self._tray_settling.get(obj_name)
                            if (
                                settling is None
                                and obj_name not in self._tray_planar_locks
                                and not is_grasped
                                and inside_tray_xy
                                and self._TRAY_SETTLING_GAP_MIN
                                <= support_gap
                                <= self._TRAY_SUPPORT_GAP_MAX
                            ):
                                self._tray_settling[obj_name] = {
                                    'tray': g_name,
                                    'stable_steps': 0,
                                }
                                settling = self._tray_settling[obj_name]
                                print(
                                    f"[TRAY SETTLING] {obj_name} -> {g_name} | "
                                    f"gap={1000.0 * support_gap:.2f}mm"
                                )

                            # LEVEL 1: gravity and the real tray floor control Z.
                            # A bounded contact-only torque damps tilt toward the
                            # object's metadata-derived support axis.
                            has_floor_contact = not obj_geom_ids.isdisjoint(
                                tray_floor_contacts[i]
                            )
                            flat_dot = np.dot(
                                self._world_support_axis(obj_cache),
                                site_mat[:, 2],
                            )
                            if (
                                has_floor_contact
                                and settling is not None
                                and settling['tray'] == g_name
                            ):
                                self._apply_tray_tilt_torque(
                                    obj_cache, site_mat[:, 2]
                                )

                            physically_supported = self._is_supported_flat_on_tray(
                                obj_cache, i, site_pos, site_mat, site_size,
                                tray_floor_contacts[i]
                            )
                            angular_speed = np.linalg.norm(
                                self.sim.data.qvel[
                                    obj_cache['qvel_adr'] + 3:
                                    obj_cache['qvel_adr'] + 6
                                ]
                            )
                            stable_now = (
                                physically_supported
                                and settling is not None
                                and settling['tray'] == g_name
                                and angular_speed
                                <= self._TRAY_SETTLING_ANGULAR_SPEED_MAX
                            )
                            if settling is not None and settling['tray'] == g_name:
                                settling['stable_steps'] = (
                                    settling['stable_steps'] + 1
                                    if stable_now else 0
                                )
                            should_stick = (
                                stable_now
                                and settling['stable_steps']
                                >= self._TRAY_SETTLING_STABLE_STEPS
                                and obj_name not in colliding_objects
                            )
                            is_inside_xyz = should_stick
                            
                        else:
                            # LEVEL 2 & 3: require contact, CoM containment, and alignment.
                            is_touching_groove = not obj_geom_ids.isdisjoint(slot_contacts[i])
                            carrier_capacity_ok = (
                                not self._is_single_object_carrier(g_name)
                                or carrier_occupants.get(g_name, set())
                                == {obj_name}
                            )
                            uses_service_lock = self._uses_v_groove_service_lock(
                                obj_name, g_name
                            )
                            service_lock = self._v_groove_pose_locks.get(obj_name)
                            if (
                                service_lock is not None
                                and service_lock['slot'] != g_name
                            ):
                                service_lock = None
                            breakable_lock_contact = (
                                self._breakable_lock_contact(obj_name, obj_cache)
                                if (
                                    "pump_bottle" in obj_name.lower()
                                    and "flat_groove" in g_name.lower()
                                )
                                else None
                            )
                            if (
                                service_lock is not None
                                and breakable_lock_contact is not None
                            ):
                                self._v_groove_pose_locks.pop(obj_name, None)
                                self._v_groove_settling[obj_name] = {
                                    'slot': g_name,
                                    'stable_steps': 0,
                                }
                                service_lock = None
                                other_geom_name = (
                                    self.sim.model.geom_id2name(
                                        breakable_lock_contact['other_geom']
                                    )
                                    or str(breakable_lock_contact['other_geom'])
                                )
                                print(
                                    f"[GROOVE LOCK BREAK] {obj_name} -> {g_name} | "
                                    f"other={breakable_lock_contact['owner']}:"
                                    f"{other_geom_name} | "
                                    f"force={breakable_lock_contact['force']:.3f}N | "
                                    f"threshold="
                                    f"{breakable_lock_contact['threshold']:.3f}N | "
                                    f"distance="
                                    f"{1000.0 * breakable_lock_contact['distance']:.3f}mm"
                                )

                            if is_touching_groove:
                                uses_floor_guard = (
                                    self._uses_flat_groove_floor_guard(
                                        obj_name, g_name
                                    )
                                ) or (
                                    "taipan" in obj_name.lower()
                                    and "v_groove" in g_name.lower()
                                )
                                floor_support_gap = np.inf
                                if uses_floor_guard and not is_grasped:
                                    floor_support_gap = self._correct_groove_floor_penetration(
                                        obj_name, obj_cache, i, site_mat, vel_vec
                                    )
                                if (
                                    uses_floor_guard
                                    and not is_grasped
                                    and floor_support_gap
                                    <= self._TRAY_SUPPORT_GAP_MAX
                                    and abs(local_obj_pos[0])
                                    <= site_size[0] + 0.005
                                    and abs(local_obj_pos[1]) <= site_size[1]
                                ):
                                    self._carry_pump_flat_groove_settling(
                                        obj_cache, site_mat, vel_vec
                                    )
                                if (
                                    "pump_bottle" in obj_name.lower()
                                    and "flat_groove" in g_name.lower()
                                    and not is_grasped
                                    and floor_support_gap
                                    <= self._TRAY_SUPPORT_GAP_MAX
                                    and abs(local_obj_pos[0])
                                    <= site_size[0] + 0.005
                                    and abs(local_obj_pos[1]) <= site_size[1]
                                ):
                                    obj_mat = self.sim.data.body_xmat[
                                        obj_cache['id']
                                    ].reshape(3, 3)
                                    long_axis = obj_mat[:, 2]
                                    projected_axis = (
                                        long_axis
                                        - np.dot(long_axis, site_mat[:, 2])
                                        * site_mat[:, 2]
                                    )
                                    projected_norm = np.linalg.norm(projected_axis)
                                    if (
                                        projected_norm > 1e-6
                                        and abs(
                                            np.dot(
                                                projected_axis / projected_norm,
                                                site_mat[:, 1],
                                            )
                                        ) > 0.9
                                    ):
                                        target_axis = site_mat[:, 1]
                                        if np.dot(long_axis, target_axis) < 0.0:
                                            target_axis = -target_axis
                                        self._apply_tray_tilt_torque(
                                            obj_cache,
                                            target_axis,
                                            max_tilt_acceleration=(
                                                self._PUMP_FLAT_SETTLING_MAX_TILT_ACCELERATION
                                            ),
                                            max_tilt_speed=(
                                                self._PUMP_FLAT_SETTLING_MAX_TILT_SPEED
                                            ),
                                        )
                                routed_now = self._is_routed_in_slot_region(
                                    obj_name,
                                    g_name,
                                    obj_cache=obj_cache,
                                    site_pos=site_pos,
                                    site_mat=site_mat,
                                    site_size=site_size,
                                ) and carrier_capacity_ok
                                is_headset_inlay = (
                                    "headset" in obj_name.lower()
                                    and "headset" in g_name.lower()
                                    and "inlay" in g_name.lower()
                                )
                                is_headset_carrier = (
                                    "headset" in obj_name.lower()
                                    and "headset" in g_name.lower()
                                )
                                if (
                                    routed_now
                                    and is_headset_carrier
                                    and not is_grasped
                                    and service_lock is None
                                ):
                                    obj_qpos_adr = obj_cache['qpos_adr']
                                    service_lock = {
                                        'slot': g_name,
                                        'local_pos': site_mat.T.dot(
                                            self.sim.data.body_xpos[
                                                obj_cache['id']
                                            ] - site_pos
                                        ),
                                        'quat': self.sim.data.qpos[
                                            obj_qpos_adr+3:obj_qpos_adr+7
                                        ].copy(),
                                    }
                                    if is_headset_inlay:
                                        service_lock['headset_inlay'] = True
                                    self._v_groove_pose_locks[
                                        obj_name
                                    ] = service_lock
                                    self._v_groove_settling.pop(obj_name, None)
                                    print(
                                        f"[HEADSET POSE LOCK] {obj_name} -> "
                                        f"{g_name}"
                                    )
                                if uses_service_lock:
                                    physical_v_service_support = bool(
                                        physical_v_service_support
                                        or (
                                            is_service_phase
                                            and service_lock is None
                                            and is_touching_groove
                                        )
                                    )
                                    obj_qvel_adr = obj_cache['qvel_adr']
                                    support_state = self._v_groove_support_state(
                                        obj_name, g_name
                                    )
                                    support_left = bool(
                                        support_state is not None
                                        and support_state['left_contact']
                                    )
                                    support_right = bool(
                                        support_state is not None
                                        and support_state['right_contact']
                                    )
                                    support_load_ready = bool(
                                        support_state is not None
                                        and support_state['load_ready']
                                    )
                                    support_up_ratio = (
                                        np.nan
                                        if support_state is None
                                        else support_state['upward_load_ratio']
                                    )
                                    support_torque_ratio = (
                                        np.nan
                                        if support_state is None
                                        else support_state['torque_ratio']
                                    )
                                    support_penetration = (
                                        np.nan
                                        if support_state is None
                                        else support_state['maximum_penetration']
                                    )
                                    actually_grasped = is_grasped
                                    if service_lock is None:
                                        try:
                                            actually_grasped = self._check_grasp(
                                                self.robots[0].gripper,
                                                self.objects_dict[
                                                    obj_name
                                                ].contact_geoms,
                                            )
                                        except (AttributeError, KeyError, TypeError):
                                            # Retain the conservative contact
                                            # fallback across Robosuite gripper
                                            # API variants.
                                            pass
                                    linear_speed = np.linalg.norm(
                                        self.sim.data.qvel[
                                            obj_qvel_adr:obj_qvel_adr+3
                                        ]
                                    )
                                    angular_speed = np.linalg.norm(
                                        self.sim.data.qvel[
                                            obj_qvel_adr+3:obj_qvel_adr+6
                                        ]
                                    )
                                    stable_now = (
                                        routed_now
                                        and not is_grasped
                                        and not actually_grasped
                                        and support_load_ready
                                        and linear_speed
                                        <= self._V_GROOVE_SETTLING_LINEAR_SPEED_MAX
                                        and angular_speed
                                        <= self._V_GROOVE_SETTLING_ANGULAR_SPEED_MAX
                                    )
                                    settling = self._v_groove_settling.get(obj_name)
                                    if settling is None or settling['slot'] != g_name:
                                        settling = {
                                            'slot': g_name,
                                            'stable_steps': 0,
                                        }
                                        self._v_groove_settling[obj_name] = settling
                                    if is_service_phase:
                                        settling['stable_steps'] = (
                                            settling['stable_steps'] + 1
                                            if stable_now else 0
                                        )

                                    if (
                                        service_lock_frame
                                        and service_lock is None
                                        and not actually_grasped
                                        and stable_now
                                        and settling['stable_steps']
                                        >= self._V_GROOVE_SETTLING_STABLE_STEPS
                                    ):
                                        obj_qpos_adr = obj_cache['qpos_adr']
                                        service_lock = {
                                            'slot': g_name,
                                            'local_pos': site_mat.T.dot(
                                                self.sim.data.body_xpos[
                                                    obj_cache['id']
                                                ] - site_pos
                                            ),
                                            'quat': self.sim.data.qpos[
                                                obj_qpos_adr+3:obj_qpos_adr+7
                                            ].copy(),
                                        }
                                        self._v_groove_pose_locks[obj_name] = service_lock
                                        self._v_groove_settling.pop(obj_name, None)
                                        print(
                                            f"[V-GROOVE SERVICE LOCK] {obj_name} -> "
                                            f"{g_name} | t={self.sim.data.time:.3f}s | "
                                            f"remaining={service_remaining:.3f}s | "
                                            f"supports=({support_left},"
                                            f"{support_right}) | "
                                            f"up_ratio={support_up_ratio:.3f} | "
                                            f"torque_ratio={support_torque_ratio:.3f} | "
                                            f"penetration="
                                            f"{1000.0 * support_penetration:.2f}mm | "
                                            f"linear_speed={linear_speed:.3f}m/s | "
                                            f"angular_speed={angular_speed:.3f}rad/s"
                                        )
                                    elif (
                                        service_lock_frame
                                        and service_lock is None
                                        and not settling.get('terminal_reported', False)
                                    ):
                                        obj_mat = self.sim.data.body_xmat[
                                            obj_cache['id']
                                        ].reshape(3, 3)
                                        semantic_top = -obj_mat[:, 1]
                                        lid_axis = site_mat.T.dot(semantic_top)
                                        settling['terminal_reported'] = True
                                        print(
                                            f"[V-GROOVE SERVICE REJECT] {obj_name} -> "
                                            f"{g_name} | t={self.sim.data.time:.3f}s | "
                                            f"remaining={service_remaining:.3f}s | "
                                            f"routed={routed_now} | "
                                            f"supports=({support_left},"
                                            f"{support_right}) | "
                                            f"up_ratio={support_up_ratio:.3f} | "
                                            f"torque_ratio={support_torque_ratio:.3f} | "
                                            f"penetration="
                                            f"{1000.0 * support_penetration:.2f}mm | "
                                            f"gripper_contact={is_grasped} | "
                                            f"grasped={actually_grasped} | "
                                            f"stable_steps={settling['stable_steps']} | "
                                            f"linear_speed={linear_speed:.3f}m/s | "
                                            f"angular_speed={angular_speed:.3f}rad/s | "
                                            f"lid=({lid_axis[0]:.3f},"
                                            f"{lid_axis[1]:.3f},{lid_axis[2]:.3f})"
                                        )

                                    # Before the terminal boundary, MuJoCo owns
                                    # translation, gravity, and every rotation.
                                    # Delivery begins only from the saved settled
                                    # pose, never from the first transient contact.
                                    should_stick = service_lock is not None
                                    is_inside_xyz = (
                                        should_stick and carrier_capacity_ok
                                    )
                                else:
                                    obj_qvel_adr = obj_cache['qvel_adr']
                                    linear_speed = np.linalg.norm(
                                        self.sim.data.qvel[
                                            obj_qvel_adr:obj_qvel_adr+3
                                        ]
                                    )
                                    angular_speed = np.linalg.norm(
                                        self.sim.data.qvel[
                                            obj_qvel_adr+3:obj_qvel_adr+6
                                        ]
                                    )
                                    uses_flat_groove_floor_guard = (
                                        self._uses_flat_groove_floor_guard(
                                            obj_name, g_name
                                        )
                                    )
                                    floor_support_gap = (
                                        self._groove_floor_support_gap(
                                            obj_cache, i, site_mat
                                        )
                                        if uses_flat_groove_floor_guard else np.inf
                                    )
                                    settling_linear_speed = (
                                        np.linalg.norm(
                                            self.sim.data.qvel[
                                                obj_qvel_adr:obj_qvel_adr+3
                                            ] - vel_vec
                                        )
                                        if uses_flat_groove_floor_guard
                                        else linear_speed
                                    )
                                    stable_now = (
                                        (
                                            is_service_phase
                                            or uses_flat_groove_floor_guard
                                        )
                                        and routed_now
                                        and not is_grasped
                                        and breakable_lock_contact is None
                                        and floor_support_gap
                                        >= self._TRAY_PENETRATION_LIMIT
                                        and (
                                            not uses_flat_groove_floor_guard
                                            or floor_support_gap
                                            <= self._TRAY_SUPPORT_GAP_MAX
                                        )
                                        and settling_linear_speed
                                        <= self._V_GROOVE_SETTLING_LINEAR_SPEED_MAX
                                        and angular_speed
                                        <= self._V_GROOVE_SETTLING_ANGULAR_SPEED_MAX
                                    )
                                    settling = self._v_groove_settling.get(obj_name)
                                    if stable_now and (
                                        settling is None
                                        or settling['slot'] != g_name
                                    ):
                                        settling = {
                                            'slot': g_name,
                                            'stable_steps': 0,
                                        }
                                        self._v_groove_settling[obj_name] = settling
                                    if (
                                        policy_step
                                        and settling is not None
                                        and settling['slot'] == g_name
                                    ):
                                        settling['stable_steps'] = (
                                            settling['stable_steps'] + 1
                                            if stable_now else 0
                                        )
                                    if (
                                        service_lock is None
                                        and obj_name not in self._v_groove_pose_locks
                                        and stable_now
                                        and settling['stable_steps']
                                        >= self._V_GROOVE_SETTLING_STABLE_STEPS
                                    ):
                                        obj_qpos_adr = obj_cache['qpos_adr']
                                        locked_world_pos = self.sim.data.body_xpos[
                                            obj_cache['id']
                                        ].copy()
                                        if (
                                            uses_flat_groove_floor_guard
                                            and floor_support_gap
                                            < self._FLAT_GROOVE_SUPPORT_GAP
                                        ):
                                            locked_world_pos += site_mat[:, 2] * (
                                                self._FLAT_GROOVE_SUPPORT_GAP
                                                - floor_support_gap
                                            )
                                        service_lock = {
                                            'slot': g_name,
                                            'local_pos': site_mat.T.dot(
                                                locked_world_pos - site_pos
                                            ),
                                            'quat': self.sim.data.qpos[
                                                obj_qpos_adr+3:obj_qpos_adr+7
                                            ].copy(),
                                        }
                                        self._v_groove_pose_locks[obj_name] = service_lock
                                        self._v_groove_settling.pop(obj_name, None)
                                        print(
                                            f"[GROOVE POSE LOCK] {obj_name} -> "
                                            f"{g_name} | t={self.sim.data.time:.3f}s | "
                                            f"linear_speed={linear_speed:.3f}m/s | "
                                            f"relative_speed={settling_linear_speed:.3f}m/s | "
                                            f"angular_speed={angular_speed:.3f}rad/s | "
                                            f"floor_gap={1000.0 * floor_support_gap:.2f}mm"
                                        )
                                    should_stick = bool(
                                        breakable_lock_contact is None
                                        and (
                                            service_lock is not None
                                            or (
                                                routed_now
                                                and not uses_flat_groove_floor_guard
                                            )
                                        )
                                    )
                                    is_inside_xyz = (
                                        should_stick and carrier_capacity_ok
                                    )
                            elif service_lock is not None:
                                should_stick = True
                                is_inside_xyz = carrier_capacity_ok
                                
                    if is_inside_xyz:
                        routed_objects.add(obj_name)

                    # E. APPLY STICKY
                    if is_tray:
                        try:
                            obj_qvel_adr = obj_cache['qvel_adr']
                            obj_qpos_adr = obj_cache['qpos_adr']

                            lock = self._tray_planar_locks.get(obj_name)
                            # A wrapped tray is still the same physical carrier.
                            # Keep its product lock so the object follows the
                            # transfer jump instead of being stranded at outfeed.
                            if actually_grasped:
                                self._tray_planar_locks.pop(obj_name, None)
                                lock = None

                            settling = self._tray_settling.get(obj_name)
                            if did_loop and settling and settling['tray'] == g_name:
                                support_gap = self._tray_support_gap(
                                    obj_cache, i, site_mat
                                )
                                angular_speed = np.linalg.norm(
                                    self.sim.data.qvel[
                                        obj_cache['qvel_adr'] + 3:
                                        obj_cache['qvel_adr'] + 6
                                    ]
                                )
                                print(
                                    f"[TRAY SETTLING LOST] {obj_name} -> "
                                    f"{g_name} | tray looped | gap="
                                    f"{1000.0 * support_gap:.2f}mm | "
                                    f"flat_dot={flat_dot:.3f} | "
                                    f"angular_speed={angular_speed:.3f}rad/s"
                                )
                                self._tray_settling.pop(obj_name, None)
                                settling = None

                            if (
                                settling
                                and settling['tray'] == g_name
                                and lock is None
                                and not is_grasped
                                and not did_loop
                            ):
                                # Follow only the tray's in-plane translation
                                # during settling. Gravity, floor contact, Z, and
                                # all rotation remain under MuJoCo physics.
                                normal = site_mat[:, 2]
                                tangent_delta = (
                                    delta_pos
                                    - normal * np.dot(delta_pos, normal)
                                )
                                tangent_velocity = (
                                    vel_vec
                                    - normal * np.dot(vel_vec, normal)
                                )
                                self.sim.data.qpos[
                                    obj_qpos_adr:obj_qpos_adr+3
                                ] += tangent_delta
                                object_velocity = self.sim.data.qvel[
                                    obj_qvel_adr:obj_qvel_adr+3
                                ]
                                normal_velocity = normal * np.dot(
                                    object_velocity, normal
                                )
                                self.sim.data.qvel[
                                    obj_qvel_adr:obj_qvel_adr+3
                                ] = tangent_velocity + normal_velocity

                            if (
                                lock is None
                                and should_stick
                                and not is_grasped
                                and not did_loop
                            ):
                                support_gap = self._tray_support_gap(
                                    obj_cache, i, site_mat
                                )
                                normal = site_mat[:, 2]
                                corrected_obj_pos = (
                                    self.sim.data.body_xpos[obj_cache['id']].copy()
                                    + normal * (
                                        self._TRAY_SUPPORT_CLEARANCE
                                        - support_gap
                                    )
                                )
                                corrected_quat = self.sim.data.qpos[
                                    obj_qpos_adr+3:obj_qpos_adr+7
                                ].copy()
                                self._tray_planar_locks[obj_name] = {
                                    'tray': g_name,
                                    'local_pos': site_mat.T.dot(
                                        corrected_obj_pos - site_pos
                                    ),
                                    'quat': corrected_quat,
                                }
                                self._tray_settling.pop(obj_name, None)
                                lock = self._tray_planar_locks[obj_name]
                                print(
                                    f"[TRAY XYZ LOCK] {obj_name} -> {g_name} | "
                                    f"measured_gap={1000.0 * support_gap:.2f}mm | "
                                    f"angular_speed={angular_speed:.3f}rad/s | "
                                    f"locked_gap={1000.0 * self._TRAY_SUPPORT_CLEARANCE:.2f}mm"
                                )

                            if (
                                lock
                                and lock['tray'] == g_name
                                and not actually_grasped
                            ):
                                # Preserve a support-corrected pose relative to
                                # the tray.  XYZ transport is deterministic while
                                # the real base geom remains a penetration guard.
                                new_site_pos = site_pos + delta_pos
                                target_pos = new_site_pos + site_mat.dot(lock['local_pos'])

                                self.sim.data.qpos[obj_qpos_adr:obj_qpos_adr+3] = target_pos
                                self.sim.data.qpos[obj_qpos_adr+3:obj_qpos_adr+7] = lock['quat']
                                self.sim.data.qvel[obj_qvel_adr:obj_qvel_adr+3] = vel_vec
                                self.sim.data.qvel[obj_qvel_adr+3:obj_qvel_adr+6] = 0.0
                        except Exception:
                            pass
                    elif should_stick and (
                        not actually_grasped or headset_inlay_lock
                    ):
                        try:
                            obj_qvel_adr = obj_cache['qvel_adr']
                            obj_qpos_adr = obj_cache['qpos_adr']
                            service_lock = self._v_groove_pose_locks.get(obj_name)
                            if (
                                service_lock is not None
                                and service_lock['slot'] == g_name
                            ):
                                new_site_pos = site_pos + delta_pos
                                target_pos = new_site_pos + site_mat.dot(
                                    service_lock['local_pos']
                                )
                                self.sim.data.qpos[
                                    obj_qpos_adr:obj_qpos_adr+3
                                ] = target_pos
                                self.sim.data.qpos[
                                    obj_qpos_adr+3:obj_qpos_adr+7
                                ] = service_lock['quat']
                                self.sim.data.qvel[
                                    obj_qvel_adr:obj_qvel_adr+3
                                ] = vel_vec
                                self.sim.data.qvel[
                                    obj_qvel_adr+3:obj_qvel_adr+6
                                ] = 0.0
                                if did_loop:
                                    print(
                                        f"[GROOVE POSE WRAP] {obj_name} -> "
                                        f"{g_name} | raw_gripper_contact="
                                        f"{is_grasped} | grasped="
                                        f"{actually_grasped}"
                                    )
                            else:
                                # Other LEVEL 2 & 3 products retain their
                                # contact-triggered transport behavior.
                                self.sim.data.qvel[
                                    obj_qvel_adr:obj_qvel_adr+3
                                ] = vel_vec
                                self.sim.data.qvel[
                                    obj_qvel_adr+3:obj_qvel_adr+6
                                ] = 0.0
                                if not did_loop:
                                    self.sim.data.qpos[
                                        obj_qpos_adr:obj_qpos_adr+3
                                    ] += delta_pos
                                    self.sim.data.qvel[obj_qvel_adr+2] = 0.0
                        except Exception as exc:
                            if service_lock is not None:
                                print(
                                    f"[GROOVE POSE LOCK ERROR] {obj_name} -> "
                                    f"{g_name}: {type(exc).__name__}: {exc}"
                                )
            if physical_v_service_support:
                self._physical_service_grooves.add(i)
                groove_pos = np.asarray(
                    self.sim.data.qpos[qpos_adr:qpos_adr+3], dtype=float
                )
                groove_quat = np.asarray(
                    self.sim.data.qpos[qpos_adr+3:qpos_adr+7], dtype=float
                )
                groove_vel = np.asarray(
                    self.sim.data.qvel[qvel_adr:qvel_adr+3], dtype=float
                )
                groove_angular_vel = np.asarray(
                    self.sim.data.qvel[qvel_adr+3:qvel_adr+6], dtype=float
                )
                groove_mass = float(self.sim.model.body_mass[g_id])
                position_kp = self._V_GROOVE_SERVICE_POSITION_KP
                position_kd = 2.0 * np.sqrt(position_kp * groove_mass)
                gravity = np.asarray(self.sim.model.opt.gravity, dtype=float)
                self.sim.data.qfrc_applied[qvel_adr:qvel_adr+3] = (
                    -groove_mass * gravity
                    + position_kp * (exact_world_pos - groove_pos)
                    + position_kd * (vel_vec - groove_vel)
                )
                rotation_error = np.zeros(3)
                mujoco.mju_subQuat(
                    rotation_error,
                    np.asarray(self.world_groove_quat_wxyz, dtype=float),
                    groove_quat,
                )
                self.sim.data.qfrc_applied[qvel_adr+3:qvel_adr+6] = (
                    self._V_GROOVE_SERVICE_ROTATION_KP * rotation_error
                    - self._V_GROOVE_SERVICE_ROTATION_KD * groove_angular_vel
                )
            else:
                self.sim.data.qpos[qpos_adr:qpos_adr+3] = exact_world_pos
                self.sim.data.qpos[qpos_adr+3:qpos_adr+7] = self.world_groove_quat_wxyz
                self.sim.data.qvel[qvel_adr:qvel_adr+3] = vel_vec

        # Update delivery progress once per object after evaluating every slot.
        # Persistent carrier locks are the DynamicOn state after settling.
        routed_objects.update(self._tray_planar_locks)
        routed_objects.update(
            obj_name
            for obj_name in self._v_groove_pose_locks
            if obj_name not in over_capacity_objects
        )
        for obj_name in self._object_info_cache:
            if obj_name in routed_objects:
                if obj_name not in grasped_objects:
                    routed_steps = self._object_routed_steps.get(obj_name, 0) + 1
                    self._object_routed_steps[obj_name] = routed_steps
                    if routed_steps >= self._routed_threshold:
                        if obj_name not in self._delivered_objects:
                            lock = self._tray_planar_locks.get(obj_name)
                            if lock is not None:
                                tray_index = self.dynamic_groove_names.index(
                                    lock['tray']
                                )
                                tray_site_mat = self._groove_site_mats[tray_index]
                                live_gap = self._tray_support_gap(
                                    self._object_info_cache[obj_name],
                                    tray_index,
                                    tray_site_mat,
                                )
                                flat_dot = np.dot(
                                    self._world_support_axis(
                                        self._object_info_cache[obj_name]
                                    ),
                                    tray_site_mat[:, 2],
                                )
                                print(
                                    f"[TRAY DELIVERY STATE] {obj_name} -> "
                                    f"{lock['tray']} | gap="
                                    f"{1000.0 * live_gap:.2f}mm | "
                                    f"flat_dot={flat_dot:.3f}"
                                )
                        self._delivered_objects.add(obj_name)
                # Gripper contact pauses progress without resetting it.
            else:
                routed_steps = self._object_routed_steps.get(obj_name, 0)
                self._object_routed_steps[obj_name] = max(0, routed_steps - 1)
