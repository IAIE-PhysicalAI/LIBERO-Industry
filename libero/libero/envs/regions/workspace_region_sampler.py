import re

import numpy as np

from .base_region_sampler import MultiRegionRandomSampler, InSiteRegionRandomSampler
from robosuite.utils.transform_utils import quat_multiply
from copy import copy
from robosuite.utils.errors import RandomizationError
from robosuite.utils.transform_utils import quat_multiply
import robosuite.utils.transform_utils as T


_DYNAMIC_CONVEYOR_REGION = re.compile(
    r"^(?P<prefix>.+)_(?P<kind>offset|tail)_(?:region(?:_d2d)?|bowl)_(?P<slot>\d+)$"
)


def _conveyor_timing_target(
    sim,
    site_name,
    timing,
    reference_pos=None,
    reference_quat=None,
):
    """Resolve a timed source position from the placed belt's centre site.

    The timing metadata is attached by the workbench problem after generic
    site ranges are initialized. Conditional object placement happens before
    MuJoCo applies the sampled fixture pose, so callers pass the belt pose from
    ``placed_objects`` when it is available. This turns the canonical XML
    sites into their eventual world positions without changing or deriving the
    centre-region geometry.
    """
    match = _DYNAMIC_CONVEYOR_REGION.match(site_name)
    if match is None:
        raise RandomizationError("Invalid dynamic conveyor site: {}".format(site_name))
    prefix = match.group("prefix")
    centre_name = "{}_centre_region".format(prefix)
    def site_world_position(name):
        position = np.asarray(sim.data.get_site_xpos(name), dtype=float)
        if reference_pos is None or reference_quat is None:
            return position
        rotation = T.quat2mat(
            T.convert_quat(np.asarray(reference_quat, dtype=float), to="xyzw")
        )
        return np.asarray(reference_pos, dtype=float) + rotation.dot(position)

    site_position = site_world_position(site_name)
    try:
        centre_position = site_world_position(centre_name)
    except Exception as error:
        raise RandomizationError(
            "Canonical centre_region is required for {}: {}".format(site_name, error)
        )

    axis = None
    for tail_name in (
        "{}_tail_region_4".format(prefix),
        "{}_tail_region_d2d_4".format(prefix),
    ):
        try:
            candidate = site_world_position(tail_name) - centre_position
        except Exception:
            continue
        length = np.linalg.norm(candidate[:2])
        if length > 1e-8:
            axis = candidate[:2] / length
            break
    if axis is None:
        candidate = site_position[:2] - centre_position[:2]
        length = np.linalg.norm(candidate)
        if length <= 1e-8:
            raise RandomizationError("Cannot derive transport axis for {}".format(site_name))
        axis = candidate / length
        if match.group("kind") == "offset":
            axis *= -1.0

    slot = int(timing["slot"])
    distance = abs(float(timing["speed"])) * (
        float(timing["offset"]) + (slot - 1) * float(timing["takt"])
    )
    motion_sign = 1.0 if float(timing["speed"]) >= 0.0 else -1.0
    target_xy = centre_position[:2] - motion_sign * distance * axis
    return target_xy, centre_position[:2], site_position


class TableRegionSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class Libero100TableRegionSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class ObjectBasedSampler(MultiRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=(np.pi / 2, np.pi / 2),
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"table-middle-{object_name}"
        super().__init__(
            object_name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def _sample_quat(self):
        """
        Samples the orientation for a given object
        Add multiple rotation options
        Returns:
            np.array: sampled (r,p,y) euler angle orientation
        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # multiple rotations
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )


class LiberoIndustrySampler(InSiteRegionRandomSampler):
    def __init__(
        self,
        object_name,
        mujoco_objects=None,
        x_ranges=None,
        y_ranges=None,
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.01,
    ):
        name = f"conveyor-{object_name}"
        super().__init__(
            name,
            mujoco_objects,
            x_ranges,
            y_ranges,
            rotation,
            rotation_axis,
            ensure_object_boundary_in_range,
            ensure_valid_placement,
            reference_pos,
            z_offset,
        )

    def sample(self, fixtures=None, reference=None, site_name="", on_top=True, sim=None):
        """
        Overrides the base sample method to make 'sim' an optional argument.
        """
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
            reference_position = None
            reference_quat = None
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            reference_position = ref_pos
            reference_quat = ref_quat
        else:
            base_offset = np.array(reference)
            reference_position = None
            reference_quat = None
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        for obj in self.mujoco_objects:
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            
            # --- THE FIX: Safe sim check ---
            if sim is not None and site_name != "":
                site_offset = sim.data.get_site_xpos(site_name)
                if 'ref_quat' in locals():
                    site_x, site_y, site_z = T.quat2mat(
                        T.convert_quat(ref_quat, to="xyzw")
                    ) @ site_offset
                else:
                    site_x, site_y, site_z = site_offset

                timing = getattr(self, "_conveyor_timing", None)
                if timing is not None:
                    target_xy, _, _ = _conveyor_timing_target(
                        sim,
                        site_name,
                        timing,
                        reference_pos=reference_position,
                        reference_quat=reference_quat,
                    )
                    # The timing position is a world-space CoM target. Keep
                    # the site Z component for on-top placement below. Timed
                    # sources intentionally may lie outside the task's XY
                    # distribution while travelling along the long belt.
                    site_x = target_xy[0] - base_offset[0]
                    site_y = target_xy[1] - base_offset[1]
            else:
                # Fallback for static table placements where sim is missing
                site_x, site_y, site_z = 0.0, 0.0, 0.0
            # -------------------------------

            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(0) + base_offset[0] + site_x
                object_y = self._sample_y(0) + base_offset[1] + site_y
                object_z = self.z_offset + base_offset[2] + site_z
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    quat = self._sample_quat()

                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # ========================================================
                    # CRITICAL FIX: THE LIBERO QUATERNION PATCH
                    # Shift [x, y, z, w] to [w, x, y, z] manually so the 
                    # BDDL loader doesn't accidentally flip the object 180 degrees!
                    # ========================================================
                    if len(obj.joints) > 0:
                        quat = np.array([quat[3], quat[0], quat[1], quat[2]])

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                import pdb
                pdb.set_trace()
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects
