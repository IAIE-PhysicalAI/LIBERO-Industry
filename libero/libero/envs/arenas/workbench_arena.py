import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    string_to_array,
    xml_path_completion,
)
import robosuite.utils.transform_utils as T

from libero.libero.envs.arenas.style import get_texture_filename
import xml.etree.ElementTree as ET


_BASE_AGENTVIEW_ORBIT = {
    "azimuth_deg": 0.0,
    "elevation_deg": 38.922340,
    "distance_m": 1.130651,
}
_CAMERA_FOCAL_OFFSET = np.array([-0.221031, 0.0, 0.0])
_CAMERA_NAME_ALIASES = {
    # Panda owns this camera on the robot model. The Aubo profile owns the
    # equivalent camera on its CTAG gripper, whose prefix differs between
    # Robosuite 1.4 and 1.5.
    "robot0_eye_in_hand": (
        "gripper0_eye_in_hand",
        "gripper0_right_eye_in_hand",
    ),
}
_LEGACY_LIGHTING_COLORS = {
    "neutral": np.array([1.0, 0.81, 0.65]),
    "warm": np.array([1.0, 0.72, 0.50]),
    "cool": np.array([0.82, 0.90, 1.0]),
}


class IndustryWorkbenchArena(Arena):
    """
    Workspace arena for LIBERO Industry tasks.

    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self,
        table_full_size=(0.8, 2, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        xml="arenas/libero_industry_workbench_base_style.xml",
        floor_style="light-gray",
        wall_style="light-gray-plaster",
        ):
        super().__init__(xml_path_completion(xml))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = (
            self.bottom_pos
            + np.array([0, 0, -self.table_half_size[2]])
            + self.table_offset
        )

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        self.configure_location()
        texplane = self.asset.find("./texture[@name='texplane']")
        plane_file = texplane.get("file")
        plane_file = "/".join(
            plane_file.split("/")[:-1]
            + [get_texture_filename(type="floor", style=floor_style)]
        )
        texplane.set("file", plane_file)

        texwall = self.asset.find("./texture[@name='tex-wall']")
        wall_file = texwall.get("file")
        wall_file = "/".join(
            wall_file.split("/")[:-1]
            + [get_texture_filename(type="wall", style=wall_style)]
        )
        texwall.set("file", wall_file)

    def configure_location(self):
        """Configures correct locations for this arena"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))
        # self.table_visual.set("rgba", array_to_string([0, 0, 0, 0]))

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

        # If we're not using legs, set their size to 0
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
        else:
            # Otherwise, set leg locations appropriately
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for leg, dx, dy in zip(self.table_legs_visual, delta_x, delta_y):
                # If x-length of table is less than a certain length, place leg in the middle between ends
                # Otherwise we place it near the edge
                x = 0
                if self.table_half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * self.table_half_size[0] - dx
                # Repeat the same process for y
                y = 0
                if self.table_half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * self.table_half_size[1] - dy
                # Get z value
                z = (self.table_offset[2] - self.table_half_size[2]) / 2.0
                # Set leg position
                leg.set("pos", array_to_string([x, y, -z]))
                # Set leg size
                leg.set("size", array_to_string([0.025, z]))
                # leg.set("rgba", array_to_string([0, 0, 0, 0]))

    @staticmethod
    def _vector(config, key, length=3, default=None, nonnegative=False):
        value = default if key not in config else config[key]
        vector = np.asarray(value, dtype=float)
        if vector.shape != (length,) or not np.all(np.isfinite(vector)):
            raise ValueError("{} must contain {} finite numbers".format(key, length))
        if nonnegative and np.any(vector < 0.0):
            raise ValueError("{} must not contain negative values".format(key))
        return vector

    @staticmethod
    def _scalar(config, key, default):
        value = float(config.get(key, default))
        if not np.isfinite(value):
            raise ValueError("{} must be finite".format(key))
        return value

    @staticmethod
    def _quat_multiply_wxyz(left, right):
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ]
        )

    @staticmethod
    def _rotation_vector_quat_wxyz(rotation_vector_deg):
        rotation_vector = np.deg2rad(rotation_vector_deg)
        angle = np.linalg.norm(rotation_vector)
        if angle == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        half_angle = angle / 2.0
        xyz = rotation_vector * (np.sin(half_angle) / angle)
        return np.concatenate(([np.cos(half_angle)], xyz))

    @staticmethod
    def _find_camera(root_element, camera_name):
        cameras = root_element.findall(".//camera")
        matches = [
            camera
            for camera in cameras
            if camera.get("name") == camera_name
        ]
        aliases = _CAMERA_NAME_ALIASES.get(camera_name, ())
        if not matches and aliases:
            matches = [camera for camera in cameras if camera.get("name") in aliases]
        if len(matches) != 1:
            alias_note = (
                "; accepted aliases: {}".format(", ".join(repr(alias) for alias in aliases))
                if aliases
                else ""
            )
            raise ValueError(
                "Configured camera {!r} matched {} MJCF cameras; expected exactly one{}".format(
                    camera_name, len(matches), alias_note
                )
            )
        return matches[0]

    def _apply_agentview_orbit(self, camera, orbit):
        azimuth = np.radians(
            self._scalar(orbit, "azimuth_deg", _BASE_AGENTVIEW_ORBIT["azimuth_deg"])
        )
        elevation = np.radians(
            self._scalar(
                orbit, "elevation_deg", _BASE_AGENTVIEW_ORBIT["elevation_deg"]
            )
        )
        distance = self._scalar(
            orbit, "distance_m", _BASE_AGENTVIEW_ORBIT["distance_m"]
        )
        if distance <= 0.0:
            raise ValueError("distance_m must be positive")

        focal_point = np.asarray(self.table_offset, dtype=float) + _CAMERA_FOCAL_OFFSET
        camera_position = focal_point + np.array(
            [
                distance * np.cos(elevation) * np.cos(azimuth),
                distance * np.cos(elevation) * np.sin(azimuth),
                distance * np.sin(elevation),
            ]
        )
        forward = focal_point - camera_position
        forward /= np.linalg.norm(forward)
        right = np.cross(np.array([0.0, 0.0, 1.0]), -forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            raise ValueError("agentview elevation cannot place the camera on the world-up axis")
        right /= right_norm
        up = np.cross(-forward, right)
        up /= np.linalg.norm(up)

        rotation_matrix = np.column_stack((right, up, -forward))
        quat_xyzw = T.mat2quat(rotation_matrix)
        quat_wxyz = np.roll(quat_xyzw, 1)

        camera.set("pos", array_to_string(camera_position))
        camera.set("quat", array_to_string(quat_wxyz / np.linalg.norm(quat_wxyz)))
        camera.attrib.pop("target", None)
        camera.set("mode", "fixed")

    def _apply_camera_extrinsic(self, camera, extrinsic):
        position_offset = self._vector(
            extrinsic, "position_m", default=[0.0, 0.0, 0.0]
        )
        rotation_vector_deg = self._vector(
            extrinsic, "rotation_vector_deg", default=[0.0, 0.0, 0.0]
        )
        position = self._vector(
            {"position": camera.get("pos", "0 0 0").split()},
            "position",
        )
        quat = self._vector(
            {"quat": camera.get("quat", "1 0 0 0").split()},
            "quat",
            length=4,
        )
        quat_norm = np.linalg.norm(quat)
        if quat_norm < 1e-12:
            raise ValueError("Camera quaternion must be nonzero")
        quat /= quat_norm

        # Both offsets are expressed in the camera element's parent frame.
        delta_quat = self._rotation_vector_quat_wxyz(rotation_vector_deg)
        perturbed_quat = self._quat_multiply_wxyz(delta_quat, quat)
        perturbed_quat /= np.linalg.norm(perturbed_quat)
        camera.set("pos", array_to_string(position + position_offset))
        camera.set("quat", array_to_string(perturbed_quat))
        for alternative in ("axisangle", "euler", "xyaxes", "zaxis"):
            camera.attrib.pop(alternative, None)

    def _apply_cameras(self, root_element, set_config):
        cameras = set_config.get("cameras")
        if cameras is None and "camera" in set_config:
            legacy = set_config["camera"]
            cameras = {
                "agentview": {
                    "orbit": {
                        "azimuth_deg": legacy.get("azimuth", 0.0),
                        "elevation_deg": legacy.get("elevation", 38.922340),
                        "distance_m": legacy.get("distance", 1.130651),
                    }
                }
            }
        if cameras is None:
            return
        if not isinstance(cameras, dict):
            raise ValueError("set_scene.cameras must be a dictionary keyed by MJCF name")

        for camera_name, camera_config in cameras.items():
            if not isinstance(camera_config, dict):
                raise ValueError("Camera {!r} configuration must be a dictionary".format(camera_name))
            camera = self._find_camera(root_element, camera_name)
            orbit = camera_config.get("orbit")
            if orbit is not None:
                if camera_name != "agentview":
                    raise ValueError("orbit is only supported for the agentview workspace camera")
                self._apply_agentview_orbit(camera, orbit)
            extrinsic = camera_config.get("extrinsic_offset")
            if extrinsic is not None:
                self._apply_camera_extrinsic(camera, extrinsic)

    @staticmethod
    def _cast_shadow_value(lighting):
        value = lighting.get("cast_shadow", lighting.get("castshadow", False))
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower()
        raise ValueError("cast_shadow must be a boolean")

    def _apply_lighting(self, worldbody, lighting):
        if "task_diffuse_rgb" in lighting or "ambient_diffuse_rgb" in lighting:
            task_diffuse = self._vector(
                lighting,
                "task_diffuse_rgb",
                default=[2.5, 2.025, 1.625],
                nonnegative=True,
            )
            ambient_diffuse = self._vector(
                lighting,
                "ambient_diffuse_rgb",
                default=[0.3, 0.243, 0.195],
                nonnegative=True,
            )
        else:
            # Backward-compatible deterministic rendering for hand-authored
            # configs.  Legacy noise_level is deliberately not sampled here.
            profile_name = lighting.get("tone", "neutral")
            if profile_name == "random":
                profile_name = "neutral"
            try:
                color = _LEGACY_LIGHTING_COLORS[profile_name]
            except KeyError:
                raise ValueError("Unknown legacy lighting tone: {!r}".format(profile_name))
            task_diffuse = color * 2.5
            ambient_diffuse = color * 0.3

        cast_shadow = self._cast_shadow_value(lighting)
        lights_setup = {
            "ind_light_task_left": {
                "pos": "0 0.7 2.25",
                "dir": "0 0 -1",
                "directional": "false",
                "diffuse": array_to_string(task_diffuse),
                "specular": "0.2 0.2 0.2",
                "attenuation": "0 0 1",
            },
            "ind_light_task_center": {
                "pos": "0 0 2.25",
                "dir": "0 0 -1",
                "directional": "false",
                "diffuse": array_to_string(task_diffuse),
                "specular": "0.2 0.2 0.2",
                "attenuation": "0 0 1",
            },
            "ind_light_task_right": {
                "pos": "0 -0.7 2.25",
                "dir": "0 0 -1",
                "directional": "false",
                "diffuse": array_to_string(task_diffuse),
                "specular": "0.2 0.2 0.2",
                "attenuation": "0 0 1",
            },
            "ind_light_facility": {
                "pos": "0 0 5.0",
                "dir": "0 0 -1",
                "directional": "true",
                "diffuse": array_to_string(ambient_diffuse),
                "specular": "0 0 0",
            },
        }
        for old_light in worldbody.findall("light"):
            worldbody.remove(old_light)
        for name, attributes in lights_setup.items():
            light = ET.SubElement(worldbody, "light", name=name)
            for key, value in attributes.items():
                light.set(key, value)
            light.set("castshadow", cast_shadow)

    def apply_set_scene(self, root_element, set_config):
        """
        Apply a deterministic scene configuration to the fully merged MJCF.

        Camera offsets are written to MJCF before model compilation.  MuJoCo
        therefore owns them as persistent model extrinsics across forward
        kinematics, rendering, substeps, and environment resets.
        """
        if not set_config:
            return

        worldbody = root_element.find("worldbody")
        if worldbody is None:
            raise ValueError("Merged MJCF is missing worldbody")

        self._apply_cameras(root_element, set_config)

        if "lighting" in set_config:
            self._apply_lighting(worldbody, set_config["lighting"])

        if "table_style" in set_config:
            mat = root_element.find(".//material[@name='table_mat']")
            if mat is not None:
                style = set_config["table_style"]
                colors = {"green": "0.2 0.5 0.2", "blue": "0.1 0.3 0.6", "black": "0.1 0.1 0.1"}
                mat.set("rgba", f"{colors.get(style.get('color', 'green'), '0.8 0.8 0.8')} 1")
                
                if style.get("finish", "matte") == "matte":
                    mat.set("reflectance", "0")
                    mat.set("shininess", "0")
                    mat.set("specular", "0")
                else:
                    mat.set("reflectance", "0.6")
                    mat.set("shininess", "0.8")
                    mat.set("specular", "0.8")
                
                mat.attrib.pop("texture", None)


    @property
    def table_top_abs(self):
        """
        Grabs the absolute position of table top

        Returns:
            np.array: (x,y,z) table position
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset
