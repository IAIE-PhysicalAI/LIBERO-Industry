import os
import sys
import json
import hashlib
import inspect
import random
import subprocess
import xml.etree.ElementTree as ET
import h5py
import argparse
import numpy as np
import mujoco
import shutil
import traceback
import imageio
import types
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# ==============================================================================
# 0. EMERGENCY MOCK PATCHES FOR 1.4.0 (MUST BE AT THE VERY TOP)
# ==============================================================================
import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import os
import sys
import types
import platform

# 1. Fools legacy robomimic 0.2.0 into passing the 'import mujoco_py' statement
dummy_mujoco_py = types.ModuleType('mujoco_py')
dummy_mujoco_py.MjRenderContextOffscreen = type('MjRenderContextOffscreen', (object,), {})
dummy_mujoco_py.BuildError = Exception
dummy_mujoco_py.builder = types.SimpleNamespace(MujocoException=Exception)
sys.modules['mujoco_py'] = dummy_mujoco_py

# 2. Fools legacy robomimic 0.2.0 into finding the removed XML processing function
import robosuite.utils.mjcf_utils
if not hasattr(robosuite.utils.mjcf_utils, 'postprocess_model_xml'):
    robosuite.utils.mjcf_utils.postprocess_model_xml = lambda xml_str: xml_str

# 3. Force native DeepMind MuJoCo headless rendering based on the OS
current_os = platform.system()
if current_os == "Linux":
    os.environ["MUJOCO_GL"] = "egl"
elif current_os == "Windows":
    os.environ["MUJOCO_GL"] = "wgl"
elif current_os == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"

# 4. Fools legacy robomimic into bypassing 'egl_probe' dependency on Windows
if current_os == "Windows":
    dummy_egl_probe = types.ModuleType('egl_probe')
    # Provide mocked methods so robomimic receives expected data types without executing hardware queries
    dummy_egl_probe.get_available_devices = lambda: [0]
    dummy_egl_probe.get_egl_displays = lambda: []
    sys.modules['egl_probe'] = dummy_egl_probe
# ==========================================
# 1. LIBERO & Robosuite Registration
# ==========================================
try:
    from robosuite.environments.base import register_env
    from libero.libero.envs.env_wrapper import TASK_MAPPING
    for env_name, env_class in TASK_MAPPING.items():
        if env_class is not None:
            register_env(env_class)
except ImportError as e:
    raise ImportError(f"Failed to register LIBERO environments: {e}")

import robosuite.environments.base as _rs_base
_original_make = _rs_base.make

def _patched_make(env_name, *args, **kwargs):
    if env_name.startswith("Libero_"):
        bddl_file = kwargs.pop("bddl_file_name", None)
        if bddl_file is not None:
            args = (bddl_file,) + args
    return _original_make(env_name, *args, **kwargs)
_rs_base.make = _patched_make

import robomimic.utils.env_utils as EnvUtils

import robosuite as suite
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.models.grippers import GRIPPER_MAPPING

_ROBOSUITE_V15 = tuple(int(part) for part in suite.__version__.split(".")[:2]) >= (1, 5)

if _ROBOSUITE_V15:
    from robosuite.robots.fixed_base_robot import FixedBaseRobot as _RobotRuntime
    _WRIST_CAMERA = "gripper0_right_eye_in_hand"
else:
    from robosuite.controllers import load_controller_config
    from robosuite.robots.single_arm import SingleArm as _RobotRuntime
    _WRIST_CAMERA = "gripper0_eye_in_hand"

_WRIST_IMAGE_KEY = f"{_WRIST_CAMERA}_image"


# LIBERO domains prepend the mounting prefix themselves, e.g. the tabletop
# domain converts ``AuboI5`` into ``MountedAuboI5``. Keep the base token in
# the metadata and validate the concrete runtime model after construction.
# This prevents a floor trajectory from silently using the mounted initial
# configuration (or vice versa).
_DOMAIN_ROBOTS = {
    "Libero_Industry_Workbench_Manipulation": ("MountedPanda", "MountedAuboI5"),
    "Libero_Floor_Manipulation": ("OnTheGroundPanda", "OnTheGroundAuboI5"),
    "Libero_Tabletop_Manipulation": ("MountedPanda", "MountedAuboI5"),
    "Libero_Kitchen_Tabletop_Manipulation": ("MountedPanda", "MountedAuboI5"),
    "Libero_Study_Tabletop_Manipulation": ("MountedPanda", "MountedAuboI5"),
    "Libero_Living_Room_Tabletop_Manipulation": (
        "OnTheGroundPanda",
        "OnTheGroundAuboI5",
    ),
}

# The copied source state contains Panda qpos, which must never be written to
# the six-DoF Aubo. Object and fixture joint state is copied by name instead.
_SCENE_POSITION_TOLERANCE_M = 1e-6
_SCENE_ROTATION_TOLERANCE_DEG = 1e-3
_TRACKING_FEEDBACK_GAIN = 3.0
_POSITION_ACTION_SCALE_M = 0.05
_ROTATION_ACTION_SCALE_RAD = 0.5
_TARGET_LOOKAHEAD_STEPS = 1
_ACTION_CLIP = (-1.0, 1.0)
_INITIAL_NULL_STEPS = 5
_CONVERSION_RECIPE_ATTR = "aubo_conversion_recipe_v2"
_CONVERSION_RECIPE_VERSION = 4
_INITIALIZATION_ORDER = (
    "source_scene_transfer_then_target_settle_then_final_source_scene_restore"
)
_INITIAL_OBSERVATION_REFRESH = "robomimic_get_observation_without_control_step"
_TARGET_STATE_SCHEMA_ATTR = "target_robot_state_schema"
_TARGET_STATE_SCHEMA_VERSION = 1
_TARGET_ROBOT_TOKEN = "AuboI5"
_TARGET_GRIPPER_TOKEN = "CTAG2F90DGripper"
_TARGET_CAMERA_NAMES = ("agentview", _WRIST_CAMERA)
_TARGET_CAMERA_HEIGHT = 256
_TARGET_CAMERA_WIDTH = 256
_EMBODIMENT_MANIFEST_VERSION = 1
_EMBODIMENT_MANIFEST_ATTR = "aubo_embodiment_manifest_v1"
_EMBODIMENT_SHA256_ATTR = "aubo_embodiment_sha256"
_RUNTIME_CONTRACT_VERSION = 1
_RUNTIME_CONTRACT_ATTR = "aubo_runtime_contract_v1"
_RUNTIME_CONTRACT_SHA256_ATTR = "aubo_runtime_contract_sha256"


def _make_legacy_osc_pose_controller_config():
    """Return the native Robosuite 1.4.x flat OSC_POSE configuration."""
    config = load_controller_config(default_controller="OSC_POSE")
    config.update(
        {
            "input_max": 1,
            "input_min": -1,
            "output_max": [
                _POSITION_ACTION_SCALE_M,
                _POSITION_ACTION_SCALE_M,
                _POSITION_ACTION_SCALE_M,
                _ROTATION_ACTION_SCALE_RAD,
                _ROTATION_ACTION_SCALE_RAD,
                _ROTATION_ACTION_SCALE_RAD,
            ],
            "output_min": [
                -_POSITION_ACTION_SCALE_M,
                -_POSITION_ACTION_SCALE_M,
                -_POSITION_ACTION_SCALE_M,
                -_ROTATION_ACTION_SCALE_RAD,
                -_ROTATION_ACTION_SCALE_RAD,
                -_ROTATION_ACTION_SCALE_RAD,
            ],
        }
    )
    return config


def _make_osc_pose_controller_config():
    """Return the controller_configs payload required by this Robosuite version."""
    if not _ROBOSUITE_V15:
        return _make_legacy_osc_pose_controller_config()

    composite = suite.load_composite_controller_config(controller="BASIC")
    right = composite["body_parts"]["right"]
    right.update(
        {
            "input_max": 1,
            "input_min": -1,
            "output_max": [
                _POSITION_ACTION_SCALE_M,
                _POSITION_ACTION_SCALE_M,
                _POSITION_ACTION_SCALE_M,
                _ROTATION_ACTION_SCALE_RAD,
                _ROTATION_ACTION_SCALE_RAD,
                _ROTATION_ACTION_SCALE_RAD,
            ],
            "output_min": [
                -_POSITION_ACTION_SCALE_M,
                -_POSITION_ACTION_SCALE_M,
                -_POSITION_ACTION_SCALE_M,
                -_ROTATION_ACTION_SCALE_RAD,
                -_ROTATION_ACTION_SCALE_RAD,
                -_ROTATION_ACTION_SCALE_RAD,
            ],
            "input_ref_frame": "world",
        }
    )
    right["gripper"] = {"type": "JOINT_POSITION"}
    return composite


def _set_controller_metadata(env_meta, *, preserve_legacy_config=False):
    env_kwargs = env_meta.setdefault("env_kwargs", {})
    # Neither LIBERO's 1.4 nor 1.5 BDDL constructor accepts `controller`.
    env_kwargs.pop("controller", None)

    # Published LIBERO 1.4 datasets already contain a complete, native
    # controller_configs dictionary for the Panda reference environment.
    if (
        not _ROBOSUITE_V15
        and preserve_legacy_config
        and isinstance(env_kwargs.get("controller_configs"), dict)
    ):
        return

    # The key is stable across versions; only its payload format differs.
    env_kwargs["controller_configs"] = _make_osc_pose_controller_config()


def _set_target_observation_metadata(env_meta):
    """Describe the exact target-native camera contract stored in the output."""
    env_kwargs = env_meta.setdefault("env_kwargs", {})
    env_kwargs.update(
        {
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
            "camera_names": list(_TARGET_CAMERA_NAMES),
            "camera_heights": _TARGET_CAMERA_HEIGHT,
            "camera_widths": _TARGET_CAMERA_WIDTH,
            "camera_depths": False,
            "camera_segmentations": None,
            "reward_shaping": False,
        }
    )

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from libero.libero.envs.robots.mounted_aubo_i5 import MountedAuboI5
    from libero.libero.envs.robots.on_the_ground_aubo_i5 import OnTheGroundAuboI5
    from libero.libero.envs.robots import MountedPanda, OnTheGroundPanda
    from libero.libero.envs.robots.grippers import CTAG2F90DGripper
    from libero.libero import get_libero_path 
    
    REGISTERED_ROBOTS.update({
        "MountedPanda": MountedPanda,
        "OnTheGroundPanda": OnTheGroundPanda,
        "MountedAuboI5": MountedAuboI5,
        "OnTheGroundAuboI5": OnTheGroundAuboI5,
    })
    
    ROBOT_CLASS_MAPPING.update({
        "MountedPanda": _RobotRuntime,
        "OnTheGroundPanda": _RobotRuntime,
        "MountedAuboI5": _RobotRuntime,
        "OnTheGroundAuboI5": _RobotRuntime,
    })
    
    GRIPPER_MAPPING["CTAG2F90DGripper"] = CTAG2F90DGripper
except ImportError as e:
    print(f"Warning: Could not import custom robots. Ensure your PYTHONPATH is correct. {e}")

# ==========================================
# 2. Math & Kinematics Utilities
# ==========================================
def _get_base_env(env):
    """Return the wrapped Robosuite environment across Robomimic versions."""
    base_env = getattr(env, "base_env", None)
    if base_env is None:
        base_env = getattr(env, "env", None)
    if base_env is None:
        raise AttributeError(
            "Expected a Robomimic environment exposing 'base_env' or 'env'; "
            f"got {type(env).__name__}."
        )
    return base_env


def _action_dimension(env):
    """Return the action dimension from either a Robomimic wrapper or raw env."""
    for attribute in ("action_dimension", "action_dim"):
        value = getattr(env, attribute, None)
        if value is not None:
            return int(value)

    base_env = _get_base_env(env)
    value = getattr(base_env, "action_dim", None)
    if value is not None:
        return int(value)

    raise AttributeError(
        "Could not determine the action dimension from "
        f"{type(env).__name__} or its wrapped Robosuite environment."
    )


def _flatten_index_container(indexes):
    """Normalize Robosuite 1.4 single-arm and 1.5 fixed-base index layouts."""
    if isinstance(indexes, dict):
        flattened = []
        for value in indexes.values():
            flattened.extend(_flatten_index_container(value))
        return flattened
    if isinstance(indexes, slice):
        raise TypeError("Slice-based joint indexes are not supported for state capture.")
    if np.isscalar(indexes):
        return [int(indexes)]
    flattened = []
    for value in indexes:
        flattened.extend(_flatten_index_container(value))
    return flattened


def get_target_robot_qpos(env):
    """Return target arm and gripper qpos from the live Aubo simulation."""
    base_env = _get_base_env(env)
    robots = getattr(base_env, "robots", None)
    if robots is None or len(robots) != 1:
        raise RuntimeError("Expected exactly one target robot for Aubo state capture.")

    robot = robots[0]
    arm_positions = np.asarray(robot._joint_positions, dtype=np.float64).reshape(-1).copy()
    if arm_positions.shape != (_TARGET_ARM_STATE_DIMENSION,):
        raise RuntimeError(
            f"Expected {_TARGET_ARM_STATE_DIMENSION} Aubo joint positions, "
            f"got {arm_positions.shape}."
        )

    gripper_indexes = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if gripper_indexes is None:
        raise RuntimeError("Target robot does not expose gripper qpos indexes.")
    gripper_indexes = _flatten_index_container(gripper_indexes)
    if not gripper_indexes:
        raise RuntimeError("Target gripper has no qpos indexes.")
    gripper_positions = np.asarray(
        base_env.sim.data.qpos[gripper_indexes], dtype=np.float64
    ).reshape(-1).copy()

    if not np.all(np.isfinite(arm_positions)) or not np.all(np.isfinite(gripper_positions)):
        raise RuntimeError("Target robot state contains non-finite values.")
    return arm_positions, gripper_positions


def get_target_sim_state(env):
    """Return the flattened target MuJoCo state using the standard wrapper."""
    state = _get_base_env(env).sim.get_state()
    flattened = np.asarray(state.flatten(), dtype=np.float64).copy()
    if flattened.ndim != 1 or not np.all(np.isfinite(flattened)):
        raise RuntimeError("Target simulator returned an invalid flattened state.")
    return flattened


def _task_success(env):
    """Return the LIBERO task predicate across Robomimic success API variants."""
    result = env.is_success()
    if isinstance(result, dict):
        if "task" not in result:
            raise KeyError(
                "Robomimic success result is missing the 'task' predicate: "
                f"{sorted(result)}."
            )
        return bool(result["task"])
    return bool(result)


def _right_eef_site_id(robot):
    """Return the controller's right-arm TCP site on both supported versions."""
    site_id = robot.eef_site_id
    return site_id["right"] if isinstance(site_id, dict) else site_id


def get_eef_pose(env):
    """Extract the active gripper pinch / TCP pose, never a mesh or fingertip."""
    base_env = _get_base_env(env)
    robot = base_env.robots[0]
    site_id = _right_eef_site_id(robot)
    site_name = base_env.sim.model.site_id2name(site_id)
    if site_name is None or not site_name.endswith("grip_site"):
        raise RuntimeError(
            "Expected the active controller TCP to resolve to a gripper pinch site, "
            f"got {site_name!r}."
        )
    pos = np.array(base_env.sim.data.site_xpos[site_id])
    mat = np.array(base_env.sim.data.site_xmat[site_id].reshape([3, 3]))
    return pos, mat


def _domain_robot_models(env_name):
    """Return expected Panda and Aubo runtime model names for a supported BDDL domain."""
    try:
        return _DOMAIN_ROBOTS[env_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_DOMAIN_ROBOTS))
        raise ValueError(
            f"Unsupported LIBERO environment {env_name!r}; supported environments: {supported}"
        ) from exc


def _assert_runtime_robot(env, expected_model, role):
    """Fail early if the BDDL selected a robot model other than the aligned one."""
    base_env = _get_base_env(env)
    actual_model = type(base_env.robots[0].robot_model).__name__
    if actual_model != expected_model:
        raise RuntimeError(
            f"{role} robot routing mismatch: expected {expected_model}, got {actual_model}."
        )


def _runtime_right_gripper(robot):
    """Return the explicit right-arm gripper across supported Robosuite versions."""
    gripper = getattr(robot, "gripper", None)
    if isinstance(gripper, dict):
        if "right" not in gripper:
            raise RuntimeError(
                f"Runtime gripper mapping has no 'right' entry: {sorted(gripper)}."
            )
        gripper = gripper["right"]
    if gripper is None:
        raise RuntimeError("Runtime robot does not expose a right-arm gripper.")
    return gripper


def _assert_target_runtime_contract(env):
    """Fail unless the requested embodiment is active, then describe it."""
    base_env = _get_base_env(env)
    if len(base_env.robots) != 1:
        raise RuntimeError(
            f"Expected one target robot, found {len(base_env.robots)}."
        )
    robot = base_env.robots[0]
    gripper = _runtime_right_gripper(robot)
    actual_gripper = type(gripper).__name__
    if actual_gripper != _TARGET_GRIPPER_TOKEN:
        raise RuntimeError(
            "Aubo target gripper routing mismatch: expected "
            f"{_TARGET_GRIPPER_TOKEN}, got {actual_gripper}."
        )

    site_id = _right_eef_site_id(robot)
    site_name = base_env.sim.model.site_id2name(site_id)
    if site_name is None or not site_name.endswith("grip_site"):
        raise RuntimeError(
            "Aubo target TCP must resolve to the explicit CTAG grip_site, "
            f"got {site_name!r}."
        )

    camera_ids = {}
    for camera_name in _TARGET_CAMERA_NAMES:
        try:
            camera_ids[camera_name] = base_env.sim.model.camera_name2id(camera_name)
        except ValueError as error:
            raise RuntimeError(
                f"Aubo target camera {camera_name!r} is not present in the runtime model."
            ) from error

    robot_model = robot.robot_model
    wrist_camera_id = camera_ids[_WRIST_CAMERA]
    return {
        "version": _RUNTIME_CONTRACT_VERSION,
        "robot_model": {
            "module": type(robot_model).__module__,
            "class": type(robot_model).__name__,
        },
        "gripper_model": {
            "module": type(gripper).__module__,
            "class": actual_gripper,
        },
        "action_dimension": _action_dimension(env),
        "tcp": {
            "site_name": site_name,
            "local_position_m": np.asarray(
                base_env.sim.model.site_pos[site_id], dtype=np.float64
            ).tolist(),
            "local_quaternion_wxyz": np.asarray(
                base_env.sim.model.site_quat[site_id], dtype=np.float64
            ).tolist(),
        },
        "wrist_camera": {
            "name": _WRIST_CAMERA,
            "local_position_m": np.asarray(
                base_env.sim.model.cam_pos[wrist_camera_id], dtype=np.float64
            ).tolist(),
            "local_quaternion_wxyz": np.asarray(
                base_env.sim.model.cam_quat[wrist_camera_id], dtype=np.float64
            ).tolist(),
            "fovy_deg": float(base_env.sim.model.cam_fovy[wrist_camera_id]),
        },
        "observation": {
            "camera_names": list(_TARGET_CAMERA_NAMES),
            "image_keys": ["agentview_image", _WRIST_IMAGE_KEY],
            "height": _TARGET_CAMERA_HEIGHT,
            "width": _TARGET_CAMERA_WIDTH,
            "dtype": "uint8",
        },
    }


def _assert_target_observation_contract(observation):
    """Validate image names, shapes, and dtypes before writing target data."""
    expected_shape = (_TARGET_CAMERA_HEIGHT, _TARGET_CAMERA_WIDTH, 3)
    for observation_key in ("agentview_image", _WRIST_IMAGE_KEY):
        if observation_key not in observation:
            raise KeyError(
                f"Target observation is missing required image {observation_key!r}."
            )
        image = np.asarray(observation[observation_key])
        if image.shape != expected_shape:
            raise RuntimeError(
                f"Target observation {observation_key!r} has shape {image.shape}; "
                f"expected {expected_shape}."
            )
        if image.dtype != np.uint8:
            raise TypeError(
                f"Target observation {observation_key!r} has dtype {image.dtype}; "
                "expected uint8 RGB."
            )


def _task_joint_names(base_env):
    """Return named non-robot joints belonging to task objects and fixtures."""
    task_joints = {}
    for collection_name in ("objects_dict", "fixtures_dict"):
        for task_name, task_model in getattr(base_env, collection_name, {}).items():
            for joint_name in getattr(task_model, "joints", ()) or ():
                if not isinstance(joint_name, str) or not joint_name:
                    continue
                previous_owner = task_joints.setdefault(joint_name, task_name)
                if previous_owner != task_name:
                    raise RuntimeError(
                        f"Task joint {joint_name!r} belongs to both {previous_owner!r} and "
                        f"{task_name!r}."
                    )
    return tuple(sorted(task_joints))


def _joint_state_ranges(sim, joint_name):
    """Return qpos and qvel slices for one named MuJoCo joint without version APIs."""
    joint_id = sim.model.joint_name2id(joint_name)

    def state_range(addresses, state_size, start):
        later_addresses = addresses[addresses > start]
        end = int(later_addresses.min()) if later_addresses.size else state_size
        return slice(int(start), end)

    qpos_start = int(sim.model.jnt_qposadr[joint_id])
    qvel_start = int(sim.model.jnt_dofadr[joint_id])
    return (
        state_range(np.asarray(sim.model.jnt_qposadr), sim.model.nq, qpos_start),
        state_range(np.asarray(sim.model.jnt_dofadr), sim.model.nv, qvel_start),
    )


def _copy_task_scene_state(source_env, target_env):
    """Copy task state by name while preserving the target Aubo arm and gripper."""
    source_base = _get_base_env(source_env)
    target_base = _get_base_env(target_env)
    source_joints = _task_joint_names(source_base)
    target_joints = _task_joint_names(target_base)
    if source_joints != target_joints:
        raise RuntimeError(
            "Source and target task joint sets differ; refusing to copy a positional "
            f"state by index. source={source_joints}, target={target_joints}"
        )

    for joint_name in source_joints:
        source_qpos, source_qvel = _joint_state_ranges(source_base.sim, joint_name)
        target_qpos, target_qvel = _joint_state_ranges(target_base.sim, joint_name)
        if (
            (source_qpos.stop - source_qpos.start) != (target_qpos.stop - target_qpos.start)
            or (source_qvel.stop - source_qvel.start) != (target_qvel.stop - target_qvel.start)
        ):
            raise RuntimeError(
                f"Task joint {joint_name!r} has incompatible source / target state dimensions."
            )
        target_base.sim.data.qpos[target_qpos] = source_base.sim.data.qpos[source_qpos]
        target_base.sim.data.qvel[target_qvel] = source_base.sim.data.qvel[source_qvel]

    # Fixed BDDL fixtures have no qpos. Their root-body transforms live in the
    # MuJoCo model, so copy them by semantic task name rather than model index.
    source_body_ids = getattr(source_base, "obj_body_id", {})
    target_body_ids = getattr(target_base, "obj_body_id", {})
    if set(source_body_ids) != set(target_body_ids):
        raise RuntimeError(
            "Source and target task body sets differ; cannot synchronize fixed fixtures."
        )
    for body_name, source_id in source_body_ids.items():
        target_id = target_body_ids[body_name]
        target_base.sim.model.body_pos[target_id] = source_base.sim.model.body_pos[source_id]
        target_base.sim.model.body_quat[target_id] = source_base.sim.model.body_quat[source_id]

    # Preserve this helper's established standalone contract for diagnostic and
    # rendering callers. This forward call updates derived body poses but does
    # not advance simulation time.
    target_base.sim.forward()



def _restore_source_simulation_time(source_env, target_env):
    """Restore recorded source time and forward without advancing target physics."""
    source_time = float(_get_base_env(source_env).sim.data.time)
    if not np.isfinite(source_time):
        raise RuntimeError(f"Source simulator time is not finite: {source_time!r}.")

    target_base = _get_base_env(target_env)
    target_base.sim.data.time = source_time
    target_base.sim.forward()
    restored_time = float(target_base.sim.data.time)
    if restored_time != source_time:
        raise RuntimeError(
            "Target simulator time changed while restoring the source scene: "
            f"source={source_time:.17g}, target={restored_time:.17g}."
        )
    return source_time


def _refresh_target_observation(target_env):
    """Refresh target observables after state transfer without a control step."""
    base_env = _get_base_env(target_env)
    post_process = getattr(base_env, "_post_process", None)
    if not callable(post_process):
        raise AttributeError(
            f"{type(base_env).__name__} does not expose the supported _post_process API."
        )
    get_observation = getattr(target_env, "get_observation", None)
    if not callable(get_observation):
        raise AttributeError(
            f"{type(target_env).__name__} does not expose the supported get_observation API."
        )

    state_before = get_target_sim_state(target_env)
    post_process()
    observation = get_observation()
    state_after = get_target_sim_state(target_env)
    if not np.array_equal(state_after, state_before):
        maximum_difference = float(np.max(np.abs(state_after - state_before)))
        raise RuntimeError(
            "Refreshing the initial target observation changed simulator state by "
            f"{maximum_difference:.3e}; a no-step refresh is required."
        )
    return observation


def _assert_task_scene_alignment(source_env, target_env):
    """Verify that every named task body begins in the source recorded pose."""
    source_base = _get_base_env(source_env)
    target_base = _get_base_env(target_env)
    source_body_ids = getattr(source_base, "obj_body_id", {})
    target_body_ids = getattr(target_base, "obj_body_id", {})
    if set(source_body_ids) != set(target_body_ids):
        raise RuntimeError(
            "Source and target task body sets differ; cannot validate scene alignment."
        )

    for body_name, source_id in source_body_ids.items():
        target_id = target_body_ids[body_name]
        position_error = np.linalg.norm(
            source_base.sim.data.body_xpos[source_id] - target_base.sim.data.body_xpos[target_id]
        )
        source_quat = source_base.sim.data.body_xquat[source_id]
        target_quat = target_base.sim.data.body_xquat[target_id]
        rotation_error_deg = np.degrees(
            2.0 * np.arccos(np.clip(abs(np.dot(source_quat, target_quat)), -1.0, 1.0))
        )
        if (
            position_error > _SCENE_POSITION_TOLERANCE_M
            or rotation_error_deg > _SCENE_ROTATION_TOLERANCE_DEG
        ):
            raise RuntimeError(
                f"Initial task body {body_name!r} is not synchronized: "
                f"{position_error * 1000:.3f} mm, {rotation_error_deg:.4f} deg."
            )


def seed_replay_environment(environment, seed):
    """Seed every supported RNG before a target-environment reset."""
    if seed < 0:
        raise ValueError("Replay seed must be non-negative.")
    random.seed(seed)
    np.random.seed(seed)
    base_env = _get_base_env(environment)
    seed_environment = getattr(base_env, "seed", None)
    if callable(seed_environment):
        seed_environment(seed)
    elif isinstance(getattr(base_env, "rng", None), np.random.Generator):
        base_env.seed = seed
        base_env.rng = np.random.default_rng(seed)
    else:
        raise AttributeError(
            f"{type(base_env).__name__} exposes neither seed(seed) nor the "
            "Robosuite 1.5 rng state."
        )


def initialize_target_replay(
    reference_env,
    target_env,
    source_initial_state,
    initial_gripper_action,
    action_dimension,
    *,
    seed,
    initial_null_steps=_INITIAL_NULL_STEPS,
):
    """Create the deterministic Panda-scene / Aubo-robot initial state."""
    if initial_null_steps < 0:
        raise ValueError("initial_null_steps must be non-negative.")
    reference_env.reset_to({"states": source_initial_state})
    seed_replay_environment(target_env, seed)
    target_env.reset()

    # Settle against the intended recorded scene, not the target reset's
    # independently randomized object layout. This first copy is provisional:
    # the recorded scene is restored again after the target-only settle steps.
    _copy_task_scene_state(reference_env, target_env)
    _restore_source_simulation_time(reference_env, target_env)
    _assert_task_scene_alignment(reference_env, target_env)

    null_action = np.zeros(action_dimension, dtype=np.float64)
    null_action[-1] = float(initial_gripper_action)
    for _ in range(initial_null_steps):
        target_env.step(null_action)

    # Restore the recorded task scene after the target robot settles. This
    # prevents movable objects from retaining five Aubo-only control intervals
    # that do not exist in the Panda demonstration.
    _copy_task_scene_state(reference_env, target_env)
    _restore_source_simulation_time(reference_env, target_env)
    _assert_task_scene_alignment(reference_env, target_env)
    observation = _refresh_target_observation(target_env)
    _assert_task_scene_alignment(reference_env, target_env)
    return observation

def get_orientation_error(target_mat, current_mat):
    r_err_mat = target_mat @ current_mat.T
    return R.from_matrix(r_err_mat).as_rotvec()

def calculate_tracking_action(
    pos_current,
    mat_current,
    pos_target,
    mat_target,
    pos_limit=_POSITION_ACTION_SCALE_M,
    ori_limit=_ROTATION_ACTION_SCALE_RAD,
    feedback_gain=_TRACKING_FEEDBACK_GAIN,
    *,
    return_raw=False,
):
    delta_pos = pos_target - pos_current
    raw_action_pos = (delta_pos / pos_limit) * feedback_gain

    delta_ori = get_orientation_error(mat_target, mat_current)
    raw_action_ori = (delta_ori / ori_limit) * feedback_gain

    action_pos = np.clip(raw_action_pos, *_ACTION_CLIP)
    action_ori = np.clip(raw_action_ori, *_ACTION_CLIP)
    if return_raw:
        return action_pos, action_ori, raw_action_pos, raw_action_ori
    return action_pos, action_ori


def _max_true_run(values):
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest

# ==========================================
# 3. Core Converter Logic (1.4.0 Native)
# ==========================================
_RESUME_PROGRESS_ATTR = "aubo_conversion_progress_v2"
_RESUME_PROGRESS_VERSION = 2
_TARGET_ACTION_DIMENSION = 7
_TARGET_ARM_STATE_DIMENSION = 6
_ARM_JOINT_OBS_KEY = "robot0_joint_pos"
_GRIPPER_QPOS_OBS_KEY = "robot0_gripper_qpos"
_EEF_POSITION_OBS_KEY = "robot0_eef_pos"
_EEF_QUATERNION_OBS_KEY = "robot0_eef_quat"


def _decode_hdf5_text(value):
    """Normalize HDF5 string attributes across h5py / Python versions."""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _close_env(env):
    """Close either a Robomimic wrapper or its underlying Robosuite environment."""
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        close()
        return
    base_env = getattr(env, "env", None)
    close = getattr(base_env, "close", None)
    if callable(close):
        close()


def _sorted_demo_names(data_group):
    """Return standard LIBERO demo names in numeric order."""
    demos = list(data_group.keys())
    try:
        return sorted(demos, key=lambda name: int(name[5:]))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Expected HDF5 demonstration names in the form 'demo_<integer>', got "
            f"{demos}."
        ) from error


def _update_hash_with_array(digest, name, values):
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))


def _git_revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not record the converter Git revision.") from error
    revision = result.stdout.strip()
    if not revision:
        raise RuntimeError("Git returned an empty converter revision.")
    return revision


def _json_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _repo_relative_path(path, repository_root):
    resolved_path = Path(path).resolve()
    resolved_root = Path(repository_root).resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"Embodiment artifact is outside the repository: {resolved_path}"
        ) from error
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Embodiment artifact is missing: {resolved_path}")
    return relative_path.as_posix()


def _resolve_xml_file_reference(xml_path, file_value, directory_hint=None):
    """Resolve one XML file reference without recording machine-specific paths."""
    xml_path = Path(xml_path).resolve()
    reference = Path(file_value)
    candidates = []
    if reference.is_absolute():
        candidates.append(reference)
    else:
        candidates.append(xml_path.parent / reference)
        if directory_hint:
            hinted_directory = Path(directory_hint)
            if hinted_directory.is_absolute():
                candidates.append(hinted_directory / reference)
            else:
                candidates.append(xml_path.parent / hinted_directory / reference)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve XML file reference {file_value!r} from {xml_path}."
    )


def _xml_dependency_files(xml_path):
    """Return every XML include and asset file reachable from one model XML."""
    pending = [Path(xml_path).resolve()]
    visited_xml = set()
    dependencies = []
    while pending:
        current_xml = pending.pop()
        if current_xml in visited_xml:
            continue
        if not current_xml.is_file():
            raise FileNotFoundError(f"Embodiment XML is missing: {current_xml}")
        visited_xml.add(current_xml)
        try:
            root = ET.parse(current_xml).getroot()
        except ET.ParseError as error:
            raise ValueError(f"Invalid embodiment XML {current_xml}: {error}") from error

        compiler = root.find("compiler")
        compiler_attributes = compiler.attrib if compiler is not None else {}
        directory_keys = {
            "mesh": "meshdir",
            "texture": "texturedir",
            "hfield": "assetdir",
            "skin": "assetdir",
        }
        for element in root.findall(".//*[@file]"):
            file_value = element.get("file")
            directory_hint = compiler_attributes.get(
                directory_keys.get(element.tag, "assetdir")
            )
            dependency = _resolve_xml_file_reference(
                current_xml, file_value, directory_hint
            )
            role = "included_xml" if element.tag == "include" else f"{element.tag}_asset"
            dependencies.append((role, dependency))
            if element.tag == "include":
                pending.append(dependency)
    return dependencies


def _xml_named_element_contract(xml_path, tag, name):
    """Return a readable pose/calibration contract for one named XML element."""
    root = ET.parse(xml_path).getroot()
    matches = []

    def visit(element, body_chain):
        next_chain = body_chain
        if element.tag == "body":
            next_chain = body_chain + [
                {
                    "name": element.get("name"),
                    "position_m": [
                        float(value) for value in element.get("pos", "0 0 0").split()
                    ],
                    "quaternion_wxyz": [
                        float(value)
                        for value in element.get("quat", "1 0 0 0").split()
                    ],
                }
            ]
        if element.tag == tag and element.get("name") == name:
            contract = {
                "name": name,
                "body_chain": next_chain,
                "position_m": [
                    float(value) for value in element.get("pos", "0 0 0").split()
                ],
                "quaternion_wxyz": [
                    float(value)
                    for value in element.get("quat", "1 0 0 0").split()
                ],
            }
            if tag == "camera":
                contract["mode"] = element.get("mode", "fixed")
                contract["fovy_deg"] = float(element.get("fovy", "45"))
            matches.append(contract)
        for child in element:
            visit(child, next_chain)

    visit(root, [])
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {tag} named {name!r} in {xml_path}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _build_embodiment_manifest_from_files(
    expected_aubo_model,
    robot_wrapper_files,
    gripper_wrapper_files,
    robot_xml_path,
    gripper_xml_path,
    controller_config,
    repository_root,
):
    """Build a path-portable manifest of every target-embodiment input."""
    repository_root = Path(repository_root).resolve()
    robot_xml_path = Path(robot_xml_path).resolve()
    gripper_xml_path = Path(gripper_xml_path).resolve()
    artifact_roles = {}

    def add_artifact(role, path):
        relative_path = _repo_relative_path(path, repository_root)
        artifact_roles.setdefault(relative_path, set()).add(role)

    for class_name, wrapper_path in robot_wrapper_files:
        add_artifact(f"robot_wrapper:{class_name}", wrapper_path)
    for class_name, wrapper_path in gripper_wrapper_files:
        add_artifact(f"gripper_wrapper:{class_name}", wrapper_path)

    for model_role, xml_path in (
        ("robot", robot_xml_path),
        ("gripper", gripper_xml_path),
    ):
        add_artifact(f"{model_role}_xml", xml_path)
        for dependency_role, dependency_path in _xml_dependency_files(xml_path):
            add_artifact(f"{model_role}_{dependency_role}", dependency_path)

    artifacts = []
    for relative_path in sorted(artifact_roles):
        artifact_path = repository_root / Path(relative_path)
        artifacts.append(
            {
                "path": relative_path,
                "roles": sorted(artifact_roles[relative_path]),
                "size_bytes": artifact_path.stat().st_size,
                "sha256": _file_sha256(artifact_path),
            }
        )

    robot_wrapper_classes = sorted(class_name for class_name, _ in robot_wrapper_files)
    gripper_wrapper_classes = sorted(
        class_name for class_name, _ in gripper_wrapper_files
    )
    manifest = {
        "version": _EMBODIMENT_MANIFEST_VERSION,
        "selection": {
            "robot_token": _TARGET_ROBOT_TOKEN,
            "expected_runtime_robot_model": expected_aubo_model,
            "robot_wrapper_classes": robot_wrapper_classes,
            "gripper_token": _TARGET_GRIPPER_TOKEN,
            "gripper_wrapper_classes": gripper_wrapper_classes,
            "action_dimension": _TARGET_ACTION_DIMENSION,
        },
        "artifacts": artifacts,
        "controller": {
            "config": controller_config,
            "config_sha256": _json_sha256(controller_config),
        },
        "tcp": {
            "runtime_site_suffix": "grip_site",
            "definition": _xml_named_element_contract(
                gripper_xml_path, "site", "grip_site"
            ),
        },
        "cameras": {
            "runtime_names": list(_TARGET_CAMERA_NAMES),
            "image_keys": ["agentview_image", _WRIST_IMAGE_KEY],
            "height": _TARGET_CAMERA_HEIGHT,
            "width": _TARGET_CAMERA_WIDTH,
            "dtype": "uint8",
            "wrist_definition": _xml_named_element_contract(
                gripper_xml_path, "camera", "eye_in_hand"
            ),
        },
        "runtime": {
            "robosuite_version": suite.__version__,
            "mujoco_version": mujoco.__version__,
        },
    }
    manifest[_EMBODIMENT_SHA256_ATTR] = _json_sha256(manifest)
    return manifest


def _build_embodiment_manifest(expected_aubo_model, controller_config):
    """Resolve the concrete Aubo/CTAG files selected by the runtime wrappers."""
    robot_class = {
        "MountedAuboI5": MountedAuboI5,
        "OnTheGroundAuboI5": OnTheGroundAuboI5,
    }.get(expected_aubo_model)
    if robot_class is None:
        raise ValueError(
            f"No embodiment provenance definition for {expected_aubo_model!r}."
        )

    # Both wrappers form the supported Aubo embodiment family. Hash both so a
    # single output root cannot silently combine files produced before and
    # after either mounted or floor routing changes.
    robot_classes = (MountedAuboI5, OnTheGroundAuboI5)
    robot_model = robot_class(idn=0)
    gripper_model = CTAG2F90DGripper(idn=0)

    def wrapper_files(classes):
        files = []
        for model_class in classes:
            source_path = inspect.getsourcefile(model_class)
            if source_path is None:
                raise RuntimeError(
                    f"Could not resolve source file for {model_class.__module__}."
                    f"{model_class.__qualname__}."
                )
            class_name = f"{model_class.__module__}.{model_class.__qualname__}"
            files.append((class_name, Path(source_path)))
        return files

    return _build_embodiment_manifest_from_files(
        expected_aubo_model=expected_aubo_model,
        robot_wrapper_files=wrapper_files(robot_classes),
        gripper_wrapper_files=wrapper_files((CTAG2F90DGripper,)),
        robot_xml_path=robot_model.file,
        gripper_xml_path=gripper_model.file,
        controller_config=controller_config,
        repository_root=_REPOSITORY_ROOT,
    )


def _read_runtime_contract(data_group, required):
    raw_contract = data_group.attrs.get(_RUNTIME_CONTRACT_ATTR)
    raw_sha256 = data_group.attrs.get(_RUNTIME_CONTRACT_SHA256_ATTR)
    if raw_contract is None or raw_sha256 is None:
        if required:
            raise RuntimeError(
                "Target output is missing its resolved runtime embodiment contract."
            )
        return None
    contract = json.loads(_decode_hdf5_text(raw_contract))
    if (
        not isinstance(contract, dict)
        or contract.get("version") != _RUNTIME_CONTRACT_VERSION
    ):
        raise RuntimeError("Target runtime embodiment contract schema is invalid.")
    declared_sha256 = _decode_hdf5_text(raw_sha256)
    calculated_sha256 = _json_sha256(contract)
    if declared_sha256 != calculated_sha256:
        raise RuntimeError(
            "Target runtime embodiment contract SHA-256 is internally inconsistent."
        )
    return contract


def _bind_runtime_contract(data_group, runtime_contract, require_existing):
    existing_contract = _read_runtime_contract(
        data_group, required=require_existing
    )
    if existing_contract is not None and existing_contract != runtime_contract:
        raise RuntimeError(
            "Cannot resume output created with a different resolved runtime "
            "embodiment contract. Use --restart and a new output path."
        )
    serialized_contract = json.dumps(runtime_contract, sort_keys=True)
    data_group.attrs[_RUNTIME_CONTRACT_ATTR] = serialized_contract
    data_group.attrs[_RUNTIME_CONTRACT_SHA256_ATTR] = _json_sha256(runtime_contract)


def _build_conversion_recipe(manifest, seed, action_source_manifest=None):
    source_env_args = json.loads(manifest["env_args_str"] or "{}")
    control_frequency = source_env_args.get("env_kwargs", {}).get("control_freq")
    if control_frequency is None:
        raise KeyError("Source env_args is missing env_kwargs.control_freq.")
    controller_config = _make_osc_pose_controller_config()
    _, expected_aubo_model = _domain_robot_models(source_env_args.get("env_name"))
    embodiment_manifest = _build_embodiment_manifest(
        expected_aubo_model, controller_config
    )
    recipe = {
        "version": _CONVERSION_RECIPE_VERSION,
        "seed": int(seed),
        "initial_null_steps": _INITIAL_NULL_STEPS,
        "feedback_gain": _TRACKING_FEEDBACK_GAIN,
        "position_action_scale_m": _POSITION_ACTION_SCALE_M,
        "rotation_action_scale_rad": _ROTATION_ACTION_SCALE_RAD,
        "target_lookahead": _TARGET_LOOKAHEAD_STEPS,
        "action_clip": list(_ACTION_CLIP),
        "control_frequency_hz": int(control_frequency),
        "robosuite_version": suite.__version__,
        "mujoco_version": mujoco.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
        "git_commit": _git_revision(),
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hdf5": manifest["input_hdf5"],
        "source_size_bytes": manifest["input_size_bytes"],
        "source_signature": manifest["signature"],
        "source_selection_sha256": manifest["content_sha256"],
        "source_demos": manifest["demo_hashes"],
        "controller_config_sha256": _json_sha256(controller_config),
        _EMBODIMENT_SHA256_ATTR: embodiment_manifest[_EMBODIMENT_SHA256_ATTR],
        _EMBODIMENT_MANIFEST_ATTR: embodiment_manifest,
        "target_state_schema_version": _TARGET_STATE_SCHEMA_VERSION,
        "initialization": {
            "source_state": "selected_demo/states[0]",
            "order": _INITIALIZATION_ORDER,
            "scene_transfer": (
                "named_non_robot_joint_qpos_qvel_and_fixed_body_transforms"
            ),
            "scene_transfer_passes": 2,
            "scene_transfer_phases": ["before_target_settle", "after_target_settle"],
            "initial_gripper_action": "selected_action_sequence[0,-1]",
            "initial_null_steps": _INITIAL_NULL_STEPS,
            "restore_source_simulation_time": True,
            "observation_refresh": _INITIAL_OBSERVATION_REFRESH,
            "observation_refresh_control_steps": 0,
            "observation_refresh_preserves_simulator_state": True,
            "scene_position_tolerance_m": _SCENE_POSITION_TOLERANCE_M,
            "scene_rotation_tolerance_deg": _SCENE_ROTATION_TOLERANCE_DEG,
        },
        "action_mode": (
            "stored_action_replay"
            if action_source_manifest is not None
            else "panda_pose_tracking"
        ),
    }
    if action_source_manifest is not None:
        recipe["action_source"] = action_source_manifest
    return recipe


def _source_manifest(input_path, n_trajs, selected_demos=None):
    """Describe the selected source demos and bind a checkpoint to this input file."""
    source_path = Path(input_path).resolve()
    source_stat = source_path.stat()
    with h5py.File(source_path, "r") as source_file:
        if "data" not in source_file:
            raise KeyError(f"Source HDF5 is missing its 'data' group: {source_path}")

        data_group = source_file["data"]
        available_demos = _sorted_demo_names(data_group)
        if selected_demos is not None:
            if n_trajs is not None:
                raise ValueError("n_trajs and selected_demos cannot be used together.")
            if len(selected_demos) != len(set(selected_demos)):
                raise ValueError("selected_demos contains duplicate demonstration names.")
            missing_demos = sorted(set(selected_demos) - set(available_demos))
            if missing_demos:
                raise KeyError(
                    f"Selected demonstrations are missing from {source_path}: {missing_demos}"
                )
            selected_set = set(selected_demos)
            demos = [demo for demo in available_demos if demo in selected_set]
        else:
            demos = available_demos
        if n_trajs is not None:
            demos = demos[:n_trajs]
        if not demos:
            raise ValueError(f"No demonstrations selected from {source_path}")

        env_args_str = _decode_hdf5_text(data_group.attrs.get("env_args", "{}"))
        bddl_file = data_group.attrs.get("bddl_file_name", None)
        if bddl_file is not None:
            bddl_file = _decode_hdf5_text(bddl_file)

        horizons = {}
        demo_hashes = {}
        content_digest = hashlib.sha256()
        content_digest.update(env_args_str.encode("utf-8"))
        if bddl_file is not None:
            content_digest.update(bddl_file.encode("utf-8"))
        for demo in demos:
            if "states" not in data_group[demo]:
                raise KeyError(f"Source demo {demo!r} is missing its 'states' dataset.")
            if "actions" not in data_group[demo]:
                raise KeyError(f"Source demo {demo!r} is missing its 'actions' dataset.")
            states = data_group[demo]["states"][()]
            actions = data_group[demo]["actions"][()]
            horizons[demo] = int(states.shape[0])
            if actions.shape[0] != horizons[demo]:
                raise ValueError(
                    f"Source demo {demo!r} has {horizons[demo]} states but "
                    f"{actions.shape[0]} actions."
                )
            _update_hash_with_array(content_digest, f"{demo}/states", states)
            _update_hash_with_array(content_digest, f"{demo}/actions", actions)
            demo_hashes[demo] = {
                "horizon": horizons[demo],
                "states_sha256": _array_sha256(f"{demo}/states", states),
                "source_actions_sha256": _array_sha256(
                    f"{demo}/actions", actions
                ),
            }

    signature_payload = {
        "input_hdf5": str(source_path),
        "input_size": source_stat.st_size,
        "input_mtime_ns": source_stat.st_mtime_ns,
        "demos": [(demo, horizons[demo]) for demo in demos],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "demos": demos,
        "horizons": horizons,
        "signature": signature,
        "content_sha256": content_digest.hexdigest(),
        "input_hdf5": str(source_path),
        "input_size_bytes": int(source_stat.st_size),
        "demo_hashes": demo_hashes,
        "env_args_str": env_args_str,
        "bddl_file": bddl_file,
    }


def _array_sha256(name, values):
    digest = hashlib.sha256()
    _update_hash_with_array(digest, name, values)
    return digest.hexdigest()


def _action_source_manifest(action_source_path, source_manifest):
    """Validate and bind a stored-action replay to exact selected action arrays."""
    action_path = Path(action_source_path).resolve()
    if not action_path.is_file():
        raise FileNotFoundError(f"Stored-action source does not exist: {action_path}")

    content_digest = hashlib.sha256()
    demos = {}
    with h5py.File(action_path, "r") as action_file:
        for demo in source_manifest["demos"]:
            dataset_path = f"data/{demo}/actions"
            if dataset_path not in action_file:
                raise KeyError(
                    f"Stored-action source {action_path} is missing {dataset_path}."
                )
            actions = action_file[dataset_path][()]
            expected_shape = (
                source_manifest["horizons"][demo],
                _TARGET_ACTION_DIMENSION,
            )
            if actions.shape != expected_shape:
                raise ValueError(
                    f"Stored Aubo actions for {demo} have shape {actions.shape}; "
                    f"expected {expected_shape}."
                )
            if not np.issubdtype(actions.dtype, np.floating):
                raise TypeError(
                    f"Stored Aubo actions for {demo} must be floating point, "
                    f"got {actions.dtype}."
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError(f"Stored Aubo actions for {demo} contain NaN or inf.")
            _update_hash_with_array(content_digest, dataset_path, actions)
            demos[demo] = {
                "shape": list(actions.shape),
                "dtype": str(actions.dtype),
                "sha256": _array_sha256(dataset_path, actions),
            }

        source_recipe = action_file["data"].attrs.get(_CONVERSION_RECIPE_ATTR)
        if source_recipe is not None:
            source_recipe = _decode_hdf5_text(source_recipe)

    return {
        "hdf5": str(action_path),
        "selected_actions_sha256": content_digest.hexdigest(),
        "source_conversion_recipe_sha256": (
            hashlib.sha256(source_recipe.encode("utf-8")).hexdigest()
            if source_recipe is not None
            else None
        ),
        "demos": demos,
    }


def _result_map(results, allowed_demos, require_all):
    """Validate result records and index them by demonstration name."""
    if not isinstance(results, list):
        raise ValueError("Conversion results must be a list.")

    allowed_demos = set(allowed_demos)
    indexed = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("Each conversion result must be a JSON object.")
        demo = result.get("trajectory")
        if demo not in allowed_demos:
            raise ValueError(f"Unexpected demo {demo!r} in conversion progress.")
        if demo in indexed:
            raise ValueError(f"Duplicate demo {demo!r} in conversion progress.")
        indexed[demo] = result

    if require_all and set(indexed) != allowed_demos:
        raise ValueError("Conversion results do not cover every selected demo.")
    return indexed


def _expected_target_state_schema(gripper_state_dimension):
    """Return the target-native low-dimensional observation contract."""
    return {
        "version": _TARGET_STATE_SCHEMA_VERSION,
        "robot": _TARGET_ROBOT_TOKEN,
        "gripper": _TARGET_GRIPPER_TOKEN,
        "arm_joint_key": f"obs/{_ARM_JOINT_OBS_KEY}",
        "arm_joint_dimension": _TARGET_ARM_STATE_DIMENSION,
        "gripper_qpos_key": f"obs/{_GRIPPER_QPOS_OBS_KEY}",
        "gripper_qpos_dimension": int(gripper_state_dimension),
        "eef_position_key": f"obs/{_EEF_POSITION_OBS_KEY}",
        "eef_position_dimension": 3,
        "eef_quaternion_key": f"obs/{_EEF_QUATERNION_OBS_KEY}",
        "eef_quaternion_dimension": 4,
        "eef_reference_frame": "world",
        "eef_quaternion_convention": "xyzw",
        "combined_state_dimension": _TARGET_ARM_STATE_DIMENSION
        + int(gripper_state_dimension),
    }


def _target_output_schema(output_file):
    """Return a trusted target schema, or ``None`` for legacy / mixed output."""
    try:
        data_group = output_file["data"]
        raw_recipe = data_group.attrs[_CONVERSION_RECIPE_ATTR]
        recipe = json.loads(_decode_hdf5_text(raw_recipe))
        if (
            recipe.get("version") != _CONVERSION_RECIPE_VERSION
            or recipe.get("target_state_schema_version")
            != _TARGET_STATE_SCHEMA_VERSION
        ):
            return None

        embodiment_manifest = recipe.get(_EMBODIMENT_MANIFEST_ATTR)
        embodiment_sha256 = recipe.get(_EMBODIMENT_SHA256_ATTR)
        if not isinstance(embodiment_manifest, dict) or not isinstance(
            embodiment_sha256, str
        ):
            return None
        manifest_payload = dict(embodiment_manifest)
        manifest_declared_sha256 = manifest_payload.pop(
            _EMBODIMENT_SHA256_ATTR, None
        )
        if (
            manifest_declared_sha256 != embodiment_sha256
            or _json_sha256(manifest_payload) != embodiment_sha256
        ):
            return None
        stored_manifest = json.loads(
            _decode_hdf5_text(data_group.attrs[_EMBODIMENT_MANIFEST_ATTR])
        )
        stored_sha256 = _decode_hdf5_text(
            data_group.attrs[_EMBODIMENT_SHA256_ATTR]
        )
        if stored_manifest != embodiment_manifest or stored_sha256 != embodiment_sha256:
            return None
        _read_runtime_contract(data_group, required=True)

        env_args = json.loads(_decode_hdf5_text(data_group.attrs["env_args"]))
        env_kwargs = env_args.get("env_kwargs", {})
        if env_kwargs.get("robots") != [_TARGET_ROBOT_TOKEN] or env_kwargs.get(
            "gripper_types"
        ) != [_TARGET_GRIPPER_TOKEN]:
            return None

        schema = json.loads(
            _decode_hdf5_text(data_group.attrs[_TARGET_STATE_SCHEMA_ATTR])
        )
        if not isinstance(schema, dict):
            return None
        return schema
    except (KeyError, TypeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None


def _output_demo_is_complete(output_file, demo, horizon, result=None):
    """Check that a checkpoint contains a complete target-native Aubo demo."""
    try:
        if not isinstance(result, dict) or result.get("horizon") != horizon:
            return False
        state_schema = _target_output_schema(output_file)
        if state_schema is None:
            return False

        demo_group = output_file[f"data/{demo}"]
        actions = demo_group["actions"]
        if len(actions.shape) != 2 or actions.shape != (horizon, _TARGET_ACTION_DIMENSION):
            return False
        action_values = actions[()]
        if not np.all(np.isfinite(action_values)):
            return False
        expected_action_sha256 = result.get("stored_action_sha256")
        if not isinstance(expected_action_sha256, str) or _array_sha256(
            f"data/{demo}/actions", action_values
        ) != expected_action_sha256:
            return False
        if result.get("action_mode") == "stored_action_replay" and (
            result.get("stored_action_matches_source") is not True
            or result.get("action_source_sha256") != expected_action_sha256
        ):
            return False

        observation_shapes = {
            "agentview_image": None,
            _WRIST_IMAGE_KEY: None,
            _ARM_JOINT_OBS_KEY: (_TARGET_ARM_STATE_DIMENSION,),
            _GRIPPER_QPOS_OBS_KEY: None,
            _EEF_POSITION_OBS_KEY: (3,),
            _EEF_QUATERNION_OBS_KEY: (4,),
        }
        for observation_key, trailing_shape in observation_shapes.items():
            observation = demo_group[f"obs/{observation_key}"]
            if len(observation.shape) < 1 or observation.shape[0] != horizon:
                return False
            if trailing_shape is not None and observation.shape[1:] != trailing_shape:
                return False

        arm_values = demo_group[f"obs/{_ARM_JOINT_OBS_KEY}"][()]
        gripper_values = demo_group[f"obs/{_GRIPPER_QPOS_OBS_KEY}"][()]
        eef_position_values = demo_group[f"obs/{_EEF_POSITION_OBS_KEY}"][()]
        eef_quaternion_values = demo_group[f"obs/{_EEF_QUATERNION_OBS_KEY}"][()]
        if gripper_values.ndim != 2 or gripper_values.shape[1] < 1:
            return False
        for values in (
            arm_values,
            gripper_values,
            eef_position_values,
            eef_quaternion_values,
        ):
            if not np.all(np.isfinite(values)):
                return False
        if not np.allclose(
            np.linalg.norm(eef_quaternion_values, axis=1), 1.0, atol=1e-5
        ):
            return False
        if result.get("eef_position_sha256") != _array_sha256(
            f"data/{demo}/obs/{_EEF_POSITION_OBS_KEY}", eef_position_values
        ) or result.get("eef_quaternion_sha256") != _array_sha256(
            f"data/{demo}/obs/{_EEF_QUATERNION_OBS_KEY}",
            eef_quaternion_values,
        ):
            return False

        expected_schema = _expected_target_state_schema(gripper_values.shape[1])
        if state_schema != expected_schema:
            return False
        if result.get("arm_joint_state_dimension") != _TARGET_ARM_STATE_DIMENSION:
            return False
        if result.get("gripper_qpos_state_dimension") != gripper_values.shape[1]:
            return False

        states = demo_group["states"]
        rewards = demo_group["rewards"]
        dones = demo_group["dones"]
        if states.ndim != 2 or states.shape[0] != horizon or states.shape[1] < 1:
            return False
        if rewards.shape != (horizon,) or dones.shape != (horizon,):
            return False
        state_values = states[()]
        if not np.all(np.isfinite(state_values)):
            return False
        if result.get("target_states_sha256") != _array_sha256(
            f"data/{demo}/states", state_values
        ):
            return False
        initial_state = np.asarray(demo_group.attrs["init_state"])
        if initial_state.shape != state_values[0].shape or not np.array_equal(
            initial_state, state_values[0]
        ):
            return False
        expected_initial_state_sha256 = hashlib.sha256(
            np.ascontiguousarray(state_values[0]).view(np.uint8)
        ).hexdigest()
        if result.get("initial_target_state_sha256") != expected_initial_state_sha256:
            return False
        if "robot_states" in demo_group:
            return False
        if result.get("initial_null_steps") != _INITIAL_NULL_STEPS:
            return False
        if result.get("initialization_order") != _INITIALIZATION_ORDER:
            return False
        if result.get("initial_observation_refresh") != _INITIAL_OBSERVATION_REFRESH:
            return False
        if result.get("initial_observation_refresh_control_steps") != 0:
            return False
        if result.get("initial_observation_refresh_state_preserved") is not True:
            return False
        source_initial_time = float(result["source_initial_simulation_time_s"])
        target_initial_time = float(result["initial_target_simulation_time_s"])
        if (
            not np.isfinite(source_initial_time)
            or not np.isfinite(target_initial_time)
            or target_initial_time != source_initial_time
        ):
            return False
        if not isinstance(result.get("reset_seed"), int):
            return False
        for dataset in (rewards, dones):
            if not np.all(np.isfinite(dataset[()])):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _verified_results(output_file, manifest, results):
    """Keep only checkpointed demos whose target data is still intact."""
    verified = {}
    for demo, result in results.items():
        if _output_demo_is_complete(
            output_file,
            demo,
            manifest["horizons"][demo],
            result=result,
        ):
            verified[demo] = result
        else:
            print(f"[RESUME] Reprocessing incomplete or damaged checkpoint: {demo}")
    return verified


def _store_resume_progress(output_file, manifest, results):
    """Persist completed demos after each trajectory so a killed job can resume."""
    ordered_results = [
        results[demo] for demo in manifest["demos"] if demo in results
    ]
    progress = {
        "version": _RESUME_PROGRESS_VERSION,
        "source_signature": manifest["signature"],
        "completed_demos": [result["trajectory"] for result in ordered_results],
        "trajectories": ordered_results,
    }
    output_file["data"].attrs[_RESUME_PROGRESS_ATTR] = json.dumps(progress)
    output_file.flush()


def _load_resume_progress(output_file, manifest):
    """Load a prior per-trajectory checkpoint, or return None if none exists."""
    raw_progress = output_file["data"].attrs.get(_RESUME_PROGRESS_ATTR)
    if raw_progress is None:
        return None

    try:
        progress = json.loads(_decode_hdf5_text(raw_progress))
        if progress.get("version") != _RESUME_PROGRESS_VERSION:
            raise ValueError("unsupported progress format version")
        if progress.get("source_signature") != manifest["signature"]:
            raise ValueError("source file or selected trajectory set changed")

        completed_demos = progress.get("completed_demos")
        if not isinstance(completed_demos, list):
            raise ValueError("missing completed_demos list")
        if len(completed_demos) != len(set(completed_demos)):
            raise ValueError("duplicate demos in completed_demos")

        results = _result_map(
            progress.get("trajectories"), manifest["demos"], require_all=False
        )
        if set(results) != set(completed_demos):
            raise ValueError("completed_demos does not match trajectory results")
        return results
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid resume checkpoint: {error}") from error


def _load_complete_report(report_path, manifest):
    """Use a validated report from older runs as a complete-file checkpoint."""
    if not report_path.is_file():
        return None

    try:
        with report_path.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
        if report.get("trajectory_count") != len(manifest["demos"]):
            return None
        return _result_map(report.get("trajectories"), manifest["demos"], require_all=True)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[RESUME] Ignoring invalid completion report {report_path}: {error}")
        return None


def _write_conversion_report(
    report_path,
    input_path,
    output_path,
    env_name,
    expected_panda_model,
    expected_aubo_model,
    action_dimension,
    manifest,
    conversion_recipe,
    runtime_contract,
    results,
    action_source_path=None,
):
    """Write the final report atomically after every selected demo is complete."""
    ordered_results = [results[demo] for demo in manifest["demos"]]
    terminal_success_count = sum(
        result["terminal_task_success"] for result in ordered_results
    )
    report = {
        "input_hdf5": str(Path(input_path).resolve()),
        "output_hdf5": str(Path(output_path).resolve()),
        "environment": env_name,
        "reference_robot": expected_panda_model,
        "target_robot": expected_aubo_model,
        "action_dimension": action_dimension,
        _CONVERSION_RECIPE_ATTR: conversion_recipe,
        _EMBODIMENT_SHA256_ATTR: conversion_recipe[_EMBODIMENT_SHA256_ATTR],
        _EMBODIMENT_MANIFEST_ATTR: conversion_recipe[_EMBODIMENT_MANIFEST_ATTR],
        _RUNTIME_CONTRACT_ATTR: runtime_contract,
        _RUNTIME_CONTRACT_SHA256_ATTR: _json_sha256(runtime_contract),
        "trajectory_count": len(ordered_results),
        "terminal_task_success_count": terminal_success_count,
        "terminal_task_failure_count": len(ordered_results) - terminal_success_count,
        "trajectories": ordered_results,
    }
    if action_source_path is not None:
        report["action_source_hdf5"] = str(Path(action_source_path).resolve())
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    os.replace(temporary_path, report_path)
    print(f"[SUCCESS REPORT] {report_path}")


def _copy_hdf5_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _create_slim_output(input_path, output_path, manifest):
    """Create a target-only shell without copying stale Panda datasets."""
    with h5py.File(input_path, "r") as source_file, h5py.File(output_path, "w") as output_file:
        source_data = source_file["data"]
        output_data = output_file.create_group("data")
        _copy_hdf5_attrs(source_data, output_data)
        output_data.attrs["num_demos"] = len(manifest["demos"])
        output_data.attrs["total"] = sum(
            manifest["horizons"][demo] for demo in manifest["demos"]
        )
        for demo in manifest["demos"]:
            source_demo = source_data[demo]
            output_demo = output_data.create_group(demo)
            for key, value in source_demo.attrs.items():
                if key not in ("init_state", "model_file"):
                    output_demo.attrs[key] = value


def _replace_output_from_source(
    input_path,
    output_path,
    report_path,
    reason,
    *,
    manifest=None,
    slim_output=False,
):
    """Create a fresh output copy after a new conversion or explicit restart."""
    print(f"[RESUME] {reason}; recreating {output_path} from the source dataset.")
    if report_path.exists():
        report_path.unlink()
    if slim_output:
        if manifest is None:
            raise ValueError("A manifest is required when creating a slim output.")
        _create_slim_output(input_path, output_path, manifest)
    else:
        shutil.copy2(input_path, output_path)
    return h5py.File(output_path, "a")


def _open_output_for_conversion(
    input_path, output_path, report_path, manifest, resume, *, slim_output=False
):
    """Open a fresh output or a verified partial output for conversion."""
    if not resume:
        return _replace_output_from_source(
            input_path,
            output_path,
            report_path,
            "Restart requested",
            manifest=manifest,
            slim_output=slim_output,
        ), {}

    if not os.path.exists(output_path):
        return _replace_output_from_source(
            input_path,
            output_path,
            report_path,
            "No prior output exists",
            manifest=manifest,
            slim_output=slim_output,
        ), {}

    try:
        output_file = h5py.File(output_path, "a")
        if "data" not in output_file:
            raise ValueError("missing HDF5 'data' group")
        if not set(manifest["demos"]).issubset(output_file["data"].keys()):
            raise ValueError("missing one or more source demo groups")
    except (OSError, ValueError) as error:
        try:
            output_file.close()
        except UnboundLocalError:
            pass
        raise RuntimeError(
            f"Cannot resume untrusted output {output_path}: {error}. "
            "Use --restart to recreate it from the source dataset."
        ) from error

    try:
        results = _load_resume_progress(output_file, manifest)
    except ValueError as error:
        output_file.close()
        raise RuntimeError(
            f"Cannot resume untrusted output {output_path}: {error}. "
            "Use --restart to recreate it from the source dataset."
        ) from error

    if results is None:
        legacy_results = _load_complete_report(report_path, manifest)
        if legacy_results is None:
            output_file.close()
            raise RuntimeError(
                f"Cannot resume untrusted output {output_path}: no valid completion "
                "report or per-trajectory checkpoint exists. Use --restart to "
                "recreate it from the source dataset."
            )
        results = _verified_results(output_file, manifest, legacy_results)
        if len(results) != len(manifest["demos"]):
            output_file.close()
            raise RuntimeError(
                f"Cannot resume untrusted output {output_path}: the completed "
                "report does not match its HDF5 data. Use --restart to recreate "
                "it from the source dataset."
            )
        _store_resume_progress(output_file, manifest, results)
        print(f"[RESUME] Validated completed output: {output_path}")
        return output_file, results

    verified_results = _verified_results(output_file, manifest, results)
    if len(verified_results) != len(results):
        _store_resume_progress(output_file, manifest, verified_results)

    if len(verified_results) < len(manifest["demos"]) and report_path.exists():
        report_path.unlink()
    print(
        f"[RESUME] Found {len(verified_results)} / {len(manifest['demos'])} "
        "completed trajectories."
    )
    return output_file, verified_results


def process_single_file(
    input_path,
    output_path,
    n_trajs=None,
    render_video=False,
    video_out_path="trial_conversion.mp4",
    resume=True,
    selected_demos=None,
    action_source_path=None,
    slim_output=False,
    seed=0,
):
    print(f"\n[INFO] Converting: {os.path.basename(input_path)}")
    report_path = Path(output_path).with_suffix(".conversion_report.json")
    f = None
    source_file = None
    action_source_file = None
    env_ref = None
    env_sim = None

    try:
        manifest = _source_manifest(input_path, n_trajs, selected_demos=selected_demos)
        action_manifest = (
            _action_source_manifest(action_source_path, manifest)
            if action_source_path is not None
            else None
        )
        conversion_recipe = _build_conversion_recipe(
            manifest,
            seed,
            action_source_manifest=action_manifest,
        )
        f, trajectory_results = _open_output_for_conversion(
            input_path,
            output_path,
            report_path,
            manifest,
            resume,
            slim_output=slim_output,
        )
        serialized_recipe = json.dumps(conversion_recipe, sort_keys=True)
        existing_recipe = f["data"].attrs.get(_CONVERSION_RECIPE_ATTR)
        if existing_recipe is None and trajectory_results:
            raise RuntimeError(
                "Cannot resume completed trajectories without a recorded conversion "
                "recipe. Use --restart and a new output path."
            )
        if existing_recipe is not None:
            existing_recipe = _decode_hdf5_text(existing_recipe)
            if existing_recipe != serialized_recipe:
                raise RuntimeError(
                    "Cannot resume output created with a different conversion recipe. "
                    "Use --restart and a new output path."
                )
        f["data"].attrs[_CONVERSION_RECIPE_ATTR] = serialized_recipe

        embodiment_manifest = conversion_recipe[_EMBODIMENT_MANIFEST_ATTR]
        serialized_embodiment_manifest = json.dumps(
            embodiment_manifest, sort_keys=True
        )
        embodiment_sha256 = conversion_recipe[_EMBODIMENT_SHA256_ATTR]
        for attribute_name, expected_value in (
            (_EMBODIMENT_MANIFEST_ATTR, serialized_embodiment_manifest),
            (_EMBODIMENT_SHA256_ATTR, embodiment_sha256),
        ):
            existing_value = f["data"].attrs.get(attribute_name)
            if existing_value is None and trajectory_results:
                raise RuntimeError(
                    "Cannot resume completed trajectories without the target "
                    f"embodiment provenance attribute {attribute_name!r}. Use "
                    "--restart and a new output path."
                )
            if existing_value is not None:
                existing_value = _decode_hdf5_text(existing_value)
                if existing_value != expected_value:
                    raise RuntimeError(
                        "Cannot resume output created with a different target "
                        f"embodiment provenance attribute {attribute_name!r}. Use "
                        "--restart and a new output path."
                    )
            f["data"].attrs[attribute_name] = expected_value

        env_args_str = manifest["env_args_str"]
        env_args = json.loads(env_args_str) if env_args_str != "{}" else {}
        if "env_kwargs" not in env_args:
            env_args["env_kwargs"] = {}

        env_name = env_args.get("env_name")
        expected_panda_model, expected_aubo_model = _domain_robot_models(env_name)

        if len(trajectory_results) == len(manifest["demos"]):
            runtime_contract = _read_runtime_contract(f["data"], required=True)
            f.close()
            f = None
            _write_conversion_report(
                report_path,
                input_path,
                output_path,
                env_name,
                expected_panda_model,
                expected_aubo_model,
                _TARGET_ACTION_DIMENSION,
                manifest,
                conversion_recipe,
                runtime_contract,
                trajectory_results,
                action_source_path=action_source_path,
            )
            print(f"[RESUME] Skipping complete file: {os.path.basename(input_path)}")
            return True
            
        # 1. Configure the Target AUBO Environment
        _set_controller_metadata(env_args)
        # ``AuboI5`` is deliberately the base token. Each BDDL domain applies
        # the correct Mounted / OnTheGround prefix during environment creation.
        env_args["env_kwargs"]["robots"] = [_TARGET_ROBOT_TOKEN]
        env_args["env_kwargs"]["gripper_types"] = [_TARGET_GRIPPER_TOKEN]
        _set_target_observation_metadata(env_args)
        # Resolve BDDL paths 
        bddl_file = manifest["bddl_file"]
        if bddl_file:
            if "bddl_files/" in bddl_file:
                relative_path = bddl_file.split("bddl_files/")[-1]
            else:
                relative_path = os.path.basename(bddl_file)
                
            absolute_bddl_path = os.path.join(get_libero_path("bddl_files"), relative_path)
            env_args["env_kwargs"]["bddl_file_name"] = absolute_bddl_path
            
        f["data"].attrs["env_args"] = json.dumps(env_args, indent=4)
        env_meta_sim = json.loads(f["data"].attrs["env_args"])

        # Validate required env_meta keys
        for key in ("env_name", "type"):
            if key not in env_meta_sim:
                raise KeyError(f"env_meta_sim is missing required key '{key}'. "
                               f"Available keys: {list(env_meta_sim.keys())}")

        # 2. Configure the Reference Panda Environment
        # Overwrite the broken HDF5 metadata with the fully populated base configuration
        env_meta_ref = json.loads(env_args_str) if env_args_str != "{}" else {}
        if "env_kwargs" not in env_meta_ref: env_meta_ref["env_kwargs"] = {}
        # A source demonstration is always interpreted through the standard
        # Panda gripper TCP. This avoids inheriting an accidental target robot
        # or gripper override from edited dataset metadata.
        env_meta_ref["env_kwargs"]["robots"] = ["Panda"]
        env_meta_ref["env_kwargs"]["gripper_types"] = ["PandaGripper"]
        
        if bddl_file:
            env_meta_ref["env_kwargs"]["bddl_file_name"] = absolute_bddl_path
            
        _set_controller_metadata(env_meta_ref, preserve_legacy_config=True)

        for key in ("env_name", "type"):
            if key not in env_meta_ref:
                raise KeyError(f"env_meta_ref is missing required key '{key}'. "
                               f"Available keys: {list(env_meta_ref.keys())}")

        # Instantiate Dual Environments
        env_ref = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta_ref, 
            camera_names=[],
            camera_height=84,
            camera_width=84,
            reward_shaping=False
        )
        
        env_sim = EnvUtils.create_env_for_data_processing(
            env_meta=env_meta_sim, 
            camera_names=list(_TARGET_CAMERA_NAMES),
            camera_height=_TARGET_CAMERA_HEIGHT,
            camera_width=_TARGET_CAMERA_WIDTH,
            reward_shaping=False
        )
        _assert_runtime_robot(env_ref, expected_panda_model, "Panda reference")
        _assert_runtime_robot(env_sim, expected_aubo_model, "Aubo target")
        runtime_contract = _assert_target_runtime_contract(env_sim)
        _bind_runtime_contract(
            f["data"],
            runtime_contract,
            require_existing=bool(trajectory_results),
        )
        action_dimension = _action_dimension(env_sim)
        if action_dimension != _TARGET_ACTION_DIMENSION:
            raise RuntimeError(
                f"Expected a 6-DoF OSC pose action plus one gripper action, got {action_dimension}."
            )

        demos = manifest["demos"]
        source_file = h5py.File(input_path, "r")
        if action_source_path is not None:
            action_source_file = h5py.File(action_source_path, "r")
        video_frames = []
        video_demo = next(
            (demo for demo in demos if demo not in trajectory_results), None
        )

        for ep in tqdm(demos, desc="Processing Trajectories", leave=False):
            if ep in trajectory_results:
                print(f"[RESUME] Skipping completed trajectory: {ep}")
                continue

            ep_grp = f[f"data/{ep}"]
            source_ep_grp = source_file[f"data/{ep}"]
            states = source_ep_grp["states"][()]
            original_actions = source_ep_grp["actions"][()]
            horizon = states.shape[0]
            if action_source_file is None:
                replay_actions = None
            else:
                action_dataset = action_source_file[f"data/{ep}/actions"]
                if action_dataset.shape != (horizon, _TARGET_ACTION_DIMENSION):
                    raise ValueError(
                        f"Stored Aubo actions for {ep} have shape {action_dataset.shape}; "
                        f"expected {(horizon, _TARGET_ACTION_DIMENSION)}."
                    )
                replay_actions = action_dataset[()]
                replay_action_sha256 = _array_sha256(
                    f"data/{ep}/actions", replay_actions
                )
                expected_action_sha256 = action_manifest["demos"][ep]["sha256"]
                if replay_action_sha256 != expected_action_sha256:
                    raise RuntimeError(
                        f"Stored actions changed after replay validation for {ep}: "
                        f"{replay_action_sha256} != {expected_action_sha256}."
                    )
            
            new_actions = []
            new_obs = {
                "agentview_image": [],
                _WRIST_IMAGE_KEY: [],
                _ARM_JOINT_OBS_KEY: [],
                _GRIPPER_QPOS_OBS_KEY: [],
                _EEF_POSITION_OBS_KEY: [],
                _EEF_QUATERNION_OBS_KEY: [],
            }
            new_states = []
            new_rewards = []
            new_dones = []
            
            initial_actions = replay_actions if replay_actions is not None else original_actions
            obs = initialize_target_replay(
                env_ref,
                env_sim,
                states[0],
                initial_actions[0][-1],
                action_dimension,
                seed=seed,
            )
            _assert_target_observation_contract(obs)
            source_initial_simulation_time_s = float(
                _get_base_env(env_ref).sim.data.time
            )
            initial_target_simulation_time_s = float(
                _get_base_env(env_sim).sim.data.time
            )
            if initial_target_simulation_time_s != source_initial_simulation_time_s:
                raise RuntimeError(
                    "Target initialization did not preserve source simulator time for "
                    f"{ep}: source={source_initial_simulation_time_s:.17g}, "
                    f"target={initial_target_simulation_time_s:.17g}."
                )

            first_success_step = None
            last_reward = None
            last_done = False
            position_errors_mm = []
            rotation_errors_deg = []
            clipped_components = []
            clipped_steps = []

            for t in range(horizon):
                current_pos, current_mat = get_eef_pose(env_sim)
                new_obs["agentview_image"].append(obs["agentview_image"])
                new_obs[_WRIST_IMAGE_KEY].append(obs[_WRIST_IMAGE_KEY])
                arm_qpos, gripper_qpos = get_target_robot_qpos(env_sim)
                new_obs[_ARM_JOINT_OBS_KEY].append(arm_qpos)
                new_obs[_GRIPPER_QPOS_OBS_KEY].append(gripper_qpos)
                new_obs[_EEF_POSITION_OBS_KEY].append(current_pos)
                new_obs[_EEF_QUATERNION_OBS_KEY].append(
                    R.from_matrix(current_mat).as_quat()
                )
                new_states.append(get_target_sim_state(env_sim))
                
                if render_video and ep == video_demo:
                    video_frames.append(obs["agentview_image"])

                target_idx = min(t + _TARGET_LOOKAHEAD_STEPS, horizon - 1)
                env_ref.reset_to({"states": states[target_idx]})
                target_pos, target_mat = get_eef_pose(env_ref)
                position_errors_mm.append(
                    float(np.linalg.norm(target_pos - current_pos) * 1000.0)
                )
                rotation_errors_deg.append(
                    float(
                        np.degrees(
                            np.linalg.norm(
                                get_orientation_error(target_mat, current_mat)
                            )
                        )
                    )
                )

                if replay_actions is None:
                    action_pos, action_ori, raw_action_pos, raw_action_ori = (
                        calculate_tracking_action(
                            current_pos,
                            current_mat,
                            target_pos,
                            target_mat,
                            return_raw=True,
                        )
                    )
                    raw_cartesian = np.concatenate(
                        [raw_action_pos, raw_action_ori]
                    )
                    gripper_action = original_actions[t][-1:]
                    tracked_action = np.concatenate(
                        [action_pos, action_ori, gripper_action]
                    )
                else:
                    tracked_action = np.asarray(replay_actions[t], dtype=np.float64).copy()
                    raw_cartesian = tracked_action[:6]
                if tracked_action.shape != (_TARGET_ACTION_DIMENSION,) or not np.all(
                    np.isfinite(tracked_action)
                ):
                    raise RuntimeError(f"Invalid target action at {ep} step {t}.")
                component_mask = np.abs(raw_cartesian) > 1.0 + 1e-12
                if replay_actions is not None:
                    component_mask = np.abs(raw_cartesian) >= 1.0 - 1e-12
                clipped_components.extend(component_mask.tolist())
                clipped_steps.append(bool(np.any(component_mask)))
                new_actions.append(tracked_action)
                
                obs, reward, done, _ = env_sim.step(tracked_action)
                new_rewards.append(float(reward))
                new_dones.append(bool(done))
                last_reward = float(reward)
                last_done = bool(done)
                if _task_success(env_sim) and first_success_step is None:
                    first_success_step = t
                
            stored_actions = np.asarray(
                new_actions,
                dtype=(replay_actions.dtype if replay_actions is not None else np.float64),
            )
            if replay_actions is not None and (
                stored_actions.dtype != replay_actions.dtype
                or not np.array_equal(stored_actions, replay_actions)
            ):
                raise RuntimeError(
                    f"Replay output actions for {ep} are not byte-identical to "
                    "the stored action source."
                )

            for dataset_name, values in (
                ("actions", stored_actions),
                ("states", new_states),
                ("rewards", new_rewards),
                ("dones", new_dones),
            ):
                if dataset_name in ep_grp:
                    del ep_grp[dataset_name]
                ep_grp.create_dataset(dataset_name, data=np.asarray(values))

            if "obs" in ep_grp:
                del ep_grp["obs"]
            obs_group = ep_grp.create_group("obs")
            for observation_key, values in new_obs.items():
                obs_group.create_dataset(
                    observation_key, data=np.asarray(values), compression="gzip"
                )
            if "robot_states" in ep_grp:
                del ep_grp["robot_states"]
            ep_grp.attrs["num_samples"] = int(horizon)
            ep_grp.attrs["init_state"] = np.asarray(new_states[0])

            arm_state_dimension = int(
                np.asarray(new_obs[_ARM_JOINT_OBS_KEY]).shape[1]
            )
            gripper_state_dimension = int(
                np.asarray(new_obs[_GRIPPER_QPOS_OBS_KEY]).shape[1]
            )
            state_schema = _expected_target_state_schema(gripper_state_dimension)
            existing_schema = f["data"].attrs.get(_TARGET_STATE_SCHEMA_ATTR)
            if existing_schema is not None:
                existing_schema = json.loads(_decode_hdf5_text(existing_schema))
                if existing_schema != state_schema:
                    raise RuntimeError(
                        f"Target state schema changed within {output_path}: "
                        f"{existing_schema} != {state_schema}."
                    )
            f["data"].attrs[_TARGET_STATE_SCHEMA_ATTR] = json.dumps(
                state_schema, sort_keys=True
            )

            terminal_success = _task_success(env_sim)
            result = {
                "trajectory": ep,
                "horizon": int(horizon),
                "terminal_task_success": bool(terminal_success),
                "ever_task_success": first_success_step is not None,
                "first_success_step": first_success_step,
                "terminal_done": last_done,
                "terminal_reward": last_reward,
                "arm_joint_state_dimension": arm_state_dimension,
                "gripper_qpos_state_dimension": gripper_state_dimension,
                "reset_seed": int(seed),
                "initial_null_steps": _INITIAL_NULL_STEPS,
                "initialization_order": _INITIALIZATION_ORDER,
                "initial_observation_refresh": _INITIAL_OBSERVATION_REFRESH,
                "initial_observation_refresh_control_steps": 0,
                "initial_observation_refresh_state_preserved": True,
                "source_initial_simulation_time_s": source_initial_simulation_time_s,
                "initial_target_simulation_time_s": initial_target_simulation_time_s,
                "initial_gripper_action": float(initial_actions[0][-1]),
                "initial_target_state_sha256": hashlib.sha256(
                    np.ascontiguousarray(new_states[0]).view(np.uint8)
                ).hexdigest(),
                "max_position_error_mm": float(np.max(position_errors_mm)),
                "p95_position_error_mm": float(
                    np.percentile(position_errors_mm, 95)
                ),
                "max_rotation_error_deg": float(np.max(rotation_errors_deg)),
                "p95_rotation_error_deg": float(
                    np.percentile(rotation_errors_deg, 95)
                ),
                "clipped_component_pct": float(
                    100.0 * np.mean(clipped_components)
                ),
                "clipped_step_pct": float(100.0 * np.mean(clipped_steps)),
                "max_clipped_run_steps": int(_max_true_run(clipped_steps)),
                "raw_action_telemetry_available": replay_actions is None,
                "action_mode": (
                    "stored_action_replay"
                    if replay_actions is not None
                    else "panda_pose_tracking"
                ),
                "stored_action_sha256": _array_sha256(
                    f"data/{ep}/actions", stored_actions
                ),
                "target_states_sha256": _array_sha256(
                    f"data/{ep}/states", np.asarray(new_states)
                ),
                "eef_position_sha256": _array_sha256(
                    f"data/{ep}/obs/{_EEF_POSITION_OBS_KEY}",
                    np.asarray(new_obs[_EEF_POSITION_OBS_KEY]),
                ),
                "eef_quaternion_sha256": _array_sha256(
                    f"data/{ep}/obs/{_EEF_QUATERNION_OBS_KEY}",
                    np.asarray(new_obs[_EEF_QUATERNION_OBS_KEY]),
                ),
            }
            if replay_actions is not None:
                result.update(
                    {
                        "action_source_sha256": expected_action_sha256,
                        "stored_action_matches_source": True,
                    }
                )
            trajectory_results[ep] = result
            _store_resume_progress(f, manifest, trajectory_results)
            status = "SUCCESS" if terminal_success else "FAIL"
            success_step = first_success_step if first_success_step is not None else "none"
            print(
                f"[TRAJECTORY {ep}] {status} | first_success_step={success_step} | "
                f"terminal_done={last_done} | terminal_reward={last_reward}"
            )

        if set(trajectory_results) != set(manifest["demos"]):
            raise RuntimeError("Not every selected trajectory has a conversion result.")

        f.close()
        f = None
        _write_conversion_report(
            report_path,
            input_path,
            output_path,
            env_name,
            expected_panda_model,
            expected_aubo_model,
            action_dimension,
            manifest,
            conversion_recipe,
            runtime_contract,
            trajectory_results,
            action_source_path=action_source_path,
        )

        if render_video and len(video_frames) > 0:
            print(f"\n[UNIT TEST] Saving trial conversion video to {video_out_path}...")
            imageio.mimsave(video_out_path, video_frames, fps=20)
            print(f"[UNIT TEST] Video successfully saved! Check: {video_out_path}")

        return True
    
    except Exception as e:
        if f is not None:
            f.close()
        if not resume and os.path.exists(output_path):
            os.remove(output_path)
        elif resume and os.path.exists(output_path):
            print(f"[RESUME] Preserved partial output for the next run: {output_path}")
        print(f"\n[ERROR] Failed to process {input_path}. Error: {e}")
        traceback.print_exc()
        return False
    finally:
        if source_file is not None:
            source_file.close()
        if action_source_file is not None:
            action_source_file.close()
        _close_env(env_ref)
        _close_env(env_sim)

def batch_process(
    input_dir,
    output_dir,
    max_files=None,
    n_trajs=None,
    render_video=False,
    video_out_path="trial_conversion.mp4",
    resume=True,
    seed=0,
):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    hdf5_files = sorted(list(in_dir.glob("*.hdf5")))
    if max_files is not None:
        hdf5_files = hdf5_files[:max_files]

    print(f"Found {len(hdf5_files)} file(s) to process from {input_dir}. Output: {output_dir}")
    
    success_count = 0
    failed_files = []

    for file_path in hdf5_files:
        out_file = out_dir / file_path.name

        success = process_single_file(
            str(file_path), 
            str(out_file), 
            n_trajs=n_trajs, 
            render_video=render_video, 
            video_out_path=video_out_path,
            resume=resume,
            # Every production output is target-only. Copying the Panda HDF5
            # shell would retain stale model XML and unknown Panda datasets.
            slim_output=True,
            seed=seed,
        )
        
        if success:
            success_count += 1
        else:
            failed_files.append(file_path.name)
            
    print("\n" + "="*50)
    print("CONVERSION COMPLETE")
    print(f"Successfully converted: {success_count} / {len(hdf5_files)}")
    if failed_files:
        print(f"Failed files ({len(failed_files)}):")
        for f in failed_files:
            print(f"  - {f}")
    print("="*50)
    return not failed_files


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
    os.replace(temporary_path, path)


def _stable_selection_rank(seed, report_relative_path, demo):
    payload = f"{seed}:{report_relative_path}:{demo}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_first_run_success_selection(
    report_root,
    episodes_per_task,
    selection_seed,
    *,
    smoke=False,
    expected_success_count=807,
):
    report_root = Path(report_root).resolve()
    report_paths = sorted(report_root.glob("*/*.conversion_report.json"))
    if not report_paths:
        raise FileNotFoundError(f"No conversion reports found below {report_root}.")
    if episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive.")

    report_records = []
    baseline_success_count = 0
    for report_path in report_paths:
        with report_path.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
        trajectories = report.get("trajectories")
        if not isinstance(trajectories, list):
            raise ValueError(f"Report has no trajectory list: {report_path}")
        successes = [
            item for item in trajectories if item.get("terminal_task_success") is True
        ]
        declared_successes = report.get("terminal_task_success_count")
        if declared_successes != len(successes):
            raise ValueError(
                f"Success count mismatch in {report_path}: "
                f"declared {declared_successes}, found {len(successes)}."
            )
        baseline_success_count += len(successes)
        if not successes:
            continue

        relative_report = report_path.relative_to(report_root).as_posix()
        ranked = sorted(
            successes,
            key=lambda item: _stable_selection_rank(
                selection_seed, relative_report, item["trajectory"]
            ),
        )
        report_records.append((report_path, report, ranked))

    if expected_success_count and baseline_success_count != expected_success_count:
        raise RuntimeError(
            f"Expected {expected_success_count} first-run successes below {report_root}, "
            f"found {baseline_success_count}."
        )

    selected_records = []
    if smoke:
        selected_suites = set()
        for report_path, report, successes in report_records:
            suite = report_path.parent.name
            if suite in selected_suites:
                continue
            selected_suites.add(suite)
            selected_records.append((report_path, report, successes[:1]))
        expected_suites = {"libero_10", "libero_goal", "libero_object", "libero_spatial"}
        if selected_suites != expected_suites:
            raise RuntimeError(
                f"Smoke selection did not cover every suite: {sorted(selected_suites)}."
            )
    else:
        for report_path, report, successes in report_records:
            selected_records.append(
                (report_path, report, successes[:episodes_per_task])
            )
    return report_root, baseline_success_count, selected_records


def build_first_run_success_subset(
    report_root,
    output_dir,
    *,
    episodes_per_task=3,
    selection_seed=0,
    smoke=False,
    expected_success_count=807,
    resume=True,
    seed=0,
):
    """Replay stored first-run actions and capture target-native Aubo observations."""
    report_root, baseline_success_count, selected_records = _load_first_run_success_selection(
        report_root,
        episodes_per_task,
        selection_seed,
        smoke=smoke,
        expected_success_count=expected_success_count,
    )
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "subset_manifest.json"

    entries = []
    for report_path, report, selected in selected_records:
        suite = report_path.parent.name
        output_hdf5 = output_root / suite / Path(report["output_hdf5"]).name
        entries.append(
            {
                "suite": suite,
                "source_report": str(report_path),
                "source_report_sha256": hashlib.sha256(
                    report_path.read_bytes()
                ).hexdigest(),
                "source_panda_hdf5": str(Path(report["input_hdf5"]).resolve()),
                "first_run_aubo_hdf5": str(Path(report["output_hdf5"]).resolve()),
                "output_hdf5": str(output_hdf5),
                "selected_demos": [item["trajectory"] for item in selected],
                "selected_horizons": {
                    item["trajectory"]: int(item["horizon"]) for item in selected
                },
                "completed": False,
            }
        )

    manifest = {
        "version": 2,
        "selection": "first_run_terminal_task_success",
        "source_report_root": str(report_root),
        "output_root": str(output_root),
        "baseline_success_count": baseline_success_count,
        "episodes_per_task": 1 if smoke else episodes_per_task,
        "selection_seed": selection_seed,
        "replay_seed": int(seed),
        "smoke": bool(smoke),
        "selected_episode_count": sum(len(entry["selected_demos"]) for entry in entries),
        "entries": entries,
    }
    _atomic_write_json(manifest_path, manifest)

    failed_files = []
    for entry in entries:
        output_hdf5 = Path(entry["output_hdf5"])
        output_hdf5.parent.mkdir(parents=True, exist_ok=True)
        success = process_single_file(
            entry["source_panda_hdf5"],
            entry["output_hdf5"],
            selected_demos=entry["selected_demos"],
            action_source_path=entry["first_run_aubo_hdf5"],
            slim_output=True,
            resume=resume,
            seed=seed,
        )
        entry["completed"] = bool(success)
        subset_report = output_hdf5.with_suffix(".conversion_report.json")
        if success and subset_report.is_file():
            with subset_report.open(encoding="utf-8") as report_file:
                replay_report = json.load(report_file)
            entry["revalidated_success_count"] = int(
                replay_report["terminal_task_success_count"]
            )
            entry["revalidated_failure_count"] = int(
                replay_report["terminal_task_failure_count"]
            )
        else:
            failed_files.append(entry["output_hdf5"])
        _atomic_write_json(manifest_path, manifest)

    manifest["revalidated_success_count"] = sum(
        entry.get("revalidated_success_count", 0) for entry in entries
    )
    manifest["revalidated_failure_count"] = sum(
        entry.get("revalidated_failure_count", 0) for entry in entries
    )
    manifest["failed_file_count"] = len(failed_files)
    manifest["failed_files"] = failed_files
    manifest["replay_verified"] = bool(
        not failed_files
        and manifest["revalidated_success_count"]
        == manifest["selected_episode_count"]
    )
    _atomic_write_json(manifest_path, manifest)
    print(
        f"[SUBSET] Revalidated {manifest['revalidated_success_count']} / "
        f"{manifest['selected_episode_count']} selected first-run successes."
    )
    print(
        "[SUBSET VERDICT] "
        + ("REPLAY_VERIFIED" if manifest["replay_verified"] else "REPLAY_FAILED")
    )
    print(f"[SUBSET MANIFEST] {manifest_path}")
    return manifest["replay_verified"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1.4.0 Native: LIBERO Panda -> Aubo i5 Dataset Converter")

    parser.add_argument(
        "--input_hdf5",
        type=str,
        help=(
            "Exact source HDF5 for a slim single-file conversion. Converts every "
            "demo unless one or more --demo values are supplied."
        ),
    )
    parser.add_argument(
        "--output_hdf5",
        type=str,
        help="Exact target HDF5 for a slim single-file conversion.",
    )
    parser.add_argument(
        "--action_source_hdf5",
        type=str,
        help=(
            "Exact Aubo HDF5 whose stored 7D actions should be replayed; "
            "valid only in exact single-file mode."
        ),
    )
    parser.add_argument(
        "--demo",
        action="append",
        help="Exact demo_N to convert in single-file mode; repeat for more demos.",
    )
    parser.add_argument("--input_dir", type=str, help="Source directory for batch conversion.")
    parser.add_argument("--output_dir", type=str, help="Destination directory for batch or subset conversion.")
    parser.add_argument("--n_trajs", type=int, default=None, help="Limit number of trajectories per HDF5 file")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of HDF5 files to convert")
    parser.add_argument("--render_video", action="store_true", help="Render an MP4 video of the first trajectory")
    parser.add_argument("--video_out", type=str, default="trial_conversion.mp4", help="Path for rendered test video")
    parser.add_argument("--trial", action="store_true", help="Run a quick unit test: 1 file, 1 trajectory, renders video")
    parser.add_argument(
        "--success_report_root",
        type=str,
        default=None,
        help="Build a target-native subset from first-run conversion reports below this root.",
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=3,
        help="Deterministic successful episodes selected per task in subset mode.",
    )
    parser.add_argument("--selection_seed", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic Python, NumPy, and LIBERO target reset seed.",
    )
    parser.add_argument(
        "--subset_smoke",
        action="store_true",
        help="In subset mode, select one successful episode from each suite.",
    )
    parser.add_argument(
        "--expected_success_count",
        type=int,
        default=807,
        help="Fail subset selection unless the first-run reports contain this many successes; use 0 to disable.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Continue verified outputs (the default; retained for explicit scripts).",
    )
    mode_group.add_argument(
        "--restart",
        dest="resume",
        action="store_false",
        help="Discard selected output HDF5 files and convert them again from source.",
    )
    parser.set_defaults(resume=True)

    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative")
    exact_file_mode = args.input_hdf5 is not None or args.output_hdf5 is not None
    if exact_file_mode and (args.input_hdf5 is None or args.output_hdf5 is None):
        parser.error("--input_hdf5 and --output_hdf5 must be supplied together")
    if args.demo and not exact_file_mode:
        parser.error("--demo requires --input_hdf5 and --output_hdf5")
    if args.action_source_hdf5 is not None and not exact_file_mode:
        parser.error(
            "--action_source_hdf5 requires --input_hdf5 and --output_hdf5"
        )
    if exact_file_mode:
        if args.success_report_root is not None or args.trial:
            parser.error(
                "Exact single-file mode cannot be combined with subset or trial mode"
            )
        if args.n_trajs is not None or args.max_files is not None:
            parser.error(
                "Exact single-file mode cannot be combined with --n_trajs or --max_files"
            )
        if Path(args.input_hdf5).resolve() == Path(args.output_hdf5).resolve():
            parser.error("--output_hdf5 must differ from --input_hdf5")
        if (
            args.action_source_hdf5 is not None
            and Path(args.action_source_hdf5).resolve()
            == Path(args.output_hdf5).resolve()
        ):
            parser.error(
                "--output_hdf5 must differ from --action_source_hdf5"
            )
        Path(args.output_hdf5).resolve().parent.mkdir(parents=True, exist_ok=True)
        ok = process_single_file(
            args.input_hdf5,
            args.output_hdf5,
            render_video=args.render_video,
            video_out_path=args.video_out,
            resume=args.resume,
            selected_demos=args.demo or None,
            action_source_path=args.action_source_hdf5,
            slim_output=True,
            seed=args.seed,
        )
        os._exit(0 if ok else 1)

    if args.output_dir is None:
        parser.error("--output_dir is required outside exact single-file mode")

    if args.success_report_root is not None:
        ok = build_first_run_success_subset(
            args.success_report_root,
            args.output_dir,
            episodes_per_task=args.episodes_per_task,
            selection_seed=args.selection_seed,
            smoke=args.subset_smoke,
            expected_success_count=args.expected_success_count,
            resume=args.resume,
            seed=args.seed,
        )
        os._exit(0 if ok else 1)

    if args.input_dir is None:
        parser.error("--input_dir is required in batch mode")

    if args.trial:
        print("\n" + "="*50)
        print("          RUNNING QUICK UNIT TEST TRIAL (1.4.0)")
        print("="*50)
        args.max_files = 1
        args.n_trajs = 1
        args.render_video = True
        args.resume = False

    ok = batch_process(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        n_trajs=args.n_trajs,
        render_video=args.render_video,
        video_out_path=args.video_out,
        resume=args.resume,
        seed=args.seed,
    )

    os._exit(0 if ok else 1)
