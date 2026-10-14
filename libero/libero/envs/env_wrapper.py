import os
import json
import numpy as np
import robosuite as suite
import matplotlib.cm as cm

from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
from libero.libero.envs.inject_config import inject_industry_config
from libero.libero.envs.noise_profiles import resolve_robot_noise

_version_parts = suite.__version__.split(".")
IS_V1_5_OR_GREATER = int(_version_parts[0]) >= 1 and int(_version_parts[1]) >= 5
if IS_V1_5_OR_GREATER:
    from robosuite.controllers import load_part_controller_config
    from robosuite.controllers.composite import ALL_COMPOSITE_CONTROLLERS
    # 1. Restore the legacy '.controller' pointer on the base Robot class
    from robosuite.robots.robot import Robot
    if not hasattr(Robot, 'controller'):
        def _get_legacy_controller(self):
            if hasattr(self, 'composite_controller') and self.composite_controller:
                cc = self.composite_controller
                p_dict = getattr(cc, 'part_controllers', getattr(cc, '_controllers', None))
                if p_dict:
                    return p_dict.get("right", list(p_dict.values())[0])
            if hasattr(self, 'part_controllers') and self.part_controllers:
                return self.part_controllers.get("right", list(self.part_controllers.values())[0])
            raise AttributeError("Could not dynamically resolve the 1.5.x arm controller.")
        Robot.controller = property(_get_legacy_controller)

    ## 2. Restore spatial properties on the OperationalSpaceController class
    from robosuite.controllers.parts.arm.osc import OperationalSpaceController
    if not hasattr(OperationalSpaceController, 'ee_ori_mat'):
        def _get_ee_ori_mat(self):
            # In 1.5.2 the controller tracks the current end-effector pose via
            # ref_ori_mat / ref_pos, updated by self.update() each timestep.
            # 1.4.0 already has ee_ori_mat / ee_pos as native properties, so
            # this injection only runs on 1.5.x where they were removed.
            return self.ref_ori_mat

        OperationalSpaceController.ee_ori_mat = property(_get_ee_ori_mat)

    if not hasattr(OperationalSpaceController, 'ee_pos'):
        def _get_ee_pos(self):
            return self.ref_pos
        OperationalSpaceController.ee_pos = property(_get_ee_pos)
    try:
        import gymnasium as gym
    except ImportError:
        pass


def _uses_aubo_robot(robots):
    """Return whether this environment has an Aubo i5 robot specification."""
    if isinstance(robots, str):
        robots = (robots,)
    return any(str(robot).endswith("AuboI5") for robot in robots)
        
class GymnasiumToGymAdapter:
    """
    Proxy adapter to down-convert Gymnasium 5-tuple step returns into legacy Gym 4-tuple returns.
    Decoupled from gym.Wrapper to prevent typing assertion conflicts with native Robosuite classes.
    """
    def __init__(self, env):
        self.env = env

    def step(self, action):
        step_return = self.env.step(action)
        if len(step_return) == 5:
            obs, reward, terminated, truncated, info = step_return
            done = bool(terminated or truncated)
            return obs, reward, done, info
        elif len(step_return) == 4:
            return step_return
        else:
            raise ValueError(f"Unexpected step return length: {len(step_return)}")

    def reset(self, **kwargs):
        reset_return = self.env.reset(**kwargs)
        if isinstance(reset_return, tuple) and len(reset_return) == 2:
            return reset_return[0]
        return reset_return

    def __getattr__(self, attr):
        # 1. Try fetching from the immediate wrapped environment (Legacy compatibility)
        if hasattr(self.env, attr):
            return getattr(self.env, attr)
        # 2. Pierce Gymnasium wrappers to access the core physics base (Modern compatibility)
        elif hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, attr):
            return getattr(self.env.unwrapped, attr)
        else:
            raise AttributeError(f"'{type(self).__name__}' and its wrapped envs have no attribute '{attr}'")
            
class ControlEnv:
    def __init__(
        self,
        bddl_file_name,
        robots=["Panda"],
        controller="OSC_POSE",
        gripper_types="default",
        initialization_noise=None,
        use_camera_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names=[
            "agentview",
            "robot0_eye_in_hand",
        ],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        renderer_config=None,
        robot_action_noise="baseline",
        **kwargs,
    ):
        assert os.path.exists(
            bddl_file_name
        ), f"[error] {bddl_file_name} does not exist!"

        # controller_configs = suite.load_controller_config(default_controller=controller)
        if IS_V1_5_OR_GREATER:
            # 1. Handle string/file inputs
            if isinstance(controller, str):
                if os.path.exists(controller):
                    with open(controller, 'r') as f:
                        controller = json.load(f)
                elif controller in ALL_COMPOSITE_CONTROLLERS:
                    if controller == "BASIC" and _uses_aubo_robot(robots):
                        # The generic BASIC config includes head, torso, and
                        # left-arm entries. MountedAuboI5 has only a right arm.
                        part_config = load_part_controller_config(
                            default_controller="OSC_POSE"
                        )
                        part_config["gripper"] = {"type": "JOINT_POSITION"}
                        controller_configs = {
                            "type": "BASIC",
                            "body_parts": {"right": part_config},
                        }
                    else:
                        controller_configs = {"type": controller}
                else:
                    part_config = load_part_controller_config(default_controller=controller)
                    part_config["gripper"] = {"type": "JOINT_POSITION"}
                    controller_configs = {
                        "type": "BASIC",
                        "body_parts": {"right": part_config}
                    }

            # 2. Intercept legacy 1.4.0 dictionaries and upgrade them
            if isinstance(controller, dict):
                # If it's a flat dictionary (1.4.0 style) and not a composite controller
                if controller.get("type") not in ALL_COMPOSITE_CONTROLLERS and "body_parts" not in controller:
                    import warnings
                    warnings.warn(
                        "Legacy Robosuite 1.4.0 controller dictionary detected. "
                        "Automatically mapping to 1.5.2 Panda default composite controller.",
                        UserWarning,
                        stacklevel=2
                    )
                    # Force the 1.5.2 default Panda composite wrapper around the legacy part config
                    controller["gripper"] = {"type": "JOINT_POSITION"}
                    controller_configs = {
                        "type": "BASIC",
                        "body_parts": {"right": controller}
                    }
                else:
                    # It's already a modern 1.5.2 dictionary
                    controller_configs = controller
        else:
            # Legacy 1.4.0 logic
            controller_configs = suite.load_controller_config(default_controller=controller)

        problem_info = BDDLUtils.get_problem_info(bddl_file_name)
        # Check if we're using a multi-armed environment and use env_configuration argument if so

        # Create environment
        self.problem_name = problem_info["problem_name"]
        self.domain_name = problem_info["domain_name"]
        self.language_instruction = problem_info["language_instruction"]
        
        inject_industry_config(self.problem_name, bddl_file_name, kwargs)

        base_env = TASK_MAPPING[self.problem_name](
            bddl_file_name,
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            **kwargs,
        )
        
        # DUAL-COMPATIBLE ADAPTER WRAPPING
        if IS_V1_5_OR_GREATER:
            self.env = GymnasiumToGymAdapter(base_env)
        else:
            self.env = base_env

        # Resolve robot action noise once; named values are converted from mm to m.
        self._robot_noise_scale = resolve_robot_noise(robot_action_noise)

    @property
    def obj_of_interest(self):
        return self.env.obj_of_interest

    def step(self, action):
        # Apply robot action noise to XYZ translation (first 3 indices of OSC_POSE)
        if self._robot_noise_scale > 0 and action is not None:
            action = action.copy()
            # XYZ translation noise (mm to m conversion handled by scale)
            noise = np.random.normal(0, self._robot_noise_scale, size=3)
            action[:3] += noise

        return self.env.step(action)

    def reset(self):
        success = False
        while not success:
            try:
                ret = self.env.reset()
                success = True
            except RandomizationError:
                pass
            finally:
                continue

        return ret
    
    def get_init_state(self):
        """
        Returns the current MuJoCo simulation state as a flattened numpy array.
        This matches the [5, 92] or [50, N] format seen in .pruned_init files.
        """
        return self.env.sim.get_state().flatten()

    def check_success(self):
        return self.env._check_success()

    @property
    def _visualizations(self):
        return self.env._visualizations

    @property
    def robots(self):
        return self.env.robots

    @property
    def sim(self):
        return self.env.sim

    def get_sim_state(self):
        return self.env.sim.get_state().flatten()

    def _post_process(self):
        return self.env._post_process()

    def _update_observables(self, force=False):
        self.env._update_observables(force=force)

    def set_state(self, mujoco_state):
        self.env.sim.set_state_from_flattened(mujoco_state)

    def reset_from_xml_string(self, xml_string):
        self.env.reset_from_xml_string(xml_string)

    def seed(self, seed):
        # 1. Probe for legacy callable seed method (Robosuite 1.4.0)
        env_seed = getattr(self.env, 'seed', None)
        if callable(env_seed):
            env_seed(seed)
            
        # 2. Modern Gymnasium fallback (Robosuite 1.5.2+)
        elif IS_V1_5_OR_GREATER:
            # Gymnasium delegates seeding to the reset() method and the spaces.
            # We explicitly seed numpy to ensure deterministic physics initialization.
            np.random.seed(seed)
            
            # Conditionally seed action/observation spaces if they exist
            if hasattr(self.env, 'action_space') and hasattr(self.env.action_space, 'seed'):
                self.env.action_space.seed(seed)
            if hasattr(self.env, 'observation_space') and hasattr(self.env.observation_space, 'seed'):
                self.env.observation_space.seed(seed)
        else:
            pass

    def set_init_state(self, init_state):
        return self.regenerate_obs_from_state(init_state)

    def regenerate_obs_from_state(self, mujoco_state):
        self.set_state(mujoco_state)
        self.env.sim.forward()
        self.check_success()
        self._post_process()
        self._update_observables(force=True)
        return self.env._get_observations()

    def close(self):
        self.env.close()
        del self.env


class OffScreenRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # Off-screen environments never open an interactive renderer, but
        # callers still choose whether camera observations require an
        # off-screen context (for example, video previews versus init-state
        # generation).
        kwargs["has_renderer"] = False
        kwargs.setdefault("has_offscreen_renderer", True)
        super().__init__(**kwargs)


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper will additionally generate the segmentation mask of objects,
    which is useful for comparing attention.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.segmentation_robot_id = None
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.segmentation_id_mapping = {}

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name == "Panda0":
                self.segmentation_robot_id = i

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        # get all instances' segmentation separately
        seg_img_dict = {}
        segmentation_image[segmentation_image > self.segmentation_robot_id] = (
            self.segmentation_robot_id + 1
        )
        seg_img_dict["robot"] = segmentation_image * (
            segmentation_image == self.segmentation_robot_id + 1
        )

        for seg_id, instance_name in self.segmentation_id_mapping.items():
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == seg_id + 1
            )
        return seg_img_dict

    def get_segmentation_of_interest(self, segmentation_image):
        # get the combined segmentation of obj of interest
        # 1 for obj_of_interest
        # -1.0 for robot
        # 0 for other things
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        # ret_seg[segmentation_image == self.segmentation_robot_id+1] = -1.0
        ret_seg[segmentation_image == 0] = -1.0
        return ret_seg

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        """
        Helper function to visualize segmentations as RGB frames.
        NOTE: assumes that geom IDs go up to 255 at most - if not,
        multiple geoms might be assigned to the same color.
        """
        # ensure all values lie within [0, 255]
        seg_im = np.mod(seg_im, 256)

        if random_colors:
            colors = randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            # use @inds to map each geom ID to a color
            return seg_img


class DemoRenderEnv(ControlEnv):
    """
    For visualization and evaluation.
    """

    def __init__(self, **kwargs):
        # This shouldn't be customized
        kwargs["has_renderer"] = False
        kwargs["has_offscreen_renderer"] = True
        kwargs["render_camera"] = "frontview"

        super().__init__(**kwargs)

    def _get_observations(self):
        return self.env._get_observations()
