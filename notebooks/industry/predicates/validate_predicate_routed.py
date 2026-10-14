import os
import sys
from pathlib import Path
import imageio
import numpy as np
import robosuite.utils.transform_utils as T

# Libero imports for dynamic task generation
from libero.libero.utils.mu_utils import InitialSceneTemplates, register_mu, MU_DICT
from libero.libero.utils.task_generation_utils import register_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

# Ensure the predicates path is available for dynamic import
PREDICATES_PATH = str(
    Path(__file__).resolve().parents[3] / "libero/libero/envs/predicates"
)
if PREDICATES_PATH not in sys.path:
    sys.path.append(PREDICATES_PATH)
from libero.libero.envs.predicates.base_predicates import Routed

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# ===========================================================================
# EASY TOGGLE CONFIGURATION
# ===========================================================================
CONFIG = {
    "task_name": "Test_Routed_Poses",
    "bddl_folder": os.path.join(CURRENT_DIR, "debug_bddl"),
    "video_name": os.path.join(CURRENT_DIR, "cream_jar_routed_poses.mp4"),
    "steps": 40,
    "seed": 42,
    
    # --- NEW SETTINGS FOR TESTING ---
    "belt_axis": "x",         # "x" or "y"
    "groove_yaw": np.pi / 4,  # 45 degrees
    "start_groove_idx": 1,    # Skip the first groove to avoid spawn clashing
    "groove_stride": 1,
    
    "env_args": {
        "robots": ["Panda"],
        "gripper_types": "default",
        "control_freq": 20,
        "horizon": 200,
        "object_sampling": "uniform",
        "camera_heights": 1024,   # High-res video output
        "camera_widths": 1024,
    },
    "set_scene": {
        "camera": {"azimuth": 30, "elevation": 47.7, "distance": 1.2},
        "lighting": {"tone": "neutral", "noise_level": 1.0, "castshadow": "true"},
        "table_style": {"color": "black", "finish": "matte"}
    }
}

# 1. Dynamically define the test scene
@register_mu(scene_type="industry_workbench")
class Scene_Test_Routed_Poses(InitialSceneTemplates):
    def __init__(self):
        super().__init__(
            workspace_name="industry_workbench", 
            fixture_num_info={
                "industry_workbench": 1,
                "conveyor_ghost": 1,
            }, 
            object_num_info={
                "cream_jar_upright": 4,
            }
        )

    def define_regions(self):
        self.regions.update(self.get_region_dict([0.0, 0.0], "ghost_init_region", self.workspace_name, 0.01, (0, 0)))
        self.regions.update(self.get_region_dict([-0.3, 0.0], "dummy_region", self.workspace_name, 0.2, (0, 0)))
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    @property
    def init_states(self):
        return [
            # Spawn conveyor
            ("On", "conveyor_ghost_1", "industry_workbench_ghost_init_region"),
            
            # Load Level 2 grooves with the dynamic belt_axis
            ("LoadSlot", "conveyor_ghost_1", "cream_jar_v_groove", CONFIG["belt_axis"], "pulse", "0.05"),
            
            # Put bottles in a dummy region initially
            ("On", "cream_jar_upright_1", "industry_workbench_dummy_region"),
            ("On", "cream_jar_upright_2", "industry_workbench_dummy_region"),
            ("On", "cream_jar_upright_3", "industry_workbench_dummy_region"),
            ("On", "cream_jar_upright_4", "industry_workbench_dummy_region"),
        ]

MU_DICT[CONFIG["task_name"].lower()] = Scene_Test_Routed_Poses

def run_test():
    os.makedirs(CONFIG["bddl_folder"], exist_ok=True)
    
    register_task_info(
        "test routed predicate poses",
        scene_name=CONFIG["task_name"],
        objects_of_interest=[],
        goal_states=[
            ("Routed", "cream_jar_upright_1", "conveyor_ghost_1"),
            ("Routed", "cream_jar_upright_2", "conveyor_ghost_1"),
            ("Routed", "cream_jar_upright_3", "conveyor_ghost_1"),
            ("Routed", "cream_jar_upright_4", "conveyor_ghost_1")
        ]
    )
    
    bddl_files, fails = generate_bddl_from_task_info(folder=CONFIG["bddl_folder"])
    if fails: raise Exception(f"BDDL Generation Failed: {fails}")

    # Initialize environment WITHOUT filtering out camera dimensions
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_files[0],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        renderer="mujoco",
        set_scene=CONFIG["set_scene"],
        **CONFIG["env_args"]
    )
    
    base_env = env.env
    theta = CONFIG["groove_yaw"]
    base_env.local_groove_quat = [np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)]
    env.seed(CONFIG["seed"])
    obs = env.reset()

    # 2. Setup the Pose Tests and Handle Clashing
    available_grooves = getattr(base_env, 'dynamic_groove_names', []).copy()
    
    # Extract settings from CONFIG
    start_idx = CONFIG["start_groove_idx"]
    stride = CONFIG.get("groove_stride", 1)
    
    if len(available_grooves) > start_idx:
        # Python slicing: [start_index : end_index : step_size]
        # This skips the start index, then pulls every Nth groove based on the stride!
        available_grooves = available_grooves[start_idx::stride]
    else:
        print(f"[WARNING] Not enough grooves to skip {start_idx}.")

    if len(available_grooves) < 4:
        print(f"[ERROR] Not enough grooves left for 4 objects! (Need at least {start_idx + (4 * stride)} total grooves spawned).")
        return

    # Define test configurations (Relative to the groove)
    test_cases = [
        {"obj": "cream_jar_upright_1", "desc": "0 Deg Aligned", "rot_z": 0.0, "rot_x": 0.0},
        {"obj": "cream_jar_upright_2", "desc": "180 Deg Flipped", "rot_z": np.pi, "rot_x": 0.0},
        {"obj": "cream_jar_upright_3", "desc": "90 Deg Perpendicular", "rot_z": np.pi/2, "rot_x": 0.0},
        {"obj": "cream_jar_upright_4", "desc": "90 Deg Upright (Wrong Pose)", "rot_z": 0.0, "rot_x": np.pi/2},
    ]

    print("\n" + "="*50)
    print(f"TELEPORTING TO {CONFIG['belt_axis'].upper()}-AXIS BELT")
    print(f"(Skipping first {start_idx} grooves, placing every {stride} groove(s))")
    print("="*50)

    # 3. Teleport Objects
    for test in test_cases:
        obj_name = test["obj"]
        g_name = available_grooves.pop(0)
        site_name = f"{g_name}_goal_region"
        
        try:
            site_id = env.sim.model.site_name2id(site_name)
            site_pos = env.sim.data.site_xpos[site_id].copy()
            
            # The site_mat naturally rotates 90 degrees when the belt is set to "x"!
            site_mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
            
            # Create local rotation matrix for this specific pose test
            local_rot_z = T.rotation_matrix(angle=test["rot_z"], direction=[0, 0, 1], point=None)[:3, :3]
            local_rot_x = T.rotation_matrix(angle=test["rot_x"], direction=[1, 0, 0], point=None)[:3, :3]
            
            # Combine site rotation with our test rotation
            final_mat = site_mat.dot(local_rot_z).dot(local_rot_x)
            
            # Convert to MuJoCo's wxyz quaternion format
            target_quat_xyzw = T.mat2quat(final_mat)
            target_quat_wxyz = T.convert_quat(target_quat_xyzw, to="wxyz")
            
            obj = base_env.objects_dict[obj_name]
            joint_name = obj.joints[-1]
            joint_qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
            joint_qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
            
            # Calculate drop position using bottom offset
            global_bottom_offset = final_mat.dot(obj.bottom_offset)
            drop_pos = site_pos - global_bottom_offset + np.array([0, 0, 0.05])
            
            # Apply Position and Orientation
            env.sim.data.qpos[joint_qpos_addr[0]: joint_qpos_addr[0] + 3] = drop_pos
            env.sim.data.qpos[joint_qpos_addr[0] + 3: joint_qpos_addr[1]] = target_quat_wxyz
            
            # Clear velocity and match conveyor speed
            env.sim.data.qvel[joint_qvel_addr[0]:joint_qvel_addr[1]] = 0.0
            conv_speed = getattr(base_env, 'conveyor_speed', 0.05)
            forward_vec = getattr(base_env, 'forward_vec', np.array([0, 1, 0]))
            
            # Allow X and Y to match belt, keep Z at 0 to fall
            env.sim.data.qvel[joint_qvel_addr[0]:joint_qvel_addr[0]+2] = (forward_vec * conv_speed)[0:2]

            print(f"[TELEPORT] {obj_name} ({test['desc']}) -> {g_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to teleport {obj_name}: {e}")

    env.sim.forward()

    # 4. Simulate physics
    frames = [obs["agentview_image"][::-1]]
    for _ in range(CONFIG["steps"]):
        obs, _, _, _ = env.step([0.] * 7)
        frames.append(obs["agentview_image"][::-1])

    # 5. Evaluate the Routed Predicate
    print("\n" + "="*50)
    print("ROUTED PREDICATE EVALUATION RESULTS")
    print("="*50)
    
    routed_checker = Routed()
    class MockArg:
        def __init__(self, name, current_env):
            self.object_name = name
            self.env = current_env

    for test in test_cases:
        obj_name = test["obj"]
        arg1 = MockArg(obj_name, base_env)
        arg2 = MockArg("conveyor_ghost_1", base_env)
        
        is_routed = routed_checker(arg1, arg2)
        status = "PASSED" if is_routed else "FAILED"
        print(f"[{status}] {obj_name} | Pose: {test['desc']}")

    # Save video
    imageio.mimsave(CONFIG["video_name"], frames, fps=20)
    png_path = CONFIG["video_name"].replace(".mp4", "_final_frame.png")
    imageio.imwrite(png_path, frames[-1])
    print(f"\n[INFO] Video saved to: {CONFIG['video_name']}")
    print(f"[INFO] Final PNG saved to: {png_path}")
    env.close()

if __name__ == "__main__":
    run_test()
